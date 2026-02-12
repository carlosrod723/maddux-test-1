# phase2_alternative_formulations.py

"""
Phase 2: Alternative MADDUX Formulations

Tests seven approaches to predicting next-year OPS change:
  1. Original MADDUX (baseline) — delta-based, known to fail
  2. Barrel Rate Delta — stickier metric, still delta-based
  3. OLS Optimized Deltas — let data find best weights (train/test split)
  4. Absolute Levels — physical tools predict future OPS change
  5. Underperformance Gap (raw) — expected OPS minus actual OPS
  6. Mean Reversion Baseline — career mean OPS minus current OPS (no tools)
  7. Combined Model — mean reversion + physical tools (recommended)

All formulations evaluated the same way:
  Predictor(Year N) → delta_ops(Year N+1)
  Metrics: Pearson r, p-value, hit rate (top quartile), sample size

Train/test split: fit on 2016-2021, test on 2022-2024

Note: Formulations 6-7 were added after validation revealed that #5's r=0.50
was largely driven by OPS mean reversion. The combined model (#7) is the
honest, corrected version that separates mean reversion from physical tools.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DB_PATH = "maddux_db.db"
OUTPUT_DIR = Path("phase2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PA = 200
TRAIN_MAX_YEAR = 2021  # feature year <= 2021 for training, >= 2022 for test


def load_data():
    """Load deltas and seasons tables, filtered to 200+ PA."""
    conn = sqlite3.connect(DB_PATH)

    deltas = pd.read_sql_query("""
        SELECT d.*, p.player_name
        FROM player_deltas d
        JOIN players p ON d.player_id = p.player_id
        WHERE d.pa >= ?
    """, conn, params=(MIN_PA,))

    seasons = pd.read_sql_query("""
        SELECT s.*
        FROM player_seasons s
        WHERE s.pa >= ?
    """, conn, params=(MIN_PA,))

    conn.close()
    return deltas, seasons


def build_prediction_dataset(deltas, seasons):
    """
    Build dataset mapping features at Year N to delta_ops at Year N+1.

    Each row contains:
      - Delta features from player_deltas at year N
      - Level features from player_seasons at year N
      - Target: delta_ops from player_deltas at year N+1
    """
    # Target: delta_ops at Year N+1
    # In deltas, delta_ops at row year=Y means OPS(Y) - OPS(Y-1)
    # We want to predict this from features at year Y-1
    targets = deltas[['player_id', 'year', 'delta_ops']].copy()
    targets['feature_year'] = targets['year'] - 1
    targets = targets.rename(columns={'delta_ops': 'target_delta_ops'})
    targets = targets[['player_id', 'feature_year', 'target_delta_ops']]

    # Delta features at feature_year
    delta_features = deltas[[
        'player_id', 'year', 'player_name',
        'delta_max_ev', 'delta_hard_hit_pct', 'delta_barrel_pct',
        'delta_sprint_speed', 'delta_iso', 'delta_bb_pct', 'delta_k_pct',
        'maddux_score'
    ]].copy()
    delta_features = delta_features.rename(columns={'year': 'feature_year'})

    # Level features at feature_year (with career mean OPS for mean reversion)
    seasons_sorted = seasons.sort_values(['player_id', 'year'])
    seasons_sorted['career_mean_ops'] = (
        seasons_sorted.groupby('player_id')['ops']
        .expanding().mean().reset_index(level=0, drop=True)
    )
    seasons_sorted['seasons_played'] = (
        seasons_sorted.groupby('player_id').cumcount() + 1
    )
    level_features = seasons_sorted[[
        'player_id', 'year',
        'barrel_pct', 'hard_hit_pct', 'max_ev',
        'bb_pct', 'k_pct', 'iso', 'sprint_speed', 'ops',
        'career_mean_ops', 'seasons_played'
    ]].copy()
    level_features = level_features.rename(columns={'year': 'feature_year'})

    # Merge: delta features + targets
    pred = pd.merge(delta_features, targets, on=['player_id', 'feature_year'], how='inner')

    # Merge: + level features
    pred = pd.merge(pred, level_features, on=['player_id', 'feature_year'], how='inner')

    # Train/test split
    train = pred[pred['feature_year'] <= TRAIN_MAX_YEAR].copy()
    test = pred[pred['feature_year'] > TRAIN_MAX_YEAR].copy()

    print(f"Prediction dataset: {len(pred)} total rows")
    print(f"  Train (feature year <= {TRAIN_MAX_YEAR}): {len(train)} rows")
    print(f"  Test  (feature year >  {TRAIN_MAX_YEAR}): {len(test)} rows")
    print(f"  Feature years: {pred['feature_year'].min()}-{pred['feature_year'].max()}")
    print(f"  Unique players: {pred['player_id'].nunique()}")

    return pred, train, test


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

def ols_fit(X, y):
    """Fit OLS with intercept. Returns coefficient vector [intercept, b1, b2, ...]."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def ols_predict(X, beta):
    """Predict using OLS coefficients."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta


def r_squared(y_true, y_pred):
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(predictor, target, label=""):
    """Compute correlation, p-value, hit rates for a predictor vs target."""
    mask = predictor.notna() & target.notna()
    pred = predictor[mask].values
    tgt = target[mask].values

    if len(pred) < 10:
        return {
            'label': label, 'n': len(pred),
            'r': np.nan, 'p': np.nan, 'significant': False,
            'hit_rate_top25': np.nan, 'hit_rate_top25_n': 0,
            'hit_rate_pos': np.nan, 'hit_rate_pos_n': 0,
        }

    r, p = stats.pearsonr(pred, tgt)

    # Hit rate: top quartile of predictor — what % improved OPS?
    q75 = np.percentile(pred, 75)
    top_quarter = tgt[pred >= q75]
    hr_top25 = (top_quarter > 0).mean() if len(top_quarter) > 0 else np.nan

    # Hit rate: positive predictor — what % improved OPS?
    positive = tgt[pred > 0]
    hr_pos = (positive > 0).mean() if len(positive) > 0 else np.nan

    return {
        'label': label,
        'n': len(pred),
        'r': round(r, 4),
        'p': round(p, 6),
        'significant': p < 0.05,
        'hit_rate_top25': round(hr_top25, 4) if not np.isnan(hr_top25) else np.nan,
        'hit_rate_top25_n': int(np.sum(pred >= q75)),
        'hit_rate_pos': round(hr_pos, 4) if not np.isnan(hr_pos) else np.nan,
        'hit_rate_pos_n': int(np.sum(pred > 0)),
    }


# ---------------------------------------------------------------------------
# Formulations
# ---------------------------------------------------------------------------

def formulation_1_original_maddux(test):
    """Original MADDUX: delta_max_ev + 2.1 * delta_hard_hit_pct."""
    return (
        test['maddux_score'],
        test['target_delta_ops'],
        evaluate(test['maddux_score'], test['target_delta_ops'], "1. Original MADDUX"),
    )


def formulation_2_barrel_delta(test):
    """Barrel rate delta as sole predictor."""
    return (
        test['delta_barrel_pct'],
        test['target_delta_ops'],
        evaluate(test['delta_barrel_pct'], test['target_delta_ops'], "2. Barrel Rate Delta"),
    )


def formulation_3_ols_deltas(train, test):
    """OLS-optimized delta weights. Trained on train set."""
    feature_cols = [
        'delta_max_ev', 'delta_hard_hit_pct', 'delta_barrel_pct',
        'delta_bb_pct', 'delta_k_pct',
    ]

    train_c = train.dropna(subset=feature_cols + ['target_delta_ops'])
    test_c = test.dropna(subset=feature_cols + ['target_delta_ops'])

    X_train = train_c[feature_cols].values
    y_train = train_c['target_delta_ops'].values
    X_test = test_c[feature_cols].values

    beta = ols_fit(X_train, y_train)

    pred_train = pd.Series(ols_predict(X_train, beta), index=train_c.index)
    pred_test = pd.Series(ols_predict(X_test, beta), index=test_c.index)

    print("\n  --- Formulation 3: OLS Optimized Delta Weights ---")
    print(f"  {'Feature':<25} {'Coefficient':>12}")
    print("  " + "-" * 38)
    print(f"  {'(intercept)':<25} {beta[0]:>12.6f}")
    for col, coeff in zip(feature_cols, beta[1:]):
        print(f"  {col:<25} {coeff:>12.6f}")

    train_r2 = r_squared(y_train, pred_train.values)
    print(f"  Train R²: {train_r2:.4f}")

    result_train = evaluate(pred_train, train_c['target_delta_ops'], "3. OLS Deltas (train)")
    result_test = evaluate(pred_test, test_c['target_delta_ops'], "3. OLS Deltas (test)")

    return pred_test, test_c['target_delta_ops'], result_train, result_test


def formulation_4_absolute_levels(train, test):
    """Absolute physical levels predict next-year delta_ops."""
    feature_cols = ['barrel_pct', 'hard_hit_pct', 'max_ev', 'bb_pct', 'k_pct']

    train_c = train.dropna(subset=feature_cols + ['target_delta_ops'])
    test_c = test.dropna(subset=feature_cols + ['target_delta_ops'])

    X_train = train_c[feature_cols].values
    y_train = train_c['target_delta_ops'].values
    X_test = test_c[feature_cols].values

    beta = ols_fit(X_train, y_train)

    pred_train = pd.Series(ols_predict(X_train, beta), index=train_c.index)
    pred_test = pd.Series(ols_predict(X_test, beta), index=test_c.index)

    print("\n  --- Formulation 4: Absolute Levels ---")
    print(f"  {'Feature':<25} {'Coefficient':>12}")
    print("  " + "-" * 38)
    print(f"  {'(intercept)':<25} {beta[0]:>12.6f}")
    for col, coeff in zip(feature_cols, beta[1:]):
        print(f"  {col:<25} {coeff:>12.6f}")

    train_r2 = r_squared(y_train, pred_train.values)
    print(f"  Train R²: {train_r2:.4f}")

    result_train = evaluate(pred_train, train_c['target_delta_ops'], "4. Abs Levels (train)")
    result_test = evaluate(pred_test, test_c['target_delta_ops'], "4. Abs Levels (test)")

    return pred_test, test_c['target_delta_ops'], result_train, result_test


def formulation_5_underperformance_gap(train, test):
    """
    Underperformance gap: expected OPS from physical tools minus actual OPS.

    Step 1: Train OPS = f(barrel_pct, hard_hit_pct, max_ev, bb_pct, k_pct)
    Step 2: gap = expected_OPS - actual_OPS
    Step 3: Test whether gap(Year N) predicts delta_ops(Year N+1)
    """
    physical_cols = ['barrel_pct', 'hard_hit_pct', 'max_ev', 'bb_pct', 'k_pct']

    train_c = train.dropna(subset=physical_cols + ['ops', 'target_delta_ops'])
    test_c = test.dropna(subset=physical_cols + ['ops', 'target_delta_ops'])

    # Step 1: Fit expected OPS model on training data
    X_train = train_c[physical_cols].values
    y_train_ops = train_c['ops'].values

    beta_ops = ols_fit(X_train, y_train_ops)

    expected_ops_train = ols_predict(X_train, beta_ops)
    expected_ops_test = ols_predict(test_c[physical_cols].values, beta_ops)

    ops_r2 = r_squared(y_train_ops, expected_ops_train)

    print("\n  --- Formulation 5: Underperformance Gap ---")
    print("  Expected OPS model (physical metrics -> OPS):")
    print(f"  {'Feature':<25} {'Coefficient':>12}")
    print("  " + "-" * 38)
    print(f"  {'(intercept)':<25} {beta_ops[0]:>12.6f}")
    for col, coeff in zip(physical_cols, beta_ops[1:]):
        print(f"  {col:<25} {coeff:>12.6f}")
    print(f"  Expected OPS model R²: {ops_r2:.4f}")

    # Step 2: Compute gap
    gap_train = pd.Series(expected_ops_train - train_c['ops'].values, index=train_c.index)
    gap_test = pd.Series(expected_ops_test - test_c['ops'].values, index=test_c.index)

    print(f"  Train gap range: [{gap_train.min():.3f}, {gap_train.max():.3f}]")
    print(f"  Test  gap range: [{gap_test.min():.3f}, {gap_test.max():.3f}]")

    # Step 3: Test gap -> next-year delta_ops
    result_train = evaluate(gap_train, train_c['target_delta_ops'], "5. Underperf Gap raw (train)")
    result_test = evaluate(gap_test, test_c['target_delta_ops'], "5. Underperf Gap raw (test)")

    return gap_test, test_c['target_delta_ops'], result_train, result_test, beta_ops


def formulation_6_mean_reversion(test):
    """
    Mean reversion baseline: career_mean_OPS - current_OPS.
    No physical metrics. Pure 'bad year → bounce back' signal.
    Requires at least 2 seasons for a meaningful career mean.
    """
    test_c = test.dropna(subset=['career_mean_ops', 'ops', 'target_delta_ops'])
    test_c = test_c[test_c['seasons_played'] >= 2]

    mr = pd.Series(
        test_c['career_mean_ops'].values - test_c['ops'].values,
        index=test_c.index
    )

    print(f"\n  --- Formulation 6: Mean Reversion Baseline ---")
    print(f"  career_mean_OPS - current_OPS (no physical metrics)")
    print(f"  N = {len(test_c)} (players with >= 2 seasons)")

    result = evaluate(mr, test_c['target_delta_ops'], "6. Mean Reversion")
    return mr, test_c['target_delta_ops'], result


def formulation_7_combined(train, test, beta_ops):
    """
    Combined model: mean reversion + physical tools signal.

    Decomposes the underperformance gap into:
      - mean_reversion = career_mean_OPS - actual_OPS
      - tools_signal = expected_OPS (from physical metrics) - career_mean_OPS
    Fits weights on train set, evaluates on test set.
    """
    physical_cols = ['barrel_pct', 'hard_hit_pct', 'max_ev', 'bb_pct', 'k_pct']

    train_c = train.dropna(subset=physical_cols + ['ops', 'career_mean_ops', 'target_delta_ops'])
    train_c = train_c[train_c['seasons_played'] >= 2]
    test_c = test.dropna(subset=physical_cols + ['ops', 'career_mean_ops', 'target_delta_ops'])
    test_c = test_c[test_c['seasons_played'] >= 2]

    # Compute components using the expected OPS model trained earlier
    exp_train = ols_predict(train_c[physical_cols].values, beta_ops)
    exp_test = ols_predict(test_c[physical_cols].values, beta_ops)

    mr_train = train_c['career_mean_ops'].values - train_c['ops'].values
    tools_train = exp_train - train_c['career_mean_ops'].values

    mr_test = test_c['career_mean_ops'].values - test_c['ops'].values
    tools_test = exp_test - test_c['career_mean_ops'].values

    # Fit combined weights on training data
    X_train = np.column_stack([mr_train, tools_train])
    y_train = train_c['target_delta_ops'].values
    beta_comb = ols_fit(X_train, y_train)

    X_test = np.column_stack([mr_test, tools_test])
    pred_train = pd.Series(ols_predict(X_train, beta_comb), index=train_c.index)
    pred_test = pd.Series(ols_predict(X_test, beta_comb), index=test_c.index)

    print(f"\n  --- Formulation 7: Combined (Mean Reversion + Tools) ---")
    print(f"  delta_OPS(N+1) = {beta_comb[0]:.4f} + {beta_comb[1]:.4f}*(mean-actual) + {beta_comb[2]:.4f}*(expected-mean)")
    print(f"  Train N = {len(train_c)}, Test N = {len(test_c)}")

    result_train = evaluate(pred_train, train_c['target_delta_ops'], "7. Combined (train)")
    result_test = evaluate(pred_test, test_c['target_delta_ops'], "7. Combined (test)")

    return pred_test, test_c['target_delta_ops'], result_train, result_test


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(scatter_data, all_test_results):
    """Generate comparison bar chart and scatter plots."""

    # --- Bar chart: all formulations compared on test set ---
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r['label'] for r in all_test_results]
    corrs = [r['r'] for r in all_test_results]
    pvals = [r['p'] for r in all_test_results]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corrs]

    bars = ax.bar(range(len(labels)), corrs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=11)
    ax.set_title('Alternative MADDUX Formulations — Out-of-Sample Test\n'
                 'Predictor(Year N) vs delta OPS(Year N+1)',
                 fontsize=12, fontweight='bold')

    for bar, r_val, p_val in zip(bars, corrs, pvals):
        sig = "*" if p_val < 0.05 else ""
        offset = 0.008 if r_val >= 0 else -0.008
        va = 'bottom' if r_val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, r_val + offset,
                f"r={r_val:.3f}{sig}", ha='center', va=va, fontsize=9, fontweight='bold')

    ymin = min(corrs) - 0.08
    ymax = max(corrs) + 0.08
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alternative_formulations_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'alternative_formulations_comparison.png'}")

    # --- Scatter plots for each formulation ---
    n_plots = len(scatter_data)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    fig.suptitle("Alternative Formulations — Test Set Scatter Plots\n"
                 "Predictor(Year N) vs delta OPS(Year N+1)",
                 fontsize=13, fontweight='bold')

    if rows == 1:
        axes = [axes]

    for idx, (title, x_data, y_data, result) in enumerate(scatter_data):
        ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]

        mask = x_data.notna() & y_data.notna()
        x_clean = x_data[mask].values
        y_clean = y_data[mask].values

        ax.scatter(x_clean, y_clean, alpha=0.2, s=10, color='steelblue')

        # Regression line
        if len(x_clean) > 2:
            z = np.polyfit(x_clean, y_clean, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(x_range, p_line(x_range), color='red', linewidth=1.5)

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predictor Score', fontsize=9)
        ax.set_ylabel('Next-Year dOPS', fontsize=9)

        info = f"r = {result['r']:.4f}\np = {result['p']:.4f}\nn = {result['n']}"
        ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Hide unused subplots
    for idx in range(len(scatter_data), rows * cols):
        ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alternative_formulations_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'alternative_formulations_scatter.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis():
    print("=" * 70)
    print("PHASE 2: ALTERNATIVE MADDUX FORMULATIONS")
    print("=" * 70)

    deltas, seasons = load_data()
    full, train, test = build_prediction_dataset(deltas, seasons)

    all_test_results = []
    all_train_results = []
    scatter_data = []  # (title, x_series, y_series, result)

    # -----------------------------------------------------------------------
    # 1. Original MADDUX
    # -----------------------------------------------------------------------
    print("\n  --- Formulation 1: Original MADDUX (baseline) ---")
    x1, y1, r1 = formulation_1_original_maddux(test)
    all_test_results.append(r1)
    scatter_data.append(("1. Original MADDUX", x1, y1, r1))

    # -----------------------------------------------------------------------
    # 2. Barrel Rate Delta
    # -----------------------------------------------------------------------
    print("\n  --- Formulation 2: Barrel Rate Delta ---")
    x2, y2, r2 = formulation_2_barrel_delta(test)
    all_test_results.append(r2)
    scatter_data.append(("2. Barrel Rate Delta", x2, y2, r2))

    # -----------------------------------------------------------------------
    # 3. OLS Optimized Deltas (train/test)
    # -----------------------------------------------------------------------
    x3, y3, r3_train, r3_test = formulation_3_ols_deltas(train, test)
    all_test_results.append(r3_test)
    all_train_results.append(r3_train)
    scatter_data.append(("3. OLS Optimized Deltas", x3, y3, r3_test))

    # -----------------------------------------------------------------------
    # 4. Absolute Levels (train/test)
    # -----------------------------------------------------------------------
    x4, y4, r4_train, r4_test = formulation_4_absolute_levels(train, test)
    all_test_results.append(r4_test)
    all_train_results.append(r4_train)
    scatter_data.append(("4. Absolute Levels", x4, y4, r4_test))

    # -----------------------------------------------------------------------
    # 5. Underperformance Gap — raw (train/test)
    # -----------------------------------------------------------------------
    x5, y5, r5_train, r5_test, beta_ops = formulation_5_underperformance_gap(train, test)
    all_test_results.append(r5_test)
    all_train_results.append(r5_train)
    scatter_data.append(("5. Underperf Gap (raw)", x5, y5, r5_test))

    # -----------------------------------------------------------------------
    # 6. Mean Reversion Baseline (no physical metrics)
    # -----------------------------------------------------------------------
    x6, y6, r6 = formulation_6_mean_reversion(test)
    all_test_results.append(r6)
    scatter_data.append(("6. Mean Reversion", x6, y6, r6))

    # -----------------------------------------------------------------------
    # 7. Combined: Mean Reversion + Physical Tools (recommended)
    # -----------------------------------------------------------------------
    x7, y7, r7_train, r7_test = formulation_7_combined(train, test, beta_ops)
    all_test_results.append(r7_test)
    all_train_results.append(r7_train)
    scatter_data.append(("7. MR + Tools (recommended)", x7, y7, r7_test))

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TEST SET RESULTS  (feature year > {TRAIN_MAX_YEAR} — out-of-sample)")
    print("=" * 70)
    header = (f"{'Formulation':<28} {'N':>5} {'r':>8} {'p-value':>10} "
              f"{'Sig':>5} {'HR Top25%':>10} {'HR Pos':>8}")
    print(header)
    print("-" * 80)
    for r in all_test_results:
        sig = "Yes" if r['significant'] else "No"
        hr25 = f"{r['hit_rate_top25']:.1%}" if not np.isnan(r.get('hit_rate_top25', np.nan)) else "N/A"
        hrp = f"{r['hit_rate_pos']:.1%}" if not np.isnan(r.get('hit_rate_pos', np.nan)) else "N/A"
        print(f"{r['label']:<28} {r['n']:>5} {r['r']:>8.4f} {r['p']:>10.6f} "
              f"{sig:>5} {hr25:>10} {hrp:>8}")

    if all_train_results:
        print("\n" + "-" * 70)
        print("TRAINING SET RESULTS (for reference — NOT the validation metric)")
        print("-" * 70)
        print(f"{'Formulation':<28} {'N':>5} {'r':>8} {'p-value':>10}")
        print("-" * 55)
        for r in all_train_results:
            print(f"{r['label']:<28} {r['n']:>5} {r['r']:>8.4f} {r['p']:>10.6f}")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    export_rows = []
    for r in all_test_results:
        export_rows.append({
            'formulation': r['label'],
            'dataset': 'test',
            'n': r['n'],
            'correlation_r': r['r'],
            'p_value': r['p'],
            'significant': r['significant'],
            'hit_rate_top25': r['hit_rate_top25'],
            'hit_rate_top25_n': r['hit_rate_top25_n'],
            'hit_rate_positive': r['hit_rate_pos'],
            'hit_rate_positive_n': r['hit_rate_pos_n'],
        })
    for r in all_train_results:
        export_rows.append({
            'formulation': r['label'],
            'dataset': 'train',
            'n': r['n'],
            'correlation_r': r['r'],
            'p_value': r['p'],
            'significant': r['significant'],
            'hit_rate_top25': r['hit_rate_top25'],
            'hit_rate_top25_n': r['hit_rate_top25_n'],
            'hit_rate_positive': r['hit_rate_pos'],
            'hit_rate_positive_n': r['hit_rate_pos_n'],
        })

    pd.DataFrame(export_rows).to_csv(
        OUTPUT_DIR / "alternative_formulations_results.csv", index=False
    )
    print(f"\nResults saved: {OUTPUT_DIR / 'alternative_formulations_results.csv'}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    generate_charts(scatter_data, all_test_results)

    return all_test_results


if __name__ == "__main__":
    run_analysis()
