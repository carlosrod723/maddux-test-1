# phase2_validate_gap_model.py

"""
Phase 2: Validation — Is the Underperformance Gap Real Signal or Mean Reversion?

The underperformance gap (expected OPS - actual OPS) showed r = 0.513 on the
test set for predicting next-year delta OPS. Before reporting this, we need
to confirm it's not just a fancy way of measuring OPS regression to the mean.

The concern: gap = expected_OPS(N) - actual_OPS(N), target = OPS(N+1) - OPS(N).
Both contain -OPS(N), creating a mechanical correlation. If OPS(N) is low,
the gap is large AND delta_OPS(N+1) tends to be positive (mean reversion).

Tests:
  1. Baseline: career_mean_OPS - current_OPS → delta_OPS(N+1)
     (pure mean reversion, no physical metrics)
  2. Residual: after controlling for mean reversion, does the physical
     gap add ANY additional predictive power?
  3. Data leakage audit: verify no Year N+1 data in Year N features
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from pathlib import Path

DB_PATH = "maddux_db.db"
OUTPUT_DIR = Path("phase2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PA = 200
TRAIN_MAX_YEAR = 2021


def load_data():
    conn = sqlite3.connect(DB_PATH)

    seasons = pd.read_sql_query("""
        SELECT s.*, p.player_name
        FROM player_seasons s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.pa >= ?
        ORDER BY s.player_id, s.year
    """, conn, params=(MIN_PA,))

    deltas = pd.read_sql_query("""
        SELECT d.*, p.player_name
        FROM player_deltas d
        JOIN players p ON d.player_id = p.player_id
        WHERE d.pa >= ?
    """, conn, params=(MIN_PA,))

    conn.close()
    return seasons, deltas


def ols_fit(X, y):
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def ols_predict(X, beta):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta


def build_dataset(seasons, deltas):
    """
    Build prediction dataset with career mean OPS (no future leakage).

    For each player-year N:
      - career_mean_ops: average OPS from all years <= N for that player
      - ops: actual OPS in year N
      - physical metrics at year N (for expected OPS model)
      - target: delta_ops at year N+1
    """
    # Compute expanding career mean OPS (only past + current, no future)
    seasons = seasons.sort_values(['player_id', 'year'])
    seasons['career_mean_ops'] = (
        seasons.groupby('player_id')['ops']
        .expanding().mean().reset_index(level=0, drop=True)
    )

    # Count seasons played up to this point
    seasons['seasons_played'] = (
        seasons.groupby('player_id').cumcount() + 1
    )

    # Target: delta_ops at Year N+1
    targets = deltas[['player_id', 'year', 'delta_ops']].copy()
    targets['feature_year'] = targets['year'] - 1
    targets = targets.rename(columns={'delta_ops': 'target_delta_ops'})
    targets = targets[['player_id', 'feature_year', 'target_delta_ops']]

    # Merge seasons (features at year N) with targets (delta_ops at N+1)
    dataset = pd.merge(
        seasons,
        targets,
        left_on=['player_id', 'year'],
        right_on=['player_id', 'feature_year'],
        how='inner'
    )

    # Require at least 2 seasons for a meaningful career mean
    dataset = dataset[dataset['seasons_played'] >= 2]

    print(f"Dataset: {len(dataset)} rows")
    print(f"  Feature years: {dataset['year'].min()}-{dataset['year'].max()}")
    print(f"  Players: {dataset['player_id'].nunique()}")

    return dataset


def run_validation():
    print("=" * 70)
    print("VALIDATION: IS THE UNDERPERFORMANCE GAP REAL OR MEAN REVERSION?")
    print("=" * 70)

    seasons, deltas = load_data()
    data = build_dataset(seasons, deltas)

    # Split train/test (same split as alternative formulations)
    train = data[data['year'] <= TRAIN_MAX_YEAR].copy()
    test = data[data['year'] > TRAIN_MAX_YEAR].copy()
    print(f"  Train: {len(train)}, Test: {len(test)}")

    physical_cols = ['barrel_pct', 'hard_hit_pct', 'max_ev', 'bb_pct', 'k_pct']

    # =======================================================================
    # TEST 0: Reproduce the original gap model (sanity check)
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST 0: Reproduce Original Underperformance Gap")
    print("=" * 70)

    train_c = train.dropna(subset=physical_cols + ['ops', 'target_delta_ops'])
    test_c = test.dropna(subset=physical_cols + ['ops', 'target_delta_ops'])

    # Train expected OPS model
    beta_ops = ols_fit(train_c[physical_cols].values, train_c['ops'].values)
    expected_ops_test = ols_predict(test_c[physical_cols].values, beta_ops)
    gap_test = expected_ops_test - test_c['ops'].values

    r_gap, p_gap = stats.pearsonr(gap_test, test_c['target_delta_ops'].values)
    print(f"  Gap → delta_OPS(N+1): r = {r_gap:.4f}, p = {p_gap:.6f}")
    print(f"  (Reproducing our r = 0.513 finding)")

    # =======================================================================
    # TEST 1: Pure Mean Reversion Baseline
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Pure Mean Reversion (NO physical metrics)")
    print("=" * 70)
    print("  Baseline: career_mean_OPS - current_OPS → delta_OPS(N+1)")

    test_mr = test.dropna(subset=['career_mean_ops', 'ops', 'target_delta_ops'])

    mean_reversion = test_mr['career_mean_ops'].values - test_mr['ops'].values
    r_mr, p_mr = stats.pearsonr(mean_reversion, test_mr['target_delta_ops'].values)
    print(f"  Mean reversion → delta_OPS(N+1): r = {r_mr:.4f}, p = {p_mr:.6f}")
    print(f"  N = {len(test_mr)}")

    # Also test simple negative OPS correlation
    r_neg_ops, p_neg_ops = stats.pearsonr(
        -test_mr['ops'].values, test_mr['target_delta_ops'].values
    )
    print(f"\n  Simpler: -current_OPS → delta_OPS(N+1): r = {r_neg_ops:.4f}, p = {p_neg_ops:.6f}")
    print(f"  (This is pure 'bad year → bounce back')")

    # =======================================================================
    # TEST 2: Residual Test — Does the Physical Gap Add Anything?
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Physical Gap AFTER Controlling for Mean Reversion")
    print("=" * 70)

    # Use test set rows that have all required data
    test_resid = test.dropna(
        subset=physical_cols + ['ops', 'career_mean_ops', 'target_delta_ops']
    )

    # Compute expected OPS from physical metrics (using training model)
    expected_ops_resid = ols_predict(test_resid[physical_cols].values, beta_ops)

    # Decompose: gap = (expected - mean) + (mean - actual)
    #   mean_reversion_component = career_mean_ops - actual_ops
    #   tools_above_mean = expected_ops - career_mean_ops
    mr_component = test_resid['career_mean_ops'].values - test_resid['ops'].values
    tools_component = expected_ops_resid - test_resid['career_mean_ops'].values
    target = test_resid['target_delta_ops'].values

    print("\n  Decomposition: gap = (expected - career_mean) + (career_mean - actual)")
    print("  Testing each component's correlation with delta_OPS(N+1):")

    r_mr2, p_mr2 = stats.pearsonr(mr_component, target)
    r_tools, p_tools = stats.pearsonr(tools_component, target)
    print(f"    Mean reversion (career_mean - actual): r = {r_mr2:.4f}, p = {p_mr2:.6f}")
    print(f"    Tools signal   (expected - career_mean): r = {r_tools:.4f}, p = {p_tools:.6f}")

    # Multiple regression: delta_OPS = b1*(mean-actual) + b2*(expected-mean) + intercept
    X_both = np.column_stack([mr_component, tools_component])
    beta_both = ols_fit(X_both, target)
    pred_both = ols_predict(X_both, beta_both)
    r_both, p_both = stats.pearsonr(pred_both, target)

    print(f"\n  Combined model:")
    print(f"    delta_OPS(N+1) = {beta_both[0]:.4f} + {beta_both[1]:.4f}*(mean-actual) + {beta_both[2]:.4f}*(expected-mean)")
    print(f"    Combined r = {r_both:.4f}")
    print(f"    N = {len(test_resid)}")

    # Test significance of tools_component coefficient via t-test
    residuals = target - pred_both
    X_aug = np.column_stack([np.ones(len(X_both)), X_both])
    mse = np.sum(residuals ** 2) / (len(target) - X_aug.shape[1])
    var_beta = mse * np.linalg.inv(X_aug.T @ X_aug)
    se_tools = np.sqrt(var_beta[2, 2])
    t_tools = beta_both[2] / se_tools
    p_tools_coeff = 2 * (1 - stats.t.cdf(abs(t_tools), len(target) - X_aug.shape[1]))

    print(f"\n  Tools coefficient significance:")
    print(f"    beta = {beta_both[2]:.4f}, SE = {se_tools:.4f}, t = {t_tools:.2f}, p = {p_tools_coeff:.6f}")
    print(f"    {'SIGNIFICANT' if p_tools_coeff < 0.05 else 'NOT SIGNIFICANT'} at p < 0.05")

    # =======================================================================
    # TEST 3: Data Leakage Audit
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Data Leakage Audit")
    print("=" * 70)

    print("\n  Checking: Does Year N+1 data leak into Year N features?")
    print()

    # Check 1: Physical metrics are from Year N only
    sample = test_resid.iloc[0]
    print(f"  Sample row: {sample['player_name']}, feature year {int(sample['year'])}")
    print(f"    barrel_pct = {sample['barrel_pct']} (Year {int(sample['year'])})")
    print(f"    ops        = {sample['ops']} (Year {int(sample['year'])})")
    print(f"    target     = delta_OPS (Year {int(sample['year'])+1})")
    print(f"    career_mean_ops includes years up to {int(sample['year'])}")
    print()

    # Check 2: The expected OPS model was trained on years <= TRAIN_MAX_YEAR
    print(f"  Expected OPS model trained on feature years <= {TRAIN_MAX_YEAR}")
    print(f"  Test set feature years: {test_resid['year'].min()}-{test_resid['year'].max()}")
    print(f"  No overlap: {'PASS' if test_resid['year'].min() > TRAIN_MAX_YEAR else 'FAIL'}")
    print()

    # Check 3: Career mean uses only past data (expanding window, not full career)
    # Verify by checking a specific player
    player_check = seasons[seasons['player_id'] == sample['player_id']].sort_values('year')
    if len(player_check) > 2:
        # Career mean at year N should be mean of all OPS values up to N
        year_n = int(sample['year'])
        past_ops = player_check[player_check['year'] <= year_n]['ops']
        computed_mean = past_ops.mean()
        stored_mean = sample['career_mean_ops']
        match = abs(computed_mean - stored_mean) < 0.001
        print(f"  Career mean verification for {sample['player_name']}:")
        print(f"    Computed (years <= {year_n}): {computed_mean:.4f}")
        print(f"    Stored in dataset:           {stored_mean:.4f}")
        print(f"    Match: {'PASS' if match else 'FAIL'}")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
  Original gap model:              r = {r_gap:.4f}
  Pure mean reversion baseline:    r = {r_mr:.4f}
  Simple -OPS baseline:            r = {r_neg_ops:.4f}
  Tools signal (expected - mean):  r = {r_tools:.4f}, p = {p_tools_coeff:.6f}

  INTERPRETATION:
""")

    if abs(r_mr) > abs(r_gap) * 0.9:
        print("  WARNING: Mean reversion explains most of the gap model's signal.")
        print("  The r = 0.513 is largely OPS regression to the mean, not physical tools.")
    elif p_tools_coeff < 0.05:
        print("  The physical tools component IS significant after controlling for")
        print("  mean reversion. The gap model captures real signal beyond just")
        print("  'bad year → bounce back'.")
        print(f"  However, mean reversion alone gives r = {r_mr:.4f}.")
        print(f"  The physical tools add: r = {r_both:.4f} - {abs(r_mr):.4f} = {abs(r_both) - abs(r_mr):.4f} incremental.")
    else:
        print("  The physical tools component is NOT significant after controlling")
        print("  for mean reversion. The gap model is essentially just mean reversion")
        print("  with extra steps.")

    # Export results
    results = {
        'test': ['Original gap', 'Mean reversion', '-OPS baseline',
                 'Tools component', 'Combined model'],
        'r': [r_gap, r_mr, r_neg_ops, r_tools, r_both],
        'p': [p_gap, p_mr, p_neg_ops, p_tools_coeff, p_both],
        'n': [len(test_c), len(test_mr), len(test_mr), len(test_resid), len(test_resid)],
    }
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "gap_validation_results.csv", index=False)
    print(f"\n  Results saved: {OUTPUT_DIR / 'gap_validation_results.csv'}")


if __name__ == "__main__":
    run_validation()
