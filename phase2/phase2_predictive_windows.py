# phase2_predictive_windows.py

"""
Phase 2: Predictive Window Analysis

Tests whether MADDUX score in Year N predicts OPS change in Year N+1.
Compares 1-year, 2-year, and 3-year rolling windows of MADDUX scores.

Key question: "How many years of metric changes are required
to be predictive of a future year?"
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_db.db")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PA = 200


def load_deltas():
    """Load player deltas with 200+ PA filter."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT d.*, p.player_name
        FROM player_deltas d
        JOIN players p ON d.player_id = p.player_id
        WHERE d.pa >= ?
    """, conn, params=(MIN_PA,))
    conn.close()
    print(f"Loaded {len(df)} delta records (PA >= {MIN_PA})")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Players: {df['player_id'].nunique()}")
    return df


def build_window_datasets(df):
    """
    Build datasets for 1-year, 2-year, and 3-year predictive windows.

    For each window size W:
      - Predictor: average MADDUX score over W consecutive years ending at Year N
      - Target: delta_ops in Year N+1

    Also builds same-year (concurrent) dataset as baseline.
    """
    datasets = {}

    # --- Same-year baseline: MADDUX(N) vs delta_ops(N) ---
    datasets['same_year'] = df[['player_id', 'player_name', 'year',
                                'maddux_score', 'delta_ops']].dropna().copy()
    datasets['same_year'].rename(columns={
        'maddux_score': 'predictor',
        'delta_ops': 'target_delta_ops'
    }, inplace=True)

    # --- 1-year window: MADDUX(N) vs delta_ops(N+1) ---
    # Self-join: match each player's MADDUX at year N with their delta_ops at year N+1
    d1 = df[['player_id', 'player_name', 'year', 'maddux_score',
             'delta_max_ev', 'delta_hard_hit_pct']].copy()
    d1_target = df[['player_id', 'year', 'delta_ops', 'pa']].copy()
    d1_target['year'] = d1_target['year'] - 1  # shift so year N in target matches year N in predictor

    merged_1yr = pd.merge(d1, d1_target, on=['player_id', 'year'],
                          suffixes=('', '_next'))
    # PA filter: target year must also have 200+ PA (already filtered in load,
    # but the target row's PA was checked at load time too)
    merged_1yr = merged_1yr[merged_1yr['pa'] >= MIN_PA].copy()
    merged_1yr.rename(columns={
        'maddux_score': 'predictor',
        'delta_ops': 'target_delta_ops'
    }, inplace=True)
    datasets['1_year'] = merged_1yr

    # --- 2-year window: avg MADDUX(N, N-1) vs delta_ops(N+1) ---
    # Get MADDUX at year N-1 for each player
    d_prev = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev['year'] = d_prev['year'] + 1  # shift so N-1 aligns with N
    d_prev.rename(columns={'maddux_score': 'maddux_prev'}, inplace=True)

    merged_2yr = pd.merge(merged_1yr, d_prev, on=['player_id', 'year'], how='inner')
    merged_2yr['predictor'] = (merged_1yr.loc[merged_2yr.index, 'predictor'] +
                                merged_2yr['maddux_prev']) / 2
    merged_2yr = merged_2yr.drop(columns=['maddux_prev'])
    datasets['2_year'] = merged_2yr.copy()

    # --- 3-year window: avg MADDUX(N, N-1, N-2) vs delta_ops(N+1) ---
    d_prev2 = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev2['year'] = d_prev2['year'] + 2  # shift so N-2 aligns with N
    d_prev2.rename(columns={'maddux_score': 'maddux_prev2'}, inplace=True)

    # Rebuild 2-year merge to keep maddux_prev for 3-year avg
    d_prev_keep = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev_keep['year'] = d_prev_keep['year'] + 1
    d_prev_keep.rename(columns={'maddux_score': 'maddux_prev'}, inplace=True)

    merged_3yr = pd.merge(merged_1yr, d_prev_keep, on=['player_id', 'year'], how='inner')
    merged_3yr = pd.merge(merged_3yr, d_prev2, on=['player_id', 'year'], how='inner')
    # Recalculate predictor from the original 1yr predictor column (which is maddux at year N)
    merged_3yr['predictor'] = (merged_1yr.loc[merged_3yr.index, 'predictor'] +
                                merged_3yr['maddux_prev'] +
                                merged_3yr['maddux_prev2']) / 3
    merged_3yr = merged_3yr.drop(columns=['maddux_prev', 'maddux_prev2'])
    datasets['3_year'] = merged_3yr.copy()

    return datasets


def analyze_window(name, data, threshold=20):
    """Compute correlation, p-value, hit rate for a given window dataset."""
    predictor = data['predictor']
    target = data['target_delta_ops']

    n = len(data)
    r, p = stats.pearsonr(predictor, target)

    # Hit rate: of players with predictor > threshold, what % improved OPS?
    high_score = data[data['predictor'] > threshold]
    if len(high_score) > 0:
        hit_rate = (high_score['target_delta_ops'] > 0).mean()
        hit_n = len(high_score)
    else:
        hit_rate = np.nan
        hit_n = 0

    # Also test multiple thresholds
    threshold_results = {}
    for t in [10, 15, 20, 25]:
        subset = data[data['predictor'] > t]
        if len(subset) >= 10:
            hr = (subset['target_delta_ops'] > 0).mean()
            threshold_results[t] = {'hit_rate': hr, 'n': len(subset)}
        else:
            threshold_results[t] = {'hit_rate': np.nan, 'n': len(subset)}

    return {
        'window': name,
        'n': n,
        'correlation_r': round(r, 4),
        'p_value': round(p, 6),
        'significant': p < 0.05,
        'hit_rate_20': round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
        'hit_rate_20_n': hit_n,
        'thresholds': threshold_results
    }


def run_analysis():
    """Main analysis pipeline."""
    print("=" * 70)
    print("PHASE 2: PREDICTIVE WINDOW ANALYSIS")
    print("=" * 70)

    df = load_deltas()
    datasets = build_window_datasets(df)

    # Analyze each window
    results = []
    for name, data in datasets.items():
        result = analyze_window(name, data)
        results.append(result)

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Window':<15} {'N':>6} {'Corr (r)':>10} {'p-value':>10} {'Sig?':>6} {'Hit Rate':>10} {'(n)':>6}")
    print("-" * 70)
    for r in results:
        sig = "Yes" if r['significant'] else "No"
        hr = f"{r['hit_rate_20']:.1%}" if not np.isnan(r['hit_rate_20']) else "N/A"
        print(f"{r['window']:<15} {r['n']:>6} {r['correlation_r']:>10.4f} {r['p_value']:>10.6f} {sig:>6} {hr:>10} {r['hit_rate_20_n']:>6}")

    # Threshold breakdown
    print("\n" + "-" * 70)
    print("HIT RATE BY THRESHOLD (% of players above threshold who improved OPS)")
    print("-" * 70)
    print(f"{'Window':<15} {'> 10':>12} {'> 15':>12} {'> 20':>12} {'> 25':>12}")
    print("-" * 70)
    for r in results:
        row = f"{r['window']:<15}"
        for t in [10, 15, 20, 25]:
            tr = r['thresholds'][t]
            if not np.isnan(tr['hit_rate']):
                row += f" {tr['hit_rate']:.1%} (n={tr['n']:>3})"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Year-by-year breakdown for 1-year window
    print("\n" + "-" * 70)
    print("1-YEAR WINDOW: YEAR-BY-YEAR BREAKDOWN")
    print("-" * 70)
    d1 = datasets['1_year']
    print(f"{'Pred Year':<12} {'N':>6} {'Corr (r)':>10} {'p-value':>10} {'Hit Rate>20':>12}")
    print("-" * 70)
    for year in sorted(d1['year'].unique()):
        subset = d1[d1['year'] == year]
        if len(subset) < 10:
            continue
        r_val, p_val = stats.pearsonr(subset['predictor'], subset['target_delta_ops'])
        high = subset[subset['predictor'] > 20]
        hr = (high['target_delta_ops'] > 0).mean() if len(high) >= 5 else np.nan
        hr_str = f"{hr:.1%}" if not np.isnan(hr) else "N/A"
        print(f"{year:<12} {len(subset):>6} {r_val:>10.4f} {p_val:>10.4f} {hr_str:>12}")

    # --- Export results CSV ---
    results_df = pd.DataFrame([{
        'window': r['window'],
        'sample_size': r['n'],
        'correlation_r': r['correlation_r'],
        'p_value': r['p_value'],
        'significant': r['significant'],
        'hit_rate_gt20': r['hit_rate_20'],
        'hit_rate_gt20_n': r['hit_rate_20_n'],
    } for r in results])
    results_df.to_csv(OUTPUT_DIR / "predictive_window_results.csv", index=False)
    print(f"\nResults saved: {OUTPUT_DIR / 'predictive_window_results.csv'}")

    # --- Generate charts ---
    generate_charts(datasets, results)

    return results, datasets


def generate_charts(datasets, results):
    """Generate scatter plots and correlation comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("MADDUX Predictive Window Analysis", fontsize=14, fontweight='bold')

    window_labels = {
        'same_year': 'Same Year (Baseline)\nMADDUX(N) vs dOPS(N)',
        '1_year': '1-Year Prediction\nMADDUX(N) vs dOPS(N+1)',
        '2_year': '2-Year Avg Prediction\nAvg MADDUX(N,N-1) vs dOPS(N+1)',
        '3_year': '3-Year Avg Prediction\nAvg MADDUX(N,N-1,N-2) vs dOPS(N+1)',
    }

    for idx, (name, data) in enumerate(datasets.items()):
        ax = axes[idx // 2][idx % 2]
        result = results[idx]

        ax.scatter(data['predictor'], data['target_delta_ops'],
                   alpha=0.15, s=8, color='steelblue')

        # Regression line
        z = np.polyfit(data['predictor'], data['target_delta_ops'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(data['predictor'].min(), data['predictor'].max(), 100)
        ax.plot(x_range, p(x_range), color='red', linewidth=1.5)

        # Reference lines
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

        ax.set_title(window_labels[name], fontsize=10)
        ax.set_xlabel('MADDUX Score (Predictor)')
        ax.set_ylabel('OPS Change (Target)')

        sig_text = f"r = {result['correlation_r']:.4f}, p = {result['p_value']:.4f}"
        hr_text = f"Hit Rate (>20): {result['hit_rate_20']:.1%}" if not np.isnan(result['hit_rate_20']) else ""
        ax.text(0.05, 0.95, f"{sig_text}\nn = {result['n']}\n{hr_text}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictive_windows_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'predictive_windows_scatter.png'}")

    # --- Correlation comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['window'] for r in results]
    corrs = [r['correlation_r'] for r in results]
    colors = ['green' if c > 0 else 'red' for c in corrs]

    bars = ax.bar(names, corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Pearson Correlation (r)')
    ax.set_title('MADDUX Score vs Next-Year OPS Change\nCorrelation by Prediction Window')

    for bar, r in zip(bars, results):
        sig = "*" if r['significant'] else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"r={r['correlation_r']:.3f}{sig}",
                ha='center', va='bottom', fontsize=9)

    ax.set_ylim(min(corrs) - 0.1, max(corrs) + 0.1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictive_windows_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'predictive_windows_correlation.png'}")


if __name__ == "__main__":
    run_analysis()
