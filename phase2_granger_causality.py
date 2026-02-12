# phase2_granger_causality.py

"""
Phase 2: Driveline Thesis Validation — Granger Causality Analysis

Tests whether the Driveline causal chain flows in the right direction:
  max EV (bat speed proxy) → hard hit % → barrel % → OPS

For each link, tests both directions at lags of 1, 2, and 3 years to answer
Key question: "Does bat speed → hard-hit rate → OPS actually flow causally,
and at what time delay? Not just correlation, but directionality."

Method: Panel Granger causality on pooled player-season data.
  Restricted model:   Y(t) = a + sum[ b_i * Y(t-i) ]       (own lags only)
  Unrestricted model: Y(t) = a + sum[ b_i * Y(t-i) ] + sum[ g_j * X(t-j) ]
  F-test on whether X lags add significant predictive power beyond Y's own lags.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from pathlib import Path

DB_PATH = "maddux_db.db"
OUTPUT_DIR = Path("phase2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PA = 200
MAX_LAG = 3


def load_panel():
    """Load player-season panel data for Granger tests."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT s.player_id, s.year, s.pa,
               s.max_ev, s.hard_hit_pct, s.barrel_pct, s.sprint_speed,
               s.ops, s.iso, s.bb_pct, s.k_pct,
               p.player_name
        FROM player_seasons s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.pa >= ?
        ORDER BY s.player_id, s.year
    """, conn, params=(MIN_PA,))
    conn.close()

    print(f"Loaded {len(df)} player-seasons (PA >= {MIN_PA})")
    print(f"  Players: {df['player_id'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")

    return df


def build_lagged_panel(df, metrics, max_lag=MAX_LAG):
    """Add lagged columns for each metric and year-tracking columns."""
    df = df.sort_values(['player_id', 'year']).copy()

    for metric in metrics:
        for lag in range(1, max_lag + 1):
            df[f'{metric}_lag{lag}'] = df.groupby('player_id')[metric].shift(lag)

    # Year lags to verify consecutive years
    for lag in range(1, max_lag + 1):
        df[f'year_lag{lag}'] = df.groupby('player_id')['year'].shift(lag)

    return df


def granger_test(panel, y_col, x_col, lag):
    """
    Panel Granger causality test: does X Granger-cause Y at a given lag?

    Restricted:   Y(t) = a + b1*Y(t-1) + ... + bL*Y(t-L)
    Unrestricted: Y(t) = a + b1*Y(t-1) + ... + bL*Y(t-L) + g1*X(t-1) + ... + gL*X(t-L)

    F-test: (RSS_r - RSS_u) / q / (RSS_u / (n - k))
    where q = number of X lag terms, k = total params in unrestricted model.
    """
    y_lag_cols = [f'{y_col}_lag{i}' for i in range(1, lag + 1)]
    x_lag_cols = [f'{x_col}_lag{i}' for i in range(1, lag + 1)]

    # Filter to complete cases with consecutive years
    needed = [y_col] + y_lag_cols + x_lag_cols
    clean = panel.dropna(subset=needed).copy()

    for L in range(1, lag + 1):
        clean = clean[clean['year'] - clean[f'year_lag{L}'] == L]

    n = len(clean)
    if n < lag * 4 + 10:
        return None

    Y = clean[y_col].values

    # Restricted model: intercept + Y lags
    X_r_cols = [np.ones(n)] + [clean[c].values for c in y_lag_cols]
    X_r = np.column_stack(X_r_cols)
    beta_r, _, _, _ = np.linalg.lstsq(X_r, Y, rcond=None)
    rss_r = np.sum((Y - X_r @ beta_r) ** 2)

    # Unrestricted model: intercept + Y lags + X lags
    X_u_cols = X_r_cols + [clean[c].values for c in x_lag_cols]
    X_u = np.column_stack(X_u_cols)
    beta_u, _, _, _ = np.linalg.lstsq(X_u, Y, rcond=None)
    rss_u = np.sum((Y - X_u @ beta_u) ** 2)

    # F-test
    q = lag  # number of added X lag terms
    k = X_u.shape[1]
    dfd = n - k

    if dfd <= 0 or rss_u == 0:
        return None

    F_stat = ((rss_r - rss_u) / q) / (rss_u / dfd)
    p_value = f_dist.sf(F_stat, q, dfd)

    return {
        'F_stat': round(F_stat, 4),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'n': n,
        'dfn': q,
        'dfd': dfd,
    }


def run_analysis():
    print("=" * 70)
    print("PHASE 2: DRIVELINE THESIS — GRANGER CAUSALITY ANALYSIS")
    print("=" * 70)

    df = load_panel()

    # Metrics to test
    metrics = ['max_ev', 'hard_hit_pct', 'barrel_pct', 'ops', 'sprint_speed']
    panel = build_lagged_panel(df, metrics)

    # Define the causal chain links to test (forward and reverse)
    causal_links = [
        # Driveline chain: bat speed → contact quality → optimal contact → production
        ('max_ev', 'hard_hit_pct', 'Max EV → Hard Hit %'),
        ('hard_hit_pct', 'max_ev', 'Hard Hit % → Max EV'),

        ('hard_hit_pct', 'barrel_pct', 'Hard Hit % → Barrel %'),
        ('barrel_pct', 'hard_hit_pct', 'Barrel % → Hard Hit %'),

        ('barrel_pct', 'ops', 'Barrel % → OPS'),
        ('ops', 'barrel_pct', 'OPS → Barrel %'),

        # Full chain shortcut
        ('max_ev', 'ops', 'Max EV → OPS'),
        ('ops', 'max_ev', 'OPS → Max EV'),

        # Biomechanics proxy
        ('sprint_speed', 'hard_hit_pct', 'Sprint Speed → Hard Hit %'),
        ('hard_hit_pct', 'sprint_speed', 'Hard Hit % → Sprint Speed'),

        ('sprint_speed', 'ops', 'Sprint Speed → OPS'),
        ('ops', 'sprint_speed', 'OPS → Sprint Speed'),
    ]

    # Run all tests
    results = []
    for x_col, y_col, description in causal_links:
        for lag in range(1, MAX_LAG + 1):
            result = granger_test(panel, y_col, x_col, lag)
            if result:
                results.append({
                    'x_causes_y': description,
                    'x': x_col,
                    'y': y_col,
                    'lag': lag,
                    **result
                })
            else:
                results.append({
                    'x_causes_y': description,
                    'x': x_col,
                    'y': y_col,
                    'lag': lag,
                    'F_stat': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'n': 0,
                    'dfn': lag,
                    'dfd': 0,
                })

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # Print results by causal chain section
    # -----------------------------------------------------------------------

    sections = [
        ("DRIVELINE CHAIN: Bat Speed → Contact → Production", [
            'Max EV → Hard Hit %', 'Hard Hit % → Max EV',
            'Hard Hit % → Barrel %', 'Barrel % → Hard Hit %',
            'Barrel % → OPS', 'OPS → Barrel %',
            'Max EV → OPS', 'OPS → Max EV',
        ]),
        ("BIOMECHANICS PROXY: Sprint Speed as Lower Body Signal", [
            'Sprint Speed → Hard Hit %', 'Hard Hit % → Sprint Speed',
            'Sprint Speed → OPS', 'OPS → Sprint Speed',
        ]),
    ]

    for section_title, link_names in sections:
        print(f"\n{'=' * 70}")
        print(section_title)
        print('=' * 70)
        print(f"{'Direction':<30} {'Lag':>4} {'F-stat':>8} {'p-value':>10} {'Sig':>5} {'N':>6}")
        print("-" * 70)

        for link_name in link_names:
            subset = results_df[results_df['x_causes_y'] == link_name]
            for _, row in subset.iterrows():
                sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                f_str = f"{row['F_stat']:.2f}" if not np.isnan(row['F_stat']) else "N/A"
                p_str = f"{row['p_value']:.6f}" if not np.isnan(row['p_value']) else "N/A"
                print(f"{row['x_causes_y']:<30} {row['lag']:>4} {f_str:>8} {p_str:>10} {sig:>5} {row['n']:>6}")
            # Separator between forward/reverse pairs
            if link_name in ['Hard Hit % → Max EV', 'Barrel % → Hard Hit %',
                             'OPS → Barrel %', 'OPS → Max EV',
                             'Hard Hit % → Sprint Speed', 'OPS → Sprint Speed']:
                print()

    # -----------------------------------------------------------------------
    # Directionality summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DIRECTIONALITY SUMMARY (Lag 1 — most relevant for annual predictions)")
    print('=' * 70)

    pairs = [
        ('Max EV → Hard Hit %', 'Hard Hit % → Max EV', 'Max EV vs Hard Hit %'),
        ('Hard Hit % → Barrel %', 'Barrel % → Hard Hit %', 'Hard Hit % vs Barrel %'),
        ('Barrel % → OPS', 'OPS → Barrel %', 'Barrel % vs OPS'),
        ('Max EV → OPS', 'OPS → Max EV', 'Max EV vs OPS (full chain)'),
        ('Sprint Speed → Hard Hit %', 'Hard Hit % → Sprint Speed', 'Sprint Speed vs Hard Hit %'),
        ('Sprint Speed → OPS', 'OPS → Sprint Speed', 'Sprint Speed vs OPS'),
    ]

    print(f"\n{'Pair':<30} {'Forward':>12} {'Reverse':>12} {'Direction':>15}")
    print("-" * 72)

    for fwd_name, rev_name, pair_label in pairs:
        fwd = results_df[(results_df['x_causes_y'] == fwd_name) & (results_df['lag'] == 1)]
        rev = results_df[(results_df['x_causes_y'] == rev_name) & (results_df['lag'] == 1)]

        fwd_sig = fwd.iloc[0]['significant'] if len(fwd) > 0 else False
        rev_sig = rev.iloc[0]['significant'] if len(rev) > 0 else False
        fwd_p = fwd.iloc[0]['p_value'] if len(fwd) > 0 else np.nan
        rev_p = rev.iloc[0]['p_value'] if len(rev) > 0 else np.nan

        fwd_str = f"p={fwd_p:.4f}" if not np.isnan(fwd_p) else "N/A"
        rev_str = f"p={rev_p:.4f}" if not np.isnan(rev_p) else "N/A"

        if fwd_sig and not rev_sig:
            direction = "FORWARD ONLY"
        elif rev_sig and not fwd_sig:
            direction = "REVERSE ONLY"
        elif fwd_sig and rev_sig:
            direction = "BIDIRECTIONAL"
        else:
            direction = "NEITHER"

        print(f"{pair_label:<30} {fwd_str:>12} {rev_str:>12} {direction:>15}")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    results_df.to_csv(OUTPUT_DIR / "granger_causality_results.csv", index=False)
    print(f"\nResults saved: {OUTPUT_DIR / 'granger_causality_results.csv'}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    generate_charts(results_df)

    return results_df


def generate_charts(results_df):
    """Generate Granger causality visualization."""

    # Chart 1: Driveline chain heatmap — F-statistics by link and lag
    chain_links = [
        'Max EV → Hard Hit %', 'Hard Hit % → Barrel %',
        'Barrel % → OPS', 'Max EV → OPS'
    ]
    reverse_links = [
        'Hard Hit % → Max EV', 'Barrel % → Hard Hit %',
        'OPS → Barrel %', 'OPS → Max EV'
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Granger Causality: Driveline Causal Chain\n"
                 "Does X help predict future Y beyond Y's own history?",
                 fontsize=12, fontweight='bold')

    for ax_idx, (links, title) in enumerate([
        (chain_links, "Forward Direction\n(Driveline thesis)"),
        (reverse_links, "Reverse Direction\n(Should be weaker)")
    ]):
        ax = axes[ax_idx]

        # Build matrix: rows = links, cols = lags
        matrix = np.zeros((len(links), MAX_LAG))
        p_matrix = np.ones((len(links), MAX_LAG))

        for i, link in enumerate(links):
            for lag in range(1, MAX_LAG + 1):
                row = results_df[(results_df['x_causes_y'] == link) & (results_df['lag'] == lag)]
                if len(row) > 0 and not np.isnan(row.iloc[0]['F_stat']):
                    matrix[i, lag - 1] = row.iloc[0]['F_stat']
                    p_matrix[i, lag - 1] = row.iloc[0]['p_value']

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0,
                        vmax=max(matrix.max(), 10))

        ax.set_xticks(range(MAX_LAG))
        ax.set_xticklabels([f'Lag {i+1}' for i in range(MAX_LAG)])
        ax.set_yticks(range(len(links)))
        short_labels = [l.replace(' → ', '\n→ ') for l in links]
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_title(title, fontsize=10)

        # Annotate cells
        for i in range(len(links)):
            for j in range(MAX_LAG):
                f_val = matrix[i, j]
                p_val = p_matrix[i, j]
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                color = 'white' if f_val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f"F={f_val:.1f}\n{sig}",
                        ha='center', va='center', fontsize=8, color=color)

        fig.colorbar(im, ax=ax, label='F-statistic', shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granger_causality_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'granger_causality_heatmap.png'}")

    # Chart 2: Forward vs Reverse comparison (lag 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    pair_labels = ['Max EV vs\nHard Hit %', 'Hard Hit %\nvs Barrel %',
                   'Barrel %\nvs OPS', 'Max EV\nvs OPS']

    fwd_f = []
    rev_f = []
    for fwd, rev in zip(chain_links, reverse_links):
        fwd_row = results_df[(results_df['x_causes_y'] == fwd) & (results_df['lag'] == 1)]
        rev_row = results_df[(results_df['x_causes_y'] == rev) & (results_df['lag'] == 1)]
        fwd_f.append(fwd_row.iloc[0]['F_stat'] if len(fwd_row) > 0 else 0)
        rev_f.append(rev_row.iloc[0]['F_stat'] if len(rev_row) > 0 else 0)

    x = np.arange(len(pair_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, fwd_f, width, label='Forward (Driveline direction)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, rev_f, width, label='Reverse',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_ylabel('F-statistic (higher = stronger Granger causality)')
    ax.set_title('Granger Causality: Forward vs Reverse Direction (Lag 1)\n'
                 'Does the Driveline chain flow in the right direction?',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.axhline(y=3.84, color='gray', linewidth=0.8, linestyle='--', label='p=0.05 approx')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granger_causality_direction.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {OUTPUT_DIR / 'granger_causality_direction.png'}")


if __name__ == "__main__":
    run_analysis()
