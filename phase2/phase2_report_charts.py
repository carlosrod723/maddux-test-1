# phase2_report_charts.py

"""
Phase 2: Publication-quality report charts.

Consistent style across all charts:
  - White background, minimal gridlines (horizontal only, light gray)
  - Title 14pt, axis labels 12pt, annotations 11pt
  - 300 DPI, sized for LaTeX column width
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sqlite3
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_db.db")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MIN_PA = 200

# ── Global style ──────────────────────────────────────────────────────────
STYLE = {
    "title_size": 14,
    "label_size": 12,
    "annot_size": 11,
    "tick_size": 11,
    "dpi": 300,
    "green": "#2ecc71",
    "red": "#e74c3c",
    "gray": "#95a5a6",
}

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
})


# ── Chart 1: Predictive Windows Correlation ───────────────────────────────

def chart_01_predictive_windows():
    df = pd.read_csv(OUTPUT_DIR / "predictive_window_results.csv")

    labels = [
        "Same Year\n(Baseline)",
        "1-Year\nPrediction",
        "2-Year Avg\nPrediction",
        "3-Year Avg\nPrediction",
    ]

    colors = []
    for i, row in df.iterrows():
        if i == 0:
            colors.append(STYLE["gray"])
        elif row["correlation_r"] >= 0:
            colors.append(STYLE["green"])
        else:
            colors.append(STYLE["red"])

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(range(len(df)), df["correlation_r"], color=colors,
                  edgecolor="black", linewidth=0.6, width=0.6)

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Annotations on each bar
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        r = row.correlation_r
        sig = "*" if row.significant else "n.s."
        n = row.sample_size

        # Position annotation above or below bar
        offset = 0.02 if r >= 0 else -0.02
        va = "bottom" if r >= 0 else "top"

        ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                f"r = {r:.3f} {sig}\nn = {n:,}",
                ha="center", va=va, fontsize=STYLE["annot_size"],
                fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=STYLE["tick_size"])
    ax.set_ylabel("Pearson Correlation (r)", fontsize=STYLE["label_size"])
    ax.set_title("MADDUX Score vs Next-Year OPS Change\nby Prediction Window",
                 fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    # Minimal gridlines
    ax.yaxis.grid(True, color="#d5d5d5", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])
    ax.set_ylim(min(df["correlation_r"]) - 0.12, max(df["correlation_r"]) + 0.12)

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_01_predictive_windows.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Data helpers ──────────────────────────────────────────────────────────

def _load_window_datasets():
    """Rebuild the 4 predictive-window datasets from the database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT d.*, p.player_name
        FROM player_deltas d
        JOIN players p ON d.player_id = p.player_id
        WHERE d.pa >= ?
    """, conn, params=(MIN_PA,))
    conn.close()

    datasets = {}

    # Same year: MADDUX(N) vs delta_ops(N)
    same = df[['player_id', 'year', 'maddux_score', 'delta_ops']].dropna().copy()
    same.rename(columns={'maddux_score': 'predictor',
                         'delta_ops': 'target_delta_ops'}, inplace=True)
    datasets['same_year'] = same

    # 1-year: MADDUX(N) vs delta_ops(N+1)
    d1 = df[['player_id', 'year', 'maddux_score']].copy()
    d1_target = df[['player_id', 'year', 'delta_ops', 'pa']].copy()
    d1_target['year'] = d1_target['year'] - 1
    merged = pd.merge(d1, d1_target, on=['player_id', 'year'], how='inner')
    merged = merged[merged['pa'] >= MIN_PA]
    merged.rename(columns={'maddux_score': 'predictor',
                           'delta_ops': 'target_delta_ops'}, inplace=True)
    datasets['1_year'] = merged

    # 2-year: avg MADDUX(N, N-1) vs delta_ops(N+1)
    d_prev = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev['year'] = d_prev['year'] + 1
    d_prev.rename(columns={'maddux_score': 'maddux_prev'}, inplace=True)
    m2 = pd.merge(merged, d_prev, on=['player_id', 'year'], how='inner')
    m2['predictor'] = (merged.loc[m2.index, 'predictor'] + m2['maddux_prev']) / 2
    m2 = m2.drop(columns=['maddux_prev'])
    datasets['2_year'] = m2.copy()

    # 3-year: avg MADDUX(N, N-1, N-2) vs delta_ops(N+1)
    d_prev_keep = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev_keep['year'] = d_prev_keep['year'] + 1
    d_prev_keep.rename(columns={'maddux_score': 'maddux_prev'}, inplace=True)
    d_prev2 = df[['player_id', 'year', 'maddux_score']].copy()
    d_prev2['year'] = d_prev2['year'] + 2
    d_prev2.rename(columns={'maddux_score': 'maddux_prev2'}, inplace=True)
    m3 = pd.merge(merged, d_prev_keep, on=['player_id', 'year'], how='inner')
    m3 = pd.merge(m3, d_prev2, on=['player_id', 'year'], how='inner')
    m3['predictor'] = (merged.loc[m3.index, 'predictor'] +
                       m3['maddux_prev'] + m3['maddux_prev2']) / 3
    m3 = m3.drop(columns=['maddux_prev', 'maddux_prev2'])
    datasets['3_year'] = m3.copy()

    return datasets


# ── Chart 2: Predictive Windows Scatter ───────────────────────────────────

def chart_02_predictive_scatter():
    datasets = _load_window_datasets()
    results = pd.read_csv(OUTPUT_DIR / "predictive_window_results.csv")

    subtitles = {
        'same_year': "Same Year (Baseline)\nMADDUX Score vs OPS Change (Same Year)",
        '1_year':    "1-Year Prediction\nMADDUX Score vs Next-Year OPS Change",
        '2_year':    "2-Year Avg Prediction\nAvg MADDUX (2 Years) vs Next-Year OPS Change",
        '3_year':    "3-Year Avg Prediction\nAvg MADDUX (3 Years) vs Next-Year OPS Change",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("MADDUX Predictive Window Analysis",
                 fontsize=STYLE["title_size"], fontweight="bold")

    for idx, (name, data) in enumerate(datasets.items()):
        ax = axes[idx // 2][idx % 2]
        row = results.iloc[idx]

        # Scatter
        ax.scatter(data['predictor'], data['target_delta_ops'],
                   alpha=0.15, s=8, color='steelblue')

        # Regression line
        z = np.polyfit(data['predictor'], data['target_delta_ops'], 1)
        poly = np.poly1d(z)
        x_range = np.linspace(data['predictor'].min(),
                              data['predictor'].max(), 100)
        ax.plot(x_range, poly(x_range), color=STYLE["red"], linewidth=1.5)

        # Reference lines
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

        ax.set_title(subtitles[name], fontsize=STYLE["annot_size"])
        ax.set_xlabel('MADDUX Score', fontsize=10)
        ax.set_ylabel('OPS Change', fontsize=10)

        # Stat box
        r_val = row['correlation_r']
        p_val = row['p_value']
        n = int(row['sample_size'])
        sig = "*" if row['significant'] else " n.s."

        # Hit rate with small-sample guard
        hr_n = int(row['hit_rate_gt20_n'])
        if hr_n < 5 or pd.isna(row['hit_rate_gt20']):
            hr_text = "Hit Rate: N/A (n<5)"
        else:
            hr_text = f"Hit Rate (>20): {row['hit_rate_gt20']:.1%} (n={hr_n})"

        box_text = f"r = {r_val:.4f}{sig}\nn = {n:,}\n{hr_text}"
        ax.text(0.05, 0.95, box_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_02_predictive_scatter.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Chart 3: Alternative Formulations Comparison ─────────────────────────

def chart_03_alternative_formulations():
    df = pd.read_csv(OUTPUT_DIR / "alternative_formulations_results.csv")

    # Test-set rows only
    df = df[df["dataset"] == "test"].copy()

    # Map formulation names to clean labels (sorted by r, low → high)
    label_map = {
        "2. Barrel Rate Delta":       "Barrel Rate\nDelta",
        "1. Original MADDUX":         "Original\nMADDUX",
        "4. Abs Levels (test)":       "Absolute\nLevels",
        "3. OLS Deltas (test)":       "OLS Optimized\nDeltas",
        "6. Mean Reversion":          "Mean Reversion\nBaseline",
        "5. Underperf Gap raw (test)":"Underperformance\nGap (Raw)",
        "7. Combined (test)":         "Combined Model\n(MR + Tools)",
    }

    df["label"] = df["formulation"].map(label_map)
    df = df.sort_values("correlation_r").reset_index(drop=True)

    colors = [STYLE["red"] if r < 0 else STYLE["green"]
              for r in df["correlation_r"]]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(range(len(df)), df["correlation_r"], color=colors,
                  edgecolor="black", linewidth=0.6, width=0.6)

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Annotations
    for bar, row in zip(bars, df.itertuples()):
        r = row.correlation_r
        sig = "*" if row.significant else " n.s."
        offset = 0.015 if r >= 0 else -0.015
        va = "bottom" if r >= 0 else "top"

        ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                f"r = {r:.3f}{sig}\nn = {row.n:,}",
                ha="center", va=va, fontsize=STYLE["annot_size"],
                fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontsize=STYLE["annot_size"])
    ax.set_ylabel("Pearson Correlation (r)", fontsize=STYLE["label_size"])
    ax.set_title(
        "Alternative MADDUX Formulations: Out-of-Sample Test\n"
        "Predictor (Year N) vs OPS Change (Year N+1)",
        fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    # Minimal gridlines
    ax.yaxis.grid(True, color="#d5d5d5", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])
    r_vals = df["correlation_r"]
    ax.set_ylim(r_vals.min() - 0.1, r_vals.max() + 0.1)

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_03_alternative_formulations.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Chart 4: Granger Causality Heatmap ────────────────────────────────────

def chart_04_granger_heatmap():
    results_df = pd.read_csv(OUTPUT_DIR / "granger_causality_results.csv")
    MAX_LAG = 3

    chain_links = [
        'Max EV → Hard Hit %', 'Hard Hit % → Barrel %',
        'Barrel % → OPS', 'Max EV → OPS',
    ]
    reverse_links = [
        'Hard Hit % → Max EV', 'Barrel % → Hard Hit %',
        'OPS → Barrel %', 'OPS → Max EV',
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        "Granger Causality: Driveline Causal Chain\n"
        "Does X help predict future Y beyond Y's own history?",
        fontsize=STYLE["title_size"], fontweight="bold")

    for ax_idx, (links, title) in enumerate([
        (chain_links, "Forward Direction\n(Driveline thesis)"),
        (reverse_links, "Reverse Direction\n(Should be weaker)"),
    ]):
        ax = axes[ax_idx]

        matrix = np.zeros((len(links), MAX_LAG))
        p_matrix = np.ones((len(links), MAX_LAG))

        for i, link in enumerate(links):
            for lag in range(1, MAX_LAG + 1):
                row = results_df[
                    (results_df['x_causes_y'] == link) &
                    (results_df['lag'] == lag)
                ]
                if len(row) > 0 and not np.isnan(row.iloc[0]['F_stat']):
                    matrix[i, lag - 1] = row.iloc[0]['F_stat']
                    p_matrix[i, lag - 1] = row.iloc[0]['p_value']

        vmax = max(matrix.max(), 10)
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=vmax)

        ax.set_xticks(range(MAX_LAG))
        ax.set_xticklabels([f'Lag {i+1}' for i in range(MAX_LAG)],
                           fontsize=STYLE["annot_size"])
        ax.set_yticks(range(len(links)))

        # Clean arrow labels (no em dashes)
        short_labels = [l.replace(' → ', '\n-> ') for l in links]
        ax.set_yticklabels(short_labels, fontsize=STYLE["annot_size"])
        ax.set_title(title, fontsize=STYLE["label_size"], pad=10)

        # Annotate cells with F-stat and significance stars
        for i in range(len(links)):
            for j in range(MAX_LAG):
                f_val = matrix[i, j]
                p_val = p_matrix[i, j]
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""
                color = 'white' if f_val > vmax * 0.6 else 'black'
                ax.text(j, i, f"F={f_val:.1f}\n{sig}",
                        ha='center', va='center',
                        fontsize=STYLE["annot_size"] - 1, color=color,
                        fontweight='bold')

        fig.colorbar(im, ax=ax, label='F-statistic', shrink=0.7, pad=0.02)

    # Callout annotation on the smoking gun: OPS -> Max EV at lag 1
    # That cell is row 3 (OPS -> Max EV), col 0 (Lag 1) in the right panel
    ax_rev = axes[1]
    ax_rev.annotate(
        "Not significant\nChain flows\nforward only",
        xy=(0, 3), xycoords='data',
        xytext=(1.0, 3.65), textcoords='data',
        fontsize=10, fontweight='bold', color=STYLE["red"],
        ha='center',
        arrowprops=dict(arrowstyle='->', color=STYLE["red"], lw=2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3f3',
                  edgecolor=STYLE["red"], linewidth=1.5),
    )

    # Significance legend below the figure
    fig.text(0.5, 0.01,
             "* p < 0.05    ** p < 0.01    *** p < 0.001",
             ha='center', fontsize=STYLE["annot_size"],
             style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = OUTPUT_DIR / "report_chart_04_granger_heatmap.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Chart 5: Granger Causality Direction (Forward vs Reverse) ─────────────

def chart_05_granger_direction():
    results_df = pd.read_csv(OUTPUT_DIR / "granger_causality_results.csv")

    chain_links = [
        'Max EV → Hard Hit %', 'Hard Hit % → Barrel %',
        'Barrel % → OPS', 'Max EV → OPS',
    ]
    reverse_links = [
        'Hard Hit % → Max EV', 'Barrel % → Hard Hit %',
        'OPS → Barrel %', 'OPS → Max EV',
    ]

    pair_labels = [
        "Max EV\n-> Hard Hit %",
        "Hard Hit %\n-> Barrel %",
        "Barrel %\n-> OPS",
        "Max EV -> OPS\n(full chain)",
    ]

    def _get_lag1(link):
        row = results_df[
            (results_df['x_causes_y'] == link) & (results_df['lag'] == 1)
        ]
        if len(row) == 0:
            return 0, 1.0
        return row.iloc[0]['F_stat'], row.iloc[0]['p_value']

    def _sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    fwd_f, fwd_p = zip(*[_get_lag1(l) for l in chain_links])
    rev_f, rev_p = zip(*[_get_lag1(l) for l in reverse_links])

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(pair_labels))
    width = 0.35

    bars_fwd = ax.bar(x - width / 2, fwd_f, width,
                       label='Forward (Driveline direction)',
                       color=STYLE["green"], alpha=0.85,
                       edgecolor='black', linewidth=0.6)
    bars_rev = ax.bar(x + width / 2, rev_f, width,
                       label='Reverse',
                       color=STYLE["red"], alpha=0.85,
                       edgecolor='black', linewidth=0.6)

    # Annotate forward bars
    for bar, f, p in zip(bars_fwd, fwd_f, fwd_p):
        sig = _sig_stars(p)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"F={f:.1f}\n{sig}",
                ha='center', va='bottom',
                fontsize=STYLE["annot_size"], fontweight='bold')

    # Annotate reverse bars
    for i, (bar, f, p) in enumerate(zip(bars_rev, rev_f, rev_p)):
        sig = _sig_stars(p)
        # Money shot: OPS -> Max EV (index 3) — simple red text above bar
        if i == 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"F={f:.1f}\nn.s. (p=0.20)",
                    ha='center', va='bottom',
                    fontsize=STYLE["annot_size"], fontweight='bold',
                    color=STYLE["red"])
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"F={f:.1f}\n{sig}",
                    ha='center', va='bottom',
                    fontsize=STYLE["annot_size"], fontweight='bold')

    # Significance threshold line with label in open space
    ax.axhline(y=3.84, color='gray', linewidth=1.0, linestyle='--')
    ax.text(0.25, 3.84 + 3,
            "F critical (p = 0.05)",
            fontsize=STYLE["annot_size"], color='#555555', ha='left')

    # Y-axis headroom so F=140.8 label doesn't clip
    ax.set_ylim(0, 165)
    ax.set_xlim(-0.5, len(pair_labels) - 0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=STYLE["annot_size"])
    ax.set_ylabel('F-statistic (higher = stronger Granger causality)',
                  fontsize=STYLE["label_size"])
    ax.set_title(
        "Granger Causality: Forward vs Reverse Direction (Lag 1)\n"
        "Does the Driveline chain flow in the right direction?",
        fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    ax.legend(fontsize=STYLE["annot_size"], loc='upper right')

    # Minimal gridlines
    ax.yaxis.grid(True, color="#d5d5d5", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_05_granger_direction.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Chart 6: Validation Decomposition ─────────────────────────────────────

def chart_06_validation_decomposition():
    val = pd.read_csv(OUTPUT_DIR / "gap_validation_results.csv")

    # Pull exact values from CSV
    r_tools = val.loc[val["test"] == "Tools component", "r"].values[0]
    r_mr = val.loc[val["test"] == "Mean reversion", "r"].values[0]
    r_gap = val.loc[val["test"] == "Original gap", "r"].values[0]
    r_comb = val.loc[val["test"] == "Combined model", "r"].values[0]

    p_tools = val.loc[val["test"] == "Tools component", "p"].values[0]
    p_mr = val.loc[val["test"] == "Mean reversion", "p"].values[0]
    p_gap = val.loc[val["test"] == "Original gap", "p"].values[0]
    p_comb = val.loc[val["test"] == "Combined model", "p"].values[0]

    # Bars sorted by r value (low to high)
    labels = [
        "Tools Signal\nAlone",
        "Mean Reversion\nAlone",
        "Underperformance\nGap (Raw)",
        "Combined Model\n(MR + Tools)",
    ]
    r_vals = [r_tools, r_mr, r_gap, r_comb]
    p_vals = [p_tools, p_mr, p_gap, p_comb]
    colors = [STYLE["red"] if r < 0 else STYLE["green"] for r in r_vals]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.bar(range(len(labels)), r_vals, color=colors,
                  edgecolor="black", linewidth=0.6, width=0.6)

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Annotate bars
    for bar, r, p in zip(bars, r_vals, p_vals):
        sig = "*" if p < 0.05 else " n.s."
        offset = 0.015 if r >= 0 else -0.015
        va = "bottom" if r >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                f"r = {r:.3f}{sig}",
                ha="center", va=va, fontsize=STYLE["annot_size"],
                fontweight="bold")

    # Mean reversion baseline dashed line
    ax.axhline(y=r_mr, color="#888888", linewidth=1.0, linestyle="--")
    ax.text(0.0, r_mr + 0.012,
            "Mean reversion baseline",
            fontsize=10, color="#666666", ha="center")

    # Bracket showing tools contribution between MR (bar 1) and Combined (bar 3)
    x_mr = 1
    x_comb = 3
    bracket_x = x_comb + 0.38
    increment = r_comb - r_mr

    # Vertical line of bracket
    ax.plot([bracket_x, bracket_x], [r_mr, r_comb],
            color="#333333", linewidth=1.5, solid_capstyle="butt")
    # Top tick
    ax.plot([bracket_x - 0.05, bracket_x], [r_comb, r_comb],
            color="#333333", linewidth=1.5)
    # Bottom tick
    ax.plot([bracket_x - 0.05, bracket_x], [r_mr, r_mr],
            color="#333333", linewidth=1.5)
    # Label
    ax.text(bracket_x + 0.06, (r_mr + r_comb) / 2,
            f"Tools\ncontribution:\n+{increment:.3f}",
            fontsize=10, fontweight="bold", va="center", ha="left",
            color="#333333")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=STYLE["annot_size"])
    ax.set_ylabel("Pearson Correlation (r)", fontsize=STYLE["label_size"])
    ax.set_title(
        "Validation: What Drives the Underperformance Gap Model?\n"
        "Decomposing r = 0.513 into Mean Reversion vs Physical Tools",
        fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    # Minimal gridlines
    ax.yaxis.grid(True, color="#d5d5d5", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=STYLE["tick_size"])
    ax.set_ylim(min(r_vals) - 0.12, max(r_vals) + 0.1)
    ax.set_xlim(-0.5, len(labels) - 0.15)

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_06_validation_decomposition.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Chart 7: Driveline Causal Flow Diagram ────────────────────────────────

def chart_07_causal_flow():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.set_title(
        "Driveline Causal Chain: Validated via Granger Causality (Lag 1)",
        fontsize=STYLE["title_size"], fontweight="bold", pad=16)

    # Node layout: evenly spaced across 14 units
    node_w, node_h = 2.2, 1.0
    node_y = 4.0
    node_xs = [1.75, 5.25, 8.75, 12.25]
    node_labels = ["Max EV", "Hard Hit %", "Barrel %", "OPS"]

    node_edge = "#2c7a8e"

    # Draw nodes: white fill, teal border
    for cx, label in zip(node_xs, node_labels):
        box = FancyBboxPatch(
            (cx - node_w / 2, node_y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.15",
            facecolor="white", edgecolor=node_edge, linewidth=2)
        ax.add_patch(box)
        ax.text(cx, node_y, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a4a54")

    # Chain arrows between adjacent nodes
    arrow_color = "#444444"

    chain_arrows = [
        (0, 1, "F = 140.8***"),
        (1, 2, "F = 71.9***"),
        (2, 3, "F = 45.6***"),
    ]

    for i_from, i_to, label in chain_arrows:
        x0 = node_xs[i_from] + node_w / 2 + 0.1
        x1 = node_xs[i_to] - node_w / 2 - 0.1

        ax.annotate(
            "", xy=(x1, node_y), xytext=(x0, node_y),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color,
                linewidth=2, mutation_scale=18))

        # F-stat label centered above arrow
        ax.text((x0 + x1) / 2, node_y + 0.7, label,
                ha="center", va="bottom",
                fontsize=STYLE["label_size"], fontweight="bold",
                color="black")

    # Full chain: curved dashed arrow below nodes from Max EV to OPS
    x_start = node_xs[0]
    x_end = node_xs[3]

    ax.annotate(
        "", xy=(x_end, node_y - node_h / 2 - 0.15), xycoords="data",
        xytext=(x_start, node_y - node_h / 2 - 0.15), textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>", color="#2c7a8e",
            linewidth=2, linestyle="--",
            connectionstyle="arc3,rad=0.35", mutation_scale=18))

    # Label centered below the arc
    ax.text((x_start + x_end) / 2, 1.15,
            "Full chain:  F = 87.4*** (forward)   vs   F = 1.6 n.s. (reverse)",
            ha="center", va="top",
            fontsize=STYLE["annot_size"], fontweight="bold",
            color="#2c7a8e")

    plt.tight_layout()
    out = OUTPUT_DIR / "report_chart_07_causal_flow.png"
    plt.savefig(out, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    chart_01_predictive_windows()
    chart_02_predictive_scatter()
    chart_03_alternative_formulations()
    chart_04_granger_heatmap()
    chart_05_granger_direction()
    chart_06_validation_decomposition()
    chart_07_causal_flow()
