"""
Phase 3, Item 2: Bat Speed vs Max Exit Velocity Correlation Analysis
=====================================================================
Pulls bat speed data from Baseball Savant's bat tracking leaderboard
for 2024 and 2025, joins against max_ev from maddux_db.db, and runs
Pearson correlation analysis.

Outputs:
  - bat_speed_vs_max_ev.png (scatter plot, 300 DPI)
  - bat_speed_correlation_results.csv
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_db.db")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
YEARS = [2024, 2025]


def pull_bat_speed(year):
    """Pull bat tracking leaderboard from Baseball Savant for a given year."""
    print(f"  Pulling bat speed data for {year}...")
    url = (
        "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?attackZone=&batSide=&contactType=&count=&dateStart={year}-01-01"
        f"&dateEnd={year}-12-31&gameType=&isHardHit=&minSwings=1"
        f"&minGroupSwings=1&pitchHand=&pitchType=&seasonStart={year}"
        f"&seasonEnd={year}&team=&type=batter&csv=true"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    print(f"    Raw columns: {list(df.columns)}")
    print(f"    {len(df)} players returned")

    # Keep player ID, name, and avg bat speed
    df = df[["id", "name", "avg_bat_speed"]].copy()
    df = df.rename(columns={"id": "player_id", "name": "player_name"})
    df["year"] = year

    # Drop rows with missing bat speed
    df = df.dropna(subset=["avg_bat_speed"])
    print(f"    {len(df)} players with valid bat speed")
    return df


def load_max_ev(years):
    """Load max_ev from player_seasons for the given years."""
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join(["?"] * len(years))
    query = f"""
        SELECT ps.player_id, p.player_name, ps.year, ps.max_ev, ps.pa
        FROM player_seasons ps
        JOIN players p ON ps.player_id = p.player_id
        WHERE ps.year IN ({placeholders})
        AND ps.max_ev IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=years)
    conn.close()
    print(f"  Loaded {len(df)} player-seasons with max_ev from DB")
    return df


def run_correlation(df, label):
    """Run Pearson correlation and return results dict."""
    r, p = stats.pearsonr(df["avg_bat_speed"], df["max_ev"])
    n = len(df)
    return {
        "subset": label,
        "n": n,
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "r_squared": round(r ** 2, 4),
        "significant": "Yes" if p < 0.05 else "No",
    }


def make_scatter(df, results):
    """Generate scatter plot: bat speed (x) vs max EV (y), color by year."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {2024: "#2196F3", 2025: "#FF5722"}
    labels = {2024: "2024", 2025: "2025"}

    for year in sorted(df["year"].unique()):
        subset = df[df["year"] == year]
        ax.scatter(
            subset["avg_bat_speed"],
            subset["max_ev"],
            c=colors.get(year, "#999"),
            label=labels.get(year, str(year)),
            alpha=0.5,
            s=30,
            edgecolors="white",
            linewidth=0.3,
        )

    # Combined regression line
    slope, intercept, r, p, se = stats.linregress(
        df["avg_bat_speed"], df["max_ev"]
    )
    x_range = np.linspace(df["avg_bat_speed"].min(), df["avg_bat_speed"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, color="#333", linewidth=2, zorder=5)

    # Annotation box
    combined = [r for r in results if r["subset"] == "Combined (2024-2025)"][0]
    r2024 = [r for r in results if r["subset"] == "2024"][0]
    r2025 = [r for r in results if r["subset"] == "2025"][0]

    annotation = (
        f"Combined: r = {combined['pearson_r']:.3f}, n = {combined['n']}\n"
        f"2024: r = {r2024['pearson_r']:.3f}, n = {r2024['n']}\n"
        f"2025: r = {r2025['pearson_r']:.3f}, n = {r2025['n']}"
    )
    ax.annotate(
        annotation,
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )

    ax.set_xlabel("Average Bat Speed (mph)", fontsize=13)
    ax.set_ylabel("Max Exit Velocity (mph)", fontsize=13)
    ax.set_title(
        "Bat Speed vs Max Exit Velocity (2024-2025)\nCan Max EV Proxy for Bat Speed in Historical Data?",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bat_speed_vs_max_ev.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR / 'bat_speed_vs_max_ev.png'}")


def main():
    print("=" * 60)
    print("Phase 3: Bat Speed vs Max Exit Velocity Correlation")
    print("=" * 60)

    # 1. Pull bat speed from Savant
    print("\n[1] Pulling bat speed from Baseball Savant...")
    bat_frames = []
    for year in YEARS:
        bat_frames.append(pull_bat_speed(year))
    bat_speed = pd.concat(bat_frames, ignore_index=True)
    print(f"  Total bat speed records: {len(bat_speed)}")

    # 2. Load max EV from database
    print("\n[2] Loading max EV from maddux_db.db...")
    max_ev = load_max_ev(YEARS)

    # 3. Join on player_id + year
    print("\n[3] Joining datasets...")
    merged = bat_speed.merge(
        max_ev[["player_id", "year", "max_ev", "pa"]],
        on=["player_id", "year"],
        how="inner",
    )
    print(f"  Matched: {len(merged)} player-seasons")
    for year in YEARS:
        n = len(merged[merged["year"] == year])
        print(f"    {year}: {n} players")

    # 4. Run correlations
    print("\n[4] Running Pearson correlations...")
    results = []
    for year in YEARS:
        subset = merged[merged["year"] == year]
        if len(subset) > 2:
            r = run_correlation(subset, str(year))
            results.append(r)
            print(f"    {year}: r = {r['pearson_r']}, n = {r['n']}, p = {r['p_value']}")

    combined_r = run_correlation(merged, "Combined (2024-2025)")
    results.append(combined_r)
    print(f"    Combined: r = {combined_r['pearson_r']}, n = {combined_r['n']}, p = {combined_r['p_value']}")

    # 5. Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "bat_speed_correlation_results.csv", index=False)
    print("\n  Saved: bat_speed_correlation_results.csv")

    # 6. Generate scatter plot
    print("\n[5] Generating scatter plot...")
    make_scatter(merged, results)

    # 7. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        sig = "*" if r["significant"] == "Yes" else "n.s."
        print(f"  {r['subset']:>20s}: r = {r['pearson_r']:+.4f}  (n={r['n']}, {sig})")
    print("=" * 60)


if __name__ == "__main__":
    main()
