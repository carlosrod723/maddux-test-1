"""
MADDUX Methodology Verification Script

Run this in your local maddux-test-1 directory to verify our approach.

KEY QUESTION: Are we measuring the right thing?

MADDUX Intent (per original spec):
  "When underlying physical metrics improve, performance results follow—usually the NEXT season"

What we SHOULD measure:
  MADDUX score in Year N → predicts → OPS change in Year N+1

What we MAY HAVE measured:
  MADDUX score in Year N → correlates with → OPS change in Year N (same year)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_db.db")

conn = sqlite3.connect(DB_PATH)

# Load our delta data
df = pd.read_sql_query("""
    SELECT
        p.player_name,
        pd.player_id,
        pd.year,
        pd.maddux_score,
        pd.delta_ops
    FROM player_deltas pd
    JOIN players p ON pd.player_id = p.player_id
    ORDER BY pd.player_id, pd.year
""", conn)

print("=" * 70)
print("MADDUX METHODOLOGY VERIFICATION")
print("=" * 70)

print(f"\nTotal delta records: {len(df)}")
print(f"Years covered: {df['year'].min()} - {df['year'].max()}")

# CURRENT APPROACH: Same-year correlation
# maddux_score (Year N) vs delta_ops (Year N)
print("\n" + "-" * 70)
print("CURRENT APPROACH: Same-Year Correlation")
print("MADDUX(Year N) vs Δ OPS(Year N)")
print("-" * 70)

same_year_corr = df['maddux_score'].corr(df['delta_ops'])
print(f"Correlation: {same_year_corr:.4f}")

# Hit rate for same year
high_maddux = df[df['maddux_score'] > 20]
same_year_hit_rate = (high_maddux['delta_ops'] > 0).mean()
print(f"Hit Rate (MADDUX > 20): {same_year_hit_rate:.1%}")

# CORRECT APPROACH: Next-year prediction
# maddux_score (Year N) vs delta_ops (Year N+1)
print("\n" + "-" * 70)
print("CORRECT APPROACH: Next-Year Prediction")
print("MADDUX(Year N) vs Δ OPS(Year N+1)")
print("-" * 70)

# Create lagged dataset
df_sorted = df.sort_values(['player_id', 'year'])
df_sorted['next_year_delta_ops'] = df_sorted.groupby('player_id')['delta_ops'].shift(-1)

# Remove rows without next year data
df_predictive = df_sorted.dropna(subset=['next_year_delta_ops'])

print(f"Records with next-year data: {len(df_predictive)}")

next_year_corr = df_predictive['maddux_score'].corr(df_predictive['next_year_delta_ops'])
print(f"Correlation: {next_year_corr:.4f}")

# Hit rate for next year
high_maddux_pred = df_predictive[df_predictive['maddux_score'] > 20]
if len(high_maddux_pred) > 0:
    next_year_hit_rate = (high_maddux_pred['next_year_delta_ops'] > 0).mean()
    print(f"Hit Rate (MADDUX > 20): {next_year_hit_rate:.1%}")
    print(f"Sample size (MADDUX > 20): {len(high_maddux_pred)}")

# Year-by-year breakdown for next-year prediction
print("\n" + "-" * 70)
print("YEAR-BY-YEAR BREAKDOWN (Next-Year Prediction)")
print("-" * 70)
print(f"{'Year':<8} {'N':<8} {'Correlation':<12} {'Hit Rate (>20)':<15}")
print("-" * 70)

for year in sorted(df_predictive['year'].unique()):
    year_data = df_predictive[df_predictive['year'] == year]
    corr = year_data['maddux_score'].corr(year_data['next_year_delta_ops'])
    high = year_data[year_data['maddux_score'] > 20]
    hr = (high['next_year_delta_ops'] > 0).mean() if len(high) > 0 else np.nan
    print(f"{year:<8} {len(year_data):<8} {corr:+.4f}      {hr:.1%} (n={len(high)})" if not np.isnan(hr) else f"{year:<8} {len(year_data):<8} {corr:+.4f}      N/A")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if next_year_corr < 0:
    print("""
The other candidate appears to be CORRECT.

When we measure MADDUX as a PREDICTIVE model (Year N → Year N+1),
the correlation is likely NEGATIVE, indicating regression to the mean.

Our positive correlation (+0.41) was measuring CONCURRENT relationship,
which is expected (better contact → better OPS in the same year).

This is not the intended use. The model should PREDICT future breakouts.
""")
else:
    print(f"""
Interesting - the next-year correlation is {next_year_corr:+.4f}.

If positive, our methodology may still be valid but we should clarify
exactly what prediction horizon was intended.
""")

conn.close()
