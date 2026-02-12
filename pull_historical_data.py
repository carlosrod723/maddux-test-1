# pull_historical_data.py

"""
MADDUX Historical Data Pull (2015-2025)

Pulls hitter data from Baseball Savant and FanGraphs for delta calculations.
Phase 1: Statcast exit velocity, barrels
Phase 2: Sprint speed (biomechanics proxy)
"""

import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import time

# Configuration
YEARS = range(2015, 2026)  # 2015-2025
MIN_PA = 50  # Lower threshold to capture more players, filter later
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}


def pull_savant_statcast(year: int) -> pd.DataFrame:
    """Pull exit velocity and barrel data from Baseball Savant."""
    url = f'https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year={year}&position=&team=&min={MIN_PA}&csv=true'
    
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    df = pd.read_csv(StringIO(response.text))
    
    # Rename and select columns
    df = df.rename(columns={
        'last_name, first_name': 'player_name',
        'max_hit_speed': 'max_ev',
        'ev95percent': 'hard_hit_pct',
        'brl_percent': 'barrel_pct'
    })
    
    df['year'] = year
    
    return df[['player_id', 'player_name', 'year', 'max_ev', 'hard_hit_pct', 'barrel_pct']]


def pull_all_savant():
    """Pull Statcast data for all years."""
    print("=" * 60)
    print("Pulling Baseball Savant Data (2015-2025)")
    print("=" * 60)
    
    all_data = []
    
    for year in YEARS:
        try:
            df = pull_savant_statcast(year)
            all_data.append(df)
            print(f"  ✓ {year}: {len(df)} players")
            time.sleep(1)  # Be nice to the server
        except Exception as e:
            print(f"  ✗ {year}: {e}")
    
    combined = pd.concat(all_data, ignore_index=True)
    output_path = OUTPUT_DIR / "savant_statcast_2015_2025.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"\n  Saved: {output_path}")
    print(f"  Total rows: {len(combined)}")
    
    return combined


def pull_sprint_speed(year: int) -> pd.DataFrame:
    """Pull sprint speed data from Baseball Savant."""
    url = (
        f'https://baseballsavant.mlb.com/leaderboard/sprint_speed'
        f'?min_season={year}&max_season={year}&position=&team=&min=0&csv=true'
    )

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df['year'] = year

    return df[['player_id', 'year', 'sprint_speed']]


def pull_all_sprint_speed():
    """Pull sprint speed data for all years."""
    print("=" * 60)
    print("Pulling Baseball Savant Sprint Speed (2015-2025)")
    print("=" * 60)

    all_data = []

    for year in YEARS:
        try:
            df = pull_sprint_speed(year)
            all_data.append(df)
            print(f"  ✓ {year}: {len(df)} players")
            time.sleep(1)
        except Exception as e:
            print(f"  ✗ {year}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    output_path = OUTPUT_DIR / "savant_sprint_speed_2015_2025.csv"
    combined.to_csv(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Total rows: {len(combined)}")

    return combined


def pull_fangraphs_batting(year: int) -> pd.DataFrame:
    """
    Pull batting stats from FanGraphs.
    """
    from pybaseball import batting_stats
    
    try:
        df = batting_stats(year, qual=MIN_PA)
        df['year'] = year
        return df[['IDfg', 'Name', 'year', 'PA', 'OPS', 'wRC+']]
    except Exception as e:
        print(f"  pybaseball failed for {year}: {e}")
        return pd.DataFrame()


def main(sprint_speed_only=False):
    print("\n" + "=" * 60)
    print("MADDUX Historical Data Pull")
    print("=" * 60 + "\n")

    if not sprint_speed_only:
        # Pull Savant statcast data
        savant_df = pull_all_savant()
        print(f"\n  Statcast: {len(savant_df)} player-seasons, {savant_df['player_id'].nunique()} unique players")

    # Pull sprint speed data
    sprint_df = pull_all_sprint_speed()
    print(f"\n  Sprint speed: {len(sprint_df)} player-seasons, {sprint_df['player_id'].nunique()} unique players")

    print("\n✅ Savant data complete!")
    print("\n⚠️  FanGraphs data requires manual export (Cloudflare blocks automation)")
    print("   Export from: https://www.fangraphs.com/leaders/major-league")
    print("   Settings: 2015-2025, Min PA=50, Stats=PA,OPS,wRC+")


if __name__ == "__main__":
    main()