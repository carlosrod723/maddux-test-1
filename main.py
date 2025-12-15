# main.py

# ==============================================================================
# MLB Hitter Analytics - 2024 Statcast & FanGraphs Data Pipeline
# ==============================================================================
# Pulls and merges 2024 MLB hitter data (200+ PA) from Baseball Savant
# (exit velocity, bat tracking) and FanGraphs (batting stats, plate discipline).
# ==============================================================================

import pandas as pd
import requests
from io import StringIO
from pybaseball import statcast_batter_exitvelo_barrels, batting_stats
from thefuzz import fuzz, process
import warnings

warnings.filterwarnings('ignore')

# Configuration
YEAR = 2024
MIN_PA = 200
MIN_PLAYERS = 50

# Output paths
OUTPUT_DIR = "data"
FILES = {
    "exit_velocity": f"{OUTPUT_DIR}/test_exit_velocity.csv",
    "bat_tracking": f"{OUTPUT_DIR}/test_bat_tracking.csv",
    "batting_stats": f"{OUTPUT_DIR}/test_batting_stats.csv",
    "plate_discipline": f"{OUTPUT_DIR}/test_plate_discipline.csv",
    "merged": "merged_data.csv"
}


def pull_exit_velocity():
    """Pull Exit Velocity & Barrel data from Baseball Savant."""
    print("📊 Pulling Exit Velocity data from Baseball Savant...")
    df = statcast_batter_exitvelo_barrels(YEAR, MIN_PA)
    df = df[['player_id', 'last_name, first_name', 'avg_hit_speed', 'max_hit_speed',
             'ev95percent', 'brl_percent']].copy()
    df = df.rename(columns={
        'last_name, first_name': 'player_name',
        'ev95percent': 'hard_hit_percent'
    })
    print(f"   ✓ {len(df)} players retrieved")
    return df


def pull_bat_tracking():
    """Pull Bat Tracking data from Baseball Savant."""
    print("🏏 Pulling Bat Tracking data from Baseball Savant...")

    url = (
        "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?attackZone=&batSide=&contactType=&count=&dateStart={YEAR}-01-01"
        f"&dateEnd={YEAR}-12-31&gameType=&isHardHit=&minSwings=1"
        f"&minGroupSwings=1&pitchHand=&pitchType=&seasonStart={YEAR}"
        f"&seasonEnd={YEAR}&team=&type=batter&csv=true"
    )

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)

    df = pd.read_csv(StringIO(response.text))
    df = df[['id', 'name', 'avg_bat_speed', 'swing_length']].copy()
    df = df.rename(columns={'id': 'player_id', 'name': 'player_name'})
    print(f"   ✓ {len(df)} players retrieved")
    return df


def pull_batting_stats():
    """Pull Batting Stats from FanGraphs CSV."""
    print("📈 Pulling Batting Stats from FanGraphs...")

    df = pd.read_csv(f"{OUTPUT_DIR}/fangraph_batting_2024.csv")
    df = df[['MLBAMID', 'Name', 'PA', 'AVG', 'OBP', 'SLG', 'ISO', 'wRC+']].copy()
    df = df.rename(columns={
        'MLBAMID': 'playerid',
        'Name': 'player_name'
    })

    print(f"   ✓ {len(df)} players retrieved")
    return df


def pull_plate_discipline():
    """Pull Plate Discipline from FanGraphs CSVs."""
    print("🎯 Pulling Plate Discipline from FanGraphs...")

    # BB% and K% from batting file
    batting = pd.read_csv(f"{OUTPUT_DIR}/fangraph_batting_2024.csv")
    batting = batting[['MLBAMID', 'Name', 'BB%', 'K%']].copy()

    # Contact% from discipline file
    discipline = pd.read_csv(f"{OUTPUT_DIR}/fangraph_discipline_2024.csv")
    discipline = discipline[['MLBAMID', 'Contact%']].copy()

    # Merge
    df = batting.merge(discipline, on='MLBAMID', how='inner')
    df = df.rename(columns={
        'MLBAMID': 'playerid',
        'Name': 'player_name'
    })

    print(f"   ✓ {len(df)} players retrieved")
    return df


def fuzzy_match_name(name, choices, threshold=85):
    """Match player name using fuzzy string matching."""
    match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


def merge_datasets(exit_velo, bat_tracking, batting, discipline):
    """Merge all datasets on player name with fuzzy matching."""
    print("\n🔗 Merging datasets...")
    
    # Start with FanGraphs batting as base
    merged = batting.merge(
        discipline.drop(columns=['player_name']), 
        on='playerid', 
        how='inner'
    )
    print(f"   • FanGraphs batting + discipline: {len(merged)} players")
    
    # Build lookup for Savant names
    exit_velo_names = exit_velo['player_name'].tolist()
    bat_tracking_names = bat_tracking['player_name'].tolist()
    
    # Fuzzy match to Exit Velocity data
    merged['exit_velo_match'] = merged['player_name'].apply(
        lambda x: fuzzy_match_name(x, exit_velo_names)
    )
    
    # Fuzzy match to Bat Tracking data  
    merged['bat_tracking_match'] = merged['player_name'].apply(
        lambda x: fuzzy_match_name(x, bat_tracking_names)
    )
    
    # Join exit velocity
    merged = merged.merge(
        exit_velo[['player_name', 'player_id', 'hard_hit_percent', 'brl_percent']],
        left_on='exit_velo_match',
        right_on='player_name',
        how='inner',
        suffixes=('', '_ev')
    )
    print(f"   • After exit velocity join: {len(merged)} players")
    
    # Join bat tracking
    merged = merged.merge(
        bat_tracking[['player_name', 'avg_bat_speed', 'swing_length']],
        left_on='bat_tracking_match',
        right_on='player_name',
        how='inner',
        suffixes=('', '_bt')
    )
    print(f"   • After bat tracking join: {len(merged)} players")
    
    # Select and rename final columns per spec
    final = merged[[
        'player_name', 'player_id', 'PA', 'avg_bat_speed',
        'hard_hit_percent', 'brl_percent', 'BB%', 'Contact%',
        'AVG', 'OBP', 'SLG', 'ISO'
    ]].copy()
    
    final.columns = [
        'player_name', 'player_id', 'pa', 'avg_bat_speed',
        'hard_hit_pct', 'barrel_pct', 'bb_pct', 'contact_pct',
        'avg', 'obp', 'slg', 'iso'
    ]

    # Convert decimals to percentages for bb_pct and contact_pct
    final['bb_pct'] = (final['bb_pct'] * 100).round(1)
    final['contact_pct'] = (final['contact_pct'] * 100).round(1)

    # Remove duplicates
    final = final.drop_duplicates(subset=['player_name'])
    
    print(f"   ✓ {len(final)} players in final merged dataset")
    return final


def validate(df):
    """Validate merged dataset against requirements."""
    print("\n🔍 Validating dataset...")

    # Hard requirements
    hard_checks = {
        "Row count >= 50": len(df) >= MIN_PLAYERS,
        "Player IDs are 6 digits": df['player_id'].astype(str).str.len().eq(6).all(),
        "No duplicates": df['player_name'].is_unique
    }

    # Soft checks (expected ranges, outliers are OK)
    soft_checks = {
        "Bat Speed 65-85 mph": df['avg_bat_speed'].between(65, 85).all(),
        "Hard Hit % 20-60%": df['hard_hit_pct'].between(20, 60).all(),
        "OBP .250-.450": df['obp'].between(0.250, 0.450).all(),
    }

    all_passed = True

    print("   Hard Requirements:")
    for check, passed in hard_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"      {status}: {check}")
        if not passed:
            all_passed = False

    print("   Range Checks (outliers noted):")
    for check, passed in soft_checks.items():
        status = "✅ PASS" if passed else "⚠️  OUTLIERS"
        print(f"      {status}: {check}")
        if not passed:
            if "Bat Speed" in check:
                outliers = df[~df['avg_bat_speed'].between(65, 85)][['player_name', 'avg_bat_speed']]
                print(f"         {len(outliers)} players outside range")
            elif "Hard Hit" in check:
                outliers = df[~df['hard_hit_pct'].between(20, 60)][['player_name', 'hard_hit_pct']]
                print(f"         {len(outliers)} players outside range")
            elif "OBP" in check:
                outliers = df[~df['obp'].between(0.250, 0.450)][['player_name', 'obp']]
                print(f"         {len(outliers)} players outside range")

    return all_passed


def export_all(exit_velo, bat_tracking, batting, discipline, merged):
    """Export all CSVs."""
    print("\n💾 Exporting CSVs...")
    exit_velo.to_csv(FILES["exit_velocity"], index=False)
    bat_tracking.to_csv(FILES["bat_tracking"], index=False)
    batting.to_csv(FILES["batting_stats"], index=False)
    discipline.to_csv(FILES["plate_discipline"], index=False)
    merged.to_csv(FILES["merged"], index=False)
    print(f"   ✓ Exported: {', '.join(FILES.values())}")


def main():
    print("=" * 60)
    print("MLB Hitter Analytics - 2024 Data Pipeline")
    print("=" * 60 + "\n")
    
    # Pull data
    exit_velo = pull_exit_velocity()
    bat_tracking = pull_bat_tracking()
    batting = pull_batting_stats()
    discipline = pull_plate_discipline()
    
    # Merge
    merged = merge_datasets(exit_velo, bat_tracking, batting, discipline)
    
    # Validate
    is_valid = validate(merged)
    
    # Export
    export_all(exit_velo, bat_tracking, batting, discipline, merged)
    
    print("\n" + "=" * 60)
    if is_valid:
        print("✅ Pipeline complete! All validations passed.")
    else:
        print("⚠️  Pipeline complete with validation warnings.")
    print("=" * 60)


if __name__ == "__main__":
    main()