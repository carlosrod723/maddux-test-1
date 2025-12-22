# build_maddux_database.py

"""
Build Database with Delta Calculations

Merges Savant + FanGraphs data, calculates year-over-year deltas,
and computes MADDUX scores.
"""

import pandas as pd
import sqlite3
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "maddux_phase1.db"


def load_and_merge_data():
    """Load Savant and FanGraphs data, merge on player_id + year."""
    print("📥 Loading data...")
    
    # Load Savant
    savant = pd.read_csv(RAW_DIR / "savant_statcast_2015_2025.csv")
    print(f"   Savant: {len(savant)} rows")
    
    # Load FanGraphs
    fangraphs = pd.read_csv(RAW_DIR / "fangraphs_2015_2025.csv")
    # Calculate OPS from components (more precise than pre-calculated)
    fangraphs['OPS'] = fangraphs['OBP'] + fangraphs['SLG']
    # Rename for merge
    fangraphs = fangraphs.rename(columns={
        'MLBAMID': 'player_id',
        'Season': 'year',
        'Name': 'fg_name',
        'wRC+': 'wrc_plus'
    })
    print(f"   FanGraphs: {len(fangraphs)} rows")
    
    # Merge on player_id + year
    merged = pd.merge(
        savant,
        fangraphs[['player_id', 'year', 'PA', 'OPS', 'OBP', 'SLG', 'wrc_plus', 'Team']],
        on=['player_id', 'year'],
        how='inner'
    )
    
    print(f"   Merged: {len(merged)} rows")
    print(f"   Unique players: {merged['player_id'].nunique()}")
    
    return merged


def calculate_deltas(df):
    """Calculate year-over-year deltas for each player."""
    print("\n📊 Calculating year-over-year deltas...")
    
    # Sort by player and year
    df = df.sort_values(['player_id', 'year'])
    
    # Calculate deltas within each player
    delta_cols = ['max_ev', 'hard_hit_pct', 'OPS', 'wrc_plus']
    
    for col in delta_cols:
        df[f'prev_{col}'] = df.groupby('player_id')[col].shift(1)
        df[f'delta_{col}'] = df[col] - df[f'prev_{col}']
    
    # Also get previous year for reference
    df['prev_year'] = df.groupby('player_id')['year'].shift(1)
    
    # Filter to rows with valid deltas (excludes first year for each player)
    df_with_deltas = df[df['prev_year'].notna()].copy()
    
    # Verify year continuity (should be consecutive years)
    df_with_deltas['year_gap'] = df_with_deltas['year'] - df_with_deltas['prev_year']
    consecutive = df_with_deltas[df_with_deltas['year_gap'] == 1].copy()
    
    print(f"   Player-seasons with deltas: {len(df_with_deltas)}")
    print(f"   Consecutive years only: {len(consecutive)}")
    
    return df, consecutive


def calculate_maddux_scores(df):
    """Calculate MADDUX Hitter score: Δ Max EV + (2.1 × Δ Hard Hit%)."""
    print("\n🔢 Calculating MADDUX scores...")
    
    # MADDUX Hitter Formula
    df['maddux_score'] = df['delta_max_ev'] + (2.1 * df['delta_hard_hit_pct'])
    
    # Round for readability
    df['maddux_score'] = df['maddux_score'].round(2)
    df['delta_max_ev'] = df['delta_max_ev'].round(2)
    df['delta_hard_hit_pct'] = df['delta_hard_hit_pct'].round(2)
    df['delta_OPS'] = df['delta_OPS'].round(3)
    
    print(f"   Score range: {df['maddux_score'].min():.2f} to {df['maddux_score'].max():.2f}")
    print(f"   Mean score: {df['maddux_score'].mean():.2f}")
    
    return df


def create_database(df_all, df_deltas):
    """Create SQLite database with proper schema."""
    print(f"\n💾 Creating database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Drop existing tables
    conn.execute("DROP TABLE IF EXISTS player_seasons")
    conn.execute("DROP TABLE IF EXISTS player_deltas")
    conn.execute("DROP TABLE IF EXISTS players")
    
    # Create players table
    players = df_all[['player_id', 'player_name']].drop_duplicates()
    players = players.groupby('player_id').first().reset_index()
    
    conn.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL
        )
    """)
    players.to_sql('players', conn, if_exists='append', index=False)
    print(f"   ✓ players: {len(players)} rows")
    
    # Create player_seasons table (all yearly data)
    seasons_cols = [
        'player_id', 'year', 'PA', 'max_ev', 'hard_hit_pct', 'barrel_pct',
        'OPS', 'OBP', 'SLG', 'wrc_plus', 'Team'
    ]
    df_seasons = df_all[seasons_cols].copy()
    
    conn.execute("""
        CREATE TABLE player_seasons (
            player_id INTEGER,
            year INTEGER,
            pa INTEGER,
            max_ev REAL,
            hard_hit_pct REAL,
            barrel_pct REAL,
            ops REAL,
            obp REAL,
            slg REAL,
            wrc_plus REAL,
            team TEXT,
            PRIMARY KEY (player_id, year),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)
    df_seasons.columns = ['player_id', 'year', 'pa', 'max_ev', 'hard_hit_pct', 
                          'barrel_pct', 'ops', 'obp', 'slg', 'wrc_plus', 'team']
    df_seasons.to_sql('player_seasons', conn, if_exists='append', index=False)
    print(f"   ✓ player_seasons: {len(df_seasons)} rows")
    
    # Create player_deltas table (year-over-year changes + MADDUX scores)
    deltas_cols = [
        'player_id', 'year', 'prev_year', 'PA',
        'max_ev', 'prev_max_ev', 'delta_max_ev',
        'hard_hit_pct', 'prev_hard_hit_pct', 'delta_hard_hit_pct',
        'OPS', 'prev_OPS', 'delta_OPS',
        'maddux_score', 'Team'
    ]
    df_deltas_clean = df_deltas[deltas_cols].copy()
    df_deltas_clean.columns = [
        'player_id', 'year', 'prev_year', 'pa',
        'max_ev', 'prev_max_ev', 'delta_max_ev',
        'hard_hit_pct', 'prev_hard_hit_pct', 'delta_hard_hit_pct',
        'ops', 'prev_ops', 'delta_ops',
        'maddux_score', 'team'
    ]
    
    conn.execute("""
        CREATE TABLE player_deltas (
            player_id INTEGER,
            year INTEGER,
            prev_year INTEGER,
            pa INTEGER,
            max_ev REAL,
            prev_max_ev REAL,
            delta_max_ev REAL,
            hard_hit_pct REAL,
            prev_hard_hit_pct REAL,
            delta_hard_hit_pct REAL,
            ops REAL,
            prev_ops REAL,
            delta_ops REAL,
            maddux_score REAL,
            team TEXT,
            PRIMARY KEY (player_id, year),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)
    df_deltas_clean.to_sql('player_deltas', conn, if_exists='append', index=False)
    print(f"   ✓ player_deltas: {len(df_deltas_clean)} rows")
    
    conn.commit()
    conn.close()
    
    return len(players), len(df_seasons), len(df_deltas_clean)


def validate_database():
    """Run validation queries."""
    print("\n🔍 Validating database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check counts
    players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    seasons = conn.execute("SELECT COUNT(*) FROM player_seasons").fetchone()[0]
    deltas = conn.execute("SELECT COUNT(*) FROM player_deltas").fetchone()[0]
    
    print(f"   Players: {players}")
    print(f"   Seasons: {seasons}")
    print(f"   Deltas: {deltas}")
    
    # Top 10 MADDUX scores
    print("\n   Top 10 MADDUX Scores (all time):")
    top10 = conn.execute("""
        SELECT p.player_name, d.year, d.delta_max_ev, d.delta_hard_hit_pct, 
               d.maddux_score, d.delta_ops
        FROM player_deltas d
        JOIN players p ON d.player_id = p.player_id
        ORDER BY d.maddux_score DESC
        LIMIT 10
    """).fetchall()
    
    print(f"   {'Player':<25} {'Year':<6} {'Δ MaxEV':<8} {'Δ HH%':<8} {'Score':<8} {'Δ OPS':<8}")
    print("   " + "-" * 63)
    for row in top10:
        print(f"   {row[0]:<25} {row[1]:<6} {row[2]:<8.1f} {row[3]:<8.1f} {row[4]:<8.2f} {row[5]:<+8.3f}")
    
    conn.close()


def export_processed_data(df_all, df_deltas):
    """Export processed CSVs."""
    print("\n📁 Exporting processed data...")
    
    df_all.to_csv(PROCESSED_DIR / "all_seasons_merged.csv", index=False)
    df_deltas.to_csv(PROCESSED_DIR / "player_deltas.csv", index=False)
    
    print(f"   ✓ {PROCESSED_DIR / 'all_seasons_merged.csv'}")
    print(f"   ✓ {PROCESSED_DIR / 'player_deltas.csv'}")


def main():
    print("=" * 60)
    print("MADDUX Phase 1: Build Database")
    print("=" * 60)
    
    # Load and merge
    df_merged = load_and_merge_data()
    
    # Calculate deltas
    df_all, df_deltas = calculate_deltas(df_merged)
    
    # Calculate MADDUX scores
    df_deltas = calculate_maddux_scores(df_deltas)
    
    # Create database
    create_database(df_all, df_deltas)
    
    # Validate
    validate_database()
    
    # Export CSVs
    export_processed_data(df_all, df_deltas)
    
    print("\n" + "=" * 60)
    print("✅ Database ready: maddux_phase1.db")
    print("=" * 60)


if __name__ == "__main__":
    main()