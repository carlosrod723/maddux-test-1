# build_maddux_database.py

"""
Build Database with Delta Calculations

Merges Savant + FanGraphs + Sprint Speed data, calculates year-over-year deltas,
and computes MADDUX scores.

Phase 2 additions: sprint_speed, iso, bb_pct, k_pct, barrel_pct deltas
"""

import pandas as pd
import sqlite3
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "maddux_db.db"


def load_and_merge_data():
    """Load Savant, FanGraphs, and Sprint Speed data, merge on player_id + year."""
    print("Loading data...")

    # Load Savant statcast (exit velocity, barrels)
    savant = pd.read_csv(RAW_DIR / "savant_statcast_2015_2025.csv")
    print(f"   Savant statcast: {len(savant)} rows")

    # Load sprint speed
    sprint = pd.read_csv(RAW_DIR / "savant_sprint_speed_2015_2025.csv")
    print(f"   Sprint speed: {len(sprint)} rows")

    # Load FanGraphs
    fangraphs = pd.read_csv(RAW_DIR / "fangraphs_2015_2025.csv")
    fangraphs['OPS'] = fangraphs['OBP'] + fangraphs['SLG']
    fangraphs = fangraphs.rename(columns={
        'MLBAMID': 'player_id',
        'Season': 'year',
        'Name': 'fg_name',
        'wRC+': 'wrc_plus',
        'BB%': 'bb_pct',
        'K%': 'k_pct',
        'ISO': 'iso',
    })
    print(f"   FanGraphs: {len(fangraphs)} rows")

    # Merge savant statcast + FanGraphs on player_id + year
    merged = pd.merge(
        savant,
        fangraphs[['player_id', 'year', 'PA', 'OPS', 'OBP', 'SLG',
                    'wrc_plus', 'iso', 'bb_pct', 'k_pct', 'Team']],
        on=['player_id', 'year'],
        how='inner'
    )

    # Left join sprint speed (not all batters have sprint data)
    merged = pd.merge(
        merged,
        sprint[['player_id', 'year', 'sprint_speed']],
        on=['player_id', 'year'],
        how='left'
    )

    sprint_coverage = merged['sprint_speed'].notna().sum()
    print(f"   Merged: {len(merged)} rows")
    print(f"   Sprint speed coverage: {sprint_coverage}/{len(merged)} ({100*sprint_coverage/len(merged):.1f}%)")
    print(f"   Unique players: {merged['player_id'].nunique()}")

    return merged


def calculate_deltas(df):
    """Calculate year-over-year deltas for each player."""
    print("\nCalculating year-over-year deltas...")

    df = df.sort_values(['player_id', 'year'])

    # All metrics to calculate deltas for
    delta_cols = [
        'max_ev', 'hard_hit_pct', 'barrel_pct',
        'OPS', 'wrc_plus',
        'iso', 'bb_pct', 'k_pct',
        'sprint_speed'
    ]

    for col in delta_cols:
        df[f'prev_{col}'] = df.groupby('player_id')[col].shift(1)
        df[f'delta_{col}'] = df[col] - df[f'prev_{col}']

    df['prev_year'] = df.groupby('player_id')['year'].shift(1)

    # Filter to rows with valid deltas
    df_with_deltas = df[df['prev_year'].notna()].copy()

    # Consecutive years only
    df_with_deltas['year_gap'] = df_with_deltas['year'] - df_with_deltas['prev_year']
    consecutive = df_with_deltas[df_with_deltas['year_gap'] == 1].copy()

    print(f"   Player-seasons with deltas: {len(df_with_deltas)}")
    print(f"   Consecutive years only: {len(consecutive)}")

    return df, consecutive


def calculate_maddux_scores(df):
    """Calculate MADDUX Hitter score: delta_max_ev + (2.1 x delta_hard_hit_pct)."""
    print("\nCalculating MADDUX scores...")

    df['maddux_score'] = df['delta_max_ev'] + (2.1 * df['delta_hard_hit_pct'])

    # Round for readability
    for col in ['maddux_score', 'delta_max_ev', 'delta_hard_hit_pct', 'delta_barrel_pct',
                'delta_sprint_speed', 'delta_iso', 'delta_bb_pct', 'delta_k_pct']:
        if col in df.columns:
            df[col] = df[col].round(3)
    df['delta_OPS'] = df['delta_OPS'].round(3)
    df['delta_wrc_plus'] = df['delta_wrc_plus'].round(3)

    print(f"   Score range: {df['maddux_score'].min():.2f} to {df['maddux_score'].max():.2f}")
    print(f"   Mean score: {df['maddux_score'].mean():.2f}")

    return df


def create_database(df_all, df_deltas):
    """Create SQLite database with proper schema."""
    print(f"\nCreating database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    conn.execute("DROP TABLE IF EXISTS player_seasons")
    conn.execute("DROP TABLE IF EXISTS player_deltas")
    conn.execute("DROP TABLE IF EXISTS players")

    # --- players table ---
    players = df_all[['player_id', 'player_name']].drop_duplicates()
    players = players.groupby('player_id').first().reset_index()

    conn.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL
        )
    """)
    players.to_sql('players', conn, if_exists='append', index=False)
    print(f"   players: {len(players)} rows")

    # --- player_seasons table ---
    seasons_cols = [
        'player_id', 'year', 'PA', 'max_ev', 'hard_hit_pct', 'barrel_pct',
        'sprint_speed', 'iso', 'bb_pct', 'k_pct',
        'OPS', 'OBP', 'SLG', 'wrc_plus', 'Team'
    ]
    df_seasons = df_all[seasons_cols].copy()
    df_seasons.columns = [
        'player_id', 'year', 'pa', 'max_ev', 'hard_hit_pct', 'barrel_pct',
        'sprint_speed', 'iso', 'bb_pct', 'k_pct',
        'ops', 'obp', 'slg', 'wrc_plus', 'team'
    ]

    conn.execute("""
        CREATE TABLE player_seasons (
            player_id INTEGER,
            year INTEGER,
            pa INTEGER,
            max_ev REAL,
            hard_hit_pct REAL,
            barrel_pct REAL,
            sprint_speed REAL,
            iso REAL,
            bb_pct REAL,
            k_pct REAL,
            ops REAL,
            obp REAL,
            slg REAL,
            wrc_plus REAL,
            team TEXT,
            PRIMARY KEY (player_id, year),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)
    df_seasons.to_sql('player_seasons', conn, if_exists='append', index=False)
    print(f"   player_seasons: {len(df_seasons)} rows")

    # --- player_deltas table ---
    deltas_cols = [
        'player_id', 'year', 'prev_year', 'PA',
        'max_ev', 'prev_max_ev', 'delta_max_ev',
        'hard_hit_pct', 'prev_hard_hit_pct', 'delta_hard_hit_pct',
        'barrel_pct', 'prev_barrel_pct', 'delta_barrel_pct',
        'sprint_speed', 'prev_sprint_speed', 'delta_sprint_speed',
        'iso', 'prev_iso', 'delta_iso',
        'bb_pct', 'prev_bb_pct', 'delta_bb_pct',
        'k_pct', 'prev_k_pct', 'delta_k_pct',
        'OPS', 'prev_OPS', 'delta_OPS',
        'wrc_plus', 'prev_wrc_plus', 'delta_wrc_plus',
        'maddux_score', 'Team'
    ]
    df_deltas_clean = df_deltas[deltas_cols].copy()
    df_deltas_clean.columns = [
        'player_id', 'year', 'prev_year', 'pa',
        'max_ev', 'prev_max_ev', 'delta_max_ev',
        'hard_hit_pct', 'prev_hard_hit_pct', 'delta_hard_hit_pct',
        'barrel_pct', 'prev_barrel_pct', 'delta_barrel_pct',
        'sprint_speed', 'prev_sprint_speed', 'delta_sprint_speed',
        'iso', 'prev_iso', 'delta_iso',
        'bb_pct', 'prev_bb_pct', 'delta_bb_pct',
        'k_pct', 'prev_k_pct', 'delta_k_pct',
        'ops', 'prev_ops', 'delta_ops',
        'wrc_plus', 'prev_wrc_plus', 'delta_wrc_plus',
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
            barrel_pct REAL,
            prev_barrel_pct REAL,
            delta_barrel_pct REAL,
            sprint_speed REAL,
            prev_sprint_speed REAL,
            delta_sprint_speed REAL,
            iso REAL,
            prev_iso REAL,
            delta_iso REAL,
            bb_pct REAL,
            prev_bb_pct REAL,
            delta_bb_pct REAL,
            k_pct REAL,
            prev_k_pct REAL,
            delta_k_pct REAL,
            ops REAL,
            prev_ops REAL,
            delta_ops REAL,
            wrc_plus REAL,
            prev_wrc_plus REAL,
            delta_wrc_plus REAL,
            maddux_score REAL,
            team TEXT,
            PRIMARY KEY (player_id, year),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)
    df_deltas_clean.to_sql('player_deltas', conn, if_exists='append', index=False)
    print(f"   player_deltas: {len(df_deltas_clean)} rows")

    conn.commit()
    conn.close()

    return len(players), len(df_seasons), len(df_deltas_clean)


def validate_database():
    """Run validation queries."""
    print("\nValidating database...")

    conn = sqlite3.connect(DB_PATH)

    players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    seasons = conn.execute("SELECT COUNT(*) FROM player_seasons").fetchone()[0]
    deltas = conn.execute("SELECT COUNT(*) FROM player_deltas").fetchone()[0]

    print(f"   Players: {players}")
    print(f"   Seasons: {seasons}")
    print(f"   Deltas: {deltas}")

    # Column counts
    seasons_cols = conn.execute("PRAGMA table_info(player_seasons)").fetchall()
    deltas_cols = conn.execute("PRAGMA table_info(player_deltas)").fetchall()
    print(f"   player_seasons columns: {len(seasons_cols)}")
    print(f"   player_deltas columns: {len(deltas_cols)}")

    # Sprint speed coverage
    sprint_total = conn.execute(
        "SELECT COUNT(*) FROM player_seasons WHERE sprint_speed IS NOT NULL"
    ).fetchone()[0]
    print(f"   Sprint speed records: {sprint_total}/{seasons}")

    # Null check for new columns
    for col in ['iso', 'bb_pct', 'k_pct']:
        nulls = conn.execute(
            f"SELECT COUNT(*) FROM player_seasons WHERE {col} IS NULL"
        ).fetchone()[0]
        print(f"   {col} nulls: {nulls}/{seasons}")

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

    print(f"   {'Player':<25} {'Year':<6} {'dMaxEV':<8} {'dHH%':<8} {'Score':<8} {'dOPS':<8}")
    print("   " + "-" * 63)
    for row in top10:
        print(f"   {row[0]:<25} {row[1]:<6} {row[2]:<8.1f} {row[3]:<8.1f} {row[4]:<8.2f} {row[5]:<+8.3f}")

    conn.close()


def export_processed_data(df_all, df_deltas):
    """Export processed CSVs."""
    print("\nExporting processed data...")

    df_all.to_csv(PROCESSED_DIR / "all_seasons_merged.csv", index=False)
    df_deltas.to_csv(PROCESSED_DIR / "player_deltas.csv", index=False)

    print(f"   {PROCESSED_DIR / 'all_seasons_merged.csv'}")
    print(f"   {PROCESSED_DIR / 'player_deltas.csv'}")


def main():
    print("=" * 60)
    print("MADDUX: Build Database (Phase 2)")
    print("=" * 60)

    df_merged = load_and_merge_data()
    df_all, df_deltas = calculate_deltas(df_merged)
    df_deltas = calculate_maddux_scores(df_deltas)
    create_database(df_all, df_deltas)
    validate_database()
    export_processed_data(df_all, df_deltas)

    print("\n" + "=" * 60)
    print(f"Database ready: {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
