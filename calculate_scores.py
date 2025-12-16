# calcualte_scores.py

# ==============================================================================
# MADDUX Test 2: Database + Score Calculation
# ==============================================================================
# Creates SQLite database, loads Test 1 data, and calculates MADDUX OBP Score.
# Score = bb_pct + (1.8 × contact_pct)
# ==============================================================================

import sqlite3
import pandas as pd
from pathlib import Path

# Configuration
DB_PATH = "maddux_test.db"
DATA_PATH = "merged_data.csv"


def create_database():
    """Create SQLite database with schema per Matt's spec."""
    print("📦 Creating database...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS batting_2024")
    cursor.execute("DROP TABLE IF EXISTS players")
    
    # Create players table
    cursor.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL
        )
    """)
    
    # Create batting_2024 table
    cursor.execute("""
        CREATE TABLE batting_2024 (
            player_id INTEGER,
            avg_bat_speed REAL,
            hard_hit_pct REAL,
            barrel_pct REAL,
            bb_pct REAL,
            contact_pct REAL,
            avg REAL,
            obp REAL,
            slg REAL,
            iso REAL,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)
    
    conn.commit()
    print("   ✓ Tables created")
    return conn


def load_data(conn):
    """Load merged data from Test 1 into database."""
    print("📥 Loading data...")
    
    df = pd.read_csv(DATA_PATH)
    
    # Insert into players table
    players_df = df[['player_id', 'player_name']].drop_duplicates()
    players_df.to_sql('players', conn, if_exists='append', index=False)
    print(f"   ✓ Loaded {len(players_df)} players")
    
    # Insert into batting_2024 table
    batting_df = df[[
        'player_id', 'avg_bat_speed', 'hard_hit_pct', 'barrel_pct',
        'bb_pct', 'contact_pct', 'avg', 'obp', 'slg', 'iso'
    ]]
    batting_df.to_sql('batting_2024', conn, if_exists='append', index=False)
    print(f"   ✓ Loaded {len(batting_df)} batting records")
    
    return len(players_df)


def calculate_top_10(conn):
    """Calculate MADDUX OBP Score and return top 10 players."""
    print("\n🔢 Calculating MADDUX OBP Scores...")
    print("   Formula: bb_pct + (1.8 × contact_pct)\n")
    
    query = """
        SELECT 
            p.player_name,
            p.player_id,
            b.bb_pct,
            b.contact_pct,
            ROUND(b.bb_pct + (1.8 * b.contact_pct), 2) AS maddux_score
        FROM players p
        JOIN batting_2024 b ON p.player_id = b.player_id
        ORDER BY maddux_score DESC
        LIMIT 10
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Display results
    print("   " + "=" * 60)
    print(f"   {'Rank':<6}{'Player':<25}{'BB%':<10}{'Contact%':<12}{'Score':<10}")
    print("   " + "=" * 60)
    
    top_10 = []
    for i, row in enumerate(results, 1):
        player_name, player_id, bb_pct, contact_pct, score = row
        print(f"   {i:<6}{player_name:<25}{bb_pct:<10.1f}{contact_pct:<12.1f}{score:<10.2f}")
        top_10.append({
            'rank': i,
            'player_name': player_name,
            'player_id': player_id,
            'bb_pct': bb_pct,
            'contact_pct': contact_pct,
            'maddux_score': score
        })
    
    print("   " + "=" * 60)
    
    return top_10


def validate_database(conn):
    """Validate database meets requirements."""
    print("\n🔍 Validating database...")
    
    cursor = conn.cursor()
    
    # Check players count
    cursor.execute("SELECT COUNT(*) FROM players")
    players_count = cursor.fetchone()[0]
    
    # Check batting count
    cursor.execute("SELECT COUNT(*) FROM batting_2024")
    batting_count = cursor.fetchone()[0]
    
    # Check foreign key
    cursor.execute("""
        SELECT COUNT(*) FROM batting_2024 b
        WHERE NOT EXISTS (SELECT 1 FROM players p WHERE p.player_id = b.player_id)
    """)
    orphan_count = cursor.fetchone()[0]
    
    checks = {
        "Players table has 50+ rows": players_count >= 50,
        "Batting table has 50+ rows": batting_count >= 50,
        "Foreign key relationship valid": orphan_count == 0
    }
    
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check}")
    
    return all(checks.values())


def main():
    print("=" * 60)
    print("MADDUX Test 2: Database + Score Calculation")
    print("=" * 60 + "\n")
    
    # Create database
    conn = create_database()
    
    # Load data
    load_data(conn)
    
    # Validate
    validate_database(conn)
    
    # Calculate top 10
    top_10 = calculate_top_10(conn)
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Database ready! Run query_claude.py next.")
    print("=" * 60)
    
    return top_10


if __name__ == "__main__":
    main()