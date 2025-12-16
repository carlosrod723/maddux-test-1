# Maddux Test 1 & 2 - MLB Hitter Analytics

2024 MLB hitter data pipeline pulling from Baseball Savant (Statcast) and FanGraphs, with SQLite database and Claude API integration.

## Data Sources

| Source | Data |
|--------|------|
| Baseball Savant | Exit Velocity, Bat Tracking |
| FanGraphs | Batting Stats, Plate Discipline |

## Deliverables

- **Google Sheet**: [View Data](https://docs.google.com/spreadsheets/d/1-YjA9YqWREQkxz1HCuTAueieFp5zj3y1d6sphsOXeU4/edit?usp=sharing)
- **GitHub Repo**: CSV files + Python pipeline + Database + Claude API

## Files
```
├── data/
│   ├── test_exit_velocity.csv    # 284 players - Baseball Savant
│   ├── test_bat_tracking.csv     # 650 players - Baseball Savant
│   ├── test_batting_stats.csv    # 365 players - FanGraphs
│   └── test_plate_discipline.csv # 365 players - FanGraphs
├── merged_data.csv               # 284 players - All sources joined
├── main.py                       # Test 1: Data pipeline script
├── maddux_test.db                # Test 2: SQLite database
├── calculate_scores.py           # Test 2: Score calculation script
├── query_claude.py               # Test 2: Claude API script
├── claude_response.txt           # Test 2: Saved Claude response
└── requirements.txt              # Python dependencies
```

## Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_api_key_here
```

## Test 1: Data Pipeline
```bash
python main.py
```

Pulls data from Baseball Savant and FanGraphs, merges into `merged_data.csv`.

## Test 2: Database + Claude API

### Calculate Scores
```bash
python calculate_scores.py
```

Creates SQLite database and calculates MADDUX OBP Score:
```
Score = BB% + (1.8 × Contact%)
```

**Sample Output:**
```
Rank  Player                   BB%       Contact%    Score
1     Steven Kwan              9.8       92.8        176.84
2     Luis Arraez              3.6       94.2        173.16
3     Geraldo Perdomo          9.3       89.9        171.12
...
```

### Query Claude
```bash
python query_claude.py
```

Sends top 10 players to Claude API for breakout analysis. Response saved to `claude_response.txt`.

## Merged Data Schema

| Column | Source |
|--------|--------|
| player_name | Any |
| player_id | Baseball Savant (MLBAM ID) |
| pa | FanGraphs |
| avg_bat_speed | Bat Tracking |
| hard_hit_pct | Exit Velocity |
| barrel_pct | Exit Velocity |
| bb_pct | FanGraphs |
| contact_pct | FanGraphs |
| avg | FanGraphs |
| obp | FanGraphs |
| slg | FanGraphs |
| iso | FanGraphs |

## Database Schema
```sql
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL
);

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
);
```

## Validation

### Test 1
- ✅ 284 players (50+ required)
- ✅ Valid 6-digit MLBAM player IDs
- ✅ No duplicate players

### Test 2
- ✅ Database opens in SQLite
- ✅ Players table has 284 rows
- ✅ Batting table has 284 rows
- ✅ Foreign key relationship valid
- ✅ Score calculation returns correct top 10
- ✅ Claude API runs without errors
- ✅ Response saved to claude_response.txt
