# Maddux Test 1 - MLB Hitter Analytics

2024 MLB hitter data pipeline pulling from Baseball Savant (Statcast) and FanGraphs.

## Data Sources

| Source | Data |
|--------|------|
| Baseball Savant | Exit Velocity, Bat Tracking |
| FanGraphs | Batting Stats, Plate Discipline |

## Deliverables

- **Google Sheet**: [View Data](https://docs.google.com/spreadsheets/d/1-YjA9YqWREQkxz1HCuTAueieFp5zj3y1d6sphsOXeU4/edit?usp=sharing)
- **GitHub Repo**: CSV files + Python pipeline

## Files
```
├── data/
│   ├── test_exit_velocity.csv    # 284 players - Baseball Savant
│   ├── test_bat_tracking.csv     # 650 players - Baseball Savant  
│   ├── test_batting_stats.csv    # 365 players - FanGraphs
│   └── test_plate_discipline.csv # 365 players - FanGraphs
├── merged_data.csv               # 284 players - All sources joined
├── main.py                       # Data pipeline script
└── requirements.txt              # Python dependencies
```

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

## Usage
```bash
pip install -r requirements.txt
python main.py
```

## Validation

- ✅ 284 players (50+ required)
- ✅ Valid 6-digit MLBAM player IDs
- ✅ No duplicate players
- ⚠️ Some outliers in bat speed/hard hit %/OBP (real data, not errors)

## Time Spent

~2.5 hours