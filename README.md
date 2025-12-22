# MADDUX Phase 1: MLB Hitter Analytics

Predictive analytics platform identifying MLB hitter breakouts through physical metrics.

**MADDUX Score = Δ Max EV + (2.1 × Δ Hard Hit%)**

## Key Results

| Metric | Result | Target |
|--------|--------|--------|
| Average Correlation | **0.410** | >0.15 |
| Hit Rate (MADDUX >20) | **73.7%** | >65% |
| Hit Rate (MADDUX >15) | **69.6%** | — |
| Hit Rate (MADDUX >10) | **64.7%** | — |

## Data Coverage

- **Years**: 2015-2025
- **Player-Seasons**: 5,178
- **Delta Records**: 3,606 (consecutive year pairs)
- **Unique Players**: 1,341

## Deliverables

### 1. Data Pipeline
- `pull_historical_data.py` - Baseball Savant data pull
- `build_maddux_database.py` - Merge + delta calculations
- `data/raw/` - Source CSVs (Savant + FanGraphs)
- `data/processed/` - Merged datasets

### 2. SQLite Database (`maddux_db.db`)
```
players (1,341 rows)
├── player_id (MLBAM ID)
└── player_name

player_seasons (5,178 rows)
├── player_id, year, pa
├── max_ev, hard_hit_pct, barrel_pct
├── ops, obp, slg, wrc_plus
└── team

player_deltas (3,606 rows)
├── player_id, year, prev_year, pa
├── max_ev, prev_max_ev, delta_max_ev
├── hard_hit_pct, prev_hard_hit_pct, delta_hard_hit_pct
├── ops, prev_ops, delta_ops
├── maddux_score
└── team
```

### 3. Claude API Integration
- `query_maddux.py` - CLI natural language queries
- Dashboard "Ask Claude" tab - Interactive queries

### 4. Interactive Dashboard (`dashboard.py`)
- **Scatter Analysis** - MADDUX vs OPS change with quadrant annotations
- **Leaderboard** - Breakout/regression candidates by year
- **Player Deep Dive** - Career trajectory + radar chart + AI analysis
- **Model Validation** - Correlation by year + hit rate analysis
- **Ask Claude** - Natural language database queries

## Quick Start
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run dashboard
streamlit run dashboard.py
```

## File Structure
```
maddux-test-1/
├── dashboard.py              # Interactive Streamlit dashboard
├── maddux_db.db              # SQLite database
├── build_maddux_database.py  # Data pipeline
├── pull_historical_data.py   # Savant data pull
├── query_maddux.py           # CLI Claude queries
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/                  # Source CSVs
│   └── processed/            # Merged data
└── .streamlit/
    └── config.toml           # Theme config
```

## Model Validation Summary

The MADDUX Hitter model shows **consistent predictive signal**:

- **Correlation (r = 0.41)**: Moderate positive relationship between physical metric improvements and OPS gains
- **Hit Rate (74% at >20 threshold)**: Players with high MADDUX scores improve their OPS nearly 3 out of 4 times
- **Stability**: Correlation remains positive across all 10 years (2016-2025)

### 2025 Top Breakout Candidates

| Player | Team | MADDUX | Δ MaxEV | Δ HH% | Δ OPS |
|--------|------|--------|---------|-------|-------|
| Rice, Ben | NYY | 43.4 | +2.7 | +19.4 | +0.223 |
| Story, Trevor | BOS | 43.2 | +6.5 | +17.5 | +0.008 |
| Turang, Brice | MIL | 40.6 | +3.4 | +17.7 | +0.129 |

## Tech Stack

- **Data**: Baseball Savant (Statcast), FanGraphs
- **Database**: SQLite
- **Dashboard**: Streamlit + Plotly
- **AI**: Claude API (Haiku 4.5)
- **Language**: Python 3.11+
