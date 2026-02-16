# MADDUX Analytics: MLB Hitter Breakout Prediction

Predictive analytics platform testing whether physical batting metrics (exit velocity, hard hit rate, barrel rate) can forecast future MLB hitter performance (OPS).

## Phase 1: Original Formula Validation

**MADDUX Score = d Max EV + (2.1 x d Hard Hit%)**

The original delta-based formula shows strong *concurrent* correlation (r = 0.41) but **fails as a predictor**. When MADDUX(Year N) is used to predict OPS change in Year N+1, the correlation flips negative (r = -0.135). This is because year-over-year deltas regress to the mean.

## Phase 2: Alternative Formulations and Causal Analysis

Phase 2 tested alternative approaches, validated the Driveline causal thesis, and decomposed what actually drives predictive signal.

### Key Findings

| Analysis | Result |
|----------|--------|
| **Predictive Windows** | No window (1/2/3 year) fixes the delta-based formula |
| **Best Predictor** | Combined Model (mean reversion + physical tools): r = 0.546 |
| **Mean Reversion** | Explains ~90% of the underperformance gap signal (r = 0.493 alone) |
| **Physical Tools** | Add modest but statistically significant incremental value (+0.068) |
| **Driveline Chain** | Max EV -> Hard Hit % -> Barrel % -> OPS confirmed as forward-causal |
| **Sprint Speed** | No Granger causality toward hitting metrics at any lag |

### Alternative Formulations (Out-of-Sample Test, 2022-2024)

| Formulation | r | Significant |
|-------------|---|-------------|
| Barrel Rate Delta | -0.232 | Yes (wrong direction) |
| Original MADDUX | -0.154 | Yes (wrong direction) |
| Absolute Levels (OLS) | 0.175 | Yes |
| OLS Optimized Deltas | 0.237 | Yes |
| Mean Reversion Baseline | 0.494 | Yes |
| Underperformance Gap (Raw) | 0.513 | Yes |
| Combined Model (MR + Tools) | 0.546 | Yes |

### Granger Causality (Driveline Chain, Lag 1)

| Link | Forward F | Reverse F | Direction |
|------|-----------|-----------|-----------|
| Max EV -> Hard Hit % | 140.8*** | 99.2*** | Bidirectional |
| Hard Hit % -> Barrel % | 71.9*** | 44.1*** | Bidirectional |
| Barrel % -> OPS | 45.6*** | 27.0*** | Bidirectional |
| Max EV -> OPS (full chain) | 87.4*** | 1.6 n.s. | **Forward only** |

## Data Coverage

- **Years**: 2015-2025
- **Player-Seasons**: 5,178
- **Delta Records**: 3,606 (consecutive year pairs)
- **Unique Players**: 1,341
- **Sprint Speed Coverage**: 99.6%

## Database Schema (`maddux_db.db`)

```
players (1,341 rows)
  player_id, player_name

player_seasons (5,178 rows, 15 columns)
  player_id, year, pa
  max_ev, hard_hit_pct, barrel_pct, sprint_speed
  ops, obp, slg, wrc_plus, iso, bb_pct, k_pct
  team

player_deltas (3,606 rows, 33 columns)
  player_id, year, prev_year, pa
  current + prev + delta for: max_ev, hard_hit_pct, barrel_pct,
    sprint_speed, ops, obp, slg, iso, bb_pct, k_pct
  maddux_score
  team
```

## File Structure

```
maddux-test-1/
├── README.md
├── requirements.txt
├── maddux_db.db                          # SQLite database (5,178 player-seasons)
├── dashboard.py                          # Interactive Streamlit dashboard
├── data/
│   ├── raw/                              # Source CSVs (Savant, FanGraphs, Sprint Speed)
│   └── processed/                        # Merged datasets
├── scripts/
│   ├── build_maddux_database.py          # Data pipeline (Savant + FanGraphs -> DB)
│   ├── pull_historical_data.py           # Baseball Savant data pull + sprint speed
│   ├── query_maddux.py                   # CLI Claude queries
│   ├── query_claude.py                   # Claude API integration (Test 2)
│   ├── calculate_scores.py              # MADDUX score calculation (Test 2)
│   ├── main.py                          # Original data pipeline (Test 1)
│   └── verify_maddux_methodology.py     # Methodology verification
├── phase2/
│   ├── phase2_predictive_windows.py      # Predictive window analysis (1/2/3 year)
│   ├── phase2_alternative_formulations.py # 7 alternative formulations with train/test
│   ├── phase2_granger_causality.py       # Driveline causal chain validation
│   ├── phase2_validate_gap_model.py      # Mean reversion decomposition
│   ├── phase2_report_charts.py           # Publication-quality report charts
│   └── outputs/                          # CSVs and PNGs from Phase 2 analyses
├── phase3/
│   ├── phase3_bat_speed_correlation.py   # Bat speed vs Max EV correlation
│   ├── phase3_model_comparison.py        # Model comparison tables
│   ├── MADDUX_V2_DEFINITION.md           # MADDUX v2 formula definition
│   ├── MILB_DATA_AUDIT.md                # MiLB data availability audit
│   ├── METRICS.md                        # Metrics glossary
│   └── outputs/                          # CSVs and PNGs from Phase 3 analyses
├── reports/
│   ├── MADDUX_Phase_2__*.pdf             # Phase 2 deliverable report
│   └── charts/                           # Publication-quality report charts
└── .streamlit/
    └── config.toml                       # Theme config
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set API key (for Claude integration)
export ANTHROPIC_API_KEY=your_key_here

# Run dashboard
streamlit run dashboard.py

# Run Phase 2 analyses
python phase2/phase2_predictive_windows.py
python phase2/phase2_alternative_formulations.py
python phase2/phase2_granger_causality.py
python phase2/phase2_validate_gap_model.py
python phase2/phase2_report_charts.py

# Run Phase 3 analyses
python phase3/phase3_bat_speed_correlation.py
python phase3/phase3_model_comparison.py
```

## Tech Stack

- **Data**: Baseball Savant (Statcast), FanGraphs
- **Database**: SQLite
- **Dashboard**: Streamlit + Plotly
- **AI**: Claude API (Haiku 4.5)
- **Analysis**: pandas, numpy, scipy, matplotlib, seaborn
- **Language**: Python 3.11+
