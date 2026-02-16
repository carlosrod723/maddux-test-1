# MiLB Data Availability Audit

**Prepared:** February 2026
**Purpose:** Inventory of publicly available minor league Statcast data, access methods, and risks to future availability.

---

## 1. Public MiLB Statcast Data — What Exists

Only **two minor league levels** have public Statcast data. Everything else is dark.

### Triple-A (AAA)

| Field | Detail |
|-------|--------|
| **Years available** | 2022 (partial), 2023-2025 (full) |
| **2022 coverage** | Pacific Coast League games + Charlotte home games only |
| **2023-2025 coverage** | All Triple-A games |
| **Source** | Baseball Savant Minor League Statcast Search |
| **Granularity** | Pitch-level (every tracked pitch) |

### Single-A / Florida State League (FSL)

| Field | Detail |
|-------|--------|
| **Years available** | 2021-2025 |
| **Coverage** | Florida State League parks equipped with ABS (Automated Ball-Strike) system |
| **Source** | Baseball Savant Minor League Statcast Search |
| **Granularity** | Pitch-level |
| **Note** | This is classified as "A" on Savant, tied to ABS deployment, not all Single-A parks |

### Levels With NO Public Statcast Data

| Level | Status |
|-------|--------|
| **Double-A (AA)** | No public Statcast data at any park |
| **High-A (A+)** | No public Statcast data (except FSL parks reclassified as Single-A) |
| **Low-A** | No public Statcast data |
| **Rookie / Complex League** | No public Statcast data |

---

## 2. Metrics Available at Public MiLB Levels

### Pitch-Level Metrics (AAA and FSL)

Available via Baseball Savant's minor league Statcast search. Same CSV schema as MLB Statcast data.

**Pitching / Tracking:**
- Release speed, release spin rate, release extension
- Spin axis
- Pitch movement (pfx_x, pfx_z)
- Pitch location at plate (plate_x, plate_z)
- Zone classification

**Batted Ball / Contact:**
- Exit velocity (launch_speed)
- Launch angle
- Hit distance
- Batted ball type (ground ball, fly ball, line drive, popup)

**Expected Stats:**
- xBA (expected batting average from speed + angle)
- xwOBA (expected weighted on-base average from speed + angle)

**Game Context:**
- Count, inning, score, runners, outs
- Plate appearance outcome (events)

### Metrics NOT Available for MiLB (Even at AAA)

| Metric | Why |
|--------|-----|
| **Bat speed** | Requires 300fps Hawk-Eye cameras, not installed at MiLB parks |
| **Swing length** | Same — bat tracking is MLB-only |
| **Attack angle** | Same |
| **Sprint speed** | Not publicly tracked for MiLB on Savant |
| **Biomechanics** | Force plates, motion capture — collected by some orgs internally, never public |

---

## 3. Data Sources and Access Methods

### A. Baseball Savant — Minor League Statcast Search (Primary)

**URL:** `https://baseballsavant.mlb.com/statcast-search-minors`

- Pitch-level data for AAA (2022-2025) and FSL/Single-A (2021-2025)
- CSV download: append `&type=details&csv=true` to query URL
- No authentication required
- Rate limiting applies — add delays between bulk requests

**Example CSV download URL:**
```
https://baseballsavant.mlb.com/statcast-search-minors
  ?hfPT=&hfAB=&hfGT=R&hfPR=&hfZ=&hfStadium=
  &hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2024|
  &hfSit=&player_type=batter&hfOuts=&hfOpponent=
  &pitcher_throws=&batter_stands=&hfSA=
  &game_date_gt=&game_date_lt=&hfMo=&hfTeam=
  &home_road=&hfRO=&position=&hfInfield=
  &hfOutfield=&hfInn=&hfBBT=&hfFlag=
  &metric_1=&group_by=name&min_pitches=0
  &min_results=0&min_pas=0&sort_col=pitches
  &player_event_sort=api_p_release_speed
  &sort_order=desc&type=details&csv=true
```

### B. MLB Stats API (Supplementary)

**Base URL:** `https://statsapi.mlb.com/api/`

- Play-by-play data for ALL MiLB levels (AAA through Rookie)
- Free, no authentication required
- Sport IDs: 11 (AAA), 12 (AA), 13 (High-A), 14 (A), 15 (Short Season), 16 (Rookie)
- **Critical limitation:** Most sensor-derived fields (velocity, spin, exit velo) are NULL for parks without Hawk-Eye hardware. Pitch locations are manually entered by stringers, not sensor-derived.
- Useful for: game schedules, rosters, box scores, play descriptions, plate appearance outcomes

**Python wrappers:**
- `MLB-StatsAPI` (`pip install MLB-StatsAPI`) — by toddrob99
- `python-mlb-statsapi` — alternative wrapper

**Workflow:**
1. Query schedule endpoint with date + sport_id to get game_pk values
2. Pass each game_pk to play-by-play endpoint
3. Parse and store

### C. FanGraphs Minor League Leaderboards

**URL:** `https://www.fangraphs.com/leaders/minor-league`

- Traditional batting/pitching stats for all MiLB levels (AVG, OBP, SLG, wOBA, wRC+, FIP, K%, BB%)
- Some batted ball profile data (GB%, FB%, LD%, HR/FB%)
- Sortable by level (AAA, AA, A+, A, CPX/Rookie)
- Historical data going back many years
- **No Statcast metrics** (exit velo, barrel rate, etc.) — FanGraphs links to Savant for that
- CSV export requires FanGraphs membership (free tier can view but not download)

### D. Other Sources

| Source | What It Provides |
|--------|-----------------|
| **MiLB.com** (`mlb.com/milb/stats`) | Official traditional stats (BA, HR, RBI, ERA) |
| **Baseball-Reference** | Minor league player pages, historical traditional stats |
| **pybaseball** | No native MiLB Statcast function — MLB-only. Custom scraper needed for Savant MiLB endpoint. |

---

## 4. 2026 Data Centralization Risk

### What Changed

In December 2025, MLB announced a sweeping data centralization regulation, reported by [Baseball America](https://www.baseballamerica.com/stories/8-takeaways-from-mlbs-new-minor-league-data-regulation-plan/), [Front Office Sports](https://frontofficesports.com/mlb-scouting-tech-winners-losers/), and others. The regulation takes effect for the **2026 MiLB season**, with full implementation phased in as existing team-vendor contracts are transitioned:

- **MLB becomes the sole data collector and distributor** for all tracking technology in MiLB. MLB will negotiate with vendors (Trackman, Hawk-Eye, Kinetrax, etc.), install hardware, and distribute standardized data packages to all 30 clubs equally.
- **No more team-proprietary hardware** at MiLB parks. Information collected by third-party vendors will no longer be proprietary to individual clubs.
- **Amateur scouting included** — MLB will regulate data and technology at college, high school, and showcase events. Teams receive all amateur data from MLB starting with the 2026 Draft.
- **Vendor relationships go through MLB** — teams can no longer negotiate directly with third-party data providers. Existing multi-year contracts are being brought in-house (expected to take time).

**Note on timeline:** This is not a hard January 1 switch. The regulation was announced December 2025, applies to the 2026 season, and the transition from team-owned to MLB-managed vendor contracts will be phased.

### What This Means for Public Access

**Short-term (2026):** Existing public data on Baseball Savant (AAA and FSL Statcast) is unlikely to disappear immediately. MLB expanded its public-facing Statcast tools as recently as March 2024 when the minor league search was launched.

**Medium-term risk:**
- Data currently public could move behind enterprise licensing
- New data types (bat tracking at MiLB, biomechanics, expanded lower-level Statcast) will likely be available to teams as paid packages but NOT made public
- MLB Stats API endpoints could become more restricted

**Potential upside:** [Baseball Prospectus](https://www.baseballprospectus.com/news/article/103608/mlb-owners-agree-to-share-their-minor-league-data/) and others have noted centralization could make MiLB tracking data *more* uniform and eventually more public — including equivalent Statcast data for Double-A and High-A — as MLB may use it for broadcasts and fan engagement, similar to how MLB Statcast data is used today.

---

## 5. Recommendation: Archive Now

Given the centralization risk, we recommend **downloading and storing all available historical MiLB Statcast data immediately:**

### Priority 1 — Archive Now

| Data | Source | Years | Action |
|------|--------|-------|--------|
| AAA pitch-level Statcast | Baseball Savant | 2022-2025 | Bulk CSV download by team/month |
| FSL/Single-A pitch-level Statcast | Baseball Savant | 2021-2025 | Bulk CSV download by team/month |
| AAA + FSL aggregated player stats | Computed from above | 2021-2025 | Aggregate exit velo, barrel rate, xwOBA per player-season |

### Priority 2 — Archive When Possible

| Data | Source | Years | Action |
|------|--------|-------|--------|
| All-level play-by-play | MLB Stats API | 2018-2025 | Query by game_pk, store locally |
| FanGraphs MiLB leaderboards | FanGraphs | All available | Export CSVs (requires membership for download) |
| Player ID crosswalk | Multiple sources | Current | Map player_id across Savant, FanGraphs, MiLB rosters |

### Priority 3 — Monitor

| Data | Source | Timeline |
|------|--------|----------|
| MiLB bat tracking | Baseball Savant (if published) | Unknown — depends on MLB's public data strategy |
| Double-A Statcast | Baseball Savant (if expanded) | No announced timeline |
| 2024 bat speed data (MLB) | Already available | Archive alongside MiLB for cross-level comparison |

---

## 6. Implications for the MADDUX Prospect Framework

The Phase 2 report proposed a 4-stage prospect translation framework. Here is its feasibility given current data:

| Stage | Requirement | Feasibility |
|-------|------------|-------------|
| **1. Level Adjustment Factors** | Exit velo, barrel rate, hard hit% at adjacent levels | Partially feasible: AAA data exists, but AA is missing. Cannot calculate AAA-to-AA adjustment. |
| **2. Adjusted Physical Tool Scores** | Normalize prospect metrics to MLB-equivalent baseline | Feasible for AAA-to-MLB translation only |
| **3. Expected MLB Performance** | Regression from adjusted tools to OPS | Feasible: can train on players who moved AAA -> MLB |
| **4. Confidence Tiers** | Multi-season data, cross-level consistency | Limited: most prospects have 1-2 seasons of AAA Statcast at best |

**Bottom line:** The prospect framework is viable as a AAA-to-MLB translation model today. Extending it below Triple-A requires either (a) MLB publishing more levels publicly, or (b) acquiring data through enterprise licensing.

---

## 7. Key Metrics for MiLB Analysis (What Analysts Use)

Published research and analysis from Pitcher List, FanGraphs, and independent analysts has identified the most predictive MiLB Statcast metrics for projecting MLB success:

| Metric | Correlation with Future MLB wOBAcon | Sample Size Needed | Available? |
|--------|-------------------------------------|--------------------|------------|
| **xwOBAcon** | r ~ 0.55 | 100+ BBE | Yes (AAA, FSL) |
| **Barrel rate per BBE** | r ~ 0.50 | 100+ BBE | Yes (AAA, FSL) |
| **90th percentile exit velocity** | r ~ 0.45 | 50+ BBE | Yes (AAA, FSL) |
| **Average exit velocity** | r ~ 0.40 | 100+ BBE | Yes (AAA, FSL) |
| **Max exit velocity** | r ~ 0.35 | 50+ BBE | Yes (AAA, FSL) |
| **Bat speed** | Unknown for MiLB | N/A | No (MLB-only) |
| **Sprint speed** | Not predictive of hitting | N/A | No (MLB-only) |
