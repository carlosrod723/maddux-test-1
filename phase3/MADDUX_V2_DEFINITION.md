# MADDUX v2: Model Definition

**Prepared:** February 2026
**Purpose:** Restate the MADDUX scoring system using findings from Phase 2. Preserve the delta concept while fixing the implementation failure.

---

## 1. The Problem with v1

The original MADDUX formula measures **delta from last year:**

```
MADDUX_v1 = dMax_EV + (2.1 x dHard_Hit%)

where d = Year N value minus Year N-1 value
```

This is anti-predictive (r = -0.135 at one-year lag). Year-over-year deltas in physical metrics are dominated by noise and regression to the mean. Players whose metrics improved the most in Year N are statistically the most likely to regress in Year N+1, regardless of whether the improvement was real or random.

No prediction window (1, 2, or 3 years), weighting scheme, or metric substitution fixes this. All delta-based formulations tested in Phase 2 produced negative or near-zero predictive correlations.

---

## 2. The Fix: Delta from Expected, Not Delta from Last Year

MADDUX v2 redefines what "delta" means.

**v1 asks:** "Did this player's tools improve from last year?"
**v2 asks:** "Is this player producing below what his tools say he should?"

The v2 delta is the gap between where a player's physical tools say he *should* be performing and where he *actually* is. This gap has two components:

1. **Mean Reversion** — How far has the player fallen below his own career baseline? (Players bounce back from down years.)
2. **Tools Signal** — Do his physical tools suggest he should be performing *above* his own career baseline? (Physical metrics identify *which* players bounce back.)

---

## 3. The v2 Formula

### Stage 1: Expected OPS from Physical Tools

Fit a regression on the relationship between physical metrics and OPS using historical data (trained on 2015-2021, players with 2+ seasons, 200+ PA):

```
Expected_OPS = 0.8407
             + 0.0176 x Barrel%
             + 0.0012 x Hard_Hit%
             - 0.0009 x Max_EV
             + 0.4813 x BB%
             - 0.9046 x K%
```

These coefficients are from the same model run that produced the combined v2 weights below (`phase2_validate_gap_model.py`). The inputs are **absolute levels** — current-season values, not deltas.

Note on Max_EV coefficient: The small negative weight on Max_EV is because barrel rate already captures the useful EV-related signal more efficiently. Max_EV's independent contribution is absorbed once barrel rate is included. Max_EV remains important as the **entry point of the Driveline causal chain** and as the **historical proxy for bat speed** (r = 0.764).

### Stage 2: Decompose the Gap

For each player-season, compute two components:

```
Mean_Reversion = Career_Mean_OPS - Actual_OPS
Tools_Signal   = Expected_OPS   - Career_Mean_OPS
```

- **Mean_Reversion** is positive when a player is having a down year relative to his own career. This captures the statistical tendency to bounce back.
- **Tools_Signal** is positive when a player's physical tools predict he should be performing above his career average. This is the Driveline-aligned component — better bat speed, exit velocity, and barrel rate pushing expected OPS above what the player has historically produced.

### Stage 3: The MADDUX v2 Score

```
MADDUX_v2 = -0.0445 + 1.023 x Mean_Reversion + 0.558 x Tools_Signal
```

The v2 score predicts **next-year OPS change** (delta_OPS in Year N+1).

| Component | Weight | Interpretation |
|-----------|--------|---------------|
| Mean_Reversion | 1.023 | ~88% of predictive signal. Every 1 point of OPS below career mean predicts ~1 point of OPS gain next year. |
| Tools_Signal | 0.558 | ~12% of predictive signal, but statistically significant (t = 8.32, p < 0.001). Physical tools matter as a refinement layer. |

**Out-of-sample performance (test: 2022-2025):** r = 0.561, n = 652.

---

## 4. How v2 Preserves the Delta Concept

Matt's original intuition — that improving physical tools should predict future breakouts — is correct. The issue was how "improvement" was measured.

| Concept | v1 (Original) | v2 (Revised) |
|---------|---------------|--------------|
| **What is the delta?** | Change from last year's metrics | Deviation from tool-based expectation |
| **Reference point** | Last year's value | Career baseline + physical tool prediction |
| **What it captures** | "Are your tools improving?" | "Are your results lagging behind your tools?" |
| **Why it works** | It doesn't (r = -0.135) | Gap from expected regresses upward (r = +0.561) |

The v2 delta is still a delta — it's the distance between where you are and where you should be. It just uses a more stable reference point (career baseline + current tool levels) instead of a noisy one (last year's metrics).

---

## 5. Connection to the Driveline Causal Chain

The Driveline thesis is validated as the theoretical backbone of MADDUX v2:

```
Bat Speed ---> Max EV ---> Hard Hit% ---> Barrel% ---> OPS
  (r=0.764)    (F=140.8)   (F=71.9)      (F=45.6)
  proxy link    Granger     Granger       Granger

Full chain: Max EV --> OPS
  Forward:  F = 87.4***
  Reverse:  F = 1.6 n.s.
  Direction: ONE-WAY (tools drive production, not the reverse)
```

In v2, this chain feeds into the Expected_OPS calculation. A player with elite bat speed (proxied by Max_EV) will generate higher exit velocities, which produce more hard-hit balls and barrels, which push Expected_OPS higher. If his actual OPS is below that expectation, the Tools_Signal component is positive, and MADDUX v2 flags him as a breakout candidate.

---

## 6. Input Mapping Across All Three Versions

| Input | Original MADDUX (v1) | Driveline Framework | MADDUX v2 |
|-------|---------------------|-------------------|-----------|
| **Bat Speed** | Not available | Root of causal chain | Proxied by Max EV (r = 0.764) for 2015-2023. Direct data from 2024+. |
| **Max Exit Velocity** | dMax_EV (delta) | Second link in chain | Absolute level, feeds Expected_OPS model. Also serves as bat speed proxy. |
| **Hard Hit %** | dHard_Hit% (delta) | Third link in chain | Absolute level, feeds Expected_OPS model |
| **Barrel %** | Not included | Fourth link in chain | Absolute level, heaviest weight in Expected_OPS (coeff = 0.0176) |
| **BB%** | Not included | Not part of Driveline chain | Included in Expected_OPS (coeff = 0.4813). Plate discipline matters. |
| **K%** | Not included | Not part of Driveline chain | Included in Expected_OPS (coeff = -0.9046). Contact quality context. |
| **Career Mean OPS** | Not used | Not used | Anchor for mean reversion component |
| **Current OPS** | Used only as delta target | Outcome metric | Used to compute both gap components |
| **Sprint Speed** | Not included | Tested, no signal | Excluded (no Granger causality) |
| **Input Type** | Year-over-year deltas | Absolute levels (causal chain) | Absolute levels + career baseline |

---

## 7. Missing Pieces and Future Integration

### Available Now — Ready to Integrate

| Piece | Status | Impact |
|-------|--------|--------|
| **Bat speed (2024-2025)** | Data pulled from Baseball Savant | Can replace Max_EV proxy for recent seasons. Test whether direct bat speed improves Expected_OPS model. |
| **Swing metrics (swing length, squared-up rate)** | Available from 2024+ | Potential additional features in Expected_OPS model |
| **AAA Statcast data** | Public for 2022-2025 | Enables prospect translation (Stage 1 of the Phase 2 framework) |

### Needs More Time

| Piece | Timeline | What's Needed |
|-------|----------|--------------|
| **Bat speed historical depth** | ~2027 for 3+ seasons | Required to test bat speed's independent predictive value beyond Max_EV |
| **MiLB bat tracking** | Unknown | Not publicly available. Requires MLB to extend bat tracking to MiLB parks. |
| **Double-A / High-A Statcast** | Possibly post-2026 centralization | MLB centralization may eventually publish these levels |

### The Bat Speed Question

Bat speed correlates with Max_EV at r = 0.764 (n = 971, 2024-2025). This is strong enough for Max_EV to serve as a proxy in the 2015-2023 data where bat speed doesn't exist. But they are not the same thing:

- Max_EV captures the **single hardest contact event** in a season — partially skill, partially luck
- Bat speed captures **average swing mechanics** across all competitive swings — more stable, more trainable

Once 3+ years of bat speed data accumulate, the v2 Expected_OPS model should be re-tested with bat speed as a direct input. If bat speed carries independent predictive signal beyond Max_EV, it would strengthen the tools component of MADDUX v2.

---

## 8. Operational Summary

### How to Score a Player with MADDUX v2

1. **Gather current-season metrics:** Barrel%, Hard_Hit%, Max_EV, BB%, K%, OPS
2. **Compute Expected_OPS** from the tools regression
3. **Look up Career_Mean_OPS** (requires 2+ seasons of history)
4. **Compute the two components:**
   - Mean_Reversion = Career_Mean_OPS - Actual_OPS
   - Tools_Signal = Expected_OPS - Career_Mean_OPS
5. **Apply the v2 formula:**
   - MADDUX_v2 = -0.0445 + 1.023 x Mean_Reversion + 0.558 x Tools_Signal
6. **Interpret:** Positive score = model predicts OPS improvement next year. Higher score = stronger breakout signal.

### Breakout Candidates

Players with the highest MADDUX v2 scores are those who:
- Had a down year relative to their career (high Mean_Reversion)
- Still show strong physical tools (high Tools_Signal)

The model's edge over pure mean reversion is identifying players whose tools confirm the bounce-back thesis — separating genuine underperformers from players in legitimate decline.
