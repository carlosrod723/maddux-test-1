# MADDUX Analytics — Metrics Glossary

Reference document for all metrics, concepts, and terminology used in the MADDUX analytics platform. Covers Phase 1 (original formula) and Phase 2 (alternative formulations, predictive analysis).

---

## Batting Metrics (What we measure)

**OPS (On-Base Plus Slugging)** — OBP + SLG combined. The primary outcome metric we're trying to predict. A player with .800+ OPS is strong, .700 is average, below .650 is poor. This is the primary forecast target.

**ISO (Isolated Power)** — SLG minus batting average. Measures raw power only, stripping out singles. A .200+ ISO is elite power.

**wRC+ (Weighted Runs Created Plus)** — A park-adjusted, league-adjusted measure of total offensive value. 100 is league average. 120+ is very good. More comprehensive than OPS but harder to interpret intuitively.

**wOBA (Weighted On-Base Average)** — Like OPS but weights each outcome (single, double, HR, walk) by its actual run value. More accurate than OPS. League average is around .310-.320.

**xwOBA (Expected wOBA)** — What a player's wOBA should be based on how hard they hit the ball and at what angle, removing luck, defense, and ballpark. When xwOBA >> wOBA, the player was unlucky and may bounce back.

---

## Statcast Physical Metrics (What a player's body does)

**Exit Velocity (EV)** — How fast the ball comes off the bat in mph. Average EV across all batted balls. League average is roughly 88-89 mph.

**Max Exit Velocity (Max EV)** — The hardest single ball a player hit all season. Used in the current MADDUX formula. Less stable year-to-year than other EV measures.

**EV on FB/LD (Exit Velocity on Fly Balls and Line Drives)** — Average exit velocity only on balls hit in the air. The stickiest Statcast metric (r ~ 0.82 year-to-year). More predictive than max EV or average EV because it filters out ground balls.

**Hard Hit % (HH%)** — Percentage of batted balls hit 95+ mph. Used in the current MADDUX formula. A blunt measure — doesn't account for launch angle.

**Barrel Rate (Barrel %)** — Percentage of batted balls that are "barreled" — hit at 98+ mph exit velocity AND at an optimal launch angle (26-30 degrees at 98 mph, range widens as EV increases). This is the king metric — it combines exit velocity AND launch angle into one number. Stickiest power metric (r ~ 0.80 year-to-year), strongest predictor of future HR output (r = 0.66-0.80 with HR/FB rate). A barrel produces a .750+ batting average and 2.400+ slugging. Elite is 15%+, average is 6-9%.

**Bat Speed** — How fast the bat moves through the zone in mph. Only publicly available from 2024 onward via Statcast bat tracking. Driveline's core training metric. Every 1 mph of bat speed ~ 1.2 mph of exit velocity ~ 7 feet of batted ball distance.

**Sprint Speed** — A player's top running speed in feet per second, measured on competitive plays. Available 2015-present. Our proxy for lower body athleticism/biomechanics since ground force data isn't public.

**Swing Length** — Distance the bat travels through the swing, in feet. Only available 2024+ like bat speed.

---

## Plate Discipline Metrics (Approach/decision-making)

**BB% (Walk Rate)** — Percentage of plate appearances resulting in a walk. Indicates patience and pitch recognition.

**K% (Strikeout Rate)** — Percentage of plate appearances resulting in a strikeout. Lower is generally better for contact.

**Contact %** — Percentage of swings where the bat makes contact with the ball.

---

## Analytical Concepts (How we analyze)

**Delta (d)** — Year-over-year change. d Hard Hit % = this year's HH% minus last year's. The MADDUX formula is built entirely on deltas. The problem: deltas regress to the mean.

**MADDUX Score** — The current formula: d Max EV + (2.1 x d Hard Hit %). A positive score means the player's physical metrics improved. Our Phase 1 analysis proved this doesn't predict next-year OPS.

**Regression to the Mean** — The statistical tendency for extreme values to move back toward average. A player who jumps 10% in hard hit rate is more likely to drop back than continue improving. This is why delta-based prediction fails — you're measuring noise, not signal.

**Stickiness** — How well a metric correlates with itself from one year to the next. High stickiness = the metric reflects real skill. Low stickiness = too much noise. Barrel rate and EV on FB/LD are very sticky. Max EV and hard hit % deltas are not.

**Underperformance Gap** — The difference between what a player's physical tools predict they should produce versus what they actually produced. Example: a player with elite barrel rate and EV but a .700 OPS is underperforming — the tools say he should be at .800+. These players tend to "catch up" the following year. This is the most promising predictive framework the research supports.

**Granger Causality** — A statistical test for whether one time series (e.g., bat speed changes) predicts future values of another time series (e.g., hard hit rate changes). It doesn't prove true causation, but it tests whether X happens before Y in a statistically significant pattern. Used to test "not just correlation, but directionality."

**Lag** — The time delay between a change in one metric and its effect on another. A 1-year lag means a bat speed improvement this year shows up as an OPS improvement next year. Key question: is the optimal lag 1, 2, or 3 years?

**Predictive Window** — The number of years of data used to make a prediction. A 1-year window uses only last year's data. A 3-year window averages the last 3 years. Our Phase 2 results showed no window fixes the delta-based MADDUX formula.
