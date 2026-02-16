"""
Phase 3, Item 1: Model Comparison Table
========================================
Compares Original MADDUX, Driveline thesis, and Phase 2 findings
across inputs, logic, and predictive performance.

Outputs:
  - model_comparison_table.csv
  - model_comparison_detailed.csv
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_core_comparison():
    """Side-by-side comparison of the three model versions."""
    rows = [
        {
            "dimension": "Core Thesis",
            "original_maddux": "Players whose exit velocity and hard contact rate are improving will see future OPS gains",
            "driveline_thesis": "Bat speed drives exit velocity, which drives hard contact, which drives offensive production",
            "phase2_combined_model": "Players whose physical tools suggest higher OPS than they actually produced will regress upward",
        },
        {
            "dimension": "Formula",
            "original_maddux": "MADDUX = dMax EV + (2.1 x dHard Hit%)",
            "driveline_thesis": "Bat Speed -> Max EV -> Hard Hit% -> Barrel% -> OPS (causal chain, no single formula)",
            "phase2_combined_model": "Gap = (Expected OPS from tools) - (Actual OPS), decomposed into Mean Reversion + Tools Signal",
        },
        {
            "dimension": "Input Metrics",
            "original_maddux": "Max Exit Velocity, Hard Hit %",
            "driveline_thesis": "Bat Speed, Max EV, Hard Hit %, Barrel %",
            "phase2_combined_model": "Barrel %, Max EV, Hard Hit %, Career Mean OPS, Current OPS",
        },
        {
            "dimension": "Input Type",
            "original_maddux": "Year-over-year DELTAS (Year N minus Year N-1)",
            "driveline_thesis": "ABSOLUTE LEVELS (current-year values at each chain link)",
            "phase2_combined_model": "GAP-BASED (deviation from tool-based expectation, using absolute levels)",
        },
        {
            "dimension": "Prediction Target",
            "original_maddux": "Next-year OPS change (dOPS Year N+1)",
            "driveline_thesis": "Next-year OPS (via forward causal chain)",
            "phase2_combined_model": "Next-year OPS change (dOPS Year N+1)",
        },
        {
            "dimension": "Predictive r (out-of-sample)",
            "original_maddux": "r = -0.135 (anti-predictive)",
            "driveline_thesis": "Not directly testable as a single score (validated as causal chain: F = 87.4 forward)",
            "phase2_combined_model": "r = +0.561 (best performer)",
        },
        {
            "dimension": "Same-Year r (concurrent)",
            "original_maddux": "r = +0.408",
            "driveline_thesis": "N/A (framework, not a formula)",
            "phase2_combined_model": "N/A (designed as a predictive model)",
        },
        {
            "dimension": "Why It Works / Fails",
            "original_maddux": "FAILS: Deltas capture noise + regression to mean. Players who improved most are most likely to regress.",
            "driveline_thesis": "WORKS as causal framework: Max EV Granger-causes OPS (F=87.4 forward, F=1.6 reverse). Chain is directional.",
            "phase2_combined_model": "WORKS: 88% of signal is mean reversion (players bounce back from down years). 12% is tools signal (physical metrics identify WHICH players bounce back). Both components are significant.",
        },
        {
            "dimension": "Bat Speed Role",
            "original_maddux": "Not included (bat speed data unavailable pre-2024)",
            "driveline_thesis": "Top of the causal chain (bat speed is the root input that Driveline trains)",
            "phase2_combined_model": "Proxied by Max EV (r = 0.764 with bat speed, validated in Phase 3). Max EV serves as the historical stand-in.",
        },
        {
            "dimension": "Barrel Rate Role",
            "original_maddux": "Not included",
            "driveline_thesis": "Middle of causal chain (Hard Hit% -> Barrel% -> OPS)",
            "phase2_combined_model": "Included as input to tools-based expected OPS. Stickiest power metric (r ~ 0.80 YoY).",
        },
        {
            "dimension": "Sprint Speed Role",
            "original_maddux": "Not included",
            "driveline_thesis": "Tested as biomechanics proxy",
            "phase2_combined_model": "Excluded: no Granger causality toward hitting metrics at any lag",
        },
        {
            "dimension": "Alignment with Driveline Logic",
            "original_maddux": "Partially aligned: uses correct metrics (EV, hard contact) but wrong input type (deltas instead of levels)",
            "driveline_thesis": "Fully aligned by definition (this IS the Driveline framework)",
            "phase2_combined_model": "Strongly aligned: uses absolute tool levels, validated causal chain direction, adds mean reversion baseline",
        },
    ]
    return pd.DataFrame(rows)


def build_formulation_results():
    """All tested formulations with out-of-sample performance."""
    rows = [
        {
            "formulation": "Original MADDUX (dMax EV + 2.1 x dHH%)",
            "input_type": "Deltas",
            "test_r": -0.154,
            "test_n": 708,
            "significant": "Yes (wrong direction)",
            "verdict": "Anti-predictive",
        },
        {
            "formulation": "Barrel Rate Delta (dBarrel%)",
            "input_type": "Deltas",
            "test_r": -0.232,
            "test_n": 708,
            "significant": "Yes (wrong direction)",
            "verdict": "Worse than original",
        },
        {
            "formulation": "OLS Optimized Deltas",
            "input_type": "Deltas (weighted)",
            "test_r": 0.237,
            "test_n": 708,
            "significant": "Yes",
            "verdict": "Positive but weak; learned weights are mean-reversion in disguise",
        },
        {
            "formulation": "Absolute Levels (Barrel%, Max EV, HH%)",
            "input_type": "Levels",
            "test_r": 0.175,
            "test_n": 708,
            "significant": "Yes",
            "verdict": "Modest signal from raw tools",
        },
        {
            "formulation": "Mean Reversion Baseline",
            "input_type": "Gap (career mean - actual)",
            "test_r": 0.494,
            "test_n": 652,
            "significant": "Yes",
            "verdict": "Strong; captures most of the underperformance gap signal",
        },
        {
            "formulation": "Underperformance Gap (raw)",
            "input_type": "Gap (expected - actual)",
            "test_r": 0.513,
            "test_n": 708,
            "significant": "Yes",
            "verdict": "Strong but ~88% is mean reversion",
        },
        {
            "formulation": "Combined Model (MR + Tools)",
            "input_type": "Gap (decomposed)",
            "test_r": 0.561,
            "test_n": 652,
            "significant": "Yes",
            "verdict": "Best performer; tools add +0.068 over mean reversion alone",
        },
    ]
    return pd.DataFrame(rows)


def build_causal_chain_summary():
    """Granger causality results validating the Driveline chain."""
    rows = [
        {
            "link": "Bat Speed -> Max EV",
            "test_type": "Pearson correlation (2024-2025)",
            "forward_stat": "r = 0.764 (n=971)",
            "reverse_stat": "N/A (same metric pair)",
            "verdict": "Strong proxy relationship; Max EV viable as historical bat speed stand-in",
        },
        {
            "link": "Max EV -> Hard Hit %",
            "test_type": "Granger causality (Lag 1)",
            "forward_stat": "F = 140.8***",
            "reverse_stat": "F = 99.2***",
            "verdict": "Bidirectional but forward stronger (1.42x ratio)",
        },
        {
            "link": "Hard Hit % -> Barrel %",
            "test_type": "Granger causality (Lag 1)",
            "forward_stat": "F = 71.9***",
            "reverse_stat": "F = 44.1***",
            "verdict": "Bidirectional but forward stronger (1.63x ratio)",
        },
        {
            "link": "Barrel % -> OPS",
            "test_type": "Granger causality (Lag 1)",
            "forward_stat": "F = 45.6***",
            "reverse_stat": "F = 27.0***",
            "verdict": "Bidirectional but forward stronger (1.69x ratio)",
        },
        {
            "link": "Max EV -> OPS (full chain)",
            "test_type": "Granger causality (Lag 1)",
            "forward_stat": "F = 87.4***",
            "reverse_stat": "F = 1.6 n.s.",
            "verdict": "ONE-DIRECTIONAL: EV predicts OPS, OPS does not predict EV (54.6x ratio)",
        },
        {
            "link": "Sprint Speed -> Hard Hit %",
            "test_type": "Granger causality (Lag 1-3)",
            "forward_stat": "Not significant at any lag",
            "reverse_stat": "Not significant at any lag",
            "verdict": "No causal relationship; sprint speed is not a hitting proxy",
        },
    ]
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 3: Model Comparison Table")
    print("=" * 60)

    # Build all three tables
    core = build_core_comparison()
    formulations = build_formulation_results()
    causal = build_causal_chain_summary()

    # Save detailed comparison
    core.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'model_comparison_table.csv'} ({len(core)} dimensions)")

    # Save formulation results
    formulations.to_csv(OUTPUT_DIR / "model_formulation_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'model_formulation_results.csv'} ({len(formulations)} formulations)")

    # Save causal chain
    causal.to_csv(OUTPUT_DIR / "model_causal_chain_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'model_causal_chain_summary.csv'} ({len(causal)} links)")

    # Print formatted summary
    print("\n" + "=" * 60)
    print("TABLE 1: CORE MODEL COMPARISON")
    print("=" * 60)
    for _, row in core.iterrows():
        print(f"\n  {row['dimension'].upper()}")
        print(f"    Original MADDUX:  {row['original_maddux']}")
        print(f"    Driveline Thesis: {row['driveline_thesis']}")
        print(f"    Phase 2 Combined: {row['phase2_combined_model']}")

    print("\n" + "=" * 60)
    print("TABLE 2: FORMULATION RESULTS (Out-of-Sample, 2022-2025)")
    print("=" * 60)
    print(f"  {'Formulation':<45} {'Type':<20} {'r':>8} {'Verdict'}")
    print(f"  {'-'*45} {'-'*20} {'-'*8} {'-'*40}")
    for _, row in formulations.iterrows():
        print(f"  {row['formulation']:<45} {row['input_type']:<20} {row['test_r']:>+8.3f} {row['verdict']}")

    print("\n" + "=" * 60)
    print("TABLE 3: CAUSAL CHAIN VALIDATION")
    print("=" * 60)
    for _, row in causal.iterrows():
        print(f"\n  {row['link']}")
        print(f"    Forward: {row['forward_stat']}")
        print(f"    Reverse: {row['reverse_stat']}")
        print(f"    Verdict: {row['verdict']}")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
  All three models share the same CORE LOGIC:
    Physical tools (bat speed / exit velocity / hard contact) drive offensive production.

  Where they diverge is IMPLEMENTATION:
    - Original MADDUX: Correct thesis, wrong input type (deltas regress to mean)
    - Driveline Thesis: Correct thesis, validated causal direction (forward-only)
    - Phase 2 Combined: Correct thesis, correct implementation (gap from expected, not delta from last year)

  The bat speed correlation (r = 0.764) confirms Max EV is a viable
  proxy for bat speed in historical data (2015-2023), bridging the
  Driveline thesis to the MADDUX framework.
""")


if __name__ == "__main__":
    main()
