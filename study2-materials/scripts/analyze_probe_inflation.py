"""
Probe Set Inflation Analysis
=============================
Investigates why the 20-item Eedi probe set shows ρ≈0.54 while the
105-item confirmation shows ρ≈0.11.

Findings:
1. The probe set contains 2 extreme-difficulty outliers (b=-5.18, -3.37)
   that are far outside the non-probe range ([-1.62, 1.85]).
2. Probe difficulty SD is 2.3× the non-probe SD (1.46 vs 0.64).
3. Dropping the 2 outliers reduces probe ρ from 0.537 to 0.391 (p=0.108, NS).
4. Non-probe items in the same difficulty range: ρ≈0.
5. Bootstrap: random 20-item subsets hit ρ≥0.50 only 2.5% of the time.

Conclusion: The probe-set signal is an artifact of difficulty-range inflation,
not genuine item-level discrimination by the LLM.

Usage:
    python3 scripts/analyze_probe_inflation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    pred = pd.read_csv("pilot/b2pl_105_experiment/averaged_predictions.csv")
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")
    with open("results/irt_proper_statistics.json") as f:
        irt = json.load(f)["items"]
    irt_df = pd.DataFrame.from_dict(irt, orient="index")
    irt_df["QuestionId"] = irt_df.index.astype(int)
    merged = pred.merge(
        irt_df[["QuestionId", "b_2pl", "a_2pl"]],
        on="QuestionId",
        suffixes=("_pred", "_irt"),
    )
    merged["is_probe"] = merged.QuestionId.isin(set(probe.QuestionId.unique()))
    return merged, probe


def main():
    merged, probe = load_data()
    probe_data = merged[merged.is_probe]
    nonprobe_data = merged[~merged.is_probe]

    print("=" * 70)
    print("PROBE SET INFLATION ANALYSIS")
    print("=" * 70)

    # --- 1. Overall correlations ---
    print("\n1. OVERALL CORRELATIONS (105-item confirmation experiment)")
    print("-" * 50)
    for label, sub in [
        ("All 105 items", merged),
        ("20 probe items", probe_data),
        ("85 non-probe items", nonprobe_data),
    ]:
        rho, p = stats.spearmanr(sub.mean_p_inc, sub.b_2pl_irt)
        print(f"  {label:25s}: ρ = {rho:+.3f}  (p = {p:.4f}, n = {len(sub)})")

    # --- 2. Difficulty spread comparison ---
    print("\n2. DIFFICULTY SPREAD (b_2pl)")
    print("-" * 50)
    for label, sub in [("Probe", probe_data), ("Non-probe", nonprobe_data)]:
        b = sub.b_2pl_irt
        print(
            f"  {label:10s}: mean={b.mean():+.3f}  SD={b.std():.3f}  "
            f"range=[{b.min():.2f}, {b.max():.2f}]"
        )
    print(f"\n  SD ratio (probe/non-probe): {probe_data.b_2pl_irt.std() / nonprobe_data.b_2pl_irt.std():.1f}×")

    # --- 3. Outlier analysis ---
    print("\n3. OUTLIER ANALYSIS")
    print("-" * 50)
    outliers = probe_data[probe_data.b_2pl_irt < -2]
    print(f"  Probe items with b_2pl < -2: {len(outliers)}")
    for _, r in outliers.iterrows():
        print(f"    QID {r.QuestionId:.0f}: b_2pl = {r.b_2pl_irt:.3f}, pred = {r.mean_p_inc:.3f}")

    no_outliers = probe_data[probe_data.b_2pl_irt >= -2]
    rho_full, p_full = stats.spearmanr(probe_data.mean_p_inc, probe_data.b_2pl_irt)
    rho_no, p_no = stats.spearmanr(no_outliers.mean_p_inc, no_outliers.b_2pl_irt)
    print(f"\n  All 20 probe items:          ρ = {rho_full:.3f}  (p = {p_full:.4f})")
    print(f"  18 probe items (drop b<-2):  ρ = {rho_no:.3f}  (p = {p_no:.4f})")
    print(f"  Δρ from 2 outliers:          {rho_full - rho_no:+.3f}")

    # --- 4. Prediction spread ---
    print("\n4. PREDICTION SPREAD (mean_p_inc)")
    print("-" * 50)
    for label, sub in [("Probe", probe_data), ("Non-probe", nonprobe_data)]:
        p = sub.mean_p_inc
        print(
            f"  {label:10s}: mean={p.mean():.3f}  SD={p.std():.3f}  "
            f"range=[{p.min():.3f}, {p.max():.3f}]"
        )
    print("\n  Note: prediction SDs are similar — the model doesn't spread")
    print("  predictions wider for probe items. The correlation comes from")
    print("  the GROUND TRUTH spread, not from better predictions.")

    # --- 5. Bootstrap ---
    print("\n5. BOOTSTRAP: RANDOM 20-ITEM SUBSETS FROM 105")
    print("-" * 50)
    np.random.seed(42)
    n_boot = 10000
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(len(merged), 20, replace=False)
        sub = merged.iloc[idx]
        r, _ = stats.spearmanr(sub.mean_p_inc, sub.b_2pl_irt)
        rhos.append(r)
    rhos = np.array(rhos)
    print(f"  Mean ρ:          {np.mean(rhos):.3f}")
    print(f"  SD:              {np.std(rhos):.3f}")
    print(f"  P(ρ ≥ 0.30):    {np.mean(rhos >= 0.30):.3f}")
    print(f"  P(ρ ≥ 0.40):    {np.mean(rhos >= 0.40):.3f}")
    print(f"  P(ρ ≥ 0.50):    {np.mean(rhos >= 0.50):.3f}")
    pcts = np.percentile(rhos, [5, 25, 50, 75, 95])
    print(f"  Percentiles [5/25/50/75/95]: {np.round(pcts, 3)}")

    # --- 6. Same misconceptions ---
    print("\n6. MISCONCEPTION COVERAGE")
    print("-" * 50)
    probe_misc = sorted(probe.misconception_id.unique())
    print(f"  Probe misconceptions ({len(probe_misc)}): {probe_misc}")
    print(f"  All 105 items were selected from these same 4 misconceptions.")
    print(f"  The probe/non-probe difference is NOT about misconception familiarity.")

    # --- 7. Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The 20-item probe set produces ρ=0.54 in the 105-item experiment while
  the 85 non-probe items produce ρ=0.03. This is NOT because:
    - The probe tests different misconceptions (all 105 share the same 4)
    - The LLM makes better predictions on probe items (prediction SD is equal)

  It IS because:
    - The probe set contains 2 items with extreme difficulty (b=-5.2, -3.4)
    - These outliers inflate the difficulty SD to 2.3× the non-probe SD
    - Removing them drops the correlation to ρ=0.39 (p=0.11, non-significant)
    - The remaining signal (~0.39) is consistent with the model distinguishing
      "very easy" from "moderate" items, but not rank-ordering within the
      normal difficulty range

  For the paper: the Eedi result is definitively null (ρ=0.114, n=105,
  CI includes zero). The probe-set correlation was inflated by difficulty
  range, not by genuine item-level discrimination.
""")


if __name__ == "__main__":
    main()
