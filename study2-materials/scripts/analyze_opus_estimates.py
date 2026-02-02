#!/usr/bin/env python3
"""
Analyze Opus 4.5 difficulty estimates vs actual student performance.

This script is run AFTER the experiment to compare predicted vs actual error rates.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_actual_data():
    """Load actual student performance data from EEDI."""
    df = pd.read_csv("data/eedi/curated_eedi_items.csv")

    # Calculate actual error rate (100 - correct_answer_percentage)
    # correct_answer uses NeurIPS positional ordering (matches pct_A/B/C/D)
    actual_data = {}
    for _, row in df.iterrows():
        qid = row["QuestionId"]
        correct = row["correct_answer"]
        correct_pct = row[f"pct_{correct}"]
        actual_error_rate = 100 - correct_pct

        if qid not in actual_data:
            actual_data[qid] = {
                "actual_error_rate": actual_error_rate,
                "correct_pct": correct_pct,
                "total_responses": row["total_responses"],
                "misconception_id": row["misconception_id"],
                "target_type": row["target_type"],
            }

    return actual_data


def load_opus_estimates(results_file):
    """Load Opus estimates from JSONL file."""
    estimates = []
    with open(results_file) as f:
        for line in f:
            estimates.append(json.loads(line))
    return estimates


def analyze_accuracy(estimates, actual_data):
    """Compute accuracy metrics comparing estimates to actual."""

    # Aggregate multiple estimates per QID
    qid_estimates = {}
    for est in estimates:
        qid = est["qid"]
        if qid not in qid_estimates:
            qid_estimates[qid] = []
        qid_estimates[qid].append(est["error_rate"])

    # Compare to actual
    comparison = []
    for qid, est_list in qid_estimates.items():
        if qid in actual_data:
            actual = actual_data[qid]
            mean_estimate = np.mean(est_list)
            comparison.append({
                "qid": qid,
                "predicted_error_rate": mean_estimate,
                "actual_error_rate": actual["actual_error_rate"],
                "abs_error": abs(mean_estimate - actual["actual_error_rate"]),
                "num_estimates": len(est_list),
                "estimate_std": np.std(est_list) if len(est_list) > 1 else 0,
                "misconception_id": actual["misconception_id"],
                "target_type": actual["target_type"],
                "total_responses": actual["total_responses"],
            })

    return pd.DataFrame(comparison)


def main():
    # Find most recent results file
    results_dir = Path("pilot/opus_difficulty_estimation")
    results_files = list(results_dir.glob("results_*.jsonl"))

    if not results_files:
        print("No results files found. Run run_opus_difficulty_estimation.py first.")
        return

    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_file}")

    # Load data
    estimates = load_opus_estimates(latest_file)
    actual_data = load_actual_data()

    print(f"\nLoaded {len(estimates)} estimates")
    print(f"Loaded {len(actual_data)} items with actual data")

    # Analyze
    df = analyze_accuracy(estimates, actual_data)

    if df.empty:
        print("No matching QIDs found between estimates and actual data.")
        return

    print("\n" + "=" * 60)
    print("OPUS 4.5 DIFFICULTY ESTIMATION ACCURACY")
    print("=" * 60)

    # Overall metrics
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["abs_error"] ** 2).mean())
    correlation = stats.pearsonr(df["predicted_error_rate"], df["actual_error_rate"])

    print(f"\nOverall Metrics (n={len(df)} items):")
    print(f"  Mean Absolute Error: {mae:.1f} percentage points")
    print(f"  RMSE: {rmse:.1f} percentage points")
    print(f"  Pearson correlation: r={correlation[0]:.3f} (p={correlation[1]:.4f})")

    # By misconception type
    print("\nBy Misconception Type:")
    for mtype in df["target_type"].unique():
        subset = df[df["target_type"] == mtype]
        subset_mae = subset["abs_error"].mean()
        subset_corr = stats.pearsonr(subset["predicted_error_rate"], subset["actual_error_rate"])
        print(f"  {mtype}: MAE={subset_mae:.1f}, r={subset_corr[0]:.3f} (n={len(subset)})")

    # By misconception ID
    print("\nBy Misconception ID:")
    for mid in df["misconception_id"].unique():
        subset = df[df["misconception_id"] == mid]
        subset_mae = subset["abs_error"].mean()
        if len(subset) >= 3:
            subset_corr = stats.pearsonr(subset["predicted_error_rate"], subset["actual_error_rate"])
            print(f"  {int(mid)}: MAE={subset_mae:.1f}, r={subset_corr[0]:.3f} (n={len(subset)})")
        else:
            print(f"  {int(mid)}: MAE={subset_mae:.1f} (n={len(subset)}, too few for correlation)")

    # Estimate consistency (for items estimated multiple times)
    multi_estimates = df[df["num_estimates"] > 1]
    if not multi_estimates.empty:
        print(f"\nEstimate Consistency (n={len(multi_estimates)} items with >1 estimate):")
        print(f"  Mean within-item std: {multi_estimates['estimate_std'].mean():.1f} pp")

    # Best and worst predictions
    print("\nBest Predictions (lowest abs error):")
    for _, row in df.nsmallest(5, "abs_error").iterrows():
        print(f"  QID {row['qid']}: predicted={row['predicted_error_rate']:.0f}%, "
              f"actual={row['actual_error_rate']:.0f}%, error={row['abs_error']:.0f}pp")

    print("\nWorst Predictions (highest abs error):")
    for _, row in df.nlargest(5, "abs_error").iterrows():
        print(f"  QID {row['qid']}: predicted={row['predicted_error_rate']:.0f}%, "
              f"actual={row['actual_error_rate']:.0f}%, error={row['abs_error']:.0f}pp")

    # Save detailed comparison
    output_file = latest_file.parent / f"analysis_{latest_file.stem.replace('results_', '')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed comparison saved to: {output_file}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
