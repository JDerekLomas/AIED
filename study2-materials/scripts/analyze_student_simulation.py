#!/usr/bin/env python3
"""
Analyze student simulation results and compare with human data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
RESULTS_DIR = Path("pilot/student_simulation")
FINAL_ITEMS_PATH = Path("results/final_items.json")


def load_latest_results():
    """Load the most recent results file."""
    results_files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
    if not results_files:
        raise FileNotFoundError("No results files found")

    latest = results_files[-1]
    print(f"Loading: {latest}")

    results = []
    with open(latest) as f:
        for line in f:
            results.append(json.loads(line))

    return pd.DataFrame(results)


def load_human_data():
    """Load human performance data from final items."""
    with open(FINAL_ITEMS_PATH) as f:
        data = json.load(f)

    items = []
    for item in data["items"]:
        items.append({
            "question_id": item["question_id"],
            "misconception_id": item["misconception_id"],
            "human_correct_rate": item.get(f"pct_{item.get('correct_answer_neurips', item['correct_answer'])}", 0),
            "human_target_rate": item.get("human_target_rate", 0),
            "pct_correct": item.get(f"pct_{item.get('correct_answer_neurips', item['correct_answer'])}", 0),
            "pct_target": item.get(f"pct_{item['target_distractor']}", 0),
        })

    return pd.DataFrame(items)


def analyze_results():
    """Main analysis."""
    df = load_latest_results()
    human_df = load_human_data()

    print(f"\n{'='*60}")
    print("STUDENT SIMULATION RESULTS")
    print(f"{'='*60}")

    # Overall statistics
    print(f"\nOverall: {len(df)} responses")
    print(f"  Correct: {df['is_correct'].mean():.1%}")
    print(f"  Target misconception: {df['selected_target'].mean():.1%}")

    # By proficiency
    print(f"\n{'='*60}")
    print("BY PROFICIENCY LEVEL")
    print(f"{'='*60}")

    prof_stats = df.groupby('proficiency').agg({
        'is_correct': ['count', 'mean'],
        'selected_target': 'mean'
    }).round(3)
    prof_stats.columns = ['n', 'correct_rate', 'target_rate']

    for prof in ['struggling', 'developing', 'secure', 'confident']:
        if prof in prof_stats.index:
            row = prof_stats.loc[prof]
            print(f"  {prof:12s}: n={int(row['n']):3d}, correct={row['correct_rate']:.1%}, target={row['target_rate']:.1%}")

    # By misconception type
    print(f"\n{'='*60}")
    print("BY MISCONCEPTION")
    print(f"{'='*60}")

    misc_stats = df.groupby('misconception_id').agg({
        'is_correct': ['count', 'mean'],
        'selected_target': 'mean'
    }).round(3)
    misc_stats.columns = ['n', 'correct_rate', 'target_rate']

    misconception_names = {
        217: "Fraction addition (add tops/bottoms)",
        1507: "Order of operations (left-to-right)",
        1597: "Neg × Neg = Neg"
    }

    for misc_id in [217, 1507, 1597]:
        if misc_id in misc_stats.index:
            row = misc_stats.loc[misc_id]
            name = misconception_names.get(misc_id, str(misc_id))
            print(f"  {misc_id}: {name}")
            print(f"       n={int(row['n']):3d}, correct={row['correct_rate']:.1%}, target={row['target_rate']:.1%}")

    # Compare with human data
    print(f"\n{'='*60}")
    print("COMPARISON WITH HUMAN DATA")
    print(f"{'='*60}")

    # Aggregate LLM responses by question
    llm_by_item = df.groupby('question_id').agg({
        'is_correct': 'mean',
        'selected_target': 'mean'
    }).reset_index()
    llm_by_item.columns = ['question_id', 'llm_correct_rate', 'llm_target_rate']

    # Merge with human data
    comparison = pd.merge(llm_by_item, human_df, on='question_id', how='inner')

    # Correlations
    corr_correct = stats.pearsonr(comparison['llm_correct_rate'], comparison['pct_correct'])
    corr_target = stats.pearsonr(comparison['llm_target_rate'], comparison['pct_target'])

    spearman_correct = stats.spearmanr(comparison['llm_correct_rate'], comparison['pct_correct'])
    spearman_target = stats.spearmanr(comparison['llm_target_rate'], comparison['pct_target'])

    print(f"\nCorrelation: LLM correct rate vs Human correct rate")
    print(f"  Pearson r = {corr_correct[0]:.3f} (p = {corr_correct[1]:.4f})")
    print(f"  Spearman ρ = {spearman_correct[0]:.3f} (p = {spearman_correct[1]:.4f})")

    print(f"\nCorrelation: LLM target rate vs Human target rate")
    print(f"  Pearson r = {corr_target[0]:.3f} (p = {corr_target[1]:.4f})")
    print(f"  Spearman ρ = {spearman_target[0]:.3f} (p = {spearman_target[1]:.4f})")

    # MAE and RMSE
    mae_correct = np.abs(comparison['llm_correct_rate'] * 100 - comparison['pct_correct']).mean()
    mae_target = np.abs(comparison['llm_target_rate'] * 100 - comparison['pct_target']).mean()

    print(f"\nMean Absolute Error:")
    print(f"  Correct rate: {mae_correct:.1f} percentage points")
    print(f"  Target rate: {mae_target:.1f} percentage points")

    # By proficiency comparison
    print(f"\n{'='*60}")
    print("LLM VS HUMAN BY PROFICIENCY")
    print(f"{'='*60}")

    for prof in ['struggling', 'developing', 'secure', 'confident']:
        prof_df = df[df['proficiency'] == prof]
        if len(prof_df) == 0:
            continue

        prof_by_item = prof_df.groupby('question_id').agg({
            'is_correct': 'mean',
            'selected_target': 'mean'
        }).reset_index()
        prof_by_item.columns = ['question_id', 'llm_correct', 'llm_target']

        merged = pd.merge(prof_by_item, human_df, on='question_id')

        if len(merged) > 5:  # Need enough data points
            r_correct = stats.pearsonr(merged['llm_correct'], merged['pct_correct'])
            r_target = stats.pearsonr(merged['llm_target'], merged['pct_target'])

            print(f"\n{prof}:")
            print(f"  Correct rate correlation: r = {r_correct[0]:.3f}")
            print(f"  Target rate correlation: r = {r_target[0]:.3f}")

    # Save detailed comparison
    comparison_file = RESULTS_DIR / "comparison_with_human.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"\nDetailed comparison saved to: {comparison_file}")

    return df, comparison


def print_interesting_cases(df, comparison):
    """Print cases where LLM and human diverge most."""
    print(f"\n{'='*60}")
    print("INTERESTING DIVERGENCES")
    print(f"{'='*60}")

    # Items where LLM target rate differs most from human
    comparison['target_diff'] = np.abs(comparison['llm_target_rate'] * 100 - comparison['pct_target'])
    comparison['correct_diff'] = np.abs(comparison['llm_correct_rate'] * 100 - comparison['pct_correct'])

    print("\nLargest target rate divergences:")
    top_target = comparison.nlargest(5, 'target_diff')
    for _, row in top_target.iterrows():
        print(f"  Q{int(row['question_id'])}: LLM={row['llm_target_rate']*100:.1f}%, Human={row['pct_target']:.1f}% (diff={row['target_diff']:.1f}pp)")

    print("\nLargest correct rate divergences:")
    top_correct = comparison.nlargest(5, 'correct_diff')
    for _, row in top_correct.iterrows():
        print(f"  Q{int(row['question_id'])}: LLM={row['llm_correct_rate']*100:.1f}%, Human={row['pct_correct']:.1f}% (diff={row['correct_diff']:.1f}pp)")


if __name__ == "__main__":
    df, comparison = analyze_results()
    print_interesting_cases(df, comparison)
