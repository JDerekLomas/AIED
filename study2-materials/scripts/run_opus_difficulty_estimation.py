#!/usr/bin/env python3
"""
Experiment: Claude Opus 4.5 difficulty estimation for EEDI items.

Design:
- 25 test items (stratified across 4 misconception types)
- 10 calls with the SAME 25 items each time
- 10 estimates per item → measure consistency
- 3 calibration examples with actual difficulty revealed (anchors)

Population context:
- UK secondary school students (ages 11-16)
- Math diagnostic assessments
- Items from NeurIPS 2020 EEDI competition
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Configuration
NUM_CALLS = 10
NUM_TEST_ITEMS = 25
NUM_CALIBRATION_ITEMS = 3
MODEL_ID = "claude-opus-4-5-20251101"
OUTPUT_DIR = Path("pilot/opus_difficulty_estimation")

# Seed for reproducibility
RANDOM_SEED = 42

# Fields that contain actual student statistics - MUST BE EXCLUDED from test items
FORBIDDEN_FIELDS = [
    "student_selection_pct",
    "total_responses",
    "pct_A",
    "pct_B",
    "pct_C",
    "pct_D",
    "resp_A",
    "resp_B",
    "resp_C",
    "resp_D",
]


def load_full_data():
    """Load EEDI items with all data (for calibration selection)."""
    df = pd.read_csv("data/eedi/curated_eedi_items.csv")

    # Get answer options from main EEDI file
    eedi_df = pd.read_csv("data/eedi/eedi_with_student_data.csv")
    answer_cols = ["QuestionId", "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
    answers_df = eedi_df[answer_cols].drop_duplicates()

    # Merge
    df = df.merge(answers_df, on="QuestionId", how="left")

    # Calculate actual error rate
    # correct_answer now uses NeurIPS positional ordering (matches pct_A/B/C/D)
    df["actual_error_rate"] = 100 - df.apply(lambda r: r[f"pct_{r['correct_answer']}"], axis=1)

    # Drop duplicates (one row per question)
    df = df.drop_duplicates(subset=["QuestionId"])

    return df


def select_calibration_items(df, n=3):
    """Select calibration items spanning the difficulty range."""
    # Sort by actual error rate
    sorted_df = df.sort_values("actual_error_rate")

    # Pick easy (~20th percentile), medium (~50th), hard (~80th)
    n_items = len(sorted_df)
    indices = [
        int(n_items * 0.2),  # easy
        int(n_items * 0.5),  # medium
        int(n_items * 0.8),  # hard
    ]

    calibration = sorted_df.iloc[indices].to_dict("records")
    return calibration


def select_test_items(df, calibration_qids, n=25, seed=42):
    """Select test items stratified by misconception type, excluding calibration items."""
    random.seed(seed)

    # Exclude calibration items
    test_pool = df[~df["QuestionId"].isin(calibration_qids)]

    # Stratify by misconception_id (4 types)
    misconception_ids = test_pool["misconception_id"].unique()
    items_per_type = n // len(misconception_ids)
    remainder = n % len(misconception_ids)

    selected = []
    for i, mid in enumerate(misconception_ids):
        pool = test_pool[test_pool["misconception_id"] == mid]
        # Add one extra to first few types if there's a remainder
        take = items_per_type + (1 if i < remainder else 0)
        take = min(take, len(pool))
        sampled = pool.sample(n=take, random_state=seed)
        selected.append(sampled)

    test_items = pd.concat(selected).to_dict("records")
    random.shuffle(test_items)

    return test_items


def strip_statistics(item):
    """Remove actual statistics from item for test prompt."""
    safe_item = {k: v for k, v in item.items() if k not in FORBIDDEN_FIELDS and k != "actual_error_rate"}
    return safe_item


def build_estimation_prompt(test_items, calibration_items, call_num):
    """Build prompt with calibration examples and test items."""

    # Calibration section
    calibration_text = ""
    for i, item in enumerate(calibration_items, 1):
        calibration_text += f"""
**Calibration Example {i}:**
Question: {item['question_text']}
Options: A) {item['AnswerAText']} | B) {item['AnswerBText']} | C) {item['AnswerCText']} | D) {item['AnswerDText']}
Correct: {item['correct_answer']}
**Actual error rate: {item['actual_error_rate']:.0f}%** (This is the real student data - use as calibration)
"""

    # Test items section
    items_text = []
    for i, item in enumerate(test_items, 1):
        item_block = f"""
### Item {i} (QID: {item['QuestionId']})

**Question:**
{item['question_text']}

**Options:**
A) {item['AnswerAText']}
B) {item['AnswerBText']}
C) {item['AnswerCText']}
D) {item['AnswerDText']}

**Correct Answer:** {item['correct_answer']}

**Misconception tested (distractor {item['target_distractor']}):**
{item['misconception_name']}
"""
        items_text.append(item_block)

    prompt = f"""You are an expert in mathematics education assessment. Your task is to estimate the difficulty of multiple-choice math items for a specific student population.

## Student Population
- **Location:** United Kingdom
- **Age range:** 11-16 years old (secondary school)
- **Context:** Diagnostic math assessments used to identify misconceptions
- **Data source:** NeurIPS 2020 EEDI competition (real student responses from 2018-2020)

## Calibration Examples
These examples show ACTUAL student performance to calibrate your estimates:
{calibration_text}

## Your Task
For each test item below, estimate the **error rate** - the percentage of students who would answer INCORRECTLY.

Consider:
1. The mathematical concept and its typical difficulty for this age group
2. How attractive the distractors are (based on common misconceptions)
3. The clarity of the question
4. Use the calibration examples above to anchor your estimates

## Response Format
Provide estimates in this exact JSON format:
```json
{{
  "estimates": [
    {{"qid": <QuestionId>, "error_rate": <0-100>, "reasoning": "<1-2 sentence explanation>"}},
    ...
  ]
}}
```

---

## Test Items (estimate these)
{"".join(items_text)}

---

Analyze each item and provide your difficulty estimates. This is call {call_num} of {NUM_CALLS}.
"""
    return prompt


def parse_response(response_text):
    """Extract JSON estimates from model response."""
    import re

    # Try to find JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r'\{[\s\S]*"estimates"[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def run_experiment():
    """Run the difficulty estimation experiment."""

    print("=" * 60)
    print("OPUS 4.5 DIFFICULTY ESTIMATION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Test items: {NUM_TEST_ITEMS}")
    print(f"Calibration items: {NUM_CALIBRATION_ITEMS}")
    print(f"Calls (estimates per item): {NUM_CALLS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # Load data
    df = load_full_data()
    print(f"Loaded {len(df)} unique items from EEDI")

    # Select calibration items (with actual stats visible)
    calibration_items = select_calibration_items(df, NUM_CALIBRATION_ITEMS)
    calibration_qids = [item["QuestionId"] for item in calibration_items]
    print(f"\nCalibration items (QIDs): {calibration_qids}")
    for item in calibration_items:
        print(f"  QID {item['QuestionId']}: {item['actual_error_rate']:.0f}% error rate")

    # Select test items (stratified, stats stripped)
    test_items = select_test_items(df, calibration_qids, NUM_TEST_ITEMS, RANDOM_SEED)
    test_qids = [item["QuestionId"] for item in test_items]
    print(f"\nTest items (QIDs): {test_qids}")

    # Store actual error rates for later analysis (NOT shown to model)
    actual_error_rates = {item["QuestionId"]: item["actual_error_rate"] for item in test_items}

    # Set up output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"results_{timestamp}.jsonl"
    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"

    # Save experiment config
    config = {
        "model": MODEL_ID,
        "num_calls": NUM_CALLS,
        "num_test_items": NUM_TEST_ITEMS,
        "random_seed": RANDOM_SEED,
        "calibration_qids": calibration_qids,
        "test_qids": test_qids,
        "actual_error_rates": actual_error_rates,
    }
    config_file = OUTPUT_DIR / f"config_{timestamp}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_file}")

    # Initialize client
    client = Anthropic()

    all_estimates = []
    call_metadata = []

    print("\n" + "-" * 60)
    print("RUNNING EXPERIMENT")
    print("-" * 60)

    for call_num in range(1, NUM_CALLS + 1):
        print(f"\n--- Call {call_num}/{NUM_CALLS} ---")

        # Build prompt (same items each time, just different call number)
        prompt = build_estimation_prompt(test_items, calibration_items, call_num)

        try:
            start_time = time.time()
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - start_time

            raw_response = response.content[0].text
            parsed = parse_response(raw_response)

            if parsed and "estimates" in parsed:
                estimates = parsed["estimates"]
                print(f"  Parsed {len(estimates)} estimates in {elapsed:.1f}s")

                # Validate and save each estimate
                for est in estimates:
                    est["call_num"] = call_num
                    est["timestamp"] = timestamp

                    # Add actual error rate for analysis
                    qid = est.get("qid")
                    if qid in actual_error_rates:
                        est["actual_error_rate"] = actual_error_rates[qid]
                        est["abs_error"] = abs(est["error_rate"] - actual_error_rates[qid])

                    all_estimates.append(est)

                    with open(results_file, "a") as f:
                        f.write(json.dumps(est) + "\n")

                call_metadata.append({
                    "call_num": call_num,
                    "num_estimates": len(estimates),
                    "elapsed_seconds": elapsed,
                    "success": True,
                })
            else:
                print(f"  ERROR: Could not parse response")
                call_metadata.append({
                    "call_num": call_num,
                    "success": False,
                    "raw_response": raw_response[:1000],
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            call_metadata.append({
                "call_num": call_num,
                "success": False,
                "error": str(e),
            })

        # Rate limiting
        if call_num < NUM_CALLS:
            time.sleep(2)

    # Summary statistics
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    successful_calls = sum(1 for c in call_metadata if c.get("success"))
    total_estimates = len(all_estimates)

    print(f"\nSuccessful calls: {successful_calls}/{NUM_CALLS}")
    print(f"Total estimates: {total_estimates}")

    if all_estimates:
        # Compute quick stats
        import numpy as np

        # Group by QID
        qid_estimates = {}
        for est in all_estimates:
            qid = est.get("qid")
            if qid and isinstance(est.get("error_rate"), (int, float)):
                if qid not in qid_estimates:
                    qid_estimates[qid] = {"estimates": [], "actual": est.get("actual_error_rate")}
                qid_estimates[qid]["estimates"].append(est["error_rate"])

        # Consistency (within-item std dev)
        stds = [np.std(v["estimates"]) for v in qid_estimates.values() if len(v["estimates"]) > 1]
        mean_std = np.mean(stds) if stds else 0

        # Accuracy (correlation with actual)
        predicted = [np.mean(v["estimates"]) for v in qid_estimates.values()]
        actual = [v["actual"] for v in qid_estimates.values()]
        if len(predicted) > 2:
            from scipy import stats
            corr, p = stats.pearsonr(predicted, actual)
        else:
            corr, p = 0, 1

        # MAE
        abs_errors = [abs(p - a) for p, a in zip(predicted, actual)]
        mae = np.mean(abs_errors)

        print(f"\nQuick Results:")
        print(f"  Items estimated: {len(qid_estimates)}")
        print(f"  Mean within-item std: {mean_std:.1f} pp (consistency)")
        print(f"  Pearson r: {corr:.3f} (p={p:.4f})")
        print(f"  MAE: {mae:.1f} percentage points")

    # Save summary
    summary = {
        "experiment": "opus_difficulty_estimation_calibrated",
        "model": MODEL_ID,
        "timestamp": timestamp,
        "config": {
            "num_calls": NUM_CALLS,
            "num_test_items": NUM_TEST_ITEMS,
            "num_calibration_items": NUM_CALIBRATION_ITEMS,
            "random_seed": RANDOM_SEED,
        },
        "results": {
            "successful_calls": successful_calls,
            "total_estimates": total_estimates,
        },
        "call_metadata": call_metadata,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nRun 'python scripts/analyze_opus_estimates.py' for detailed analysis")

    return all_estimates


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    run_experiment()
