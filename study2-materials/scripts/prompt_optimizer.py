#!/usr/bin/env python3
"""
Automated Prompt Optimization for Student Error Simulation

Uses iterative refinement: test prompt → analyze gaps → revise → repeat
"""

import json
import os
import csv
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional
import re
import sys
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

from anthropic import Anthropic

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path(__file__).parent.parent / "results" / "final_items.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "prompt_optimization"

# Use Haiku for cheap iteration, Sonnet for final eval
ITERATION_MODEL = "claude-3-5-haiku-latest"
EVAL_MODEL = "claude-sonnet-4-20250514"
OPTIMIZER_MODEL = "claude-sonnet-4-20250514"  # For generating prompt revisions

N_RUNS_PER_ITEM = 5  # Runs per item to estimate distribution
N_ITEMS_PER_ROUND = 8  # Items to test per optimization round (smaller for 36-item set)
MAX_ROUNDS = 6  # Fewer rounds for smaller dataset
TEST_FRACTION = 0.25  # Hold out 25% for final eval

# =============================================================================
# DATA LOADING
# =============================================================================

def load_items(misconception_id: Optional[int] = None, n_sample: int = 0) -> list[dict]:
    """Load Eedi items from CSV."""
    items = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "question_id": int(row["question_id"]),
                "question_text": row["question_text"],
                "correct_answer": row["correct_answer"],
                "target_distractor": row["target_distractor"],
                "misconception_id": int(row["misconception_id"]),
                "misconception_name": row["misconception_name"],
                "misconception_type": row.get("misconception_type", "unknown"),
                "real_distribution": {
                    "A": float(row["pct_A"]) / 100,
                    "B": float(row["pct_B"]) / 100,
                    "C": float(row["pct_C"]) / 100,
                    "D": float(row["pct_D"]) / 100,
                },
                "total_responses": int(row["total_responses"]),
                "options": {
                    "A": row.get("option_A", ""),
                    "B": row.get("option_B", ""),
                    "C": row.get("option_C", ""),
                    "D": row.get("option_D", ""),
                },
            }

            if misconception_id is None or item["misconception_id"] == misconception_id:
                items.append(item)

    if n_sample > 0 and len(items) > n_sample:
        items = random.sample(items, n_sample)

    return items


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_student(client: Anthropic, prompt: str, item: dict, model: str) -> dict:
    """Run one simulation and return the response."""
    # Format options
    options_text = "\n".join([
        f"{letter}) {item['options'][letter]}"
        for letter in "ABCD"
        if item['options'].get(letter)
    ])

    # Format the item into the prompt
    formatted_prompt = prompt.format(
        question=item["question_text"],
        options=options_text,
        correct_answer=item["correct_answer"],
        misconception=item["misconception_name"],
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.8,  # Higher temp for diversity
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    text = response.content[0].text
    answer = parse_answer(text)

    return {
        "raw_response": text,
        "parsed_answer": answer,
        "is_correct": answer == item["correct_answer"],
        "hit_target": answer == item["target_distractor"],
    }


def parse_answer(response: str) -> Optional[str]:
    """Extract answer letter from response."""
    patterns = [
        r'Answer:\s*\*?\*?([A-D])',
        r'\*\*([A-D])\)',
        r'I (?:choose|select|pick|would pick)\s*\*?\*?([A-D])',
        r'(?:is|be)\s*\*?\*?([A-D])\)?[\.\s]*$',
        r'\b([A-D])\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def run_simulation_batch(
    client: Anthropic,
    prompt: str,
    items: list[dict],
    n_runs: int,
    model: str
) -> list[dict]:
    """Run simulations for multiple items, multiple times each."""
    results = []

    for item in items:
        item_results = {
            "question_id": item["question_id"],
            "misconception_id": item["misconception_id"],
            "real_distribution": item["real_distribution"],
            "correct_answer": item["correct_answer"],
            "target_distractor": item["target_distractor"],
            "responses": [],
            "simulated_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
        }

        for _ in range(n_runs):
            try:
                result = simulate_student(client, prompt, item, model)
                item_results["responses"].append(result)
                if result["parsed_answer"]:
                    item_results["simulated_distribution"][result["parsed_answer"]] += 1
            except Exception as e:
                print(f"  Error on Q{item['question_id']}: {e}")

        # Normalize distribution
        total = sum(item_results["simulated_distribution"].values())
        if total > 0:
            for k in item_results["simulated_distribution"]:
                item_results["simulated_distribution"][k] /= total

        results.append(item_results)

    return results


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from simulation results."""
    correlations = []
    accuracy_diffs = []
    misconception_hits = []

    for r in results:
        # Distribution correlation
        real = [r["real_distribution"][k] for k in "ABCD"]
        sim = [r["simulated_distribution"][k] for k in "ABCD"]

        if sum(sim) > 0:  # Valid simulation
            corr, _ = stats.pearsonr(real, sim)
            if not np.isnan(corr):
                correlations.append(corr)

        # Accuracy difference
        real_correct = r["real_distribution"][r["correct_answer"]]
        sim_correct = r["simulated_distribution"][r["correct_answer"]]
        accuracy_diffs.append(abs(real_correct - sim_correct))

        # Misconception hit rate (when wrong, did we pick target?)
        wrong_responses = [
            resp for resp in r["responses"]
            if resp["parsed_answer"] and not resp["is_correct"]
        ]
        if wrong_responses:
            hits = sum(1 for resp in wrong_responses if resp["hit_target"])
            misconception_hits.append(hits / len(wrong_responses))

    return {
        "mean_correlation": np.mean(correlations) if correlations else 0,
        "mean_accuracy_diff": np.mean(accuracy_diffs),
        "misconception_hit_rate": np.mean(misconception_hits) if misconception_hits else 0,
        "n_items": len(results),
        "n_valid_correlations": len(correlations),
    }


# =============================================================================
# PROMPT OPTIMIZATION
# =============================================================================

INITIAL_PROMPT = """You are a student taking a math test. You have a specific misconception: {misconception}

Because of this misconception, you often make mistakes on problems like this.

Question: {question}

Options:
{options}

Think through this problem using your (flawed) understanding, then give your answer.
End with "Answer: X" where X is A, B, C, or D."""


def generate_analysis(client: Anthropic, prompt: str, results: list[dict], metrics: dict) -> str:
    """Use LLM to analyze what went wrong and suggest improvements."""

    # Prepare examples of failures
    failures = []
    for r in results:
        if r["simulated_distribution"][r["correct_answer"]] > r["real_distribution"][r["correct_answer"]] + 0.15:
            failures.append({
                "question_id": r["question_id"],
                "issue": "Too correct",
                "real_correct": f"{r['real_distribution'][r['correct_answer']]:.0%}",
                "sim_correct": f"{r['simulated_distribution'][r['correct_answer']]:.0%}",
                "sample_response": r["responses"][0]["raw_response"][:300] if r["responses"] else "N/A",
            })

        wrong_resps = [resp for resp in r["responses"] if not resp["is_correct"] and resp["parsed_answer"]]
        target_hits = [resp for resp in wrong_resps if resp["hit_target"]]
        if wrong_resps and len(target_hits) / len(wrong_resps) < 0.3:
            failures.append({
                "question_id": r["question_id"],
                "issue": "Wrong misconception",
                "target": r["target_distractor"],
                "actual_wrong": Counter(resp["parsed_answer"] for resp in wrong_resps).most_common(1),
                "sample_response": wrong_resps[0]["raw_response"][:300] if wrong_resps else "N/A",
            })

    analysis_prompt = f"""You are optimizing a prompt for simulating student errors on math problems.

CURRENT PROMPT:
```
{prompt}
```

METRICS:
- Mean correlation with real student distributions: {metrics['mean_correlation']:.3f} (target: >0.7)
- Mean accuracy difference: {metrics['mean_accuracy_diff']:.3f} (target: <0.1)
- Misconception hit rate: {metrics['misconception_hit_rate']:.1%} (target: >50%)

FAILURE EXAMPLES:
{json.dumps(failures[:5], indent=2)}

Analyze:
1. Why is the prompt producing responses that don't match real students?
2. What specific changes would help?

Be specific about prompt engineering techniques that could help:
- Role framing
- Explicit vs implicit misconception injection
- Temperature/randomness cues
- Response format
- Worked examples

Output your analysis in 2-3 paragraphs."""

    response = client.messages.create(
        model=OPTIMIZER_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": analysis_prompt}],
    )

    return response.content[0].text


def generate_revised_prompt(client: Anthropic, current_prompt: str, analysis: str, history: list[dict]) -> str:
    """Generate a revised prompt based on analysis."""

    history_summary = "\n".join([
        f"Round {h['round']}: corr={h['metrics']['mean_correlation']:.3f}, misc_hit={h['metrics']['misconception_hit_rate']:.1%}"
        for h in history[-3:]  # Last 3 rounds
    ])

    revision_prompt = f"""You are optimizing a prompt for simulating student errors.

CURRENT PROMPT:
```
{current_prompt}
```

ANALYSIS OF PROBLEMS:
{analysis}

HISTORY:
{history_summary}

Generate an improved version of the prompt. The prompt must:
1. Include {{question}} placeholder for the math question
2. Include {{options}} placeholder for the answer choices
3. Include {{misconception}} placeholder for the misconception description
4. End by asking for "Answer: X" format

Focus on making the simulated student more authentically wrong - matching real student error patterns.

Output ONLY the new prompt, nothing else. Start directly with the prompt text."""

    response = client.messages.create(
        model=OPTIMIZER_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": revision_prompt}],
    )

    new_prompt = response.content[0].text.strip()

    # Validate it has required placeholders
    if "{question}" not in new_prompt or "{misconception}" not in new_prompt or "{options}" not in new_prompt:
        print("  Warning: Generated prompt missing placeholders, keeping current")
        return current_prompt

    return new_prompt


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def run_optimization(
    misconception_id: int,
    n_rounds: int = MAX_ROUNDS,
    n_items: int = N_ITEMS_PER_ROUND,
    n_runs: int = N_RUNS_PER_ITEM,
):
    """Run the full optimization loop for one misconception type."""

    client = Anthropic()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"misc_{misconception_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load items
    all_items = load_items(misconception_id)
    print(f"Loaded {len(all_items)} items for misconception {misconception_id}")

    # Split into optimization and held-out test sets
    random.shuffle(all_items)
    n_test = max(2, int(len(all_items) * TEST_FRACTION))
    test_items = all_items[:n_test]
    train_items = all_items[n_test:]

    print(f"Train: {len(train_items)}, Test: {len(test_items)}")

    # Initialize
    current_prompt = INITIAL_PROMPT
    history = []
    best_prompt = current_prompt
    best_score = -1

    print("\n" + "=" * 60)
    print(f"STARTING OPTIMIZATION FOR MISCONCEPTION {misconception_id}")
    print("=" * 60)

    for round_num in range(1, n_rounds + 1):
        print(f"\n--- Round {round_num}/{n_rounds} ---")

        # Sample items for this round
        round_items = random.sample(train_items, min(n_items, len(train_items)))

        # Run simulations
        print(f"Running {len(round_items)} items × {n_runs} runs...")
        results = run_simulation_batch(client, current_prompt, round_items, n_runs, ITERATION_MODEL)

        # Compute metrics
        metrics = compute_metrics(results)
        print(f"  Correlation: {metrics['mean_correlation']:.3f}")
        print(f"  Accuracy diff: {metrics['mean_accuracy_diff']:.3f}")
        print(f"  Misconception hit: {metrics['misconception_hit_rate']:.1%}")

        # Track best
        score = metrics['mean_correlation'] * 0.4 + metrics['misconception_hit_rate'] * 0.6
        if score > best_score:
            best_score = score
            best_prompt = current_prompt
            print(f"  ★ New best! Score: {score:.3f}")

        # Save round results
        round_data = {
            "round": round_num,
            "prompt": current_prompt,
            "metrics": metrics,
            "results": results,
        }
        history.append(round_data)

        with open(output_dir / f"round_{round_num:02d}.json", "w") as f:
            json.dump(round_data, f, indent=2, default=str)

        # Generate analysis and revision (except last round)
        if round_num < n_rounds:
            print("Analyzing and generating revision...")
            analysis = generate_analysis(client, current_prompt, results, metrics)
            print(f"  Analysis: {analysis[:200]}...")

            current_prompt = generate_revised_prompt(client, current_prompt, analysis, history)
            print(f"  New prompt generated ({len(current_prompt)} chars)")

    # Final evaluation on held-out test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    print(f"\nTesting best prompt on {len(test_items)} held-out items...")
    test_results = run_simulation_batch(client, best_prompt, test_items, n_runs, EVAL_MODEL)
    test_metrics = compute_metrics(test_results)

    print(f"\nFINAL METRICS:")
    print(f"  Correlation: {test_metrics['mean_correlation']:.3f}")
    print(f"  Accuracy diff: {test_metrics['mean_accuracy_diff']:.3f}")
    print(f"  Misconception hit: {test_metrics['misconception_hit_rate']:.1%}")

    # Save final results
    final_output = {
        "misconception_id": misconception_id,
        "best_prompt": best_prompt,
        "final_metrics": test_metrics,
        "history": [{"round": h["round"], "metrics": h["metrics"]} for h in history],
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    with open(output_dir / "best_prompt.txt", "w") as f:
        f.write(best_prompt)

    print(f"\nResults saved to: {output_dir}")

    return best_prompt, test_metrics


# =============================================================================
# CLI
# =============================================================================

def run_all_misconceptions(n_rounds: int, n_items: int, n_runs: int):
    """Run optimization for all misconception types in the dataset."""
    # Get unique misconception IDs from data
    all_items = load_items()
    misc_ids = sorted(set(item["misconception_id"] for item in all_items))

    print(f"Found {len(misc_ids)} misconception types: {misc_ids}")
    print(f"Total items: {len(all_items)}")
    print()

    results = {}
    for misc_id in misc_ids:
        misc_items = [i for i in all_items if i["misconception_id"] == misc_id]
        print(f"\n{'='*60}")
        print(f"MISCONCEPTION {misc_id}: {misc_items[0]['misconception_name']}")
        print(f"Items: {len(misc_items)}")
        print(f"{'='*60}")

        best_prompt, metrics = run_optimization(
            misconception_id=misc_id,
            n_rounds=n_rounds,
            n_items=min(n_items, len(misc_items) - 2),  # Leave room for test set
            n_runs=n_runs,
        )
        results[misc_id] = {"prompt": best_prompt, "metrics": metrics}

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for misc_id, res in results.items():
        m = res["metrics"]
        print(f"  {misc_id}: corr={m['mean_correlation']:.3f}, misc_hit={m['misconception_hit_rate']:.1%}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize prompts for student error simulation")
    parser.add_argument("--misconception", "-m", type=int, default=None,
                       help="Misconception ID (217=frac_add, 1507=order_ops, 1597=neg_mult). Omit for all.")
    parser.add_argument("--rounds", "-r", type=int, default=MAX_ROUNDS)
    parser.add_argument("--items", "-i", type=int, default=N_ITEMS_PER_ROUND)
    parser.add_argument("--runs", "-n", type=int, default=N_RUNS_PER_ITEM)
    parser.add_argument("--all", "-a", action="store_true", help="Run all misconception types")

    args = parser.parse_args()

    if args.all or args.misconception is None:
        run_all_misconceptions(
            n_rounds=args.rounds,
            n_items=args.items,
            n_runs=args.runs,
        )
    else:
        run_optimization(
            misconception_id=args.misconception,
            n_rounds=args.rounds,
            n_items=args.items,
            n_runs=args.runs,
        )
