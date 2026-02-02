#!/usr/bin/env python3
"""
Replication 3: Confusion Tuples Validation
Paper: "Towards Valid Student Simulation" (arXiv:2601.05473)

Research Question: Does explicit misconception specification improve alignment
with student errors?

Method:
1. Map Eedi misconceptions to confusion tuples
2. Compare P1 (generic persona) vs P3 (confusion tuple) alignment
3. Measure whether errors are "causally attributable" to specified confusion

Key Comparisons:
- P1: "You are a struggling student"
- P3: "You confuse {KC_A} with {KC_B}"
"""

import json
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from dotenv import load_dotenv
import re

load_dotenv()

# =============================================================================
# API CLIENTS
# =============================================================================

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not installed")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: Anthropic not installed")


# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "rpm": 500,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "rpm": 500,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "rpm": 50,
    },
}

# Prompt levels from the validation paper
PROMPT_LEVELS = {
    "P1_generic": """You are a student who sometimes struggles with math.
You're taking a test and trying your best.
Think through the problem, then give your answer.

Question: {question}

{options}

Show your thinking briefly, then state your answer as "Answer: X" where X is A, B, C, or D.""",

    "P2_knowledge_gap": """You are a student with the following math knowledge:

KNOW WELL: Basic arithmetic with single operations
STILL LEARNING: {topic_area}
OFTEN CONFUSED ABOUT: {confusion_area}

Think through the problem, then give your answer.

Question: {question}

{options}

Show your thinking briefly, then state your answer as "Answer: X" where X is A, B, C, or D.""",

    "P3_confusion_tuple": """You are a student who has a specific confusion about math.

YOUR CONFUSION: You think that {misconception_description}

This confusion affects how you approach certain problems. Apply your understanding (including this confusion) to solve:

Question: {question}

{options}

Show your thinking briefly, then state your answer as "Answer: X" where X is A, B, C, or D.""",

    "P4_procedural": """You follow this specific procedure when solving math problems:

YOUR PROCEDURE:
{procedure}

Apply this exact procedure step by step.

Question: {question}

{options}

Show each step of your procedure, then state your answer as "Answer: X" where X is A, B, C, or D."""
}

# Misconception configurations for Eedi items
MISCONCEPTION_CONFIGS = {
    1507: {  # "Carries out operations from left to right regardless of priority order"
        "name": "Left-to-Right Order of Operations",
        "topic_area": "Problems with multiple operations",
        "confusion_area": "The priority of different operations",
        "misconception_description": "math expressions should be solved left-to-right, operation by operation, just like reading a sentence. When you see 2+3×4, you do 2+3=5 first, then 5×4=20",
        "procedure": """1. Start at the left of the expression
2. Find the first operation you see
3. Do that operation with the numbers on either side
4. Replace those with the result
5. Repeat from step 1 until done

Example: 2 + 3 × 4
- First operation is +
- Do 2 + 3 = 5
- Now have 5 × 4
- Do 5 × 4 = 20
- Answer: 20"""
    },
    1672: {  # "Does not need brackets" - misunderstanding bracket necessity
        "name": "Bracket Misunderstanding",
        "topic_area": "Using brackets to change order of operations",
        "confusion_area": "When brackets are needed vs optional",
        "misconception_description": "brackets don't change anything because you just solve left to right anyway. Brackets are just for grouping things visually, not for changing the order of calculation",
        "procedure": """1. Ignore any brackets in the expression
2. Solve left to right, operation by operation
3. Each operation uses the result of the previous one"""
    },
    2142: {  # Related to algebraic simplification errors
        "name": "Algebraic Simplification Error",
        "topic_area": "Simplifying algebraic fractions",
        "confusion_area": "When terms cancel and when they don't",
        "misconception_description": "you can cancel any terms that look the same in a fraction, even if they're being added or subtracted. If you see 'm' on top and bottom, you can just cross them out",
        "procedure": """1. Look for any matching letters or numbers in numerator and denominator
2. Cancel anything that appears in both
3. What remains is the simplified answer"""
    },
}


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=300,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def call_model(model_name: str, prompt: str, temperature: float = 0.7) -> str:
    """Route to appropriate API."""
    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "openai":
        return call_openai(model_id, prompt, temperature)
    elif provider == "anthropic":
        return call_anthropic(model_id, prompt, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_answer(response: str) -> str:
    """Extract answer letter from response."""
    response_upper = response.upper()

    # Try "Answer: X" pattern
    match = re.search(r'ANSWER:\s*([A-D])', response_upper)
    if match:
        return match.group(1)

    # Try "The answer is X"
    match = re.search(r'THE ANSWER IS\s*([A-D])', response_upper)
    if match:
        return match.group(1)

    # Try standalone letter at end
    match = re.search(r'\b([A-D])\)?\.?\s*$', response.strip().upper())
    if match:
        return match.group(1)

    # Find any mention of A, B, C, D
    matches = re.findall(r'\b([A-D])\b', response_upper)
    if matches:
        return matches[-1]

    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eedi_items_with_misconceptions(csv_path: str, limit: int = None) -> list:
    """Load Eedi items that have misconception mappings."""
    df = pd.read_csv(csv_path)

    items = []
    for _, row in df.iterrows():
        # Check if any misconception ID is in our configs
        misconception_id = None
        target_distractor = None

        for col in ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']:
            if col in row and pd.notna(row[col]):
                mid = int(row[col])
                if mid in MISCONCEPTION_CONFIGS:
                    misconception_id = mid
                    # Determine which option this misconception corresponds to
                    target_distractor = col.replace('Misconception', '').replace('Id', '')
                    break

        if misconception_id is None:
            continue

        if 'AnswerAText' in row:
            options = f"A) {row['AnswerAText']}\nB) {row['AnswerBText']}\nC) {row['AnswerCText']}\nD) {row['AnswerDText']}"

            correct = row['CorrectAnswer']  # Kaggle letter (for LLM prompt)
            neurips_correct = row['neurips_correct_pos']  # NeurIPS position (for pct lookup)
            correct_pct = row[f'pct_{neurips_correct}'] / 100
            # NOTE: target_distractor uses Kaggle letter ordering — pct lookup
            # is misaligned with NeurIPS pct columns. Cannot fix without
            # per-option Kaggle→NeurIPS mapping.
            target_pct = row[f'pct_{target_distractor}'] / 100 if target_distractor else 0

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": correct,
                "target_distractor": target_distractor,
                "misconception_id": misconception_id,
                "misconception_config": MISCONCEPTION_CONFIGS[misconception_id],
                "student_correct_pct": correct_pct,
                "student_target_pct": target_pct,
                "student_dist": {
                    "A": row['pct_A'] / 100,
                    "B": row['pct_B'] / 100,
                    "C": row['pct_C'] / 100,
                    "D": row['pct_D'] / 100,
                },
                "total_responses": row['total_responses']
            }
            items.append(item)

        if limit and len(items) >= limit:
            break

    return items


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_confusion_tuples_experiment(
    items: list,
    model_name: str,
    reps: int = 5,
    output_dir: Path = None
):
    """Run confusion tuples validation experiment."""

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODELS[model_name]
    rpm = config["rpm"]
    delay = 60.0 / rpm

    print(f"\n{'='*60}")
    print(f"CONFUSION TUPLES EXPERIMENT: {model_name}")
    print(f"Prompt levels: P1 (generic) vs P3 (confusion tuple)")
    print(f"Repetitions: {reps}")
    print('='*60)

    results = []

    for i, item in enumerate(items):
        print(f"\nItem {i+1}/{len(items)}: {item['id']}")
        print(f"  Misconception: {item['misconception_config']['name']}")
        print(f"  Target distractor: {item['target_distractor']}")

        item_results = {
            "item_id": item['id'],
            "correct_answer": item['correct'],
            "target_distractor": item['target_distractor'],
            "misconception_id": item['misconception_id'],
            "misconception_name": item['misconception_config']['name'],
            "student_correct_pct": item['student_correct_pct'],
            "student_target_pct": item['student_target_pct'],
            "student_dist": item['student_dist'],
            "responses": {}
        }

        # Test each prompt level
        for level in ["P1_generic", "P3_confusion_tuple"]:
            print(f"\n  {level}:")
            responses = []

            for rep in range(reps):
                # Format prompt
                if level == "P1_generic":
                    prompt = PROMPT_LEVELS[level].format(
                        question=item['question'],
                        options=item['options']
                    )
                elif level == "P3_confusion_tuple":
                    prompt = PROMPT_LEVELS[level].format(
                        question=item['question'],
                        options=item['options'],
                        misconception_description=item['misconception_config']['misconception_description']
                    )
                else:
                    continue

                try:
                    raw = call_model(model_name, prompt, temperature=0.7)
                    answer = parse_answer(raw)

                    responses.append({
                        "rep": rep,
                        "answer": answer,
                        "is_correct": answer == item['correct'],
                        "hit_target": answer == item['target_distractor'],
                        "raw_response": raw[:200]  # Truncate for storage
                    })

                    status = "✓" if answer == item['correct'] else ("⊛" if answer == item['target_distractor'] else "✗")
                    print(f"    {status} Rep {rep+1}: {answer}")

                    time.sleep(delay)

                except Exception as e:
                    print(f"    ERROR Rep {rep+1}: {e}")
                    time.sleep(delay * 2)

            # Compute level summary
            valid = [r for r in responses if r['answer']]
            if valid:
                correct_rate = sum(1 for r in valid if r['is_correct']) / len(valid)
                target_rate = sum(1 for r in valid if r['hit_target']) / len(valid)

                # Distribution
                dist = defaultdict(int)
                for r in valid:
                    dist[r['answer']] += 1
                dist_norm = {k: v/len(valid) for k, v in dist.items()}
            else:
                correct_rate = 0
                target_rate = 0
                dist_norm = {}

            item_results['responses'][level] = {
                "n_valid": len(valid),
                "correct_rate": correct_rate,
                "target_rate": target_rate,
                "distribution": dist_norm,
                "raw_responses": responses
            }

            print(f"    Correct: {correct_rate:.2f}, Target: {target_rate:.2f}")

        results.append(item_results)

        # Save intermediate
        if output_dir:
            with open(output_dir / f"{model_name}_intermediate.json", 'w') as f:
                json.dump(results, f, indent=2)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_confusion_tuples(results: list) -> dict:
    """Analyze confusion tuples validation results."""

    print(f"\n{'='*60}")
    print("ANALYSIS: Confusion Tuples Validation")
    print('='*60)

    print(f"\nItems analyzed: {len(results)}")

    # Aggregate by prompt level
    level_stats = {}

    for level in ["P1_generic", "P3_confusion_tuple"]:
        correct_rates = []
        target_rates = []

        # KL divergence from student distribution
        kl_divs = []

        for r in results:
            if level in r['responses']:
                resp = r['responses'][level]
                correct_rates.append(resp['correct_rate'])
                target_rates.append(resp['target_rate'])

                # Compute KL divergence
                student_dist = r['student_dist']
                sim_dist = resp['distribution']

                kl = 0
                for opt in ['A', 'B', 'C', 'D']:
                    p = student_dist.get(opt, 0.001)  # True dist
                    q = sim_dist.get(opt, 0.001)  # Simulated dist
                    if p > 0:
                        kl += p * np.log(p / q)
                kl_divs.append(kl)

        level_stats[level] = {
            "mean_correct": np.mean(correct_rates) if correct_rates else 0,
            "mean_target": np.mean(target_rates) if target_rates else 0,
            "mean_kl_div": np.mean(kl_divs) if kl_divs else 0,
            "n_items": len(correct_rates)
        }

    print("\nComparison by Prompt Level:")
    print("-" * 50)
    print(f"{'Level':<20} {'Correct%':<12} {'Target%':<12} {'KL Div':<10}")
    print("-" * 50)
    for level, stats in level_stats.items():
        print(f"{level:<20} {stats['mean_correct']*100:>8.1f}%    {stats['mean_target']*100:>8.1f}%    {stats['mean_kl_div']:>8.3f}")

    # Key comparison: P3 should have higher target rate than P1
    p1_target = level_stats['P1_generic']['mean_target']
    p3_target = level_stats['P3_confusion_tuple']['mean_target']

    print(f"\n{'='*50}")
    print("KEY FINDING: Target Distractor Selection")
    print(f"{'='*50}")
    print(f"P1 (generic persona): {p1_target*100:.1f}%")
    print(f"P3 (confusion tuple): {p3_target*100:.1f}%")
    print(f"Improvement: {(p3_target - p1_target)*100:+.1f} percentage points")

    if p3_target > p1_target:
        print("\n✓ P3 shows HIGHER target rate - confusion tuples help!")
    else:
        print("\n✗ P3 does NOT show higher target rate")

    # Distribution alignment
    print(f"\nDistribution Alignment (lower KL = better):")
    print(f"P1: KL={level_stats['P1_generic']['mean_kl_div']:.3f}")
    print(f"P3: KL={level_stats['P3_confusion_tuple']['mean_kl_div']:.3f}")

    # Statistical test
    p1_targets = [r['responses']['P1_generic']['target_rate']
                  for r in results if 'P1_generic' in r['responses']]
    p3_targets = [r['responses']['P3_confusion_tuple']['target_rate']
                  for r in results if 'P3_confusion_tuple' in r['responses']]

    if len(p1_targets) > 1 and len(p3_targets) > 1:
        t_stat, p_value = stats.ttest_rel(p3_targets, p1_targets)
        print(f"\nPaired t-test (P3 vs P1 target rates):")
        print(f"  t={t_stat:.3f}, p={p_value:.4f}")

        analysis_stats = {
            "t_statistic": t_stat,
            "p_value": p_value,
        }
    else:
        analysis_stats = {}

    analysis = {
        "n_items": len(results),
        "level_stats": level_stats,
        "target_improvement": (p3_target - p1_target) * 100,
        "statistical_test": analysis_stats,
    }

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Confusion Tuples Study")
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    parser.add_argument("--data", type=str,
                       default=str(script_dir / "data/eedi/eedi_with_student_data.csv"),
                       help="Path to Eedi data CSV")
    parser.add_argument("--output", type=str, default=str(script_dir / "pilot/replications/confusion_tuples"),
                       help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to use")
    parser.add_argument("--reps", type=int, default=5,
                       help="Repetitions per condition")
    parser.add_argument("--items", type=int, default=None,
                       help="Limit number of items")
    parser.add_argument("--analyze-only", type=str, default=None,
                       help="Just analyze existing results")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        with open(args.analyze_only) as f:
            results = json.load(f)
        analysis = analyze_confusion_tuples(results)
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        return

    print("="*60)
    print("CONFUSION TUPLES VALIDATION REPLICATION")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Reps per condition: {args.reps}")
    print(f"Item limit: {args.items or 'all'}")
    print()

    # Load data
    items = load_eedi_items_with_misconceptions(args.data, limit=args.items)
    print(f"Loaded {len(items)} items with known misconceptions")

    if len(items) == 0:
        print("No items with misconception mappings found!")
        return

    # Run experiment
    results = run_confusion_tuples_experiment(
        items=items,
        model_name=args.model,
        reps=args.reps,
        output_dir=output_dir
    )

    # Save results
    results_file = output_dir / f"{args.model}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Analyze
    analysis = analyze_confusion_tuples(results)

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
