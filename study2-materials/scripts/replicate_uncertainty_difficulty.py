#!/usr/bin/env python3
"""
Replication 5: Model Uncertainty as Difficulty Signal
Paper: EDM 2025 proceedings

Research Question: Does LLM uncertainty correlate with item difficulty?

Method:
1. For each item, get logprobs of answer tokens
2. Compute 1st-token probability (confidence)
3. Randomize option order, measure consistency
4. Use uncertainty features to predict difficulty

Key Metric: RMSE improvement over text-only baseline
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
import random
import math
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API CLIENT (OpenAI only - needs logprobs)
# =============================================================================

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not installed - logprobs require OpenAI API")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Only models that support logprobs
MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
}

PROMPT_TEMPLATE = """Answer this multiple choice math question with just the letter (A, B, C, or D).

Question: {question}

{options}

Answer:"""


# =============================================================================
# UNCERTAINTY MEASUREMENT
# =============================================================================

def get_answer_with_logprobs(model_id: str, prompt: str) -> dict:
    """Get answer with first-token logprobs."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5
    )

    choice = response.choices[0]
    content = choice.message.content.strip().upper() if choice.message.content else None

    # Extract logprobs
    logprobs_data = {}
    if choice.logprobs and choice.logprobs.content:
        token_logprobs = choice.logprobs.content[0]
        # Main token
        logprobs_data['token'] = token_logprobs.token
        logprobs_data['logprob'] = token_logprobs.logprob
        logprobs_data['prob'] = math.exp(token_logprobs.logprob)

        # Top alternatives
        if token_logprobs.top_logprobs:
            logprobs_data['top_logprobs'] = [
                {"token": lp.token, "logprob": lp.logprob, "prob": math.exp(lp.logprob)}
                for lp in token_logprobs.top_logprobs
            ]

    # Determine answer letter
    answer = None
    if content and content[0] in ['A', 'B', 'C', 'D']:
        answer = content[0]
    elif 'token' in logprobs_data:
        token_upper = logprobs_data['token'].strip().upper()
        if token_upper in ['A', 'B', 'C', 'D']:
            answer = token_upper

    return {
        "answer": answer,
        "logprobs": logprobs_data,
        "raw_content": content
    }


def compute_uncertainty_features(
    item: dict,
    model_id: str,
    n_permutations: int = 4
) -> dict:
    """Compute various uncertainty features for an item."""

    # Parse options
    options_text = item['options']
    option_lines = [l.strip() for l in options_text.strip().split('\n') if l.strip()]

    # Original order response
    prompt = PROMPT_TEMPLATE.format(
        question=item['question'],
        options=item['options']
    )

    original_result = get_answer_with_logprobs(model_id, prompt)

    # Uncertainty features
    features = {
        "original_answer": original_result['answer'],
        "original_logprob": original_result['logprobs'].get('logprob', -10),
        "original_prob": original_result['logprobs'].get('prob', 0),
    }

    # Top token entropy (from top logprobs)
    if 'top_logprobs' in original_result['logprobs']:
        probs = [lp['prob'] for lp in original_result['logprobs']['top_logprobs']]
        # Normalize
        total = sum(probs)
        if total > 0:
            probs_norm = [p/total for p in probs]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs_norm)
            features['top_token_entropy'] = entropy
        else:
            features['top_token_entropy'] = 0
    else:
        features['top_token_entropy'] = 0

    # Answer distribution from top logprobs (for A, B, C, D)
    answer_probs = {"A": 0, "B": 0, "C": 0, "D": 0}
    if 'top_logprobs' in original_result['logprobs']:
        for lp in original_result['logprobs']['top_logprobs']:
            token = lp['token'].strip().upper()
            if token in answer_probs:
                answer_probs[token] = lp['prob']

    # Normalize answer probs
    total_answer_prob = sum(answer_probs.values())
    if total_answer_prob > 0:
        for k in answer_probs:
            answer_probs[k] /= total_answer_prob
    else:
        # No clear answer tokens in top 5
        answer_probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    features['answer_probs'] = answer_probs

    # Answer entropy
    probs_list = list(answer_probs.values())
    answer_entropy = -sum(p * math.log(p + 1e-10) for p in probs_list if p > 0)
    features['answer_entropy'] = answer_entropy

    # Permutation consistency (shuffle option order)
    permutation_answers = [original_result['answer']]
    permutation_probs = [features['original_prob']]

    # Create mapping for permutations
    original_options = {}
    for line in option_lines:
        if line and ')' in line:
            letter = line[0].upper()
            text = line[2:].strip()
            original_options[letter] = text

    for perm in range(n_permutations - 1):
        # Shuffle options
        letters = ['A', 'B', 'C', 'D']
        shuffled = letters.copy()
        random.shuffle(shuffled)

        # Build mapping: new_letter -> original_letter
        mapping = {shuffled[i]: letters[i] for i in range(4)}
        reverse_mapping = {letters[i]: shuffled[i] for i in range(4)}

        # Build shuffled options text
        shuffled_options = []
        for new_letter in ['A', 'B', 'C', 'D']:
            orig_letter = mapping[new_letter]
            if orig_letter in original_options:
                shuffled_options.append(f"{new_letter}) {original_options[orig_letter]}")

        shuffled_prompt = PROMPT_TEMPLATE.format(
            question=item['question'],
            options='\n'.join(shuffled_options)
        )

        result = get_answer_with_logprobs(model_id, shuffled_prompt)

        # Map answer back to original ordering
        if result['answer'] and result['answer'] in mapping:
            original_answer = mapping[result['answer']]
            permutation_answers.append(original_answer)
        else:
            permutation_answers.append(None)

        permutation_probs.append(result['logprobs'].get('prob', 0))

        time.sleep(0.05)  # Rate limiting

    # Consistency: fraction of permutations with same answer
    valid_answers = [a for a in permutation_answers if a]
    if valid_answers:
        mode_answer = max(set(valid_answers), key=valid_answers.count)
        consistency = valid_answers.count(mode_answer) / len(valid_answers)
    else:
        consistency = 0

    features['permutation_consistency'] = consistency
    features['avg_permutation_prob'] = np.mean(permutation_probs)
    features['std_permutation_prob'] = np.std(permutation_probs)

    return features


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eedi_items(csv_path: str, limit: int = None) -> list:
    """Load Eedi items from CSV."""
    df = pd.read_csv(csv_path)

    items = []
    for _, row in df.iterrows():
        if 'AnswerAText' in row:
            options = f"A) {row['AnswerAText']}\nB) {row['AnswerBText']}\nC) {row['AnswerCText']}\nD) {row['AnswerDText']}"

            correct = row['CorrectAnswer']
            correct_pct = row[f'pct_{correct}'] / 100
            difficulty = 1 - correct_pct

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": correct,
                "difficulty": difficulty,
                "total_responses": row['total_responses']
            }
            items.append(item)

        if limit and len(items) >= limit:
            break

    return items


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_uncertainty_experiment(
    items: list,
    model_name: str,
    n_permutations: int = 4,
    output_dir: Path = None
):
    """Run uncertainty-based difficulty estimation."""

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_id = MODELS[model_name]

    print(f"\n{'='*60}")
    print(f"UNCERTAINTY EXPERIMENT: {model_name}")
    print(f"Permutations per item: {n_permutations}")
    print('='*60)

    results = []

    for i, item in enumerate(items):
        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        try:
            features = compute_uncertainty_features(
                item=item,
                model_id=model_id,
                n_permutations=n_permutations
            )

            result = {
                "item_id": item['id'],
                "correct_answer": item['correct'],
                "difficulty": item['difficulty'],
                "model_answer": features['original_answer'],
                "is_correct": features['original_answer'] == item['correct'],
                "uncertainty_features": {
                    "answer_prob": features['original_prob'],
                    "answer_logprob": features['original_logprob'],
                    "answer_entropy": features['answer_entropy'],
                    "top_token_entropy": features['top_token_entropy'],
                    "permutation_consistency": features['permutation_consistency'],
                    "avg_prob": features['avg_permutation_prob'],
                    "std_prob": features['std_permutation_prob'],
                },
                "answer_probs": features['answer_probs'],
            }
            results.append(result)

            print(f"  Difficulty: {item['difficulty']:.2f}")
            print(f"  Answer prob: {features['original_prob']:.3f}")
            print(f"  Consistency: {features['permutation_consistency']:.2f}")
            print(f"  Answer entropy: {features['answer_entropy']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(0.1)  # Rate limiting

        # Save intermediate
        if output_dir and (i + 1) % 10 == 0:
            with open(output_dir / f"{model_name}_intermediate.json", 'w') as f:
                json.dump(results, f, indent=2)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_uncertainty_results(results: list) -> dict:
    """Analyze relationship between uncertainty and difficulty."""

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print('='*60)

    # Extract data
    difficulties = []
    answer_probs = []
    answer_logprobs = []
    answer_entropies = []
    top_token_entropies = []
    consistencies = []
    avg_probs = []

    for r in results:
        difficulties.append(r['difficulty'])
        uf = r['uncertainty_features']
        answer_probs.append(uf['answer_prob'])
        answer_logprobs.append(uf['answer_logprob'])
        answer_entropies.append(uf['answer_entropy'])
        top_token_entropies.append(uf['top_token_entropy'])
        consistencies.append(uf['permutation_consistency'])
        avg_probs.append(uf['avg_prob'])

    difficulties = np.array(difficulties)
    answer_probs = np.array(answer_probs)

    # Accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results)

    print(f"\nItems analyzed: {len(results)}")
    print(f"Model accuracy: {accuracy:.3f}")

    # Correlations with difficulty
    print("\nCorrelations with Difficulty:")

    # Answer probability (expect negative: higher prob = easier)
    r_prob, p_prob = stats.pearsonr(answer_probs, difficulties)
    print(f"  Answer probability: r={r_prob:.3f} (p={p_prob:.4f})")

    # Answer entropy (expect positive: higher entropy = harder)
    r_entropy, p_entropy = stats.pearsonr(answer_entropies, difficulties)
    print(f"  Answer entropy: r={r_entropy:.3f} (p={p_entropy:.4f})")

    # Consistency (expect negative: lower consistency = harder)
    r_consist, p_consist = stats.pearsonr(consistencies, difficulties)
    print(f"  Permutation consistency: r={r_consist:.3f} (p={p_consist:.4f})")

    # Combined uncertainty signal: 1 - prob (so higher = harder)
    uncertainty = 1 - np.array(answer_probs)
    r_unc, p_unc = stats.pearsonr(uncertainty, difficulties)
    print(f"  Uncertainty (1-prob): r={r_unc:.3f} (p={p_unc:.4f})")

    # RMSE if using uncertainty as difficulty predictor
    rmse_uncertainty = np.sqrt(np.mean((uncertainty - difficulties)**2))
    rmse_prob = np.sqrt(np.mean((1 - answer_probs - difficulties)**2))
    mae_uncertainty = np.mean(np.abs(uncertainty - difficulties))

    print(f"\nUsing (1-prob) as difficulty predictor:")
    print(f"  RMSE: {rmse_uncertainty:.3f}")
    print(f"  MAE: {mae_uncertainty:.3f}")

    # Baseline: always predict mean difficulty
    baseline_pred = np.full_like(difficulties, difficulties.mean())
    rmse_baseline = np.sqrt(np.mean((baseline_pred - difficulties)**2))
    print(f"\nBaseline (mean prediction):")
    print(f"  RMSE: {rmse_baseline:.3f}")

    print(f"\nImprovement over baseline: {(rmse_baseline - rmse_uncertainty)/rmse_baseline*100:.1f}%")

    analysis = {
        "n_items": len(results),
        "model_accuracy": accuracy,
        "correlations": {
            "answer_prob_vs_difficulty": {"r": r_prob, "p": p_prob},
            "answer_entropy_vs_difficulty": {"r": r_entropy, "p": p_entropy},
            "consistency_vs_difficulty": {"r": r_consist, "p": p_consist},
            "uncertainty_vs_difficulty": {"r": r_unc, "p": p_unc},
        },
        "prediction_metrics": {
            "uncertainty_rmse": rmse_uncertainty,
            "uncertainty_mae": mae_uncertainty,
            "baseline_rmse": rmse_baseline,
            "improvement_pct": (rmse_baseline - rmse_uncertainty)/rmse_baseline*100,
        }
    }

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Uncertainty-Difficulty Study")
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    parser.add_argument("--data", type=str,
                       default=str(script_dir / "data/eedi/eedi_with_student_data.csv"),
                       help="Path to Eedi data CSV")
    parser.add_argument("--output", type=str, default=str(script_dir / "pilot/replications/uncertainty_difficulty"),
                       help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       choices=list(MODELS.keys()),
                       help="Model to use (must support logprobs)")
    parser.add_argument("--permutations", type=int, default=4,
                       help="Number of option orderings to test")
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
        analysis = analyze_uncertainty_results(results)
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        return

    print("="*60)
    print("UNCERTAINTY-DIFFICULTY REPLICATION")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Permutations: {args.permutations}")
    print(f"Item limit: {args.items or 'all'}")
    print()

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items loaded!")
        return

    # Run experiment
    results = run_uncertainty_experiment(
        items=items,
        model_name=args.model,
        n_permutations=args.permutations,
        output_dir=output_dir
    )

    # Save results
    results_file = output_dir / f"{args.model}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Analyze
    analysis = analyze_uncertainty_results(results)

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
