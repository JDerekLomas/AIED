#!/usr/bin/env python3
"""
Test LLM-as-Difficulty-Labeler Approach

Instead of simulating students, ask LLMs to directly predict
what percentage of students would select each option.

Compare predictions to actual student distributions from Eedi data.

Usage:
    python scripts/test_population_labeling.py --sample 10
    python scripts/test_population_labeling.py --all --model gpt-4o
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse
import re
import os

# Optional imports for API calls
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


DATA_DIR = Path(__file__).parent.parent / "data" / "eedi"
CURATED_FILE = DATA_DIR / "curated_eedi_items.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "pilot" / "labeling_experiment"


@dataclass
class PredictionResult:
    question_id: int
    model: str
    predicted: dict[str, float]  # {"A": 25.0, "B": 30.0, ...}
    actual: dict[str, float]
    raw_response: str


LABELING_PROMPT = """You are an expert in mathematics education and student misconceptions.

Given the following multiple choice math problem, estimate what percentage of middle school students (ages 11-14) would select each answer option.

Base your estimates on:
- Common mathematical misconceptions at this level
- The plausibility of each distractor
- Typical student error patterns

**Question:**
{question_text}

**Options:**
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

**Correct answer:** {correct_answer}

Provide your estimates as percentages that sum to 100%. Use this exact format:
A: [percentage]%
B: [percentage]%
C: [percentage]%
D: [percentage]%

Then briefly explain your reasoning for the distribution."""


def load_items(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load curated items with student distributions."""
    df = pd.read_csv(CURATED_FILE)

    if sample_size and sample_size < len(df):
        # Stratified sample by misconception type
        df = df.groupby('target_key').apply(
            lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
        ).reset_index(drop=True)

    return df


def parse_predictions(response: str) -> dict[str, float]:
    """Extract percentage predictions from LLM response."""
    predictions = {}

    # Pattern: "A: 25%" or "A: 25.5%" or "A: 25"
    pattern = r'([A-D]):\s*(\d+(?:\.\d+)?)\s*%?'
    matches = re.findall(pattern, response, re.IGNORECASE)

    for letter, pct in matches:
        predictions[letter.upper()] = float(pct)

    # Validate we got all 4
    if len(predictions) != 4:
        return {}

    # Normalize to sum to 100 if close
    total = sum(predictions.values())
    if 95 <= total <= 105:
        for k in predictions:
            predictions[k] = (predictions[k] / total) * 100

    return predictions


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Call Anthropic API."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_llm_prediction(item: pd.Series, model: str) -> PredictionResult:
    """Get LLM prediction for a single item."""

    # Build prompt
    prompt = LABELING_PROMPT.format(
        question_text=item['question_text'],
        answer_a=item.get('AnswerAText', 'A'),
        answer_b=item.get('AnswerBText', 'B'),
        answer_c=item.get('AnswerCText', 'C'),
        answer_d=item.get('AnswerDText', 'D'),
        correct_answer=item['correct_answer']
    )

    # Call appropriate API
    if model.startswith('gpt'):
        raw_response = call_openai(prompt, model)
    elif model.startswith('claude'):
        raw_response = call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Parse predictions
    predicted = parse_predictions(raw_response)

    # Get actual distribution
    actual = {
        'A': item['pct_A'],
        'B': item['pct_B'],
        'C': item['pct_C'],
        'D': item['pct_D']
    }

    return PredictionResult(
        question_id=item['QuestionId'],
        model=model,
        predicted=predicted,
        actual=actual,
        raw_response=raw_response
    )


def calculate_metrics(results: list[PredictionResult]) -> dict:
    """Calculate alignment metrics between predicted and actual distributions."""

    valid_results = [r for r in results if r.predicted]

    if not valid_results:
        return {"error": "No valid predictions parsed"}

    # Per-option MAE
    option_errors = {opt: [] for opt in ['A', 'B', 'C', 'D']}
    all_errors = []
    correlations = []

    for r in valid_results:
        pred_vec = [r.predicted.get(opt, 0) for opt in ['A', 'B', 'C', 'D']]
        actual_vec = [r.actual[opt] for opt in ['A', 'B', 'C', 'D']]

        # Per-option errors
        for i, opt in enumerate(['A', 'B', 'C', 'D']):
            error = abs(pred_vec[i] - actual_vec[i])
            option_errors[opt].append(error)
            all_errors.append(error)

        # Correlation per item
        if np.std(pred_vec) > 0 and np.std(actual_vec) > 0:
            corr = np.corrcoef(pred_vec, actual_vec)[0, 1]
            correlations.append(corr)

    # Target distractor analysis
    target_pred_errors = []
    target_rank_correct = 0

    for r in valid_results:
        # Find which option was the target distractor for this item
        # (We'd need to join back to get target_distractor column)
        pass  # TODO: Add target-specific analysis

    return {
        "n_items": len(valid_results),
        "n_failed_parse": len(results) - len(valid_results),
        "overall_mae": np.mean(all_errors),
        "overall_mae_std": np.std(all_errors),
        "per_option_mae": {opt: np.mean(errs) for opt, errs in option_errors.items()},
        "mean_correlation": np.mean(correlations) if correlations else None,
        "median_correlation": np.median(correlations) if correlations else None,
    }


def run_experiment(
    sample_size: Optional[int] = None,
    models: list[str] = None,
    dry_run: bool = False
) -> dict:
    """Run the labeling experiment."""

    if models is None:
        models = ["gpt-4o-mini"]

    # Load items
    items = load_items(sample_size)
    print(f"Loaded {len(items)} items")

    # We need the full item data with answer texts
    # Join back to get answer texts
    full_data = pd.read_csv(DATA_DIR / "eedi_with_student_data.csv")
    items = items.merge(
        full_data[['QuestionId', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']],
        on='QuestionId',
        how='left'
    )

    results_by_model = {}

    for model in models:
        print(f"\nRunning {model}...")
        results = []

        for idx, item in items.iterrows():
            if dry_run:
                print(f"  Would call {model} for Q{item['QuestionId']}")
                continue

            try:
                result = get_llm_prediction(item, model)
                results.append(result)

                # Progress
                if result.predicted:
                    mae = np.mean([
                        abs(result.predicted[opt] - result.actual[opt])
                        for opt in ['A', 'B', 'C', 'D']
                    ])
                    print(f"  Q{item['QuestionId']}: MAE={mae:.1f}pp")
                else:
                    print(f"  Q{item['QuestionId']}: Failed to parse")

            except Exception as e:
                print(f"  Q{item['QuestionId']}: Error - {e}")

        if not dry_run:
            metrics = calculate_metrics(results)
            results_by_model[model] = {
                "metrics": metrics,
                "results": [
                    {
                        "question_id": r.question_id,
                        "predicted": r.predicted,
                        "actual": r.actual,
                        "raw_response": r.raw_response
                    }
                    for r in results
                ]
            }

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Test LLM population labeling")
    parser.add_argument('--sample', type=int, default=10,
                        help='Number of items to sample (default: 10)')
    parser.add_argument('--all', action='store_true',
                        help='Use all 113 items')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use (gpt-4o-mini, gpt-4o, claude-3-haiku-20240307)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without API calls')
    args = parser.parse_args()

    sample_size = None if args.all else args.sample

    print("=" * 60)
    print("LLM-AS-DIFFICULTY-LABELER EXPERIMENT")
    print("=" * 60)

    results = run_experiment(
        sample_size=sample_size,
        models=[args.model],
        dry_run=args.dry_run
    )

    if not args.dry_run and results:
        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"labeling_results_{args.model.replace('-', '_')}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Print summary
        for model, data in results.items():
            print(f"\n{'=' * 40}")
            print(f"Model: {model}")
            print(f"{'=' * 40}")
            metrics = data['metrics']
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
                continue
            print(f"Items tested: {metrics['n_items']}")
            print(f"Failed parses: {metrics['n_failed_parse']}")
            print(f"Overall MAE: {metrics['overall_mae']:.1f} ± {metrics['overall_mae_std']:.1f} pp")
            print(f"Per-option MAE: {metrics['per_option_mae']}")
            if metrics['mean_correlation']:
                print(f"Mean correlation: {metrics['mean_correlation']:.3f}")


if __name__ == "__main__":
    main()
