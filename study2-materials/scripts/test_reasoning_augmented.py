#!/usr/bin/env python3
"""
Test Reasoning-Augmented Difficulty Estimation

Based on literature finding that analyzing each option improves prediction:
- AIED 2025: 10-28% MSE reduction with per-option reasoning
- Student Choice Prediction: 67.5% accuracy with misconception reasoning

This script asks LLMs to:
1. Analyze what misconception would lead to each distractor
2. Rate plausibility of each option for middle schoolers
3. Then estimate the distribution

Usage:
    python scripts/test_reasoning_augmented.py --sample 5
    python scripts/test_reasoning_augmented.py --model gpt-4o
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse
import re

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
OUTPUT_DIR = Path(__file__).parent.parent / "pilot" / "reasoning_experiment"


# Based on literature: explicit per-option reasoning before estimation
REASONING_PROMPT = """You are an expert in mathematics education and student misconceptions.

**Task:** Analyze this multiple-choice math problem and predict how middle school students (ages 11-14) would respond.

**Question:**
{question_text}

**Options:**
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

**Correct answer:** {correct_answer}

## Step 1: Analyze Each Option

For each option, explain:
- What misconception or error would lead a student to select this?
- How plausible is this error for a typical middle schooler?

**Option A analysis:**
[Your analysis]

**Option B analysis:**
[Your analysis]

**Option C analysis:**
[Your analysis]

**Option D analysis:**
[Your analysis]

## Step 2: Estimate Distribution

Based on your analysis above, estimate what percentage of middle school students would select each option. Consider:
- The correct answer should get the highest percentage (typically 40-70%)
- Distractors with common misconceptions get higher percentages
- Implausible distractors get lower percentages

Your estimates must sum to 100%.

A: [percentage]%
B: [percentage]%
C: [percentage]%
D: [percentage]%"""


@dataclass
class ReasoningResult:
    question_id: int
    model: str
    predicted: dict[str, float]
    actual: dict[str, float]
    reasoning: str
    raw_response: str


def load_items(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load curated items with student distributions."""
    df = pd.read_csv(CURATED_FILE)

    if sample_size and sample_size < len(df):
        # Stratified sample
        df = df.groupby('target_key', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // 4)), random_state=42)
        ).reset_index(drop=True)

    return df


def parse_predictions(response: str) -> dict[str, float]:
    """Extract percentage predictions from response."""
    predictions = {}

    # Look for pattern after "Step 2" or "Estimate Distribution"
    step2_match = re.search(r'(?:Step 2|Estimate Distribution).*', response, re.IGNORECASE | re.DOTALL)
    search_text = step2_match.group(0) if step2_match else response

    # Pattern: "A: 25%" or "A: 25.5%" or "A: 25"
    pattern = r'([A-D]):\s*(\d+(?:\.\d+)?)\s*%?'
    matches = re.findall(pattern, search_text, re.IGNORECASE)

    for letter, pct in matches:
        predictions[letter.upper()] = float(pct)

    # If we didn't find in Step 2, search whole response
    if len(predictions) != 4:
        matches = re.findall(pattern, response, re.IGNORECASE)
        predictions = {}
        for letter, pct in matches[-4:]:  # Take last 4 matches
            predictions[letter.upper()] = float(pct)

    if len(predictions) != 4:
        return {}

    # Normalize
    total = sum(predictions.values())
    if 90 <= total <= 110:
        for k in predictions:
            predictions[k] = (predictions[k] / total) * 100

    return predictions


def extract_reasoning(response: str) -> str:
    """Extract the reasoning portion of the response."""
    # Find Step 1 section
    step1_match = re.search(r'(?:Step 1|Analyze Each Option).*?(?=Step 2|Estimate Distribution|$)',
                            response, re.IGNORECASE | re.DOTALL)
    if step1_match:
        return step1_match.group(0).strip()
    return response[:1000]  # First 1000 chars as fallback


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_reasoning_prediction(item: pd.Series, model: str) -> ReasoningResult:
    """Get reasoning-augmented prediction for an item."""

    prompt = REASONING_PROMPT.format(
        question_text=item['question_text'],
        answer_a=item.get('AnswerAText', 'A'),
        answer_b=item.get('AnswerBText', 'B'),
        answer_c=item.get('AnswerCText', 'C'),
        answer_d=item.get('AnswerDText', 'D'),
        correct_answer=item['correct_answer']
    )

    if model.startswith('gpt'):
        raw_response = call_openai(prompt, model)
    elif model.startswith('claude'):
        raw_response = call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown model: {model}")

    predicted = parse_predictions(raw_response)
    reasoning = extract_reasoning(raw_response)

    actual = {
        'A': item['pct_A'],
        'B': item['pct_B'],
        'C': item['pct_C'],
        'D': item['pct_D']
    }

    return ReasoningResult(
        question_id=item['QuestionId'],
        model=model,
        predicted=predicted,
        actual=actual,
        reasoning=reasoning,
        raw_response=raw_response
    )


def calculate_metrics(results: list[ReasoningResult]) -> dict:
    """Calculate alignment metrics."""
    valid_results = [r for r in results if r.predicted]

    if not valid_results:
        return {"error": "No valid predictions"}

    all_errors = []
    option_errors = {opt: [] for opt in ['A', 'B', 'C', 'D']}
    correlations = []
    target_errors = []

    for r in valid_results:
        pred_vec = [r.predicted.get(opt, 0) for opt in ['A', 'B', 'C', 'D']]
        actual_vec = [r.actual[opt] for opt in ['A', 'B', 'C', 'D']]

        for i, opt in enumerate(['A', 'B', 'C', 'D']):
            error = abs(pred_vec[i] - actual_vec[i])
            option_errors[opt].append(error)
            all_errors.append(error)

        if np.std(pred_vec) > 0 and np.std(actual_vec) > 0:
            corr = np.corrcoef(pred_vec, actual_vec)[0, 1]
            correlations.append(corr)

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
    """Run the reasoning-augmented experiment."""

    if models is None:
        models = ["gpt-4o-mini"]

    items = load_items(sample_size)
    print(f"Loaded {len(items)} items")

    # Join to get answer texts
    full_data = pd.read_csv(DATA_DIR / "eedi_with_student_data.csv")
    items = items.merge(
        full_data[['QuestionId', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']],
        on='QuestionId',
        how='left'
    )

    results_by_model = {}

    for model in models:
        print(f"\nRunning {model} with reasoning-augmented prompt...")
        results = []

        for idx, item in items.iterrows():
            if dry_run:
                print(f"  Would analyze Q{item['QuestionId']}")
                continue

            try:
                result = get_reasoning_prediction(item, model)
                results.append(result)

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

        if not dry_run and results:
            metrics = calculate_metrics(results)
            results_by_model[model] = {
                "metrics": metrics,
                "results": [
                    {
                        "question_id": r.question_id,
                        "predicted": r.predicted,
                        "actual": r.actual,
                        "reasoning": r.reasoning[:500],  # Truncate for storage
                        "raw_response": r.raw_response
                    }
                    for r in results
                ]
            }

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Test reasoning-augmented difficulty estimation")
    parser.add_argument('--sample', type=int, default=5,
                        help='Number of items to sample (default: 5)')
    parser.add_argument('--all', action='store_true',
                        help='Use all items')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done')
    args = parser.parse_args()

    sample_size = None if args.all else args.sample

    print("=" * 60)
    print("REASONING-AUGMENTED DIFFICULTY ESTIMATION")
    print("=" * 60)
    print("Method: Analyze misconceptions for each option before estimating")
    print("Based on: AIED 2025 (10-28% improvement over direct estimation)")
    print("=" * 60)

    results = run_experiment(
        sample_size=sample_size,
        models=[args.model],
        dry_run=args.dry_run
    )

    if not args.dry_run and results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"reasoning_results_{args.model.replace('-', '_')}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        for model, data in results.items():
            metrics = data['metrics']
            print(f"\n{'=' * 40}")
            print(f"Model: {model}")
            print(f"{'=' * 40}")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
                continue
            print(f"Items tested: {metrics['n_items']}")
            print(f"Failed parses: {metrics['n_failed_parse']}")
            print(f"Overall MAE: {metrics['overall_mae']:.1f} ± {metrics['overall_mae_std']:.1f} pp")
            print(f"Per-option MAE: {metrics['per_option_mae']}")
            if metrics['mean_correlation']:
                print(f"Mean correlation: {metrics['mean_correlation']:.3f}")
                print(f"Median correlation: {metrics['median_correlation']:.3f}")


if __name__ == "__main__":
    main()
