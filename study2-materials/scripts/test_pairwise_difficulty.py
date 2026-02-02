#!/usr/bin/env python3
"""
Test Pairwise Difficulty Estimation (Elo-style)

Instead of asking for absolute percentages, ask:
"Which option would MORE students select: A or B?"

Then use Bradley-Terry / Elo to convert pairwise comparisons
into difficulty rankings.

Usage:
    python scripts/test_pairwise_difficulty.py --sample 10
    python scripts/test_pairwise_difficulty.py --all --model gpt-4o
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional
import argparse
import re
import os
import math

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
OUTPUT_DIR = Path(__file__).parent.parent / "pilot" / "pairwise_experiment"


@dataclass
class PairwiseResult:
    question_id: int
    option_a: str
    option_b: str
    predicted_winner: str  # Which option LLM thinks more students pick
    actual_winner: str     # Which option actually had more selections
    confidence: Optional[str] = None
    raw_response: str = ""


@dataclass
class EloRatings:
    """Simple Elo rating system for options within an item."""
    ratings: dict = field(default_factory=lambda: {'A': 1000, 'B': 1000, 'C': 1000, 'D': 1000})
    k: float = 32  # K-factor

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected probability that A beats B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner: str, loser: str):
        """Update ratings after a comparison."""
        expected = self.expected_score(self.ratings[winner], self.ratings[loser])
        self.ratings[winner] += self.k * (1 - expected)
        self.ratings[loser] += self.k * (0 - (1 - expected))

    def get_ranking(self) -> list[tuple[str, float]]:
        """Return options sorted by rating (highest first)."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])


PAIRWISE_PROMPT = """You are an expert in mathematics education and student misconceptions.

Given this math problem, compare TWO specific answer options and predict which one MORE middle school students (ages 11-14) would select.

**Question:**
{question_text}

**All options:**
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

**Correct answer:** {correct_answer}

**Compare these two options:**
- Option {opt1}: {opt1_text}
- Option {opt2}: {opt2_text}

Which option would MORE students select: {opt1} or {opt2}?

Answer with just the letter ({opt1} or {opt2}), then briefly explain why."""


def load_items(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load curated items with student distributions."""
    df = pd.read_csv(CURATED_FILE)

    if sample_size and sample_size < len(df):
        df = df.groupby('target_key', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // 4)), random_state=42)
        ).reset_index(drop=True)

    return df


def parse_winner(response: str, opt1: str, opt2: str) -> Optional[str]:
    """Extract which option the LLM picked as winner."""
    response_upper = response.upper().strip()

    # Check first character
    if response_upper.startswith(opt1):
        return opt1
    if response_upper.startswith(opt2):
        return opt2

    # Check for "Option X" pattern
    match = re.search(rf'OPTION\s*({opt1}|{opt2})', response_upper)
    if match:
        return match.group(1)

    # Check which letter appears first
    pos1 = response_upper.find(opt1)
    pos2 = response_upper.find(opt2)

    if pos1 >= 0 and (pos2 < 0 or pos1 < pos2):
        return opt1
    if pos2 >= 0:
        return opt2

    return None


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_pairwise_comparisons(item: pd.Series, model: str) -> list[PairwiseResult]:
    """Get all 6 pairwise comparisons for an item."""
    results = []
    options = ['A', 'B', 'C', 'D']

    # Get actual percentages
    actual_pcts = {
        'A': item['pct_A'],
        'B': item['pct_B'],
        'C': item['pct_C'],
        'D': item['pct_D']
    }

    # All 6 pairs
    for opt1, opt2 in combinations(options, 2):
        prompt = PAIRWISE_PROMPT.format(
            question_text=item['question_text'],
            answer_a=item.get('AnswerAText', 'A'),
            answer_b=item.get('AnswerBText', 'B'),
            answer_c=item.get('AnswerCText', 'C'),
            answer_d=item.get('AnswerDText', 'D'),
            correct_answer=item['correct_answer'],
            opt1=opt1,
            opt2=opt2,
            opt1_text=item.get(f'Answer{opt1}Text', opt1),
            opt2_text=item.get(f'Answer{opt2}Text', opt2),
        )

        # Call API
        if model.startswith('gpt'):
            raw_response = call_openai(prompt, model)
        elif model.startswith('claude'):
            raw_response = call_anthropic(prompt, model)
        else:
            raise ValueError(f"Unknown model: {model}")

        predicted_winner = parse_winner(raw_response, opt1, opt2)
        actual_winner = opt1 if actual_pcts[opt1] > actual_pcts[opt2] else opt2

        results.append(PairwiseResult(
            question_id=item['QuestionId'],
            option_a=opt1,
            option_b=opt2,
            predicted_winner=predicted_winner,
            actual_winner=actual_winner,
            raw_response=raw_response
        ))

    return results


def compute_elo_from_comparisons(comparisons: list[PairwiseResult], use_predicted: bool = True) -> EloRatings:
    """Convert pairwise comparisons to Elo ratings."""
    elo = EloRatings()

    for comp in comparisons:
        if use_predicted:
            winner = comp.predicted_winner
            if winner is None:
                continue
            loser = comp.option_b if winner == comp.option_a else comp.option_a
        else:
            winner = comp.actual_winner
            loser = comp.option_b if winner == comp.option_a else comp.option_a

        elo.update(winner, loser)

    return elo


def rank_correlation(ranking1: list[str], ranking2: list[str]) -> float:
    """Kendall's tau-like correlation between two rankings."""
    concordant = 0
    discordant = 0

    for i in range(len(ranking1)):
        for j in range(i + 1, len(ranking1)):
            # Get positions in ranking2
            pos1_in_2 = ranking2.index(ranking1[i])
            pos2_in_2 = ranking2.index(ranking1[j])

            if pos1_in_2 < pos2_in_2:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0
    return (concordant - discordant) / total


def run_experiment(
    sample_size: Optional[int] = None,
    models: list[str] = None,
    dry_run: bool = False
) -> dict:
    """Run the pairwise comparison experiment."""

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
        print(f"\nRunning {model}...")
        all_comparisons = []
        item_results = []

        for idx, item in items.iterrows():
            if dry_run:
                print(f"  Would make 6 comparisons for Q{item['QuestionId']}")
                continue

            try:
                comparisons = get_pairwise_comparisons(item, model)
                all_comparisons.extend(comparisons)

                # Compute Elo for this item
                predicted_elo = compute_elo_from_comparisons(comparisons, use_predicted=True)
                actual_elo = compute_elo_from_comparisons(comparisons, use_predicted=False)

                pred_ranking = [x[0] for x in predicted_elo.get_ranking()]
                actual_ranking = [x[0] for x in actual_elo.get_ranking()]

                # Accuracy on this item
                correct = sum(1 for c in comparisons if c.predicted_winner == c.actual_winner)
                accuracy = correct / len(comparisons)

                # Rank correlation
                tau = rank_correlation(pred_ranking, actual_ranking)

                item_results.append({
                    'question_id': item['QuestionId'],
                    'pairwise_accuracy': accuracy,
                    'rank_correlation': tau,
                    'predicted_ranking': pred_ranking,
                    'actual_ranking': actual_ranking,
                    'actual_pcts': {
                        'A': item['pct_A'], 'B': item['pct_B'],
                        'C': item['pct_C'], 'D': item['pct_D']
                    }
                })

                print(f"  Q{item['QuestionId']}: {correct}/6 correct, tau={tau:.2f}")

            except Exception as e:
                print(f"  Q{item['QuestionId']}: Error - {e}")

        if not dry_run and item_results:
            # Overall metrics
            overall_accuracy = np.mean([r['pairwise_accuracy'] for r in item_results])
            overall_tau = np.mean([r['rank_correlation'] for r in item_results])

            # Does predicted #1 match actual #1?
            top1_accuracy = np.mean([
                1 if r['predicted_ranking'][0] == r['actual_ranking'][0] else 0
                for r in item_results
            ])

            results_by_model[model] = {
                'metrics': {
                    'n_items': len(item_results),
                    'pairwise_accuracy': overall_accuracy,
                    'mean_rank_correlation': overall_tau,
                    'top1_accuracy': top1_accuracy,
                },
                'item_results': item_results,
                'comparisons': [
                    {
                        'question_id': c.question_id,
                        'pair': f"{c.option_a} vs {c.option_b}",
                        'predicted': c.predicted_winner,
                        'actual': c.actual_winner,
                        'correct': c.predicted_winner == c.actual_winner
                    }
                    for c in all_comparisons
                ]
            }

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Test pairwise difficulty estimation")
    parser.add_argument('--sample', type=int, default=10,
                        help='Number of items to sample (default: 10)')
    parser.add_argument('--all', action='store_true',
                        help='Use all items')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done')
    args = parser.parse_args()

    sample_size = None if args.all else args.sample

    print("=" * 60)
    print("PAIRWISE DIFFICULTY ESTIMATION (ELO-STYLE)")
    print("=" * 60)

    results = run_experiment(
        sample_size=sample_size,
        models=[args.model],
        dry_run=args.dry_run
    )

    if not args.dry_run and results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"pairwise_results_{args.model.replace('-', '_')}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        for model, data in results.items():
            metrics = data['metrics']
            print(f"\n{'=' * 40}")
            print(f"Model: {model}")
            print(f"{'=' * 40}")
            print(f"Items: {metrics['n_items']}")
            print(f"Pairwise accuracy: {metrics['pairwise_accuracy']:.1%}")
            print(f"Mean rank correlation (tau): {metrics['mean_rank_correlation']:.3f}")
            print(f"Top-1 accuracy: {metrics['top1_accuracy']:.1%}")


if __name__ == "__main__":
    main()
