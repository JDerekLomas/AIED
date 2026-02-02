#!/usr/bin/env python3
"""
Replication 2: Classroom Simulation + Aggregation
Paper: "Take Out Your Calculators" (Kröger et al., arXiv:2601.09953)

Research Question: Can aggregating responses from simulated students at different
ability levels predict item difficulty?

Method:
1. Define ability levels: Below Basic (25%), Basic (35%), Proficient (25%), Advanced (15%)
2. Simulate N students at each level
3. Aggregate responses to get predicted distribution
4. Compare to actual student distribution

Key Metric: Correlation with student difficulty (benchmark: r=0.75-0.82)
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

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: Groq not installed")


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
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "rpm": 6000,
    },
}

# Ability levels from Kröger et al.
# Population distribution: Below Basic 25%, Basic 35%, Proficient 25%, Advanced 15%
ABILITY_LEVELS = {
    "below_basic": {
        "proportion": 0.25,
        "description": """You are a student who struggles significantly with math.
You often feel confused by math problems and frequently make errors.
You have difficulty with multi-step problems and often forget procedures.
You tend to guess when you're unsure, which is often.""",
    },
    "basic": {
        "proportion": 0.35,
        "description": """You are a student with basic math skills.
You can handle straightforward problems but struggle with anything complex.
You sometimes mix up procedures or apply the wrong method.
You can usually get simple calculations right but make errors on harder ones.""",
    },
    "proficient": {
        "proportion": 0.25,
        "description": """You are a proficient math student.
You understand most concepts and can apply procedures correctly.
You occasionally make careless errors but generally perform well.
You can handle moderately complex problems without much difficulty.""",
    },
    "advanced": {
        "proportion": 0.15,
        "description": """You are an advanced math student who excels at the subject.
You rarely make errors and can tackle complex problems confidently.
You understand the underlying concepts, not just the procedures.
You might occasionally make a careless mistake but almost always get problems right.""",
    },
}

PROMPT_TEMPLATE = """You are taking a math test. {ability_description}

Answer the following question. Think briefly about how you would approach it given your skill level, then give your final answer.

Question: {question}

{options}

First show your thinking (1-2 sentences), then state your answer as "Answer: X" where X is A, B, C, or D."""


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=200,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def call_groq(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def call_model(model_name: str, prompt: str, temperature: float = 0.7) -> str:
    """Route to appropriate API."""
    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "openai":
        return call_openai(model_id, prompt, temperature)
    elif provider == "anthropic":
        return call_anthropic(model_id, prompt, temperature)
    elif provider == "groq":
        return call_groq(model_id, prompt, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# RESPONSE PARSING
# =============================================================================

import re

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
        return matches[-1]  # Return last mentioned

    return None


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

            # Calculate difficulty as 1 - accuracy
            correct = row['CorrectAnswer']  # Kaggle letter (for LLM prompt)
            neurips_correct = row['neurips_correct_pos']  # NeurIPS position (for pct lookup)
            correct_pct = row[f'pct_{neurips_correct}'] / 100
            difficulty = 1 - correct_pct

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": correct,
                "student_dist": {
                    "A": row['pct_A'] / 100,
                    "B": row['pct_B'] / 100,
                    "C": row['pct_C'] / 100,
                    "D": row['pct_D'] / 100,
                },
                "student_difficulty": difficulty,
                "total_responses": row['total_responses']
            }
            items.append(item)

        if limit and len(items) >= limit:
            break

    return items


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_classroom_simulation(
    items: list,
    model_name: str,
    students_per_level: int = 5,
    output_dir: Path = None
):
    """Run classroom simulation with ability levels."""

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODELS[model_name]
    rpm = config["rpm"]
    delay = 60.0 / rpm

    print(f"\n{'='*60}")
    print(f"CLASSROOM SIMULATION: {model_name}")
    print(f"Students per level: {students_per_level}")
    print(f"Total simulated per item: {students_per_level * len(ABILITY_LEVELS)}")
    print('='*60)

    results = []

    for i, item in enumerate(items):
        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        # Collect responses from simulated classroom
        all_responses = []
        level_responses = {level: [] for level in ABILITY_LEVELS}

        for level_name, level_config in ABILITY_LEVELS.items():
            print(f"  {level_name}: ", end="", flush=True)

            for student in range(students_per_level):
                prompt = PROMPT_TEMPLATE.format(
                    ability_description=level_config['description'],
                    question=item['question'],
                    options=item['options']
                )

                try:
                    raw = call_model(model_name, prompt, temperature=0.7)
                    answer = parse_answer(raw)

                    if answer:
                        all_responses.append({
                            "level": level_name,
                            "answer": answer,
                            "weight": level_config['proportion']
                        })
                        level_responses[level_name].append(answer)
                        print(answer, end="", flush=True)
                    else:
                        print("?", end="", flush=True)

                    time.sleep(delay)

                except Exception as e:
                    print(f"E", end="", flush=True)
                    time.sleep(delay * 2)

            print()

        # Compute aggregated distribution with population weighting
        weighted_counts = defaultdict(float)
        for resp in all_responses:
            # Weight by ability level proportion
            weighted_counts[resp['answer']] += resp['weight']

        # Normalize
        total_weight = sum(weighted_counts.values())
        if total_weight > 0:
            sim_dist = {opt: weighted_counts.get(opt, 0) / total_weight for opt in ['A', 'B', 'C', 'D']}
        else:
            sim_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        # Also compute unweighted (equal counts per level)
        unweighted_counts = defaultdict(int)
        for resp in all_responses:
            unweighted_counts[resp['answer']] += 1
        total_unweighted = sum(unweighted_counts.values())
        if total_unweighted > 0:
            unweighted_dist = {opt: unweighted_counts.get(opt, 0) / total_unweighted for opt in ['A', 'B', 'C', 'D']}
        else:
            unweighted_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        # Compute simulated difficulty (1 - accuracy)
        correct = item['correct']
        sim_difficulty = 1 - sim_dist.get(correct, 0)
        unweighted_difficulty = 1 - unweighted_dist.get(correct, 0)

        result = {
            "item_id": item['id'],
            "correct_answer": correct,
            "student_difficulty": item['student_difficulty'],
            "simulated_difficulty_weighted": sim_difficulty,
            "simulated_difficulty_unweighted": unweighted_difficulty,
            "student_dist": item['student_dist'],
            "simulated_dist_weighted": sim_dist,
            "simulated_dist_unweighted": unweighted_dist,
            "level_responses": {k: list(v) for k, v in level_responses.items()},
            "n_responses": len(all_responses),
        }
        results.append(result)

        print(f"  Student difficulty: {item['student_difficulty']:.2f}")
        print(f"  Sim difficulty (weighted): {sim_difficulty:.2f}")
        print(f"  Sim difficulty (unweighted): {unweighted_difficulty:.2f}")

        # Save intermediate
        if output_dir:
            with open(output_dir / f"{model_name}_intermediate.json", 'w') as f:
                json.dump(results, f, indent=2)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_classroom_simulation(results: list) -> dict:
    """Compute difficulty prediction metrics."""

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print('='*60)

    student_difficulties = [r['student_difficulty'] for r in results]
    sim_difficulties_weighted = [r['simulated_difficulty_weighted'] for r in results]
    sim_difficulties_unweighted = [r['simulated_difficulty_unweighted'] for r in results]

    # Correlation between simulated and actual difficulty
    r_weighted, p_weighted = stats.pearsonr(sim_difficulties_weighted, student_difficulties)
    r_unweighted, p_unweighted = stats.pearsonr(sim_difficulties_unweighted, student_difficulties)

    # Spearman (rank) correlation
    rho_weighted, rho_p_weighted = stats.spearmanr(sim_difficulties_weighted, student_difficulties)
    rho_unweighted, rho_p_unweighted = stats.spearmanr(sim_difficulties_unweighted, student_difficulties)

    # RMSE
    rmse_weighted = np.sqrt(np.mean((np.array(sim_difficulties_weighted) - np.array(student_difficulties))**2))
    rmse_unweighted = np.sqrt(np.mean((np.array(sim_difficulties_unweighted) - np.array(student_difficulties))**2))

    # MAE
    mae_weighted = np.mean(np.abs(np.array(sim_difficulties_weighted) - np.array(student_difficulties)))
    mae_unweighted = np.mean(np.abs(np.array(sim_difficulties_unweighted) - np.array(student_difficulties)))

    print(f"\nItems analyzed: {len(results)}")
    print(f"\nDifficulty Prediction (Weighted by ability level proportions):")
    print(f"  Pearson r: {r_weighted:.3f} (p={p_weighted:.4f})")
    print(f"  Spearman ρ: {rho_weighted:.3f} (p={rho_p_weighted:.4f})")
    print(f"  RMSE: {rmse_weighted:.3f}")
    print(f"  MAE: {mae_weighted:.3f}")
    print(f"\nDifficulty Prediction (Unweighted - equal samples per level):")
    print(f"  Pearson r: {r_unweighted:.3f} (p={p_unweighted:.4f})")
    print(f"  Spearman ρ: {rho_unweighted:.3f} (p={rho_p_unweighted:.4f})")
    print(f"  RMSE: {rmse_unweighted:.3f}")
    print(f"  MAE: {mae_unweighted:.3f}")
    print(f"\nBenchmark from Kröger et al.: r=0.75-0.82")

    # Distribution correlation (all options, all items)
    all_sim_probs = []
    all_student_probs = []
    for r in results:
        for opt in ['A', 'B', 'C', 'D']:
            all_sim_probs.append(r['simulated_dist_weighted'].get(opt, 0))
            all_student_probs.append(r['student_dist'].get(opt, 0))

    dist_r, dist_p = stats.pearsonr(all_sim_probs, all_student_probs)
    print(f"\nDistribution Correlation (all options):")
    print(f"  r: {dist_r:.3f} (p={dist_p:.4f})")

    analysis = {
        "n_items": len(results),
        "weighted": {
            "pearson_r": r_weighted,
            "pearson_p": p_weighted,
            "spearman_rho": rho_weighted,
            "spearman_p": rho_p_weighted,
            "rmse": rmse_weighted,
            "mae": mae_weighted,
        },
        "unweighted": {
            "pearson_r": r_unweighted,
            "pearson_p": p_unweighted,
            "spearman_rho": rho_unweighted,
            "spearman_p": rho_p_unweighted,
            "rmse": rmse_unweighted,
            "mae": mae_unweighted,
        },
        "distribution_correlation": {
            "r": dist_r,
            "p": dist_p,
        }
    }

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Classroom Simulation Study")
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    parser.add_argument("--data", type=str,
                       default=str(script_dir / "data/eedi/eedi_with_student_data.csv"),
                       help="Path to Eedi data CSV")
    parser.add_argument("--output", type=str, default=str(script_dir / "pilot/replications/classroom_simulation"),
                       help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to use")
    parser.add_argument("--students", type=int, default=5,
                       help="Students per ability level")
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
        analysis = analyze_classroom_simulation(results)
        return

    print("="*60)
    print("CLASSROOM SIMULATION REPLICATION")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Students per level: {args.students}")
    print(f"Total per item: {args.students * len(ABILITY_LEVELS)}")
    print(f"Item limit: {args.items or 'all'}")
    print()

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items loaded!")
        return

    # Run experiment
    results = run_classroom_simulation(
        items=items,
        model_name=args.model,
        students_per_level=args.students,
        output_dir=output_dir
    )

    # Save results
    results_file = output_dir / f"{args.model}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Analyze
    analysis = analyze_classroom_simulation(results)

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
