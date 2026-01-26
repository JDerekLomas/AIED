#!/usr/bin/env python3
"""
Replication 1: Error Alignment (Index-Based)
Paper: "Do LLMs Make Mistakes Like Students?" (arXiv:2502.15140)

Research Question: When LLMs make errors, do they select the same wrong answers as students?

Method:
1. Present items with A/B/C/D options to models (no student prompt)
2. Run N=10 samples per item with temperature=1.0
3. Compute probability distribution over options
4. Compare LLM error distribution to student error distribution

Key Metric: Error selection correlation (benchmark: 0.73-0.80)
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
    "mistral-7b": {
        "provider": "groq",
        "model_id": "mistral-saba-24b",  # Using available Groq model
        "rpm": 30,
    },
}

# Simple direct prompt - no student persona
PROMPT_TEMPLATE = """Answer this multiple choice math question. Select only the letter (A, B, C, or D).

Question: {question}

{options}

Answer:"""


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str, temperature: float = 1.0) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_id: str, prompt: str, temperature: float = 1.0) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def call_groq(model_id: str, prompt: str, temperature: float = 1.0) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()


def call_model(model_name: str, prompt: str, temperature: float = 1.0) -> str:
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

def parse_answer(response: str) -> str:
    """Extract answer letter from response."""
    response = response.upper().strip()

    # Direct letter
    if response in ['A', 'B', 'C', 'D']:
        return response

    # Letter with punctuation
    for letter in ['A', 'B', 'C', 'D']:
        if response.startswith(letter):
            return letter

    # Search for letter in response
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter

    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eedi_items(csv_path: str, limit: int = None) -> list:
    """Load Eedi items from CSV."""
    df = pd.read_csv(csv_path)

    items = []
    for _, row in df.iterrows():
        # Build options string from the curated CSV format
        if 'target_answer' in row:
            # curated_eedi_items.csv format - need to reconstruct options
            # This file doesn't have individual option texts, skip for now
            continue
        elif 'AnswerAText' in row:
            # eedi_with_student_data.csv format
            options = f"A) {row['AnswerAText']}\nB) {row['AnswerBText']}\nC) {row['AnswerCText']}\nD) {row['AnswerDText']}"

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": row['CorrectAnswer'],
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

def run_error_alignment(
    items: list,
    models: list,
    samples_per_item: int = 10,
    output_dir: Path = None
):
    """Run error alignment experiment."""

    results = []

    for model_name in models:
        if model_name not in MODELS:
            print(f"Skipping unknown model: {model_name}")
            continue

        config = MODELS[model_name]
        rpm = config["rpm"]
        delay = 60.0 / rpm

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print('='*60)

        model_results = []

        for i, item in enumerate(items):
            print(f"\n  Item {i+1}/{len(items)}: {item['id']}")

            # Collect samples
            responses = defaultdict(int)
            prompt = PROMPT_TEMPLATE.format(
                question=item['question'],
                options=item['options']
            )

            for sample in range(samples_per_item):
                try:
                    raw = call_model(model_name, prompt, temperature=1.0)
                    answer = parse_answer(raw)

                    if answer:
                        responses[answer] += 1
                    else:
                        responses['INVALID'] += 1

                    time.sleep(delay)

                except Exception as e:
                    print(f"    ERROR sample {sample}: {e}")
                    responses['ERROR'] += 1
                    time.sleep(delay * 2)

            # Compute LLM distribution
            valid_total = sum(responses.get(l, 0) for l in ['A', 'B', 'C', 'D'])
            if valid_total > 0:
                llm_dist = {
                    "A": responses.get('A', 0) / valid_total,
                    "B": responses.get('B', 0) / valid_total,
                    "C": responses.get('C', 0) / valid_total,
                    "D": responses.get('D', 0) / valid_total,
                }
            else:
                llm_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

            # Record result
            result = {
                "model": model_name,
                "item_id": item['id'],
                "correct_answer": item['correct'],
                "samples": samples_per_item,
                "valid_responses": valid_total,
                "llm_dist": llm_dist,
                "student_dist": item['student_dist'],
                "raw_counts": dict(responses),
            }
            model_results.append(result)

            # Print progress
            correct = item['correct']
            llm_acc = llm_dist.get(correct, 0)
            student_acc = item['student_dist'].get(correct, 0)
            print(f"    LLM correct: {llm_acc:.2f}, Student correct: {student_acc:.2f}")

        results.extend(model_results)

        # Save intermediate results
        if output_dir:
            with open(output_dir / f"{model_name}_results.json", 'w') as f:
                json.dump(model_results, f, indent=2)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_error_alignment(results: list) -> dict:
    """Compute error alignment metrics."""

    analysis = {}

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r['model']].append(r)

    for model, model_results in by_model.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print('='*60)

        # Collect error distributions (excluding correct answer)
        llm_error_probs = []
        student_error_probs = []

        # Also track per-option alignment
        all_llm = []
        all_student = []

        items_with_errors = 0

        for r in model_results:
            correct = r['correct_answer']

            for opt in ['A', 'B', 'C', 'D']:
                llm_p = r['llm_dist'].get(opt, 0)
                student_p = r['student_dist'].get(opt, 0)

                all_llm.append(llm_p)
                all_student.append(student_p)

                # Only errors (wrong answers)
                if opt != correct:
                    llm_error_probs.append(llm_p)
                    student_error_probs.append(student_p)

            # Count items where LLM made errors
            if r['llm_dist'].get(correct, 0) < 1.0:
                items_with_errors += 1

        # Overall correlation
        overall_r, overall_p = stats.pearsonr(all_llm, all_student)

        # Error-only correlation (when both have errors)
        error_r, error_p = stats.pearsonr(llm_error_probs, student_error_probs)

        # Accuracy
        llm_accuracy = np.mean([r['llm_dist'].get(r['correct_answer'], 0) for r in model_results])
        student_accuracy = np.mean([r['student_dist'].get(r['correct_answer'], 0) for r in model_results])

        print(f"\nItems analyzed: {len(model_results)}")
        print(f"Items where LLM made errors: {items_with_errors}")
        print(f"\nAccuracy:")
        print(f"  LLM: {llm_accuracy:.3f}")
        print(f"  Students: {student_accuracy:.3f}")
        print(f"\nDistribution Correlations:")
        print(f"  Overall (all options): r={overall_r:.3f}, p={overall_p:.4f}")
        print(f"  Errors only (wrong answers): r={error_r:.3f}, p={error_p:.4f}")
        print(f"\nBenchmark from paper: r=0.73-0.80")

        analysis[model] = {
            "n_items": len(model_results),
            "items_with_errors": items_with_errors,
            "llm_accuracy": llm_accuracy,
            "student_accuracy": student_accuracy,
            "overall_correlation": overall_r,
            "overall_p_value": overall_p,
            "error_correlation": error_r,
            "error_p_value": error_p,
        }

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Error Alignment Study")
    parser.add_argument("--data", type=str,
                       default="/Users/dereklomas/AIED/study2-materials/data/eedi/eedi_with_student_data.csv",
                       help="Path to Eedi data CSV")
    parser.add_argument("--output", type=str, default="pilot/replications/error_alignment",
                       help="Output directory")
    parser.add_argument("--models", nargs="+",
                       default=["gpt-4o-mini"],
                       help="Models to test")
    parser.add_argument("--samples", type=int, default=10,
                       help="Samples per item")
    parser.add_argument("--items", type=int, default=None,
                       help="Limit number of items (default: all)")
    parser.add_argument("--analyze-only", type=str, default=None,
                       help="Just analyze existing results")

    args = parser.parse_args()

    # Setup output
    output_dir = Path("/Users/dereklomas/AIED/study2-materials") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        # Load and analyze existing results
        with open(args.analyze_only) as f:
            results = json.load(f)
        analysis = analyze_error_alignment(results)
        return

    print("="*60)
    print("ERROR ALIGNMENT REPLICATION")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Samples per item: {args.samples}")
    print(f"Item limit: {args.items or 'all'}")
    print()

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items loaded! Check data format.")
        return

    # Run experiment
    results = run_error_alignment(
        items=items,
        models=args.models,
        samples_per_item=args.samples,
        output_dir=output_dir
    )

    # Save full results
    results_file = output_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Analyze
    analysis = analyze_error_alignment(results)

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
