#!/usr/bin/env python3
"""
Replication: Direct Difficulty Estimation
Multiple papers: Benedetto et al., Attali et al., Yaneva et al.

Research Question: Can LLMs directly estimate item difficulty from text alone?

Method:
1. Present item text + options to LLM
2. Ask for direct difficulty estimate (p-value: proportion correct)
3. Compare to actual student performance

Benchmarks from literature:
- Attali (2024): r=0.63-0.82 depending on prompt
- Yaneva et al.: r=0.65-0.78 for medical items
- Benedetto et al.: r=0.54-0.71 for math items
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
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "claude-3-haiku": {"provider": "anthropic", "model_id": "claude-3-haiku-20240307"},
    "claude-3.5-sonnet": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022"},
    "qwen3-32b": {"provider": "groq", "model_id": "qwen/qwen3-32b"},
    "llama-3.3-70b": {"provider": "groq", "model_id": "llama-3.3-70b-versatile"},
    "llama-4-scout": {"provider": "groq", "model_id": "meta-llama/llama-4-scout-17b-16e-instruct"},
    "llama-4-maverick": {"provider": "groq", "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct"},
    "gemini-2.5-flash": {"provider": "gemini", "model_id": "gemini-2.5-flash"},
    "gemini-3-flash": {"provider": "gemini", "model_id": "gemini-3-flash-preview"},
    "deepseek-v3": {"provider": "deepseek", "model_id": "deepseek-chat"},
    "deepseek-r1": {"provider": "deepseek", "model_id": "deepseek-reasoner"},
}

# Different prompt strategies from literature
PROMPTS = {
    "basic": """Estimate the difficulty of this math question for middle school students.

Question: {question}

Options:
{options}

Correct Answer: {correct}

What percentage of students do you think would answer this correctly?
Reply with just a number between 0 and 100.""",

    "expert": """You are an expert in educational measurement and item response theory.

Analyze this math question and estimate what proportion of typical middle school students would answer it correctly.

Question: {question}

Options:
{options}

Correct Answer: {correct}

Consider:
- The cognitive complexity required
- Common student misconceptions
- The quality of distractors
- Typical student knowledge at this level

Provide your estimate as a single number from 0 to 100 (percentage correct).
Reply with just the number.""",

    "irt": """You are calibrating items for an adaptive test using Item Response Theory.

Estimate the difficulty parameter for this item. In IRT, difficulty represents the ability level at which a student has a 50% chance of answering correctly.

However, for this task, provide your estimate as the expected proportion of a typical middle school population that would answer correctly (0-100%).

Question: {question}

Options:
{options}

Correct Answer: {correct}

Provide only a number from 0 to 100.""",

    "error_analysis_direct": """You are an experienced UK maths teacher marking a set of Year 9 mock exams.

For this question, before estimating difficulty, think about:
- What specific calculation error leads to each wrong answer?
- Which errors are "attractive" — where the wrong method feels natural?
- Which students would catch themselves vs. fall into the trap?

Question: {question}

Options:
{options}

Correct Answer: {correct}

First, briefly analyze what misconception or error produces each wrong option (1-2 sentences each). Then, based on your analysis, estimate what percentage of Year 9 students would answer this correctly.

End your response with: ESTIMATE: [number]""",

    "error_analysis_sim": """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

For this question, based on your experience of what students actually wrote, predict what percentage at each ability level chose each option.

Before predicting, think about:
- What specific calculation error leads to each wrong answer?
- Which errors are "attractive" — where the wrong method feels natural?
- Which students would catch themselves vs. fall into the trap?

Question: {question}

Options:
{options}

Correct Answer: {correct}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "comparative": """Rate this math question's difficulty on a scale where:
- 0-20%: Very easy (most students get it right)
- 20-40%: Easy
- 40-60%: Medium difficulty
- 60-80%: Hard
- 80-100%: Very hard (most students get it wrong)

Question: {question}

Options:
{options}

Correct Answer: {correct}

What percentage of students would get this WRONG? (0-100)
Reply with just a number.""",
}


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_id: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def call_groq(model_id: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def call_deepseek(model_id: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def call_gemini(model_id: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )
    text = response.text
    if text is None:
        # Try extracting from parts
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text.strip()
        return ""
    return text.strip()


def call_model(model_name: str, prompt: str, temperature: float = 0, max_tokens: int = 50) -> str:
    config = MODELS[model_name]
    if config["provider"] == "openai":
        return call_openai(config["model_id"], prompt, temperature, max_tokens)
    elif config["provider"] == "anthropic":
        return call_anthropic(config["model_id"], prompt, temperature, max_tokens)
    elif config["provider"] == "groq":
        return call_groq(config["model_id"], prompt, temperature, max_tokens)
    elif config["provider"] == "gemini":
        return call_gemini(config["model_id"], prompt, temperature, max_tokens)
    elif config["provider"] == "deepseek":
        return call_deepseek(config["model_id"], prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


def parse_percentage(response: str) -> float:
    """Extract percentage from response."""
    # Remove % symbol and any text
    response = response.replace('%', '').strip()

    # Find first number
    match = re.search(r'(\d+(?:\.\d+)?)', response)
    if match:
        val = float(match.group(1))
        # Normalize to 0-1 if given as percentage
        if val > 1:
            val = val / 100
        return val
    return None


def parse_estimate_tag(response: str) -> float:
    """Extract number after ESTIMATE: tag."""
    match = re.search(r'ESTIMATE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val > 1:
            val = val / 100
        return val
    # Fallback: try last number in response
    numbers = re.findall(r'(\d+(?:\.\d+)?)', response)
    if numbers:
        val = float(numbers[-1])
        if val > 1:
            val = val / 100
        return val
    return None


def parse_simulation_distribution(response: str, correct_letter: str) -> float:
    """Parse level distributions and derive difficulty (proportion incorrect).

    Expects format like:
    below_basic: A=10% B=60% C=20% D=10%
    basic: A=30% B=40% C=20% D=10%
    ...

    Returns proportion correct (p-value) using weighted ability levels.
    """
    weights = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}
    total_correct = 0.0
    levels_found = 0

    for level, weight in weights.items():
        pattern = rf'{level}\s*:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            pcts = {"A": float(match.group(1)), "B": float(match.group(2)),
                    "C": float(match.group(3)), "D": float(match.group(4))}
            correct_pct = pcts.get(correct_letter.upper(), 0) / 100
            total_correct += weight * correct_pct
            levels_found += 1

    if levels_found >= 3:  # Need at least 3 of 4 levels
        # Renormalize weights for found levels
        return total_correct
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

            correct = row['CorrectAnswer']  # Kaggle letter (for LLM prompt)
            neurips_correct = row['neurips_correct_pos']  # NeurIPS position (for pct lookup)
            p_value = row[f'pct_{neurips_correct}'] / 100  # Actual proportion correct

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": correct,
                "p_value": p_value,  # Actual difficulty (proportion correct)
                "difficulty": 1 - p_value,  # Difficulty = 1 - p_value
                "total_responses": row['total_responses']
            }
            items.append(item)

        if limit and len(items) >= limit:
            break

    return items


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_direct_difficulty_experiment(
    items: list,
    model_name: str,
    prompt_type: str = "expert",
    output_dir: Path = None,
    temperature: float = 0.0
):
    """Run direct difficulty estimation."""

    print(f"\n{'='*60}")
    print(f"DIRECT DIFFICULTY ESTIMATION")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt_type}")
    print(f"Temperature: {temperature}")
    print(f"Items: {len(items)}")
    print('='*60)

    prompt_template = PROMPTS[prompt_type]
    # Use more tokens for error analysis prompts (need reasoning space)
    max_tokens = 500 if prompt_type.startswith("error_analysis") else 50
    results = []

    for i, item in enumerate(items):
        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        # Format prompt
        prompt = prompt_template.format(
            question=item['question'],
            options=item['options'],
            correct=item['correct']
        )

        try:
            raw = call_model(model_name, prompt, temperature=temperature, max_tokens=max_tokens)

            # Parse response based on prompt type
            if prompt_type == "comparative":
                parsed = parse_percentage(raw)
                if parsed is not None:
                    predicted_p = 1 - parsed
                else:
                    predicted_p = None
            elif prompt_type == "error_analysis_direct":
                predicted_p = parse_estimate_tag(raw)
            elif prompt_type == "error_analysis_sim":
                predicted_p = parse_simulation_distribution(raw, item['correct'])
            else:
                predicted_p = parse_percentage(raw)

            result = {
                "item_id": item['id'],
                "actual_p_value": item['p_value'],
                "predicted_p_value": predicted_p,
                "raw_response": raw,
            }

            if predicted_p is not None:
                error = abs(predicted_p - item['p_value'])
                result["absolute_error"] = error
                print(f"  Actual: {item['p_value']:.2f}, Predicted: {predicted_p:.2f}, Error: {error:.2f}")
            else:
                print(f"  Failed to parse: {raw[:50]}")

            results.append(result)
            time.sleep(0.1)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "item_id": item['id'],
                "actual_p_value": item['p_value'],
                "predicted_p_value": None,
                "error": str(e)
            })
            time.sleep(0.5)

        # Save intermediate
        if output_dir and (i + 1) % 20 == 0:
            with open(output_dir / f"{model_name}_{prompt_type}_intermediate.json", 'w') as f:
                json.dump(results, f, indent=2)

    return results


def analyze_results(results: list, prompt_type: str) -> dict:
    """Analyze difficulty estimation results."""

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {prompt_type}")
    print('='*60)

    # Filter valid results
    valid = [r for r in results if r.get('predicted_p_value') is not None]

    if len(valid) < 3:
        print(f"Not enough valid predictions: {len(valid)}")
        return {"error": "insufficient_data", "n_valid": len(valid)}

    actual = [r['actual_p_value'] for r in valid]
    predicted = [r['predicted_p_value'] for r in valid]

    # Correlation
    r, p = stats.pearsonr(predicted, actual)
    rho, rho_p = stats.spearmanr(predicted, actual)

    # Error metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # Bias (systematic over/under estimation)
    bias = np.mean(np.array(predicted) - np.array(actual))

    print(f"\nItems analyzed: {len(valid)}/{len(results)}")
    print(f"\nCorrelations:")
    print(f"  Pearson r: {r:.3f} (p={p:.4f})")
    print(f"  Spearman ρ: {rho:.3f} (p={rho_p:.4f})")
    print(f"\nError Metrics:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  Bias: {bias:+.3f} ({'overestimates' if bias > 0 else 'underestimates'} difficulty)")
    print(f"\nBenchmarks from literature:")
    print(f"  Attali (2024): r=0.63-0.82")
    print(f"  Yaneva et al.: r=0.65-0.78")
    print(f"  Benedetto et al.: r=0.54-0.71")

    return {
        "n_total": len(results),
        "n_valid": len(valid),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Direct Difficulty Estimation Replication")

    script_dir = Path(__file__).parent.parent
    parser.add_argument("--data", type=str,
                       default=str(script_dir / "data/eedi/eedi_with_student_data.csv"))
    parser.add_argument("--output", type=str,
                       default=str(script_dir / "pilot/replications/direct_difficulty"))
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       choices=list(MODELS.keys()))
    parser.add_argument("--prompt", type=str, default="expert",
                       choices=list(PROMPTS.keys()))
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--all-prompts", action="store_true",
                       help="Run all prompt types")
    parser.add_argument("--analyze-only", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        with open(args.analyze_only) as f:
            results = json.load(f)
        analyze_results(results, "loaded")
        return

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items loaded!")
        return

    # Run experiment(s)
    prompt_types = list(PROMPTS.keys()) if args.all_prompts else [args.prompt]

    all_analyses = {}

    for prompt_type in prompt_types:
        results = run_direct_difficulty_experiment(
            items=items,
            model_name=args.model,
            prompt_type=prompt_type,
            output_dir=output_dir,
            temperature=args.temperature
        )

        # Save results
        results_file = output_dir / f"{args.model}_{prompt_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Analyze
        analysis = analyze_results(results, prompt_type)
        all_analyses[prompt_type] = analysis

    # Save combined analysis
    analysis_file = output_dir / f"{args.model}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(all_analyses, f, indent=2)

    # Summary
    if len(prompt_types) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY: All Prompt Types")
        print('='*60)
        print(f"{'Prompt':<15} {'r':<8} {'RMSE':<8} {'MAE':<8} {'Bias':<8}")
        print('-'*50)
        for pt, analysis in all_analyses.items():
            if 'error' not in analysis:
                print(f"{pt:<15} {analysis['pearson_r']:.3f}    {analysis['rmse']:.3f}    {analysis['mae']:.3f}    {analysis['bias']:+.3f}")


if __name__ == "__main__":
    main()
