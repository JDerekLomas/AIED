#!/usr/bin/env python3
"""
Direct difficulty estimation on SmartPaper items.

Purpose: Establish the null-result baseline on SmartPaper, parallel to the
Eedi direct estimation experiments (H1 in the megaexperiment table).
If direct estimation also fails on SmartPaper (open-ended, Indian govt school),
this shows the problem isn't Eedi-specific.

Method: Present item text + rubric to LLM, ask for estimated proportion correct.
Compare to actual classical_difficulty from student data.

Design mirrors replicate_direct_difficulty.py (Eedi):
- Same 4 prompt variants from prior literature (basic, expert, irt, comparative)
- Same metrics (Pearson r, Spearman ρ, RMSE, MAE, bias)
- Temperature=0.0 for deterministic comparison (matching Eedi runs)
- All 140 SmartPaper items (vs Eedi's 1869)

Eedi baseline: r=-0.03 to +0.01 across 5 models × 4 prompts (n=1869)
Prior literature claims: r=0.54-0.82 (Benedetto, Attali, Yaneva)

API calls: 140 items × 4 prompts = 560 calls per model.
"""

import json, os, re, time, argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/smartpaper_direct_estimation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompts — adapted from replicate_direct_difficulty.py for open-ended items
# Same structure as Eedi prompts but with rubric instead of MCQ options.
# ---------------------------------------------------------------------------
PROMPTS = {
    "basic": """Estimate the difficulty of this question for Class {grade} students in Indian government schools.

Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

What proportion of students do you think would answer this correctly (earn full marks)?
Reply with just a number between 0.0 and 1.0.""",

    "expert": """You are an expert in educational measurement and item response theory.

Analyze this question and estimate what proportion of Class {grade} students in Indian government schools would answer it correctly (earn full marks).

Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Consider:
- The cognitive complexity required
- Common student misconceptions in this subject
- Typical knowledge levels in Indian government schools
- Many students have limited English proficiency

Provide your estimate as a single number from 0.0 to 1.0 (proportion correct).
Reply with just the number.""",

    "irt": """You are calibrating items for an adaptive test using Item Response Theory.

Estimate the difficulty parameter for this item. In IRT, difficulty represents the ability level at which a student has a 50% chance of answering correctly.

However, for this task, provide your estimate as the expected proportion of Class {grade} Indian government school students that would answer correctly (0.0 to 1.0).

Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Provide only a number from 0.0 to 1.0.""",

    "comparative": """Rate this question's difficulty on a scale where:
- 0.80-1.00: Very easy (most students get it right)
- 0.60-0.80: Easy
- 0.40-0.60: Medium difficulty
- 0.20-0.40: Hard
- 0.00-0.20: Very hard (most students get it wrong)

The students are Class {grade} in Indian government schools.

Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

What proportion of students would get this correct? (0.0 to 1.0)
Reply with just a number.""",
}


# ---------------------------------------------------------------------------
# API callers — supports multiple providers like the Eedi script
# ---------------------------------------------------------------------------

def call_gemini(client, prompt, model_id, temperature=0.0):
    from google.genai import types
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def call_openai(client, prompt, model_id, temperature=0.0):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def call_groq(client, prompt, model_id, temperature=0.0):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


MODELS = {
    "gemini-3-flash": {"provider": "gemini", "model_id": "gemini-3-flash-preview"},
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "llama-3.3-70b": {"provider": "groq", "model_id": "llama-3.3-70b-versatile"},
}


def make_client(provider):
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "groq":
        from groq import Groq
        return Groq()
    raise ValueError(f"Unknown provider: {provider}")


def call_model(client, provider, model_id, prompt, temperature=0.0):
    if provider == "gemini":
        return call_gemini(client, prompt, model_id, temperature)
    elif provider == "openai":
        return call_openai(client, prompt, model_id, temperature)
    elif provider == "groq":
        return call_groq(client, prompt, model_id, temperature)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_items():
    """Load all 140 SmartPaper items with ground truth difficulty."""
    with open("data/smartpaper/item_statistics.json") as f:
        items = json.load(f)
    for item in items:
        item["item_key"] = f"{item['assessment']}_q{item['question_number']}"
    return items


def parse_estimate(text):
    """Extract a float between 0 and 1 from model response."""
    matches = re.findall(r'(\d*\.?\d+)', text.strip())
    for m in matches:
        val = float(m)
        if 0 <= val <= 1:
            return val
        if 1 < val <= 100:
            return val / 100
    return None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(client, provider, model_id, model_name, items, prompt_name,
                   template, temperature, output_dir):
    """Run one model × one prompt on all items. Returns analysis dict."""
    raw_dir = output_dir / model_name / prompt_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for item in items:
        key = item["item_key"]
        raw_path = raw_dir / f"{key}.txt"

        if raw_path.exists():
            text = raw_path.read_text()
        else:
            prompt = template.format(
                grade=item["grade"],
                subject=item["subject"],
                question_text=item["question_text"],
                rubric=item["rubric"],
                max_score=item["max_score"],
            )
            try:
                text = call_model(client, provider, model_id, prompt, temperature)
                raw_path.write_text(text)
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
                time.sleep(2)
                continue
            time.sleep(0.15)

        estimate = parse_estimate(text)
        predictions.append({
            "item_key": key,
            "actual_difficulty": item["classical_difficulty"],
            "estimated_p_correct": estimate,
            "raw_response": text.strip()[:100],
        })

    df = pd.DataFrame(predictions)
    df.to_csv(raw_dir / "predictions.csv", index=False)

    # Analyze
    valid = df.dropna(subset=["estimated_p_correct"])
    if len(valid) < 5:
        return {"n_valid": len(valid), "error": "insufficient_data"}

    actual = valid["actual_difficulty"].values
    predicted = valid["estimated_p_correct"].values

    rho, rho_p = stats.spearmanr(predicted, actual)
    r, r_p = stats.pearsonr(predicted, actual)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    bias = float(np.mean(predicted - actual))

    result = {
        "n_total": len(df),
        "n_valid": len(valid),
        "n_parse_fail": len(df) - len(valid),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }

    print(f"  {model_name}/{prompt_name}: r={r:.3f}, ρ={rho:.3f} (p={rho_p:.3f}), "
          f"n={len(valid)}, RMSE={rmse:.3f}, bias={bias:+.3f}", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SmartPaper Direct Difficulty Estimation — parallel to Eedi H1"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gemini-3-flash",
                        choices=list(MODELS.keys()),
                        help="Model to use (default: gemini-3-flash)")
    parser.add_argument("--prompt", default=None,
                        choices=list(PROMPTS.keys()),
                        help="Single prompt to run (default: all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all available models")
    args = parser.parse_args()

    items = load_items()
    print(f"SmartPaper items: {len(items)}")
    print(f"Difficulty range: [{min(i['classical_difficulty'] for i in items):.3f}, "
          f"{max(i['classical_difficulty'] for i in items):.3f}]")
    print(f"Subjects: {sorted(set(i['subject'] for i in items))}")
    print(f"Grades: {sorted(set(i['grade'] for i in items))}")
    print()

    prompt_types = [args.prompt] if args.prompt else list(PROMPTS.keys())
    model_names = list(MODELS.keys()) if args.all_models else [args.model]

    total_calls = len(items) * len(prompt_types) * len(model_names)
    print(f"Plan: {len(items)} items × {len(prompt_types)} prompts × "
          f"{len(model_names)} models = {total_calls} calls")
    print(f"Temperature: {args.temperature}")
    print()

    if args.dry_run:
        print("DRY RUN — no API calls")
        return

    all_results = {}
    for model_name in model_names:
        model_cfg = MODELS[model_name]
        client = make_client(model_cfg["provider"])

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = {}
        for prompt_name in prompt_types:
            result = run_experiment(
                client, model_cfg["provider"], model_cfg["model_id"],
                model_name, items, prompt_name, PROMPTS[prompt_name],
                args.temperature, OUTPUT_DIR,
            )
            model_results[prompt_name] = result

        all_results[model_name] = model_results

        # Save per-model analysis
        with open(OUTPUT_DIR / model_name / "analysis.json", "w") as f:
            json.dump(model_results, f, indent=2, default=str)

    # ---------------------------------------------------------------------------
    # Summary — format matches Eedi replication output
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SMARTPAPER DIRECT ESTIMATION RESULTS")
    print(f"{'='*60}")
    print(f"Items: {len(items)} (all SmartPaper, open-ended, Grades 6-8)")
    print(f"Temperature: {args.temperature}")
    print()

    print(f"{'Model':<20} {'Prompt':<15} {'r':<8} {'ρ':<8} {'RMSE':<8} {'Bias':<8} {'n':<6}")
    print("-" * 70)
    for model_name, model_results in all_results.items():
        for prompt_name, res in model_results.items():
            if "error" in res:
                print(f"{model_name:<20} {prompt_name:<15} INSUFFICIENT DATA")
            else:
                print(f"{model_name:<20} {prompt_name:<15} "
                      f"{res['pearson_r']:.3f}   {res['spearman_rho']:.3f}   "
                      f"{res['rmse']:.3f}   {res['bias']:+.3f}   {res['n_valid']}")

    print()
    print("EEDI COMPARISON (replicate_direct_difficulty.py, n=1869):")
    print("  gpt-4o-mini/basic:  r=0.007, ρ=-0.016")
    print("  gpt-4o-mini/expert: r=0.010, ρ=0.007")
    print("  gpt-4o-mini/irt:    r=-0.029, ρ=-0.018")
    print("  gpt-4o-mini/comp:   r=-0.003, ρ=-0.008")
    print("  (5 models tested, all r≈0)")
    print()
    print("PRIOR LITERATURE CLAIMS:")
    print("  Attali (2024):    r=0.63-0.82")
    print("  Yaneva et al.:    r=0.65-0.78")
    print("  Benedetto et al.: r=0.54-0.71")

    # Save combined results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({
            "meta": {
                "n_items": len(items),
                "temperature": args.temperature,
                "models": model_names,
                "prompts": prompt_types,
                "dataset": "SmartPaper (Indian govt schools, open-ended, Grades 6-8)",
                "ground_truth": "classical_difficulty (proportion correct from student data)",
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
