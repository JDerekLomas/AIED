#!/usr/bin/env python3
"""
DBE-KT22 Out-of-Sample Validation
===================================
Supports both direct estimation (original) and contrastive 3-rep pipeline.

Usage:
  python run_dbe_kt22_validation.py                  # direct, gemini-2.5-flash
  python run_dbe_kt22_validation.py --mode contrastive  # 3-rep, gemini-3-flash
"""

import json
import os
import re
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "dbe-kt22"
OUTPUT_DIR = Path(__file__).parent.parent / "pilot" / "dbe_kt22_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_PROMPT = """Estimate the difficulty of this university database systems question.

Question: {question}

Options:
{options}

Correct Answer: {correct}

What percentage of undergraduate students would answer this correctly?
Reply with just a number between 0 and 100."""

CONTRASTIVE_PROMPT = """You are an experienced university database systems instructor. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK complex are actually straightforward because they test a single definition. Other questions that look simple have subtle traps involving multi-step reasoning.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right?

Question: {question}

{options}

Respond in this exact format:
struggling: {option_format}
average: {option_format}
good: {option_format}
advanced: {option_format}"""


def load_data():
    """Load DBE-KT22, compute empirical p-correct from student responses."""
    questions = pd.read_csv(DATA_DIR / "Questions.csv")
    choices = pd.read_csv(DATA_DIR / "Question_Choices.csv")
    transactions = pd.read_csv(DATA_DIR / "Transaction.csv")

    empirical = transactions.groupby("question_id").agg(
        n_responses=("answer_state", "count"),
        n_correct=("answer_state", "sum"),
    ).reset_index()
    empirical["p_correct"] = empirical["n_correct"] / empirical["n_responses"]
    questions = questions.merge(empirical, left_on="id", right_on="question_id", how="left")

    choice_map = {}
    for qid, group in choices.groupby("question_id"):
        group = group.sort_values("id")  # Deterministic label assignment regardless of CSV order
        opts = []
        labels = []
        correct_label = None
        for i, (_, row) in enumerate(group.iterrows()):
            label = chr(65 + i)
            opts.append(f"{label}. {row['choice_text']}")
            labels.append(label)
            if row["is_correct"]:
                correct_label = label
        choice_map[qid] = {
            "options_text": "\n".join(opts),
            "correct_label": correct_label,
            "labels": labels,
        }

    questions["options_text"] = questions["id"].map(lambda x: choice_map.get(x, {}).get("options_text", ""))
    questions["correct_label"] = questions["id"].map(lambda x: choice_map.get(x, {}).get("correct_label", ""))
    questions["option_labels"] = questions["id"].map(lambda x: choice_map.get(x, {}).get("labels", []))

    has_img = questions["question_rich_text"].str.contains("<img", na=False)
    questions["usable"] = (
        (questions["question_text"].str.len() > 20) &
        (questions["n_responses"] >= 50) &
        (~has_img)
    )

    usable = questions[questions["usable"]]
    print(f"Items: {len(usable)} usable / {len(questions)} total")
    print(f"Student responses: {usable['n_responses'].sum():,}")
    print(f"Empirical p_correct: {usable['p_correct'].mean():.2f} mean, "
          f"{usable['p_correct'].min():.2f}–{usable['p_correct'].max():.2f} range")
    return questions


def call_llm(prompt, model="gemini-2.5-flash", temperature=0.0, max_tokens=50):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        return response.text
    except Exception as e:
        print(f"  API error: {e}")
        return None


def parse_number(response):
    if not response:
        return None
    numbers = re.findall(r'(\d+(?:\.\d+)?)', response.strip())
    if numbers:
        val = float(numbers[0])
        if 0 <= val <= 100:
            return val / 100
    return None


def parse_contrastive(text, correct_answer, option_labels):
    """Parse contrastive response with variable number of options."""
    weights = {"struggling": 0.25, "average": 0.35, "good": 0.25, "advanced": 0.15}
    correct_idx = option_labels.index(correct_answer) if correct_answer in option_labels else 0
    n_opts = len(option_labels)

    weighted_p_correct = 0.0
    for level, w in weights.items():
        # Build pattern for variable options: A=XX% B=XX% C=XX% [D=XX%]
        opt_pattern = r'\s*'.join(rf'{lab}\s*=\s*(\d+)%?' for lab in option_labels)
        pattern = rf'{level}:\s*{opt_pattern}'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, n_opts + 1)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
        else:
            weighted_p_correct += 0.5 * w
    return 1 - weighted_p_correct


def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(rho)
    return np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def run_direct(questions):
    """Original direct estimation with Gemini 2.5 Flash."""
    model = "gemini-2.5-flash"
    usable = questions[questions["usable"]].copy()

    cache_file = OUTPUT_DIR / "predictions.json"
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached predictions")

    for idx, (_, row) in enumerate(usable.iterrows()):
        qid = str(row["id"])
        if qid in cache:
            continue
        prompt = DIRECT_PROMPT.format(
            question=row["question_text"],
            options=row["options_text"],
            correct=row["correct_label"],
        )
        raw = call_llm(prompt, model=model, temperature=0.0, max_tokens=50)
        cache[qid] = {"raw": raw, "predicted_p": parse_number(raw)}
        if (idx + 1) % 20 == 0:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  {len(cache)}/{len(usable)} done")
        time.sleep(0.1)

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    usable["predicted_p"] = usable["id"].map(lambda x: cache.get(str(x), {}).get("predicted_p"))
    valid = usable.dropna(subset=["predicted_p", "p_correct"])
    print(f"\nValid predictions: {len(valid)}/{len(usable)}")

    rho, p = stats.spearmanr(valid["predicted_p"], valid["p_correct"])
    ci_lo, ci_hi = bootstrap_ci(valid["predicted_p"].values, valid["p_correct"].values)
    rho_author, p_author = stats.spearmanr(valid["predicted_p"], -valid["difficulty"])

    print(f"\nRESULT (direct, {model})")
    print(f"ρ = {rho:.3f} (p = {p:.4f}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"ρ vs author difficulty = {rho_author:.3f}")

    result = {
        "dataset": "DBE-KT22", "model": model, "mode": "direct",
        "n_items": len(valid), "n_student_responses": int(valid["n_responses"].sum()),
        "rho": round(rho, 3), "p_value": round(p, 4),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "rho_author_difficulty": round(rho_author, 3),
    }
    with open(OUTPUT_DIR / "result.json", 'w') as f:
        json.dump(result, f, indent=2)


def run_contrastive(questions, n_reps=3):
    """Contrastive 3-rep pipeline with Gemini 3 Flash."""
    model = "gemini-3-flash-preview"
    temperature = 1.5
    usable = questions[questions["usable"]].copy()

    print(f"\nRunning contrastive pipeline: {model}, T={temperature}, {n_reps} reps")
    print(f"Total calls needed: {len(usable) * n_reps}")

    per_rep_rhos = []

    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_contrastive_g3f_rep{rep}.json"
        cache = {}
        if cache_file.exists():
            with open(cache_file) as f:
                cache = json.load(f)

        n_cached = len(cache)
        n_called = 0

        for idx, (_, row) in enumerate(usable.iterrows()):
            qid = str(row["id"])
            if qid in cache:
                continue

            labels = row["option_labels"]
            option_format = " ".join(f"{lab}=XX%" for lab in labels)
            prompt = CONTRASTIVE_PROMPT.format(
                question=row["question_text"],
                options=row["options_text"],
                option_format=option_format,
            )
            raw = call_llm(prompt, model=model, temperature=temperature, max_tokens=512)
            if raw:
                p_inc = parse_contrastive(raw, row["correct_label"], labels)
                cache[qid] = {"raw": raw, "predicted": p_inc}
                n_called += 1
            else:
                continue

            if n_called % 20 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
                print(f"  rep{rep}: {n_called} called, {n_cached} cached ({n_cached + n_called}/{len(usable)})")

            time.sleep(0.2)

        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

        # Per-rep correlation
        preds = []
        for _, row in usable.iterrows():
            qid = str(row["id"])
            if qid in cache:
                preds.append({"qid": qid, "p_correct": row["p_correct"],
                              "p_inc": cache[qid]["predicted"]})
        df_rep = pd.DataFrame(preds).dropna()
        if len(df_rep) >= 5:
            rho, p = stats.spearmanr(df_rep["p_inc"], 1 - df_rep["p_correct"])
            per_rep_rhos.append(rho)
            print(f"  rep{rep}: ρ={rho:.3f} (p={p:.2e}), n={len(df_rep)}, cached={n_cached}, called={n_called}")

    # Averaged predictions
    all_preds = []
    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_contrastive_g3f_rep{rep}.json"
        if not cache_file.exists():
            continue
        with open(cache_file) as f:
            cache = json.load(f)
        for _, row in usable.iterrows():
            qid = str(row["id"])
            if qid in cache:
                all_preds.append({"qid": qid, "p_correct": row["p_correct"],
                                  "p_inc": cache[qid]["predicted"], "rep": rep,
                                  "difficulty": row["difficulty"],
                                  "n_responses": row["n_responses"]})

    pdf = pd.DataFrame(all_preds)
    avg = pdf.groupby("qid").agg(
        mean_p_inc=("p_inc", "mean"),
        p_correct=("p_correct", "first"),
        difficulty=("difficulty", "first"),
        n_responses=("n_responses", "first"),
    ).dropna()

    # Correlate p_inc (predicted difficulty) with 1-p_correct (actual difficulty)
    avg_rho, avg_p = stats.spearmanr(avg["mean_p_inc"], 1 - avg["p_correct"])
    ci_lo, ci_hi = bootstrap_ci(avg["mean_p_inc"].values, (1 - avg["p_correct"]).values)
    rho_author, _ = stats.spearmanr(avg["mean_p_inc"], avg["difficulty"])

    print(f"\n{'=' * 50}")
    print(f"RESULT (contrastive 3-rep, {model}, T={temperature})")
    print(f"{'=' * 50}")
    print(f"Per-rep ρ: {[f'{r:.3f}' for r in per_rep_rhos]}")
    print(f"Averaged ρ = {avg_rho:.3f} (p = {avg_p:.2e}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"ρ vs author difficulty = {rho_author:.3f}")
    print(f"n = {len(avg)} items, {int(avg['n_responses'].sum()):,} student responses")

    result = {
        "dataset": "DBE-KT22", "model": model, "mode": "contrastive_3rep",
        "temperature": temperature, "n_reps": n_reps,
        "n_items": len(avg), "n_student_responses": int(avg["n_responses"].sum()),
        "per_rep_rhos": [round(r, 3) for r in per_rep_rhos],
        "averaged_rho": round(avg_rho, 3), "p_value": float(f"{avg_p:.4e}"),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "rho_author_difficulty": round(rho_author, 3),
    }
    with open(OUTPUT_DIR / "result_contrastive_g3f.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'result_contrastive_g3f.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="direct", choices=["direct", "contrastive"])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    print("=" * 50)
    print(f"DBE-KT22 Validation: {args.mode} mode")
    print("=" * 50)

    questions = load_data()

    if args.mode == "direct":
        run_direct(questions)
    else:
        run_contrastive(questions, n_reps=args.reps)


if __name__ == "__main__":
    main()
