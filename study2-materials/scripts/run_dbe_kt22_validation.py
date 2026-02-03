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


DECOMPOSED_PROMPT = """You are an experienced university database systems instructor.

For this question, estimate what percentage of students AT EACH LEVEL would answer correctly.

Question: {question}

{options}

Reply in this EXACT format (numbers only, no explanation):
struggling: XX%
average: XX%
good: XX%
advanced: XX%"""


DIRECT_CLEAN_PROMPT = """Estimate the difficulty of this university database systems question.

Question: {question}

Options:
{options}

What percentage of undergraduate students would answer this correctly?
Reply with just a number between 0 and 100."""


def parse_decomposed_dbe(text):
    """Parse proficiency-decomposed output and return weighted p_correct."""
    if not text:
        return None
    weights = {"struggling": 0.25, "average": 0.35, "good": 0.25, "advanced": 0.15}
    scores = []
    for level, w in weights.items():
        m = re.search(rf'{level}\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        if m:
            scores.append((float(m.group(1)) / 100.0, w))
    if scores:
        return sum(p * w for p, w in scores) / sum(w for _, w in scores)
    return None


def run_decomposed(questions, n_reps=3):
    """Decomposed by proficiency level, no per-option breakdown. Gemini 3 Flash."""
    model = "gemini-3-flash-preview"
    temperature = 1.5
    usable = questions[questions["usable"]].copy()

    print(f"\nRunning decomposed pipeline: {model}, T={temperature}, {n_reps} reps")
    print(f"Total calls needed: {len(usable) * n_reps}")

    per_rep_rhos = []

    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_decomposed_g3f_rep{rep}.json"
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

            prompt = DECOMPOSED_PROMPT.format(
                question=row["question_text"],
                options=row["options_text"],
            )
            raw = call_llm(prompt, model=model, temperature=temperature, max_tokens=512)
            if raw:
                p_correct = parse_decomposed_dbe(raw)
                cache[qid] = {"raw": raw, "predicted_p_correct": p_correct}
                n_called += 1

            if n_called % 20 == 0 and n_called > 0:
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
            if qid in cache and cache[qid].get("predicted_p_correct") is not None:
                preds.append({"qid": qid, "p_correct": row["p_correct"],
                              "predicted": cache[qid]["predicted_p_correct"]})
        df_rep = pd.DataFrame(preds).dropna()
        if len(df_rep) >= 5:
            rho, p = stats.spearmanr(df_rep["predicted"], df_rep["p_correct"])
            per_rep_rhos.append(rho)
            print(f"  rep{rep}: ρ={rho:.3f} (p={p:.2e}), n={len(df_rep)}, cached={n_cached}, called={n_called}")

    # Averaged predictions
    all_preds = []
    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_decomposed_g3f_rep{rep}.json"
        if not cache_file.exists():
            continue
        with open(cache_file) as f:
            cache = json.load(f)
        for _, row in usable.iterrows():
            qid = str(row["id"])
            if qid in cache and cache[qid].get("predicted_p_correct") is not None:
                all_preds.append({"qid": qid, "p_correct": row["p_correct"],
                                  "predicted": cache[qid]["predicted_p_correct"], "rep": rep,
                                  "difficulty": row["difficulty"],
                                  "n_responses": row["n_responses"]})

    pdf = pd.DataFrame(all_preds)
    avg = pdf.groupby("qid").agg(
        mean_predicted=("predicted", "mean"),
        p_correct=("p_correct", "first"),
        difficulty=("difficulty", "first"),
        n_responses=("n_responses", "first"),
    ).dropna()

    avg_rho, avg_p = stats.spearmanr(avg["mean_predicted"], avg["p_correct"])
    ci_lo, ci_hi = bootstrap_ci(avg["mean_predicted"].values, avg["p_correct"].values)
    rho_author, _ = stats.spearmanr(avg["mean_predicted"], -avg["difficulty"])

    print(f"\n{'=' * 50}")
    print(f"RESULT (decomposed 3-rep, {model}, T={temperature})")
    print(f"{'=' * 50}")
    print(f"Per-rep ρ: {[f'{r:.3f}' for r in per_rep_rhos]}")
    print(f"Averaged ρ = {avg_rho:.3f} (p = {avg_p:.2e}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"ρ vs author difficulty = {rho_author:.3f}")
    print(f"n = {len(avg)} items, {int(avg['n_responses'].sum()):,} student responses")

    result = {
        "dataset": "DBE-KT22", "model": model, "mode": "decomposed_3rep",
        "temperature": temperature, "n_reps": n_reps,
        "n_items": len(avg), "n_student_responses": int(avg["n_responses"].sum()),
        "per_rep_rhos": [round(r, 3) for r in per_rep_rhos],
        "averaged_rho": round(avg_rho, 3), "p_value": float(f"{avg_p:.4e}"),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "rho_author_difficulty": round(rho_author, 3),
    }
    with open(OUTPUT_DIR / "result_decomposed_g3f.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'result_decomposed_g3f.json'}")


def run_direct_clean(questions):
    """Direct estimation WITHOUT correct answer leakage."""
    model = "gemini-2.5-flash"
    usable = questions[questions["usable"]].copy()

    cache_file = OUTPUT_DIR / "predictions_direct_clean.json"
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached predictions")

    for idx, (_, row) in enumerate(usable.iterrows()):
        qid = str(row["id"])
        if qid in cache:
            continue
        prompt = DIRECT_CLEAN_PROMPT.format(
            question=row["question_text"],
            options=row["options_text"],
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

    print(f"\nRESULT (direct_clean, {model})")
    print(f"ρ = {rho:.3f} (p = {p:.4f}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    result = {
        "dataset": "DBE-KT22", "model": model, "mode": "direct_clean",
        "n_items": len(valid), "n_student_responses": int(valid["n_responses"].sum()),
        "rho": round(rho, 3), "p_value": round(p, 4),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
    }
    with open(OUTPUT_DIR / "result_direct_clean.json", 'w') as f:
        json.dump(result, f, indent=2)


def run_synthetic_mcq(questions, n_students=10, temp=1.5):
    """Synthetic student simulation for MCQ items.
    Stage 1: Generate university student personas with misconceptions
    Stage 2: Each student picks an option for each question
    Stage 3: Aggregate % choosing correct = predicted difficulty
    """
    model = "gemini-3-flash-preview"
    usable = questions[questions["usable"]].copy()

    config_dir = OUTPUT_DIR / "synthetic_mcq"
    config_dir.mkdir(exist_ok=True)

    # Stage 1: Generate personas
    persona_path = config_dir / "personas.txt"
    if persona_path.exists():
        persona_text = persona_path.read_text()
    else:
        prompt = f"""Generate {n_students} diverse undergraduate student profiles for a university Database Systems course.

Distribution reflecting a typical class:
- 3 students: Struggling (rarely attend, cram before exams, confuse basic concepts)
- 3 students: Average (attend most classes, understand basics, struggle with multi-step reasoning)
- 2 students: Good (consistent study, solid conceptual understanding, occasional careless errors)
- 2 students: Advanced (deep understanding, can apply concepts to novel problems)

For each student, include:
- Name and study habits
- Proficiency level
- SPECIFIC DATABASE MISCONCEPTIONS they hold (e.g., "thinks JOIN always requires ON clause", "confuses HAVING with WHERE", "believes NULL = NULL is true", "thinks normalization always improves performance")
- Test-taking behavior

Format:
STUDENT 1: [Name] | Level: [Struggling/Average/Good/Advanced] | Misconceptions: [2-3 specific DB misconceptions] | Behavior: [test behavior]
...
STUDENT {n_students}: ..."""
        persona_text = call_llm(prompt, model=model, temperature=1.0, max_tokens=2048)
        if persona_text:
            persona_path.write_text(persona_text)

    personas = [l.strip() for l in persona_text.split('\n') if re.match(r'\**STUDENT\s+\d', l.strip())]
    print(f"  {len(personas)} student personas generated", flush=True)

    if not personas:
        print("  ERROR: No personas generated")
        return

    # Stage 2: Each item — batch all students in one call
    n_api = 0
    n_parse_fail = 0
    item_results = {}

    for idx, (_, row) in enumerate(usable.iterrows()):
        qid = str(row["id"])
        raw_path = config_dir / f"q{qid}_batch.txt"

        if raw_path.exists():
            text = raw_path.read_text()
        else:
            persona_block = "\n".join(personas)
            correct = row["correct_label"]
            labels = row["option_labels"]
            prompt = f"""STUDENT PROFILES (with known database misconceptions):
{persona_block}

EXAM QUESTION:
{row['question_text']}

{row['options_text']}

For each student, consider their specific misconceptions and knowledge gaps. Which option would they choose? Think about:
- Would their misconceptions lead them to a specific wrong answer?
- Would they eliminate some options but guess among the rest?
- Would they recognize the correct answer despite surface-level confusion?

Format:
STUDENT 1: [reasoning based on their profile] → ANSWER: [A/B/C/D]
...
STUDENT {len(personas)}: [reasoning] → ANSWER: [A/B/C/D]"""
            text = call_llm(prompt, model=model, temperature=temp, max_tokens=2048)
            if text:
                raw_path.write_text(text)
            n_api += 1
            time.sleep(0.2)

        # Parse answers
        correct = row["correct_label"]
        answers = re.findall(r'ANSWER:\s*([A-D])', text or "")
        if answers:
            n_correct = sum(1 for a in answers if a == correct)
            item_results[qid] = {
                "predicted_p": n_correct / len(answers),
                "n_students": len(answers),
                "n_correct": n_correct,
            }
        else:
            n_parse_fail += 1

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(usable)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

    # Evaluate
    preds, actuals = [], []
    for _, row in usable.iterrows():
        qid = str(row["id"])
        if qid in item_results:
            preds.append(item_results[qid]["predicted_p"])
            actuals.append(row["p_correct"])

    if len(preds) >= 10:
        preds, actuals = np.array(preds), np.array(actuals)
        rho, p_val = stats.spearmanr(preds, actuals)
        mae = float(np.mean(np.abs(preds - actuals)))
        bias = float(np.mean(preds - actuals))
        ci_lo, ci_hi = bootstrap_ci(preds, actuals)

        print(f"\n{'=' * 50}")
        print(f"RESULT (synthetic_mcq, {model}, T={temp})")
        print(f"{'=' * 50}")
        print(f"ρ = {rho:.3f} (p = {p_val:.2e}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"MAE = {mae:.3f}, bias = {bias:+.3f}")
        print(f"n = {len(preds)} items, {n_api} api calls, {n_parse_fail} parse fails")

        result = {
            "dataset": "DBE-KT22", "model": model, "mode": "synthetic_mcq",
            "temperature": temp, "n_students": len(personas),
            "n_items": len(preds), "n_student_responses": int(usable["n_responses"].sum()),
            "rho": round(rho, 3), "p_value": float(f"{p_val:.4e}"),
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "mae": round(mae, 3), "bias": round(bias, 3),
        }
        with open(OUTPUT_DIR / "result_synthetic_mcq.json", 'w') as f:
            json.dump(result, f, indent=2)

        # Save per-item predictions
        with open(config_dir / "predictions.json", 'w') as f:
            json.dump(item_results, f, indent=2)
        print(f"Saved to {OUTPUT_DIR / 'result_synthetic_mcq.json'}")


BUGGY_RULES_PROMPT = """You are an expert in computer science education and systematic student errors (Brown & Burton, 1978).

For the following university-level database systems question, analyze the cognitive demands:

Question: {question}

Options:
{options}

Step 1: List the specific knowledge and reasoning steps a student must execute correctly to identify the right answer.
Step 2: For each step, identify any known "buggy rules" — systematic conceptual errors students commonly make (e.g., confusing WHERE with HAVING, thinking NULL = NULL is true, misunderstanding JOIN semantics, confusing normalization forms).
Step 3: Consider the target student population (undergraduate database systems course, multiple-choice format).
Step 4: Taking into account ALL of the above analysis holistically — the number of steps, the severity and commonality of bugs, and the MCQ format where guessing is possible — estimate what proportion of students would select the correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def parse_proportion_last(text):
    """Parse last decimal 0.XX from text (matching SmartPaper parser)."""
    if not text:
        return None
    decs = re.findall(r'(?<!\d)0\.\d+|1\.0(?:0*)?(?!\d)', text)
    if decs:
        v = float(decs[-1])
        if 0 <= v <= 1:
            return v
    return None


def run_buggy_rules(questions, n_reps=3):
    """Buggy rules holistic estimation, 3-rep, Gemini 3 Flash."""
    model = "gemini-3-flash-preview"
    temperature = 1.0
    usable = questions[questions["usable"]].copy()

    print(f"\nRunning buggy_rules pipeline: {model}, T={temperature}, {n_reps} reps")
    print(f"Total calls needed: {len(usable) * n_reps}")

    per_rep_rhos = []

    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_buggy_rules_rep{rep}.json"
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

            prompt = BUGGY_RULES_PROMPT.format(
                question=row["question_text"],
                options=row["options_text"],
            )
            raw = call_llm(prompt, model=model, temperature=temperature, max_tokens=1024)
            if raw:
                p = parse_proportion_last(raw)
                cache[qid] = {"raw": raw, "predicted_p": p}
                n_called += 1
            else:
                continue

            if n_called % 20 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
                print(f"  rep{rep}: {n_called} called, {n_cached} cached ({n_cached + n_called}/{len(usable)})")

            time.sleep(0.1)

        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

        # Per-rep correlation
        preds = []
        for _, row in usable.iterrows():
            qid = str(row["id"])
            if qid in cache and cache[qid].get("predicted_p") is not None:
                preds.append({"qid": qid, "p_correct": row["p_correct"],
                              "predicted": cache[qid]["predicted_p"]})
        df_rep = pd.DataFrame(preds).dropna()
        if len(df_rep) >= 5:
            rho, p = stats.spearmanr(df_rep["predicted"], df_rep["p_correct"])
            per_rep_rhos.append(rho)
            print(f"  rep{rep}: ρ={rho:.3f} (p={p:.2e}), n={len(df_rep)}, cached={n_cached}, called={n_called}")

    # Averaged predictions
    all_preds = []
    for rep in range(n_reps):
        cache_file = OUTPUT_DIR / f"predictions_buggy_rules_rep{rep}.json"
        if not cache_file.exists():
            continue
        with open(cache_file) as f:
            cache = json.load(f)
        for _, row in usable.iterrows():
            qid = str(row["id"])
            if qid in cache and cache[qid].get("predicted_p") is not None:
                all_preds.append({"qid": qid, "p_correct": row["p_correct"],
                                  "predicted": cache[qid]["predicted_p"], "rep": rep,
                                  "difficulty": row["difficulty"],
                                  "n_responses": row["n_responses"]})

    pdf = pd.DataFrame(all_preds)
    avg = pdf.groupby("qid").agg(
        mean_predicted=("predicted", "mean"),
        p_correct=("p_correct", "first"),
        difficulty=("difficulty", "first"),
        n_responses=("n_responses", "first"),
    ).dropna()

    avg_rho, avg_p = stats.spearmanr(avg["mean_predicted"], avg["p_correct"])
    ci_lo, ci_hi = bootstrap_ci(avg["mean_predicted"].values, avg["p_correct"].values)
    rho_author, _ = stats.spearmanr(avg["mean_predicted"], -avg["difficulty"])
    mae = float(np.mean(np.abs(avg["mean_predicted"].values - avg["p_correct"].values)))
    bias = float(np.mean(avg["mean_predicted"].values - avg["p_correct"].values))

    print(f"\n{'=' * 50}")
    print(f"RESULT (buggy_rules 3-rep, {model}, T={temperature})")
    print(f"{'=' * 50}")
    print(f"Per-rep ρ: {[f'{r:.3f}' for r in per_rep_rhos]}")
    print(f"Averaged ρ = {avg_rho:.3f} (p = {avg_p:.2e}), 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"ρ vs author difficulty = {rho_author:.3f}")
    print(f"MAE = {mae:.3f}, bias = {bias:+.3f}")
    print(f"n = {len(avg)} items, {int(avg['n_responses'].sum()):,} student responses")

    result = {
        "dataset": "DBE-KT22", "model": model, "mode": "buggy_rules_3rep",
        "temperature": temperature, "n_reps": n_reps,
        "n_items": len(avg), "n_student_responses": int(avg["n_responses"].sum()),
        "per_rep_rhos": [round(r, 3) for r in per_rep_rhos],
        "averaged_rho": round(avg_rho, 3), "p_value": float(f"{avg_p:.4e}"),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "rho_author_difficulty": round(rho_author, 3),
        "mae": round(mae, 3), "bias": round(bias, 3),
    }
    with open(OUTPUT_DIR / "result_buggy_rules.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'result_buggy_rules.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="direct",
                        choices=["direct", "contrastive", "decomposed", "direct_clean", "synthetic_mcq", "buggy_rules"])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    print("=" * 50)
    print(f"DBE-KT22 Validation: {args.mode} mode")
    print("=" * 50)

    questions = load_data()

    if args.mode == "direct":
        run_direct(questions)
    elif args.mode == "contrastive":
        run_contrastive(questions, n_reps=args.reps)
    elif args.mode == "decomposed":
        run_decomposed(questions, n_reps=args.reps)
    elif args.mode == "direct_clean":
        run_direct_clean(questions)
    elif args.mode == "synthetic_mcq":
        run_synthetic_mcq(questions, temp=1.5)
    elif args.mode == "buggy_rules":
        run_buggy_rules(questions, n_reps=args.reps)


if __name__ == "__main__":
    main()
