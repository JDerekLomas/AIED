#!/usr/bin/env python3
"""
Response Surface Method experiment for SmartPaper open-ended item difficulty estimation.

Box-Behnken design over 5 factors to find the LLM configuration that best
recovers classical difficulty from simulated student responses to open-ended items.

Adapted from run_rsm_experiment.py (Eedi MCQ version).

Factors:
  1. temperature: 0.3, 0.9, 1.5
  2. students_per_call: 1, 5, 20
  3. prompt_style: individual_roleplay, classroom_batch, teacher_prediction
  4. rubric_hint: hidden, partial, full
  5. model: gemini-2.5-flash, gemini-3-flash

Usage:
    python3 scripts/run_smartpaper_rsm.py --phase sweep
    python3 scripts/run_smartpaper_rsm.py --phase analyze
    python3 scripts/run_smartpaper_rsm.py --phase confirm
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

RANDOM_SEED = 42
OUTPUT_DIR = Path("pilot/smartpaper_rsm")
STUDENTS_PER_ITEM = 20
PROBE_ITEMS = 20

# --- Factor definitions ---
FACTORS = {
    "temperature":       {"levels": [0.3, 0.9, 1.5], "type": "continuous"},
    "students_per_call": {"levels": [1, 5, 20], "type": "continuous"},
    "prompt_style":      {"levels": ["individual_roleplay", "classroom_batch", "teacher_prediction"], "type": "categorical"},
    "rubric_hint":       {"levels": ["hidden", "partial", "full"], "type": "categorical"},
    "model":             {"levels": ["gemini-2.5-flash", "gemini-3-flash", "gemini-2.5-flash"], "type": "categorical"},
}

MODEL_IDS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash": "gemini-3-flash-preview",
}

# Indian govt school proficiency distribution (skewed lower given mean difficulty ~0.29)
PROFICIENCY_DISTRIBUTION = [
    ("struggling", 0.30),
    ("basic", 0.35),
    ("competent", 0.25),
    ("advanced", 0.10),
]

LEVEL_DESCRIPTIONS = {
    "struggling": (
        "You are a student in an Indian government school who struggles significantly "
        "with academics. You are in Class {grade} and find most topics confusing. "
        "You often cannot understand what the question is asking. Your written English "
        "is very limited and you mostly think in Hindi. You frequently leave answers "
        "blank or write something only vaguely related to the question."
    ),
    "basic": (
        "You are a student in an Indian government school with basic academic ability. "
        "You are in Class {grade} and can handle simple recall questions but struggle "
        "with anything requiring explanation or multiple steps. Your English writing "
        "is basic — you can write short phrases but make many grammatical errors. "
        "You sometimes misunderstand questions."
    ),
    "competent": (
        "You are a student in an Indian government school who is fairly good academically. "
        "You are in Class {grade} and usually understand the material. You can write "
        "short answers correctly for most questions but sometimes miss key details "
        "in longer responses. Your English is functional."
    ),
    "advanced": (
        "You are a student in an Indian government school who excels academically. "
        "You are in Class {grade} and understand concepts deeply. You write clear, "
        "correct answers and rarely make errors. You can handle questions that require "
        "reasoning or application of concepts."
    ),
}


# --- Data loading ---

def load_data():
    """Load SmartPaper items with classical statistics."""
    items = json.loads(Path("data/smartpaper/item_statistics.json").read_text())
    items = [i for i in items if i.get("question_text")]
    return items


def select_probe_items(items, n=PROBE_ITEMS):
    """Select n items stratified by difficulty quintile."""
    np.random.seed(RANDOM_SEED)
    sorted_items = sorted(items, key=lambda x: x["classical_difficulty"])
    per_q = n // 5
    selected = []
    quintile_size = len(sorted_items) // 5
    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else len(sorted_items)
        pool = sorted_items[start:end]
        indices = np.random.choice(len(pool), size=min(per_q, len(pool)), replace=False)
        selected.extend([pool[i] for i in indices])
    return selected


# --- Box-Behnken Design ---

def generate_box_behnken_5():
    n_factors = 5
    runs = []
    for i, j in combinations(range(n_factors), 2):
        for vi in [-1, 1]:
            for vj in [-1, 1]:
                row = [0] * n_factors
                row[i] = vi
                row[j] = vj
                runs.append(row)
    for _ in range(6):
        runs.append([0] * n_factors)
    return np.array(runs)


def generate_design_matrix():
    coded = generate_box_behnken_5()
    factor_names = list(FACTORS.keys())
    rows = []
    for run_idx, coded_row in enumerate(coded):
        config = {"config_id": run_idx}
        for fi, fname in enumerate(factor_names):
            level_idx = int(coded_row[fi]) + 1
            config[fname] = FACTORS[fname]["levels"][level_idx]
            config[f"{fname}_coded"] = int(coded_row[fi])
        rows.append(config)
    return pd.DataFrame(rows)


# --- Item formatting ---

def format_item_text(item, rubric_hint="hidden"):
    """Format open-ended item with variable rubric hint level."""
    lines = [
        f"Subject: {item['subject']} (Class {item['grade']})",
        f"Question: {item['question_text']}",
    ]
    if rubric_hint == "partial":
        lines.append(f"Skill tested: {item['skill']}")
        lines.append(f"Maximum marks: {item['max_score']}")
    elif rubric_hint == "full":
        lines.append(f"Skill tested: {item['skill']}")
        lines.append(f"Maximum marks: {item['max_score']}")
        lines.append(f"Scoring rubric: {item['rubric']}")
    return "\n".join(lines)


def item_key(item):
    return f"{item['assessment']}_q{item['question_number']}"


# --- Prompt builders ---

def build_individual_roleplay_prompt(item, level, rubric_hint):
    item_text = format_item_text(item, rubric_hint)
    level_desc = LEVEL_DESCRIPTIONS[level].format(grade=item["grade"])
    return f"""{level_desc}

Answer this question in writing, as you would on a paper test. Write your answer naturally — it can be a word, phrase, or short sentence. If you don't know, write what you think or leave it blank.

{item_text}

YOUR ANSWER:"""


def build_classroom_batch_prompt(item, n_students, rubric_hint):
    item_text = format_item_text(item, rubric_hint)

    roster = []
    for level, weight in PROFICIENCY_DISTRIBUTION:
        count = max(1, round(n_students * weight))
        roster.extend([level] * count)
    roster = roster[:n_students]
    np.random.shuffle(roster)

    roster_str = ", ".join(f"Student {i+1} ({level})" for i, level in enumerate(roster))

    return f"""Simulate a classroom of {n_students} Indian government school students (Class {item['grade']}) answering this question on a written test. Student answers are handwritten and may contain spelling errors.

The students have these ability levels: {roster_str}

{item_text}

For each student, write their answer as they would write it on paper. Some students may leave it blank or write something wrong.

Format:
{chr(10).join(f'Student {i+1} ({roster[i]}): [their written answer]' for i in range(min(3, n_students)))}""", roster


def build_teacher_prediction_prompt(item, rubric_hint):
    item_text = format_item_text(item, rubric_hint)
    return f"""You are an experienced teacher in an Indian government school teaching Class {item['grade']} {item['subject']}.

For this open-ended question, predict what proportion of your students at each ability level would score full marks.

{item_text}

Scoring rubric: {item['rubric']}

Respond in this exact format:
struggling: XX% full marks
basic: XX% full marks
competent: XX% full marks
advanced: XX% full marks"""


# --- Response scoring ---

def score_open_response(student_answer, item, client=None, model_key="gemini-2.5-flash"):
    """Score an open-ended response against the rubric using LLM-as-judge.

    Returns 1 (correct/adequate) or 0 (incorrect/inadequate).
    Uses a fast LLM call to judge whether the response meets the rubric.
    Falls back to keyword matching if no client provided.
    """
    if not student_answer or student_answer.strip().lower() in ["", "blank", "[blank]", "i don't know", "idk", "-", "..."]:
        return 0

    if client is None:
        # Fallback: simple keyword match
        rubric = item.get("rubric", "").lower()
        answer = student_answer.lower().strip()
        accept_matches = re.findall(r"'([^']+)'", rubric)
        accept_matches += re.findall(r'"([^"]+)"', rubric)
        key_terms = [t.strip().lower() for t in accept_matches if len(t.strip()) > 1]
        return 1 if any(t in answer for t in key_terms) else 0

    # LLM-as-judge scoring
    score_prompt = f"""Score this student answer against the rubric. Reply with ONLY "1" (meets criteria) or "0" (does not meet criteria).

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}
Student answer: {student_answer}

Score (0 or 1):"""

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=MODEL_IDS[model_key],
            contents=score_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip()
        if "1" in text[:3]:
            return 1
        return 0
    except Exception:
        return 0


# --- Response parsers ---

def parse_single_answer(text):
    """Extract written answer from individual response."""
    # Look for text after "YOUR ANSWER:" or just take the last line
    match = re.search(r'(?:YOUR ANSWER|ANSWER)\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def parse_classroom_batch(text, n_students):
    """Extract written answers from classroom batch response."""
    answers = []
    for i in range(1, n_students + 1):
        pattern = rf'Student\s*{i}\s*\([^)]*\)\s*:\s*(.+?)(?=\nStudent\s*\d|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answers.append(match.group(1).strip())
        else:
            answers.append("")
    return answers


def parse_teacher_prediction(text):
    """Parse teacher prediction into proportion correct per level."""
    results = {}
    for level, _ in PROFICIENCY_DISTRIBUTION:
        pattern = rf'{level}\s*:\s*(\d+)\s*%'
        match = re.search(pattern, text, re.IGNORECASE)
        results[level] = int(match.group(1)) / 100.0 if match else 0.5
    return results


# --- API call ---

def make_api_call(client, model_key, prompt, temperature=0.7, max_tokens=1024):
    from google.genai import types
    model_id = MODEL_IDS[model_key]
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


# --- Config runner ---

def run_single_config(config, probe_items, client):
    """Run a single RSM config on all probe items. Returns prop_correct per item."""
    config_id = config["config_id"]
    temperature = config["temperature"]
    students_per_call = config["students_per_call"]
    prompt_style = config["prompt_style"]
    rubric_hint = config["rubric_hint"]
    model = config["model"]

    raw_dir = OUTPUT_DIR / "raw_responses" / f"config_{config_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    item_scores = {}  # item_key -> list of 0/1

    for item in probe_items:
        ik = item_key(item)
        item_scores[ik] = []

        if prompt_style == "teacher_prediction":
            raw_path = raw_dir / f"{ik}_teacher.txt"
            if raw_path.exists():
                text = raw_path.read_text()
            else:
                prompt = build_teacher_prediction_prompt(item, rubric_hint)
                try:
                    text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=512)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"    ERROR config={config_id} {ik}: {e}")
                    time.sleep(1)
                    continue
                time.sleep(0.1)

            preds = parse_teacher_prediction(text)
            for level, weight in PROFICIENCY_DISTRIBUTION:
                n_students = max(1, round(STUDENTS_PER_ITEM * weight))
                p_correct = preds.get(level, 0.5)
                for _ in range(n_students):
                    item_scores[ik].append(1 if np.random.random() < p_correct else 0)

        elif prompt_style == "classroom_batch":
            n_calls = max(1, STUDENTS_PER_ITEM // students_per_call)
            for batch_idx in range(n_calls):
                raw_path = raw_dir / f"{ik}_batch{batch_idx}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    prompt, roster = build_classroom_batch_prompt(item, students_per_call, rubric_hint)
                    try:
                        text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=2048)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"    ERROR config={config_id} {ik} batch={batch_idx}: {e}")
                        time.sleep(1)
                        continue
                    time.sleep(0.1)

                answers = parse_classroom_batch(text, students_per_call)
                for ans in answers:
                    item_scores[ik].append(score_open_response(ans, item))

        elif prompt_style == "individual_roleplay":
            student_idx = 0
            for level, weight in PROFICIENCY_DISTRIBUTION:
                n_this_level = max(1, round(STUDENTS_PER_ITEM * weight))
                for si in range(n_this_level):
                    if student_idx >= STUDENTS_PER_ITEM:
                        break
                    raw_path = raw_dir / f"{ik}_s{student_idx}.txt"
                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = build_individual_roleplay_prompt(item, level, rubric_hint)
                        try:
                            text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=300)
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"    ERROR config={config_id} {ik} s={student_idx}: {e}")
                            time.sleep(0.5)
                            student_idx += 1
                            continue
                        time.sleep(0.1)

                    ans = parse_single_answer(text)
                    item_scores[ik].append(score_open_response(ans, item))
                    student_idx += 1

    # Compute mean score proportion per item
    mean_scores = {}
    for ik, scores in item_scores.items():
        if scores:
            mean_scores[ik] = sum(scores) / len(scores)

    return mean_scores


def evaluate_config(mean_scores, probe_items):
    """Compute correlation between simulated mean score and actual classical difficulty."""
    sim_vals = []
    actual_vals = []

    for item in probe_items:
        ik = item_key(item)
        if ik in mean_scores:
            sim_vals.append(mean_scores[ik])
            actual_vals.append(item["classical_difficulty"])

    if len(sim_vals) < 5:
        return {"spearman_rho": np.nan, "spearman_p": np.nan, "pearson_r": np.nan,
                "pearson_p": np.nan, "n_items": len(sim_vals)}

    rho, p_rho = stats.spearmanr(sim_vals, actual_vals)
    r, p_r = stats.pearsonr(sim_vals, actual_vals)
    return {"spearman_rho": rho, "spearman_p": p_rho, "pearson_r": r,
            "pearson_p": p_r, "n_items": len(sim_vals)}


# --- Phase: Sweep ---

def run_sweep():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_data()
    print(f"Loaded {len(items)} items")

    probe = select_probe_items(items)
    print(f"Selected {len(probe)} probe items")
    # Save probe items
    with open(OUTPUT_DIR / "probe_items.json", "w") as f:
        json.dump(probe, f, indent=2)

    design = generate_design_matrix()
    design.to_csv(OUTPUT_DIR / "design_matrix.csv", index=False)
    print(f"Design matrix: {len(design)} configurations")

    # Check for existing results
    results_path = OUTPUT_DIR / "results.csv"
    completed = set()
    all_results = []
    if results_path.exists():
        existing = pd.read_csv(results_path)
        completed = set(existing["config_id"].values)
        all_results = existing.to_dict("records")
        print(f"Resuming: {len(completed)} configs already done")

    for _, config in design.iterrows():
        cid = config["config_id"]
        if cid in completed:
            continue

        print(f"\n--- Config {cid}/{len(design)-1}: temp={config['temperature']}, "
              f"spc={config['students_per_call']}, style={config['prompt_style']}, "
              f"hint={config['rubric_hint']}, model={config['model']} ---")

        mean_scores = run_single_config(config.to_dict(), probe, client)
        metrics = evaluate_config(mean_scores, probe)

        result = config.to_dict()
        result.update(metrics)
        all_results.append(result)

        pd.DataFrame(all_results).to_csv(results_path, index=False)

        print(f"  -> Spearman rho={metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f}), "
              f"Pearson r={metrics['pearson_r']:.3f}, n={metrics['n_items']}")

    print(f"\nSweep complete. Results saved to {results_path}")


# --- Phase: Analyze ---

def run_analyze():
    results_path = OUTPUT_DIR / "results.csv"
    if not results_path.exists():
        print("No results.csv found. Run --phase sweep first.")
        return

    results = pd.read_csv(results_path)
    print(f"Loaded {len(results)} config results")
    print(f"\nSpearman rho summary:")
    print(f"  Mean: {results['spearman_rho'].mean():.3f}")
    print(f"  Std:  {results['spearman_rho'].std():.3f}")
    print(f"  Min:  {results['spearman_rho'].min():.3f}")
    print(f"  Max:  {results['spearman_rho'].max():.3f}")

    print(f"\nTop 5 configs by Spearman rho:")
    top5 = results.nlargest(5, "spearman_rho")
    for _, row in top5.iterrows():
        print(f"  Config {int(row['config_id'])}: rho={row['spearman_rho']:.3f} "
              f"| temp={row['temperature']}, spc={int(row['students_per_call'])}, "
              f"style={row['prompt_style']}, hint={row['rubric_hint']}, model={row['model']}")

    print(f"\nBottom 5 configs by Spearman rho:")
    bot5 = results.nsmallest(5, "spearman_rho")
    for _, row in bot5.iterrows():
        print(f"  Config {int(row['config_id'])}: rho={row['spearman_rho']:.3f} "
              f"| temp={row['temperature']}, spc={int(row['students_per_call'])}, "
              f"style={row['prompt_style']}, hint={row['rubric_hint']}, model={row['model']}")

    # Main effects
    print(f"\n--- Main Effects ---")
    for factor in ["temperature", "students_per_call", "prompt_style", "rubric_hint", "model"]:
        print(f"\n  {factor}:")
        for val, grp in results.groupby(factor):
            rhos = grp["spearman_rho"].dropna()
            print(f"    {val}: mean rho={rhos.mean():.3f} (n={len(rhos)}, std={rhos.std():.3f})")

    # Fit response surface
    coded_cols = [c for c in results.columns if c.endswith("_coded")]
    if len(coded_cols) == 5:
        _fit_response_surface(results, coded_cols)


def _fit_response_surface(results, coded_cols):
    y = results["spearman_rho"].values
    valid = ~np.isnan(y)
    y = y[valid]
    X_coded = results[coded_cols].values[valid]

    n = len(y)
    k = X_coded.shape[1]
    cols = ["intercept"]
    X_parts = [np.ones((n, 1))]

    for i in range(k):
        cols.append(coded_cols[i].replace("_coded", "_lin"))
        X_parts.append(X_coded[:, i:i+1])
    for i in range(k):
        cols.append(coded_cols[i].replace("_coded", "_quad"))
        X_parts.append((X_coded[:, i:i+1]) ** 2)
    for i, j in combinations(range(k), 2):
        cols.append(f"{coded_cols[i]}x{coded_cols[j]}".replace("_coded", ""))
        X_parts.append((X_coded[:, i] * X_coded[:, j]).reshape(-1, 1))

    X = np.hstack(X_parts)

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n--- Response Surface Model ---")
        print(f"  R² = {r_squared:.3f}")
        print(f"  Top coefficients (by magnitude):")
        coefs = sorted(zip(cols, beta), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs[:10]:
            print(f"    {name}: {coef:+.4f}")

        model = {"r_squared": r_squared, "coefficients": dict(zip(cols, beta.tolist()))}
        (OUTPUT_DIR / "surface_model.json").write_text(json.dumps(model, indent=2))
    except Exception as e:
        print(f"  Could not fit model: {e}")


# --- Phase: Confirm ---

def run_confirm():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    results_path = OUTPUT_DIR / "results.csv"
    if not results_path.exists():
        print("No results.csv found. Run --phase sweep first.")
        return

    results = pd.read_csv(results_path)
    best = results.loc[results["spearman_rho"].idxmax()]
    print(f"Best config: id={int(best['config_id'])}, rho={best['spearman_rho']:.3f}")

    config = {
        "config_id": "confirm",
        "temperature": best["temperature"],
        "students_per_call": int(best["students_per_call"]),
        "prompt_style": best["prompt_style"],
        "rubric_hint": best["rubric_hint"],
        "model": best["model"],
    }
    print(f"Config: {config}")

    items = load_data()
    print(f"Running confirmation on all {len(items)} items")

    global STUDENTS_PER_ITEM
    original_spi = STUDENTS_PER_ITEM
    STUDENTS_PER_ITEM = 50

    mean_scores = run_single_config(config, items, client)
    metrics = evaluate_config(mean_scores, items)

    STUDENTS_PER_ITEM = original_spi

    print(f"\n{'='*60}")
    print(f"CONFIRMATION RESULTS")
    print(f"{'='*60}")
    print(f"Items evaluated: {metrics['n_items']}")
    print(f"Spearman rho: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f})")
    print(f"Pearson r: {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.4f})")

    confirm_result = {**config, **metrics, "students_per_item": 50}
    (OUTPUT_DIR / "confirmation_result.json").write_text(json.dumps(confirm_result, indent=2))
    print(f"\nSaved to {OUTPUT_DIR / 'confirmation_result.json'}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="RSM experiment for SmartPaper difficulty estimation")
    parser.add_argument("--phase", choices=["sweep", "analyze", "confirm"], required=True)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    np.random.seed(RANDOM_SEED)

    if args.phase == "sweep":
        run_sweep()
    elif args.phase == "analyze":
        run_analyze()
    elif args.phase == "confirm":
        run_confirm()


if __name__ == "__main__":
    main()
