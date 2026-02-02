#!/usr/bin/env python3
"""
Response Surface Method experiment for student simulation difficulty estimation.

Box-Behnken design over 5 factors (temperature, students_per_call, prompt_style,
misconception_hint, model) to find the configuration that best recovers IRT b_2pl
from simulated student responses.

Usage:
    python3 scripts/run_rsm_experiment.py --phase sweep
    python3 scripts/run_rsm_experiment.py --phase analyze
    python3 scripts/run_rsm_experiment.py --phase confirm
"""

import argparse
import hashlib
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
OUTPUT_DIR = Path("pilot/rsm_experiment")
STUDENTS_PER_ITEM = 20  # per config
PROBE_ITEMS = 20

# --- Factor definitions ---
FACTORS = {
    "temperature":       {"levels": [0.3, 0.9, 1.5], "type": "continuous"},
    "students_per_call": {"levels": [1, 5, 30], "type": "continuous"},
    "prompt_style":      {"levels": ["individual_roleplay", "classroom_batch", "teacher_prediction"], "type": "categorical"},
    "misconception_hint": {"levels": ["hidden", "partial", "full"], "type": "categorical"},
    "model":             {"levels": ["gemini-2.5-flash", "gemini-3-flash", "gemini-2.5-flash"], "type": "categorical"},
}

MODEL_IDS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash": "gemini-3-flash-preview",
}

PROFICIENCY_DISTRIBUTION = [
    ("below_basic", 0.25),
    ("basic", 0.35),
    ("proficient", 0.25),
    ("advanced", 0.15),
]

LEVEL_DESCRIPTIONS = {
    "below_basic": (
        "You are a UK secondary school student who struggles significantly with maths. "
        "You are in Year 7 or 8 and find most maths topics confusing. You often guess "
        "or pick answers that look familiar. You frequently make basic arithmetic errors "
        "and misremember rules."
    ),
    "basic": (
        "You are a UK secondary school student with basic maths ability. "
        "You can handle straightforward problems but get confused by multi-step questions "
        "or unfamiliar formats. You sometimes apply rules incorrectly, especially with "
        "negatives, fractions, and algebra."
    ),
    "proficient": (
        "You are a UK secondary school student who is good at maths. "
        "You usually get questions right but occasionally make errors on tricky problems. "
        "You understand most concepts but can be caught out by subtle distractors "
        "or unusual question wording."
    ),
    "advanced": (
        "You are a UK secondary school student who excels at maths. "
        "You rarely make errors and can handle complex multi-step problems. "
        "You have strong number sense and algebraic fluency. You only get "
        "questions wrong when they involve genuinely difficult or unusual concepts."
    ),
}


# --- Data loading (from run_difficulty_experiments.py) ---

def load_data():
    """Load and merge item data with IRT parameters."""
    irt = json.loads(Path("results/irt_proper_statistics.json").read_text())
    curated = pd.read_csv("data/eedi/curated_eedi_items.csv")
    eedi = pd.read_csv("data/eedi/eedi_with_student_data.csv")

    answer_cols = ["QuestionId", "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
    answers = eedi[answer_cols].drop_duplicates(subset=["QuestionId"])
    df = curated.merge(answers, on="QuestionId", how="left").drop_duplicates(subset=["QuestionId"])

    irt_items = irt["items"]
    irt_rows = []
    for qid_str, params in irt_items.items():
        row = {"QuestionId": int(qid_str)}
        row.update(params)
        irt_rows.append(row)
    irt_df = pd.DataFrame(irt_rows)
    df = df.merge(irt_df, on="QuestionId", how="inner")
    return df


def select_probe_items(df, n=PROBE_ITEMS):
    """Select n items stratified by b_2pl quintile (n/5 per quintile)."""
    np.random.seed(RANDOM_SEED)
    sorted_df = df.sort_values("b_2pl").reset_index(drop=True)
    per_q = n // 5
    selected = []
    quintile_size = len(sorted_df) // 5
    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else len(sorted_df)
        pool = sorted_df.iloc[start:end]
        chosen = pool.sample(n=min(per_q, len(pool)), random_state=RANDOM_SEED + q)
        selected.append(chosen)
    result = pd.concat(selected).reset_index(drop=True)
    return result


# --- Box-Behnken Design ---

def generate_box_behnken_5():
    """Generate Box-Behnken design for 5 factors. Returns coded levels (-1, 0, +1)."""
    # Box-Behnken: for each pair of factors, run a 2^2 factorial with others at center
    n_factors = 5
    runs = []

    for i, j in combinations(range(n_factors), 2):
        for vi in [-1, 1]:
            for vj in [-1, 1]:
                row = [0] * n_factors
                row[i] = vi
                row[j] = vj
                runs.append(row)

    # Add center points
    for _ in range(6):
        runs.append([0] * n_factors)

    return np.array(runs)


def generate_design_matrix():
    """Generate the full design matrix with decoded factor levels."""
    coded = generate_box_behnken_5()
    factor_names = list(FACTORS.keys())

    rows = []
    for run_idx, coded_row in enumerate(coded):
        config = {"config_id": run_idx}
        for fi, fname in enumerate(factor_names):
            level_idx = int(coded_row[fi]) + 1  # -1->0, 0->1, +1->2
            config[fname] = FACTORS[fname]["levels"][level_idx]
            config[f"{fname}_coded"] = int(coded_row[fi])
        rows.append(config)

    return pd.DataFrame(rows)


# --- Item formatting ---

def format_item_text(row, misconception_hint="hidden"):
    """Format item with variable misconception hint level."""
    lines = [
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ]
    if misconception_hint == "partial":
        if pd.notna(row.get('misconception_name', None)):
            lines.append(f"Common misconception tested: {row['misconception_name']}")
    elif misconception_hint == "full":
        lines.append(f"Correct answer: {row['correct_answer_kaggle']}")
        if pd.notna(row.get('misconception_name', None)):
            lines.append(f"Misconception tested: {row['misconception_name']}")
    return "\n".join(lines)


# --- Prompt builders ---

def build_individual_roleplay_prompt(row, level, misconception_hint):
    """Single student roleplay prompt."""
    item_text = format_item_text(row, misconception_hint)
    level_desc = LEVEL_DESCRIPTIONS[level]
    return f"""{level_desc}

Answer this multiple choice maths question. Just give the letter of your answer (A, B, C, or D).

{item_text}

ANSWER:"""


def build_classroom_batch_prompt(row, n_students, misconception_hint):
    """Batch classroom simulation prompt."""
    item_text = format_item_text(row, misconception_hint)

    # Build student roster matching proficiency distribution
    roster = []
    for level, weight in PROFICIENCY_DISTRIBUTION:
        count = max(1, round(n_students * weight))
        roster.extend([level] * count)
    roster = roster[:n_students]
    np.random.shuffle(roster)

    roster_str = ", ".join(f"Student {i+1} ({level.replace('_', ' ')})" for i, level in enumerate(roster))

    return f"""Simulate a classroom of {n_students} UK secondary students answering this maths question.
The students have these ability levels: {roster_str}

{item_text}

For each student, give their answer (A, B, C, or D). Use this exact format:
{chr(10).join(f'Student {i+1} ({roster[i].replace("_", " ")}): [letter]' for i in range(min(3, n_students)))}
{"...and so on for all " + str(n_students) + " students." if n_students > 3 else ""}""", roster


def build_teacher_prediction_prompt(row, misconception_hint):
    """Teacher prediction prompt (outputs distribution)."""
    item_text = format_item_text(row, misconception_hint)
    return f"""You are an experienced UK maths teacher. For this question, predict the percentage of students at each ability level who would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""


# --- Response parsers ---

def parse_single_answer(text):
    """Extract answer letter from individual response."""
    match = re.search(r'\b([A-Da-d])\b', text[:100])
    return match.group(1).upper() if match else None


def parse_classroom_batch(text, n_students):
    """Extract answers from classroom batch response."""
    answers = []
    for i in range(1, n_students + 1):
        pattern = rf'Student\s*{i}\s*\([^)]*\)\s*:\s*([A-Da-d])'
        match = re.search(pattern, text)
        if match:
            answers.append(match.group(1).upper())
        else:
            # Fallback: look for any letter after "Student N"
            pattern2 = rf'Student\s*{i}[^A-Da-d]*([A-Da-d])'
            match2 = re.search(pattern2, text)
            answers.append(match2.group(1).upper() if match2 else None)
    return answers


def parse_teacher_prediction(text, correct_answer):
    """Parse teacher prediction into simulated student responses.

    Returns list of (level, is_correct) based on predicted distributions.
    """
    results = []
    for level, weight in PROFICIENCY_DISTRIBUTION:
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            correct_idx = ord(correct_answer) - ord('A')
            p_correct = pcts[correct_idx] / total
            # Generate simulated students from this distribution
            n_students = max(1, round(STUDENTS_PER_ITEM * weight))
            for _ in range(n_students):
                is_correct = np.random.random() < p_correct
                results.append((level, is_correct))
        else:
            # Fallback: assume 50% correct
            n_students = max(1, round(STUDENTS_PER_ITEM * weight))
            for _ in range(n_students):
                results.append((level, np.random.random() < 0.5))
    return results


# --- API call ---

def make_api_call(client, model_key, prompt, temperature=0.7, max_tokens=1024):
    """Call Gemini API."""
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
    """Run a single RSM configuration on all probe items. Returns (prop_incorrect_per_item, n_valid)."""
    config_id = config["config_id"]
    temperature = config["temperature"]
    students_per_call = config["students_per_call"]
    prompt_style = config["prompt_style"]
    misconception_hint = config["misconception_hint"]
    model = config["model"]

    raw_dir = OUTPUT_DIR / "raw_responses" / f"config_{config_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    item_results = {}  # qid -> list of is_correct booleans

    for _, row in probe_items.iterrows():
        qid = row["QuestionId"]
        correct = row["correct_answer_kaggle"]
        item_results[qid] = []

        if prompt_style == "teacher_prediction":
            # Single call per item, generates distribution
            raw_path = raw_dir / f"qid{qid}_teacher.txt"
            if raw_path.exists():
                text = raw_path.read_text()
            else:
                prompt = build_teacher_prediction_prompt(row, misconception_hint)
                try:
                    text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=512)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"    ERROR config={config_id} qid={qid}: {e}")
                    time.sleep(1)
                    continue
                time.sleep(0.1)

            parsed = parse_teacher_prediction(text, correct)
            item_results[qid].extend([ic for _, ic in parsed])

        elif prompt_style == "classroom_batch":
            # Batch calls
            n_calls = max(1, STUDENTS_PER_ITEM // students_per_call)
            for batch_idx in range(n_calls):
                raw_path = raw_dir / f"qid{qid}_batch{batch_idx}.txt"
                roster = None
                if raw_path.exists():
                    text = raw_path.read_text()
                    # Reconstruct roster for parsing
                    _, roster = build_classroom_batch_prompt(row, students_per_call, misconception_hint)
                else:
                    prompt, roster = build_classroom_batch_prompt(row, students_per_call, misconception_hint)
                    try:
                        text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=1024)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"    ERROR config={config_id} qid={qid} batch={batch_idx}: {e}")
                        time.sleep(1)
                        continue
                    time.sleep(0.1)

                answers = parse_classroom_batch(text, students_per_call)
                for ans in answers:
                    if ans is not None:
                        item_results[qid].append(ans == correct)

        elif prompt_style == "individual_roleplay":
            # Individual calls
            student_idx = 0
            for level, weight in PROFICIENCY_DISTRIBUTION:
                n_this_level = max(1, round(STUDENTS_PER_ITEM * weight))
                for si in range(n_this_level):
                    if student_idx >= STUDENTS_PER_ITEM:
                        break
                    raw_path = raw_dir / f"qid{qid}_s{student_idx}.txt"
                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = build_individual_roleplay_prompt(row, level, misconception_hint)
                        try:
                            text = make_api_call(client, model, prompt, temperature=temperature, max_tokens=200)
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"    ERROR config={config_id} qid={qid} s={student_idx}: {e}")
                            time.sleep(0.5)
                            student_idx += 1
                            continue
                        time.sleep(0.1)

                    ans = parse_single_answer(text)
                    if ans is not None:
                        item_results[qid].append(ans == correct)
                    student_idx += 1

    # Compute proportion incorrect per item
    prop_incorrect = {}
    for qid, results in item_results.items():
        if len(results) > 0:
            prop_incorrect[qid] = 1.0 - (sum(results) / len(results))

    return prop_incorrect


def evaluate_config(prop_incorrect, probe_items):
    """Compute Spearman rho between simulated difficulty and actual b_2pl."""
    qids = []
    sim_diff = []
    actual_diff = []

    for _, row in probe_items.iterrows():
        qid = row["QuestionId"]
        if qid in prop_incorrect:
            qids.append(qid)
            sim_diff.append(prop_incorrect[qid])
            actual_diff.append(row["b_2pl"])

    if len(sim_diff) < 5:
        return {"spearman_rho": np.nan, "spearman_p": np.nan, "pearson_r": np.nan,
                "pearson_p": np.nan, "n_items": len(sim_diff)}

    rho, p_rho = stats.spearmanr(sim_diff, actual_diff)
    r, p_r = stats.pearsonr(sim_diff, actual_diff)
    return {"spearman_rho": rho, "spearman_p": p_rho, "pearson_r": r,
            "pearson_p": p_r, "n_items": len(sim_diff)}


# --- Phase: Sweep ---

def run_sweep():
    """Run the full Box-Behnken sweep."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} items")

    probe = select_probe_items(df)
    print(f"Selected {len(probe)} probe items")
    probe.to_csv(OUTPUT_DIR / "probe_items.csv", index=False)

    design = generate_design_matrix()
    design.to_csv(OUTPUT_DIR / "design_matrix.csv", index=False)
    print(f"Design matrix: {len(design)} configurations")

    # Check for existing results
    results_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        completed = set(existing["config_id"].values)
        print(f"Resuming: {len(completed)} configs already done")

    all_results = []
    if results_path.exists():
        all_results = pd.read_csv(results_path).to_dict("records")

    for _, config in design.iterrows():
        cid = config["config_id"]
        if cid in completed:
            continue

        print(f"\n--- Config {cid}/{len(design)-1}: temp={config['temperature']}, "
              f"spc={config['students_per_call']}, style={config['prompt_style']}, "
              f"hint={config['misconception_hint']}, model={config['model']} ---")

        prop_incorrect = run_single_config(config.to_dict(), probe, client)
        metrics = evaluate_config(prop_incorrect, probe)

        result = config.to_dict()
        result.update(metrics)
        all_results.append(result)

        # Save incrementally
        pd.DataFrame(all_results).to_csv(results_path, index=False)

        print(f"  -> Spearman rho={metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f}), "
              f"Pearson r={metrics['pearson_r']:.3f}, n={metrics['n_items']}")

    print(f"\nSweep complete. Results saved to {results_path}")


# --- Phase: Analyze ---

def run_analyze():
    """Fit response surface model and identify optimal configuration."""
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

    # Best configs
    print(f"\nTop 5 configs by Spearman rho:")
    top5 = results.nlargest(5, "spearman_rho")
    for _, row in top5.iterrows():
        print(f"  Config {int(row['config_id'])}: rho={row['spearman_rho']:.3f} "
              f"| temp={row['temperature']}, spc={int(row['students_per_call'])}, "
              f"style={row['prompt_style']}, hint={row['misconception_hint']}, model={row['model']}")

    # Worst configs
    print(f"\nBottom 5 configs by Spearman rho:")
    bot5 = results.nsmallest(5, "spearman_rho")
    for _, row in bot5.iterrows():
        print(f"  Config {int(row['config_id'])}: rho={row['spearman_rho']:.3f} "
              f"| temp={row['temperature']}, spc={int(row['students_per_call'])}, "
              f"style={row['prompt_style']}, hint={row['misconception_hint']}, model={row['model']}")

    # Fit quadratic response surface using coded variables
    coded_cols = [c for c in results.columns if c.endswith("_coded")]
    if len(coded_cols) == 5:
        _fit_response_surface(results, coded_cols)

    # Factor main effects
    print(f"\n--- Main Effects ---")
    for factor in ["temperature", "students_per_call", "prompt_style", "misconception_hint", "model"]:
        print(f"\n  {factor}:")
        for val, grp in results.groupby(factor):
            rhos = grp["spearman_rho"].dropna()
            print(f"    {val}: mean rho={rhos.mean():.3f} (n={len(rhos)}, std={rhos.std():.3f})")

    # Save report
    report_lines = _build_report(results)
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


def _fit_response_surface(results, coded_cols):
    """Fit quadratic model: y = b0 + sum(bi*xi) + sum(bii*xi^2) + sum(bij*xi*xj)."""
    y = results["spearman_rho"].values
    valid = ~np.isnan(y)
    y = y[valid]

    X_coded = results[coded_cols].values[valid]

    # Build design matrix: intercept, linear, quadratic, interactions
    n = len(y)
    k = X_coded.shape[1]
    cols = ["intercept"]
    X_parts = [np.ones((n, 1))]

    # Linear
    for i in range(k):
        cols.append(coded_cols[i].replace("_coded", "_lin"))
        X_parts.append(X_coded[:, i:i+1])

    # Quadratic
    for i in range(k):
        cols.append(coded_cols[i].replace("_coded", "_quad"))
        X_parts.append((X_coded[:, i:i+1]) ** 2)

    # Interactions
    for i, j in combinations(range(k), 2):
        cols.append(f"{coded_cols[i]}x{coded_cols[j]}".replace("_coded", ""))
        X_parts.append((X_coded[:, i] * X_coded[:, j]).reshape(-1, 1))

    X = np.hstack(X_parts)

    # Least squares fit
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n--- Response Surface Model ---")
        print(f"  R² = {r_squared:.3f}")
        print(f"  Coefficients:")
        for name, coef in zip(cols, beta):
            print(f"    {name}: {coef:.4f}")

        # Save model
        model = {"r_squared": r_squared, "coefficients": dict(zip(cols, beta.tolist()))}
        (OUTPUT_DIR / "surface_model.json").write_text(json.dumps(model, indent=2))
    except Exception as e:
        print(f"  Could not fit model: {e}")


def _build_report(results):
    """Build analysis report."""
    lines = [
        "=" * 60,
        "RSM EXPERIMENT REPORT",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"\nTotal configs: {len(results)}",
        f"Valid results: {results['spearman_rho'].notna().sum()}",
        f"\nSpearman rho: mean={results['spearman_rho'].mean():.3f}, "
        f"std={results['spearman_rho'].std():.3f}, "
        f"range=[{results['spearman_rho'].min():.3f}, {results['spearman_rho'].max():.3f}]",
    ]

    best = results.loc[results["spearman_rho"].idxmax()]
    lines.append(f"\nBest config (id={int(best['config_id'])}):")
    lines.append(f"  temperature: {best['temperature']}")
    lines.append(f"  students_per_call: {int(best['students_per_call'])}")
    lines.append(f"  prompt_style: {best['prompt_style']}")
    lines.append(f"  misconception_hint: {best['misconception_hint']}")
    lines.append(f"  model: {best['model']}")
    lines.append(f"  Spearman rho: {best['spearman_rho']:.3f}")
    lines.append(f"  Pearson r: {best['pearson_r']:.3f}")

    lines.append(f"\nBaseline comparison: teacher_perspective r=0.095, classroom_sim r=-0.03, direct r=0.14")
    improvement = best['spearman_rho'] - 0.095
    lines.append(f"Improvement over best baseline: {improvement:+.3f}")

    return lines


# --- Phase: Confirm ---

def run_confirm():
    """Run best config on full item set with more students."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    results_path = OUTPUT_DIR / "results.csv"
    if not results_path.exists():
        print("No results.csv found. Run --phase sweep first.")
        return

    results = pd.read_csv(results_path)
    best = results.loc[results["spearman_rho"].idxmax()]
    print(f"Best config: id={int(best['config_id'])}, rho={best['spearman_rho']:.3f}")

    # Build config dict for full run
    config = {
        "config_id": "confirm",
        "temperature": best["temperature"],
        "students_per_call": int(best["students_per_call"]),
        "prompt_style": best["prompt_style"],
        "misconception_hint": best["misconception_hint"],
        "model": best["model"],
    }
    print(f"Config: {config}")

    df = load_data()
    print(f"Running confirmation on all {len(df)} items")

    # Override globals for confirmation: more students
    global STUDENTS_PER_ITEM
    original_spi = STUDENTS_PER_ITEM
    STUDENTS_PER_ITEM = 50  # 50 students per item for confirmation

    prop_incorrect = run_single_config(config, df, client)
    metrics = evaluate_config(prop_incorrect, df)

    STUDENTS_PER_ITEM = original_spi

    print(f"\n{'='*60}")
    print(f"CONFIRMATION RESULTS")
    print(f"{'='*60}")
    print(f"Items evaluated: {metrics['n_items']}")
    print(f"Spearman rho: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.4f})")
    print(f"Pearson r: {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.4f})")

    # Save
    confirm_result = {**config, **metrics, "students_per_item": 50}
    (OUTPUT_DIR / "confirmation_result.json").write_text(json.dumps(confirm_result, indent=2))
    print(f"\nSaved to {OUTPUT_DIR / 'confirmation_result.json'}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="RSM experiment for student simulation")
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
