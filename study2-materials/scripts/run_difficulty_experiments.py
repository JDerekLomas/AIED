#!/usr/bin/env python3
"""
Teacher Perspective + Student Simulation difficulty estimation experiments.

Two experiments on the same 100 test items (same calibration split as structured estimation):

1. Teacher Perspective: LLM as experienced UK maths teacher predicts % incorrect
   by year group. 5 reps × 100 items = 500 calls.

2. Student Simulation: LLM role-plays students at 4 proficiency levels answering
   items. 200 simulated students per item = 20,000 calls.

Usage:
    python3 scripts/run_difficulty_experiments.py --experiment teacher --model gemini-3-flash
    python3 scripts/run_difficulty_experiments.py --experiment simulation --model gemini-3-flash
    python3 scripts/run_difficulty_experiments.py --experiment both --model gemini-3-flash
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

# --- Configuration ---
RANDOM_SEED = 42
NUM_TEACHER_REPS = 5

# Student simulation: 200 students per item
# Distribution: 25% Below Basic, 35% Basic, 25% Proficient, 15% Advanced
STUDENT_LEVELS = {
    "below_basic": {"count": 50, "description": "below-basic", "weight": 0.25},
    "basic": {"count": 70, "description": "basic", "weight": 0.35},
    "proficient": {"count": 50, "description": "proficient", "weight": 0.25},
    "advanced": {"count": 30, "description": "advanced", "weight": 0.15},
}

PROVIDER_CONFIGS = {
    "gemini-3-flash": {
        "provider": "gemini",
        "model_id": "gemini-3-flash-preview",
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
    },
}

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


# --- Data loading (reused from structured estimation) ---

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


def select_calibration_items(df, n=5):
    """Select 5 calibration items at 10th/30th/50th/70th/90th percentile of b_2pl."""
    sorted_df = df.sort_values("b_2pl").reset_index(drop=True)
    percentiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    indices = [int(p * (len(sorted_df) - 1)) for p in percentiles]
    return sorted_df.iloc[indices]


def format_item_text(row, include_correct=True):
    """Format a single item for the prompt."""
    lines = [
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ]
    if include_correct:
        lines.append(f"Correct answer: {row['correct_answer_kaggle']}")
        if pd.notna(row.get('misconception_name', None)):
            lines.append(f"Misconception tested: {row['misconception_name']}")
    return "\n".join(lines)


# --- API call ---

def make_api_call(client, provider, model_id, prompt, temperature=0.7, max_tokens=1024):
    """Call LLM API, return text response."""
    if provider == "gemini":
        from google.genai import types
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
    else:
        raise ValueError(f"Unknown provider: {provider}")


# --- Experiment 1: Teacher Perspective ---

def build_teacher_prompt(row):
    """Build teacher perspective prompt for a single item."""
    item_text = format_item_text(row, include_correct=True)
    return f"""You are an experienced UK secondary maths teacher with 20 years of experience teaching across all year groups (Year 7 to Year 11, ages 11-16). You have deep knowledge of common student misconceptions and typical error patterns.

For this diagnostic maths question, predict what percentage of students would answer INCORRECTLY at each year group, based on your teaching experience.

{item_text}

Respond in this exact XML format (percentages should be integers 0-100):

<predictions>
<year7_pct_incorrect>N</year7_pct_incorrect>
<year8_pct_incorrect>N</year8_pct_incorrect>
<year9_pct_incorrect>N</year9_pct_incorrect>
<year10_pct_incorrect>N</year10_pct_incorrect>
<year11_pct_incorrect>N</year11_pct_incorrect>
<overall_pct_incorrect>N</overall_pct_incorrect>
<most_common_wrong_answer>X</most_common_wrong_answer>
<explanation>Why students get this wrong...</explanation>
</predictions>"""


TEACHER_TAGS = [
    "year7_pct_incorrect", "year8_pct_incorrect", "year9_pct_incorrect",
    "year10_pct_incorrect", "year11_pct_incorrect", "overall_pct_incorrect",
]


def parse_teacher_response(text):
    """Extract teacher predictions from XML response."""
    result = {}
    for tag in TEACHER_TAGS:
        match = re.search(rf"<{tag}>\s*(\d+)\s*</{tag}>", text)
        result[tag] = int(match.group(1)) if match else None

    # Most common wrong answer
    match = re.search(r"<most_common_wrong_answer>\s*([A-Da-d])\s*</most_common_wrong_answer>", text)
    result["most_common_wrong_answer"] = match.group(1).upper() if match else None

    # Explanation
    match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    result["explanation"] = match.group(1).strip()[:500] if match else None

    result["_n_extracted"] = sum(1 for t in TEACHER_TAGS if result[t] is not None)
    return result


def run_teacher_experiment(model_key, test_df):
    """Run teacher perspective experiment."""
    config = PROVIDER_CONFIGS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    output_dir = Path(f"pilot/teacher_perspective/{model_key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    all_results = []
    total_calls = len(test_df) * NUM_TEACHER_REPS
    call_num = 0

    for _, row in test_df.iterrows():
        qid = row["QuestionId"]
        for rep in range(NUM_TEACHER_REPS):
            call_num += 1

            raw_path = raw_dir / f"qid{qid}_rep{rep}.txt"
            if raw_path.exists():
                text = raw_path.read_text()
                parsed = parse_teacher_response(text)
                parsed["QuestionId"] = qid
                parsed["replication"] = rep
                all_results.append(parsed)
                print(f"  [{call_num}/{total_calls}] QID={qid} rep={rep+1} (cached)")
                continue

            prompt = build_teacher_prompt(row)
            try:
                text = make_api_call(client, provider, model_id, prompt)
                raw_path.write_text(text)
                parsed = parse_teacher_response(text)
                parsed["QuestionId"] = qid
                parsed["replication"] = rep
                all_results.append(parsed)

                overall = parsed.get("overall_pct_incorrect")
                print(f"  [{call_num}/{total_calls}] QID={qid} rep={rep+1} → {overall}% incorrect")

            except Exception as e:
                print(f"  [{call_num}/{total_calls}] QID={qid} rep={rep+1} ERROR: {e}")
                all_results.append({"QuestionId": qid, "replication": rep, "_error": str(e)})

            time.sleep(0.5)

    # Save raw results
    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(output_dir / "raw_results.csv", index=False)

    # Aggregate across replications
    agg_rows = []
    for qid in test_df["QuestionId"].values:
        qid_data = raw_df[(raw_df["QuestionId"] == qid) & (raw_df.get("_n_extracted", 0) > 0)]
        if len(qid_data) == 0:
            qid_data = raw_df[raw_df["QuestionId"] == qid]
        if len(qid_data) == 0:
            continue
        agg = {"QuestionId": qid}
        for tag in TEACHER_TAGS:
            vals = pd.to_numeric(qid_data[tag], errors="coerce").dropna()
            if len(vals) > 0:
                agg[f"{tag}_mean"] = vals.mean()
                agg[f"{tag}_std"] = vals.std() if len(vals) > 1 else 0.0
        # Weighted average across years (equal weight since we don't know year distribution)
        year_means = [agg.get(f"year{y}_pct_incorrect_mean") for y in range(7, 12)]
        year_means = [v for v in year_means if v is not None and not np.isnan(v)]
        if year_means:
            agg["weighted_avg_pct_incorrect"] = np.mean(year_means)
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(output_dir / "aggregated.csv", index=False)

    print(f"\nSaved {len(agg_df)} aggregated rows to {output_dir / 'aggregated.csv'}")
    return agg_df, output_dir


# --- Experiment 2: Student Simulation ---

def build_simulation_prompt(row, level):
    """Build student simulation prompt."""
    item_text = format_item_text(row, include_correct=False)
    level_desc = LEVEL_DESCRIPTIONS[level]
    return f"""{level_desc}

Answer this multiple choice maths question. Just give the letter of your answer (A, B, C, or D) and a brief explanation of your thinking.

{item_text}

ANSWER: """


def parse_simulation_response(text):
    """Extract answer letter from simulation response."""
    # Look for a letter A-D at the start or after ANSWER:
    match = re.search(r'\b([A-Da-d])\b', text[:50])
    if match:
        return match.group(1).upper()
    return None


def run_simulation_experiment(model_key, test_df):
    """Run student simulation experiment."""
    config = PROVIDER_CONFIGS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    output_dir = Path(f"pilot/student_simulation_v2/{model_key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    total_students = sum(cfg["count"] for cfg in STUDENT_LEVELS.values())
    total_calls = len(test_df) * total_students
    call_num = 0
    all_results = []

    for _, row in test_df.iterrows():
        qid = row["QuestionId"]
        correct = row["correct_answer_kaggle"]

        for level, cfg in STUDENT_LEVELS.items():
            for student_idx in range(cfg["count"]):
                call_num += 1
                sid = f"{level}_{student_idx}"

                raw_path = raw_dir / f"qid{qid}_{sid}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                    answer = parse_simulation_response(text)
                    all_results.append({
                        "QuestionId": qid, "level": level, "student_idx": student_idx,
                        "answer": answer, "is_correct": answer == correct,
                    })
                    if call_num % 500 == 0:
                        print(f"  [{call_num}/{total_calls}] (cached)")
                    continue

                prompt = build_simulation_prompt(row, level)
                try:
                    text = make_api_call(client, provider, model_id, prompt,
                                         temperature=0.9, max_tokens=200)
                    raw_path.write_text(text)
                    answer = parse_simulation_response(text)
                    all_results.append({
                        "QuestionId": qid, "level": level, "student_idx": student_idx,
                        "answer": answer, "is_correct": answer == correct,
                    })

                    if call_num % 100 == 0:
                        print(f"  [{call_num}/{total_calls}] QID={qid} {level}#{student_idx} → {answer}")

                except Exception as e:
                    print(f"  [{call_num}/{total_calls}] ERROR: {e}")
                    all_results.append({
                        "QuestionId": qid, "level": level, "student_idx": student_idx,
                        "answer": None, "is_correct": False, "_error": str(e),
                    })

                time.sleep(0.1)

    # Save raw
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "raw_results.csv", index=False)

    # Aggregate: proportion correct per item (weighted by level distribution)
    agg_rows = []
    for qid in test_df["QuestionId"].values:
        qid_data = results_df[results_df["QuestionId"] == qid]
        agg = {"QuestionId": qid}

        # Per-level accuracy
        weighted_correct = 0.0
        total_weight = 0.0
        for level, cfg in STUDENT_LEVELS.items():
            level_data = qid_data[qid_data["level"] == level]
            if len(level_data) > 0:
                acc = level_data["is_correct"].mean()
                agg[f"{level}_pct_correct"] = acc * 100
                agg[f"{level}_n"] = len(level_data)
                weighted_correct += acc * cfg["weight"]
                total_weight += cfg["weight"]

        if total_weight > 0:
            agg["weighted_pct_correct"] = (weighted_correct / total_weight) * 100
            agg["weighted_pct_incorrect"] = 100 - agg["weighted_pct_correct"]

        # Simple overall
        if len(qid_data) > 0:
            agg["overall_pct_correct"] = qid_data["is_correct"].mean() * 100
            agg["overall_pct_incorrect"] = 100 - agg["overall_pct_correct"]

        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(output_dir / "aggregated.csv", index=False)

    print(f"\nSaved {len(agg_df)} aggregated rows to {output_dir / 'aggregated.csv'}")
    return agg_df, output_dir


# --- Analysis ---

def run_analysis(test_df, teacher_agg=None, teacher_dir=None,
                 sim_agg=None, sim_dir=None):
    """Compute correlations with actual IRT parameters and print report."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DIFFICULTY ESTIMATION EXPERIMENT RESULTS")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 60)

    actual = test_df[["QuestionId", "difficulty_classical", "b_2pl", "a_2pl"]].copy()

    def correlate(predicted_col, actual_col, label):
        merged = actual.merge(predicted_col, on="QuestionId")
        col = [c for c in merged.columns if c not in actual.columns][0]
        x = merged[col].dropna()
        y = merged.loc[x.index, actual_col]
        if len(x) < 5:
            return f"  {label}: insufficient data (n={len(x)})"
        r_pearson, p_pearson = stats.pearsonr(x, y)
        r_spearman, p_spearman = stats.spearmanr(x, y)
        return (f"  {label} (n={len(x)}):\n"
                f"    Pearson r={r_pearson:.3f} (p={p_pearson:.4f})\n"
                f"    Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})")

    if teacher_agg is not None:
        report_lines.append("\n--- TEACHER PERSPECTIVE ---")
        # overall_pct_incorrect_mean vs classical difficulty (also pct incorrect)
        pred = teacher_agg[["QuestionId", "overall_pct_incorrect_mean"]].copy()
        pred["overall_pct_incorrect_mean"] /= 100.0  # scale to 0-1
        report_lines.append(correlate(pred, "difficulty_classical",
                                       "overall_pct_incorrect vs classical difficulty"))

        # weighted avg vs b_2pl
        if "weighted_avg_pct_incorrect" in teacher_agg.columns:
            pred2 = teacher_agg[["QuestionId", "weighted_avg_pct_incorrect"]].copy()
            report_lines.append(correlate(pred2, "b_2pl",
                                           "weighted_avg_pct_incorrect vs b_2pl"))

    if sim_agg is not None:
        report_lines.append("\n--- STUDENT SIMULATION ---")
        # weighted_pct_incorrect vs classical difficulty
        if "weighted_pct_incorrect" in sim_agg.columns:
            pred = sim_agg[["QuestionId", "weighted_pct_incorrect"]].copy()
            pred["weighted_pct_incorrect"] /= 100.0
            report_lines.append(correlate(pred, "difficulty_classical",
                                           "weighted_pct_incorrect vs classical difficulty"))
            report_lines.append(correlate(
                sim_agg[["QuestionId", "weighted_pct_incorrect"]].copy(),
                "b_2pl", "weighted_pct_incorrect vs b_2pl"))

        # Per-level accuracy
        report_lines.append("\n  Per-level mean accuracy:")
        for level in STUDENT_LEVELS:
            col = f"{level}_pct_correct"
            if col in sim_agg.columns:
                report_lines.append(f"    {level}: {sim_agg[col].mean():.1f}% correct")

    # Tercile analysis
    if teacher_agg is not None or sim_agg is not None:
        report_lines.append("\n--- TERCILE ANALYSIS ---")
        actual_sorted = actual.sort_values("difficulty_classical")
        n = len(actual_sorted)
        terciles = {
            "easy": actual_sorted.iloc[:n//3]["QuestionId"].values,
            "medium": actual_sorted.iloc[n//3:2*n//3]["QuestionId"].values,
            "hard": actual_sorted.iloc[2*n//3:]["QuestionId"].values,
        }
        for source_name, agg, col in [
            ("Teacher", teacher_agg, "overall_pct_incorrect_mean"),
            ("Simulation", sim_agg, "weighted_pct_incorrect"),
        ]:
            if agg is None or col not in agg.columns:
                continue
            report_lines.append(f"\n  {source_name} predicted difficulty by actual tercile:")
            for terc_name, qids in terciles.items():
                vals = agg[agg["QuestionId"].isin(qids)][col].dropna()
                if len(vals) > 0:
                    report_lines.append(f"    {terc_name}: mean={vals.mean():.1f}, sd={vals.std():.1f}")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    out_dir = teacher_dir or sim_dir
    if out_dir:
        (out_dir.parent / "analysis_report.txt").write_text(report)
        print(f"\nReport saved to {out_dir.parent / 'analysis_report.txt'}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Teacher perspective + student simulation experiments")
    parser.add_argument("--experiment", choices=["teacher", "simulation", "both"], required=True)
    parser.add_argument("--model", default="gemini-3-flash", choices=list(PROVIDER_CONFIGS.keys()))
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    print(f"Model: {args.model} ({PROVIDER_CONFIGS[args.model]['model_id']})")
    print(f"Experiment: {args.experiment}")

    np.random.seed(RANDOM_SEED)

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} items with IRT parameters")

    cal_items = select_calibration_items(df)
    cal_qids = set(cal_items["QuestionId"].values)
    test_df = df[~df["QuestionId"].isin(cal_qids)].reset_index(drop=True)
    print(f"Test items: {len(test_df)} (excluded {len(cal_qids)} calibration)")

    teacher_agg, teacher_dir = None, None
    sim_agg, sim_dir = None, None

    if args.experiment in ("teacher", "both"):
        print(f"\n{'='*40}")
        print("EXPERIMENT 1: TEACHER PERSPECTIVE")
        print(f"{'='*40}")
        print(f"Calls: {len(test_df)} items × {NUM_TEACHER_REPS} reps = {len(test_df)*NUM_TEACHER_REPS}")
        teacher_agg, teacher_dir = run_teacher_experiment(args.model, test_df)

    if args.experiment in ("simulation", "both"):
        print(f"\n{'='*40}")
        print("EXPERIMENT 2: STUDENT SIMULATION")
        print(f"{'='*40}")
        total_students = sum(cfg["count"] for cfg in STUDENT_LEVELS.values())
        print(f"Calls: {len(test_df)} items × {total_students} students = {len(test_df)*total_students}")
        sim_agg, sim_dir = run_simulation_experiment(args.model, test_df)

    # Analysis
    run_analysis(test_df, teacher_agg, teacher_dir, sim_agg, sim_dir)


if __name__ == "__main__":
    main()
