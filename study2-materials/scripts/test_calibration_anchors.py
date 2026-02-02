#!/usr/bin/env python3
"""
Calibration anchor experiment: Give the model a few items with known
real response data, then predict the rest.

Uses leave-K-out cross-validation to avoid overfitting:
  - Pick 5 anchor items (with real distributions)
  - Predict remaining 15
  - Rotate which 5 are anchors
  - Report average held-out rho

Inspired by SmartPaper calibration which showed +0.16 rho boost.
"""
import json, re, os, time, sys, itertools
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/calibration_anchors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3
N_FOLDS = 4  # 4-fold: 5 anchors, 15 test items each fold
SEED = 42

CALIBRATED_PROMPT = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

Here are the ACTUAL response distributions for some questions from this same exam. Use these to calibrate your sense of how these specific students perform:

{anchor_text}

Now, for this NEW question from the same exam, predict what percentage at each ability level chose each option.

Before predicting, think about:
- What specific calculation error leads to each wrong answer?
- Which errors are "attractive" — where the wrong method feels natural?
- How does the difficulty compare to the calibration questions above?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""

# Control: same prompt without anchors (error_analysis baseline)
UNCALIBRATED_PROMPT = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

For this question, based on your experience of what students actually wrote, predict what percentage at each ability level chose each option.

Before predicting, think about:
- What specific calculation error leads to each wrong answer?
- Which errors are "attractive" — where the wrong method feels natural?
- Which students would catch themselves vs. fall into the trap?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""


def format_item_text(row):
    return "\n".join([
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ])


def format_anchor_item(row):
    """Format an anchor item showing real response distributions."""
    return "\n".join([
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
        f"Correct answer: {row['correct_answer_kaggle']}",
        f"Actual responses: A={row['pct_A']:.0f}% B={row['pct_B']:.0f}% C={row['pct_C']:.0f}% D={row['pct_D']:.0f}%",
        f"(Difficulty: {row['difficulty_classical']:.0f}% got it right)",
        "",
    ])


def parse_predictions(text, correct_answer):
    weights = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}
    correct_idx = ord(correct_answer) - ord('A')
    weighted_p_correct = 0.0
    for level, w in weights.items():
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
        else:
            weighted_p_correct += 0.5 * w
    return 1 - weighted_p_correct


def main():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(probe))
    fold_size = len(probe) // N_FOLDS

    for method_name, use_anchors in [("calibrated", True), ("uncalibrated", False)]:
        print(f"\n{'='*60}", flush=True)
        print(f"{method_name.upper()} — {'with' if use_anchors else 'without'} anchor items", flush=True)
        print(f"{'='*60}", flush=True)

        fold_rhos = {rep: [] for rep in range(N_REPS)}
        all_preds = []

        for fold in range(N_FOLDS):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            anchor_idx = np.setdiff1d(indices, test_idx)

            test_items = probe.iloc[test_idx]
            anchor_items = probe.iloc[anchor_idx]

            # Format anchor text
            anchor_text = ""
            if use_anchors:
                anchor_text = "---\n".join(format_anchor_item(row) for _, row in anchor_items.iterrows())

            print(f"\n  Fold {fold}: {len(anchor_items)} anchors, {len(test_items)} test items", flush=True)

            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / method_name / f"fold{fold}" / f"rep{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)

                items_pred = []
                for _, row in test_items.iterrows():
                    qid = row["QuestionId"]
                    correct = row["correct_answer_kaggle"]
                    raw_path = raw_dir / f"qid{qid}.txt"

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        item_text = format_item_text(row)
                        if use_anchors:
                            prompt = CALIBRATED_PROMPT.format(
                                anchor_text=anchor_text, item_text=item_text)
                        else:
                            prompt = UNCALIBRATED_PROMPT.format(item_text=item_text)
                        try:
                            resp = client.models.generate_content(
                                model="gemini-3-flash-preview",
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=2.0, max_output_tokens=1024,
                                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                                ),
                            )
                            text = resp.text
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"    ERROR fold{fold} rep{rep} qid={qid}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        time.sleep(0.15)

                    p_inc = parse_predictions(text, correct)
                    items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                       "weighted_p_incorrect": p_inc, "fold": fold, "rep": rep})
                    all_preds.append(items_pred[-1])

                df = pd.DataFrame(items_pred)
                if len(df) >= 5:
                    rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                    fold_rhos[rep].append(rho)
                    print(f"    fold{fold} rep{rep}: rho={rho:.3f} (n={len(df)})", flush=True)

        # Aggregate across folds
        print(f"\n  Cross-validated results ({method_name}):", flush=True)
        rep_means = []
        for rep in range(N_REPS):
            if fold_rhos[rep]:
                mean = np.mean(fold_rhos[rep])
                rep_means.append(mean)
                print(f"    rep{rep} mean across folds: {mean:.3f}", flush=True)

        if rep_means:
            overall = np.mean(rep_means)
            std = np.std(rep_means)
            print(f"    OVERALL: {overall:.3f} ± {std:.3f}", flush=True)

        # Also: pool all test predictions and compute single rho
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            # Average across reps per item
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("weighted_p_incorrect", "mean"),
                b_2pl=("b_2pl", "first")
            ).dropna()
            if len(avg) >= 5:
                rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
                print(f"    Pooled averaged: rho={rho:.3f} (p={p:.4f}, n={len(avg)})", flush=True)

        summary = {"method": method_name, "n_folds": N_FOLDS,
                   "rep_means": [float(r) for r in rep_means],
                   "overall_mean": float(np.mean(rep_means)) if rep_means else None,
                   "overall_std": float(np.std(rep_means)) if rep_means else None}
        with open(OUTPUT_DIR / f"{method_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_preds = []  # reset for next method


if __name__ == "__main__":
    main()
