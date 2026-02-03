#!/usr/bin/env python3
"""
Run difficulty estimation on expanded 50-item probe set + 10-rep Scout experiment.

Experiments:
1. Gemini 3 Flash (error_analysis @ t=2.0) on 50 items, 3 reps
2. Llama-4-Scout (error_analysis @ t=2.0) on 50 items, 3 reps
3. Llama-4-Scout (error_analysis @ t=2.0) on original 20 items, 10 reps

Usage:
    python3 scripts/run_expanded_probe.py [experiment_name]
    experiment_name: gemini50, scout50, scout10rep, all (default: all)
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

ERROR_ANALYSIS_PROMPT = """You are an experienced UK maths teacher. For this multiple-choice question, analyze what specific calculation error or misconception leads to each wrong answer, then predict what percentage of students at each ability level would choose each option.

For each wrong answer, explain: What specific error would a student make to arrive at this answer?

Then predict the response distribution:

{item_text}

Respond in this exact format:

Error Analysis:
A) [what error/reasoning leads to choosing A]
B) [what error/reasoning leads to choosing B]
C) [what error/reasoning leads to choosing C]
D) [what error/reasoning leads to choosing D]

Predictions:
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


def call_gemini(prompt, temp=2.0):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_scout(prompt, temp=2.0):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def run_experiment(name, probe_df, call_fn, n_reps, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    rhos_b2pl = []
    rhos_brasch = []

    for rep in range(n_reps):
        raw_dir = output_dir / f"rep{rep}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        items_pred = []
        for _, row in probe_df.iterrows():
            qid = row["QuestionId"]
            correct = row["correct_answer_kaggle"]
            raw_path = raw_dir / f"qid{qid}.txt"

            if raw_path.exists() and raw_path.stat().st_size > 10:
                text = raw_path.read_text()
            else:
                item_text = format_item_text(row)
                prompt = ERROR_ANALYSIS_PROMPT.format(item_text=item_text)
                try:
                    text = call_fn(prompt)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"  ERROR rep{rep} qid={qid}: {e}")
                    time.sleep(3)
                    try:
                        text = call_fn(prompt)
                        raw_path.write_text(text)
                    except Exception as e2:
                        print(f"  RETRY FAILED: {e2}")
                        continue
                time.sleep(0.5)

            p_inc = parse_predictions(text, correct)
            pred = {"QuestionId": qid, "b_2pl": row.get("b_2pl"),
                    "b_rasch": row.get("b_rasch"), "p_correct": row.get("p_correct"),
                    "rep": rep, "p_inc": p_inc}
            items_pred.append(pred)
            all_preds.append(pred)

        df = pd.DataFrame(items_pred).dropna(subset=["p_inc"])
        if len(df) >= 5:
            valid_b2pl = df.dropna(subset=["b_2pl"])
            if len(valid_b2pl) >= 5:
                rho, p = stats.spearmanr(valid_b2pl["p_inc"], valid_b2pl["b_2pl"])
                rhos_b2pl.append(rho)
                print(f"  {name} rep{rep}: rho(b_2pl)={rho:.3f} (n={len(valid_b2pl)}, p={p:.4f})")

    # Averaged predictions
    pdf = pd.DataFrame(all_preds)
    avg = pdf.groupby("QuestionId").agg(
        mean_p_inc=("p_inc", "mean"),
        b_2pl=("b_2pl", "first"),
        b_rasch=("b_rasch", "first"),
        p_correct=("p_correct", "first"),
    ).dropna(subset=["mean_p_inc"])

    results = {"name": name, "n_items": len(probe_df), "n_reps": n_reps}

    valid = avg.dropna(subset=["b_2pl"])
    if len(valid) >= 5:
        avg_rho, avg_p = stats.spearmanr(valid["mean_p_inc"], valid["b_2pl"])
        results["avg_pred_rho_b2pl"] = float(avg_rho)
        results["avg_pred_p_b2pl"] = float(avg_p)
        print(f"\n  {name} avg-pred rho(b_2pl): {avg_rho:.3f} (p={avg_p:.4f}, n={len(valid)})")

    if rhos_b2pl:
        results["mean_rho_b2pl"] = float(np.mean(rhos_b2pl))
        results["std_rho_b2pl"] = float(np.std(rhos_b2pl))
        results["rhos_b2pl"] = [float(r) for r in rhos_b2pl]
        print(f"  {name} mean rho(b_2pl): {np.mean(rhos_b2pl):.3f} ± {np.std(rhos_b2pl):.3f}")

    # Save
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    avg.to_csv(output_dir / "averaged_predictions.csv")

    return results


def main():
    selected = sys.argv[1] if len(sys.argv) > 1 else "all"

    expanded = pd.read_csv("pilot/rsm_experiment/probe_items_expanded50.csv")
    original = pd.read_csv("pilot/rsm_experiment/probe_items.csv")
    # Add b_rasch to original
    orig_with_irt = original.merge(
        expanded[['QuestionId', 'b_rasch', 'p_correct']].drop_duplicates(),
        on='QuestionId', how='left'
    )

    all_results = {}

    if selected in ("gemini50", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT: Gemini 3 Flash on 50 items (error_analysis @ t=2.0)")
        print("="*60)
        r = run_experiment("gemini_flash_50", expanded, call_gemini, 3,
                          "pilot/rsm_experiment/expanded50/gemini_flash")
        all_results["gemini_flash_50"] = r

    if selected in ("scout50", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT: Llama-4-Scout on 50 items (error_analysis @ t=2.0)")
        print("="*60)
        r = run_experiment("scout_50", expanded, call_scout, 3,
                          "pilot/rsm_experiment/expanded50/llama4_scout")
        all_results["scout_50"] = r

    if selected in ("scout10rep", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT: Llama-4-Scout 10 reps on 20 items (error_analysis @ t=2.0)")
        print("="*60)
        r = run_experiment("scout_10rep", orig_with_irt, call_scout, 10,
                          "pilot/rsm_experiment/scout_10rep")
        all_results["scout_10rep"] = r

    # Save combined summary
    with open("pilot/rsm_experiment/expanded_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    for name, r in all_results.items():
        rho = r.get('mean_rho_b2pl', 'N/A')
        avg = r.get('avg_pred_rho_b2pl', 'N/A')
        print(f"  {name}: mean_rho={rho}, avg_pred_rho={avg}")


if __name__ == "__main__":
    main()
