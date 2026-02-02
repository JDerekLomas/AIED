#!/usr/bin/env python3
"""
10-rep stability test for Llama-4-Scout with error_analysis @ t=2.0.
Reuses 3 cached reps, generates 7 new ones.
"""
import json, re, os, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

BASE_DIR = Path("pilot/rsm_experiment/cross_model")
N_REPS = 3
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPS = [1.0, 1.5, 2.0, 2.5]

ERROR_ANALYSIS_PROMPT = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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


def call_groq(prompt, temp, max_retries=5):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "503" in str(e) or "over capacity" in str(e):
                wait = 2 ** (attempt + 1)
                print(f"    503 error, retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


def main():
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")
    print(f"Llama-4-Scout temperature sweep (error_analysis)")
    print(f"Temps: {TEMPS}, {N_REPS} reps × {len(probe)} items\n")

    summary = {}

    for temp in TEMPS:
        cache_dir = BASE_DIR / f"groq_llama4scout_erroranalysis_t{temp}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"Temperature = {temp}")
        print(f"{'='*50}")

        all_preds = []
        rhos = []

        for rep in range(N_REPS):
            raw_dir = cache_dir / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            items_pred = []
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    item_text = format_item_text(row)
                    prompt = ERROR_ANALYSIS_PROMPT.format(item_text=item_text)
                    try:
                        text = call_groq(prompt, temp)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR t={temp} rep{rep} qid={qid}: {e}", flush=True)
                        continue
                    time.sleep(0.5)

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})
                all_preds.append({"rep": rep, "QuestionId": qid,
                                  "b_2pl": row["b_2pl"], "p_inc": p_inc})

            df = pd.DataFrame(items_pred)
            valid = df.dropna()
            if len(valid) >= 5:
                rho, p = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
                rhos.append(rho)
                print(f"  rep{rep}: ρ={rho:.3f} (p={p:.4f})", flush=True)

        # Averaged-prediction ρ
        avg_rho = None
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("p_inc", "mean"),
                b_2pl=("b_2pl", "first")
            ).dropna()
            if len(avg) >= 5:
                avg_rho, avg_p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])

        mean_rho = float(np.mean(rhos)) if rhos else 0
        std_rho = float(np.std(rhos)) if rhos else 0
        summary[temp] = {"mean": mean_rho, "std": std_rho, "rhos": rhos, "avg_pred": avg_rho}
        print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}, avg-pred: {avg_rho:.3f}" if avg_rho else f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}")

    # Final ranking
    print(f"\n{'='*60}")
    print("TEMPERATURE SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Temp':<6} {'Mean ρ':<10} {'SD':<8} {'Avg-pred ρ':<12}")
    for temp in TEMPS:
        d = summary[temp]
        avg = f"{d['avg_pred']:.3f}" if d['avg_pred'] else "N/A"
        print(f"{temp:<6} {d['mean']:<10.3f} {d['std']:<8.3f} {avg:<12}")

    with open(BASE_DIR / "scout_temp_sweep.json", "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2, default=str)
    print(f"\nSaved to {BASE_DIR}/scout_temp_sweep.json")


if __name__ == "__main__":
    main()
