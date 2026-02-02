#!/usr/bin/env python3
"""
High-rep stability test: Run the top 2 configs for 10 reps each.
Also computes averaged-prediction rho (the real payoff of more reps).

Configs:
  - v5_error_analysis at t=2.0 (current best, ρ=0.604±0.062)
  - v3_contrastive at t=1.5 (runner-up, ρ=0.577±0.075)

Reuses cached reps 0–2 from metaprompt_sweep.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/high_reps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_DIR = Path("pilot/rsm_experiment/metaprompt_sweep")

N_REPS = 10

CONFIGS = {
    "v5_error_analysis_t2.0": {
        "prompt": """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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
advanced: A=XX% B=XX% C=XX% D=XX%""",
        "temp": 2.0,
        "sweep_key": "v5_error_analysis_t2.0",
    },
    "v3_contrastive_t1.5": {
        "prompt": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",
        "temp": 1.5,
        "sweep_key": "v3_contrastive_t1.5",
    },
}


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


def main():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    selected = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    configs_to_run = {k: v for k, v in CONFIGS.items() if k in selected}

    for cname, cfg in configs_to_run.items():
        print(f"\n{'='*60}", flush=True)
        print(f"{cname} — {N_REPS} reps", flush=True)
        print(f"{'='*60}", flush=True)

        rhos = []
        all_preds = []

        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / cname / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Try to reuse cached data from metaprompt_sweep (reps 0-2)
            sweep_dir = SWEEP_DIR / cfg["sweep_key"] / f"rep{rep}"

            items_pred = []
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                elif sweep_dir.exists() and (sweep_dir / f"qid{qid}.txt").exists():
                    # Copy from sweep cache
                    text = (sweep_dir / f"qid{qid}.txt").read_text()
                    raw_path.write_text(text)
                else:
                    item_text = format_item_text(row)
                    prompt = cfg["prompt"].format(item_text=item_text)
                    try:
                        resp = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=cfg["temp"],
                                max_output_tokens=1024,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                        text = resp.text
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {cname} rep{rep} qid={qid}: {e}", flush=True)
                        time.sleep(2)
                        continue
                    time.sleep(0.15)

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})
                all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                  "rep": rep, "p_inc": p_inc})

            df = pd.DataFrame(items_pred)
            if len(df) >= 5:
                rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                rhos.append(rho)
                print(f"  rep{rep}: rho={rho:.3f} (p={p:.4f})", flush=True)

        # Per-rep stats
        if rhos:
            print(f"\n  Per-rep: mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}", flush=True)
            print(f"  Individual: {[f'{r:.3f}' for r in rhos]}", flush=True)

        # Averaged-prediction rho (cumulative)
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            print(f"\n  Averaged-prediction rho (cumulative):", flush=True)
            for n in [3, 5, 7, 10]:
                sub = pdf[pdf["rep"] < n]
                avg = sub.groupby("QuestionId").agg(
                    mean_p_inc=("p_inc", "mean"),
                    b_2pl=("b_2pl", "first")
                ).dropna()
                if len(avg) >= 5:
                    rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
                    print(f"    {n} reps averaged: rho={rho:.3f} (p={p:.4f})", flush=True)

        # Save
        summary = {"config": cname, "n_reps": len(rhos),
                   "per_rep_mean": float(np.mean(rhos)) if rhos else None,
                   "per_rep_std": float(np.std(rhos)) if rhos else None,
                   "rhos": [float(r) for r in rhos]}
        with open(OUTPUT_DIR / f"{cname}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
