#!/usr/bin/env python3
"""
Run each prompt variant 3 times to get stable estimates.
"""
import json, re, os, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/prompt_variants_reps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

PROMPTS = {
    "v0_baseline": """You are an experienced UK maths teacher. For this question, predict the percentage of students at each ability level who would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v1_anchored": """You are an experienced UK maths teacher with 20 years experience across Years 7-11.

For this question, predict what percentage of students at each ability level would choose each option.

Important calibration notes:
- Below-basic students (bottom 25%) often guess or apply incorrect procedures. On hard topics they perform near chance (25% per option). On familiar topics they may reach 40-50% correct.
- Basic students (next 35%) can handle routine problems but struggle with multi-step or unfamiliar formats. Typical correct rates: 30-60%.
- Proficient students (next 25%) usually get questions right but can be caught by subtle distractors. Typical correct rates: 60-85%.
- Advanced students (top 15%) rarely make errors. Typical correct rates: 85-98%.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v3_contrastive": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v4_contrastive_anchored": """You are an experienced UK maths teacher with 20 years experience across Years 7-11.

For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps.

Calibration:
- Below-basic (bottom 25%): guess on hard topics (~25% each), reach 40-50% correct on familiar ones
- Basic (next 35%): 30-60% correct depending on complexity
- Proficient (next 25%): 60-85% correct, caught by subtle distractors
- Advanced (top 15%): 85-98% correct, rarely wrong

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",
}


def format_item_text(row):
    lines = [
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ]
    return "\n".join(lines)


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


def make_api_call(client, prompt):
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1.5,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    print(f"Running {len(PROMPTS)} variants x {N_REPS} reps x {len(probe)} items "
          f"= {len(PROMPTS) * N_REPS * len(probe)} calls\n")

    summary = {}
    for vname, template in PROMPTS.items():
        rhos = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / vname / f"rep{rep}"
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
                    prompt = template.format(item_text=item_text)
                    try:
                        text = make_api_call(client, prompt)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {vname} rep{rep} qid={qid}: {e}")
                        time.sleep(1)
                        continue
                    time.sleep(0.1)

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})

            df = pd.DataFrame(items_pred)
            valid = df.dropna()
            if len(valid) >= 5:
                rho, _ = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
                rhos.append(rho)
                print(f"  {vname} rep{rep}: rho={rho:.3f}")

        if rhos:
            mean_rho = np.mean(rhos)
            std_rho = np.std(rhos)
            summary[vname] = {"mean": mean_rho, "std": std_rho, "rhos": rhos}
            print(f"  {vname} MEAN: {mean_rho:.3f} ± {std_rho:.3f}\n")

    # Also average across reps per item for each variant
    print(f"\n{'='*60}")
    print("AVERAGED PREDICTIONS (mean p_incorrect across reps)")
    print(f"{'='*60}")
    for vname in PROMPTS:
        all_preds = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / vname / f"rep{rep}"
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                    p_inc = parse_predictions(text, correct)
                    all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                      "rep": rep, "p_inc": p_inc})

        if all_preds:
            pdf = pd.DataFrame(all_preds)
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("p_inc", "mean"),
                b_2pl=("b_2pl", "first")
            ).dropna()
            if len(avg) >= 5:
                rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
                print(f"  {vname}: rho={rho:.3f} (p={p:.4f})")

    print(f"\n{'='*60}")
    print("FINAL RANKING (by mean single-rep rho)")
    print(f"{'='*60}")
    for name, data in sorted(summary.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name}: {data['mean']:.3f} ± {data['std']:.3f}  ({data['rhos']})")


if __name__ == "__main__":
    main()
