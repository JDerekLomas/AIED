#!/usr/bin/env python3
"""
Two-stage difficulty estimation:
  Stage 1: Generate 5 diverse student backstories at high temp (2.0) using Gemini Flash
  Stage 2: For each backstory, use it as context for a teacher_prediction prompt at temp=1.5
  Aggregate the 5 predictions per item

Objective: Spearman rho between predicted weighted_p_incorrect and IRT b_2pl
"""
import json, re, os, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/two_stage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BACKSTORIES = 5
N_REPS = 3

BACKSTORY_PROMPT = """Generate a brief but vivid backstory for a UK Year 9 maths student. Include:
- Their name, personality, and attitude toward maths
- Their specific strengths and weaknesses in maths topics
- Their typical approach when stuck on a problem (guess? skip? try something wrong?)
- Their approximate ability level (pick one: below_basic, basic, proficient, or advanced)

Be creative and specific. Make this feel like a real student, not a stereotype.
Write 3-4 sentences only."""

PREDICTION_PROMPT = """You are an experienced UK maths teacher. You know this student well:

{backstory}

This student is about to answer the following question. Based on what you know about them, predict which option they would choose and why. Then predict what percentage of students LIKE THEM (similar ability, similar weaknesses) would choose each option.

{item_text}

Respond in this exact format (the ability level row matching this student's level):
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


def call_gemini(client, prompt, temp):
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    # Stage 1: Generate backstories (shared across reps)
    backstory_dir = OUTPUT_DIR / "backstories"
    backstory_dir.mkdir(parents=True, exist_ok=True)

    backstories = []
    for i in range(N_BACKSTORIES):
        path = backstory_dir / f"backstory_{i}.txt"
        if path.exists():
            backstories.append(path.read_text())
        else:
            text = call_gemini(client, BACKSTORY_PROMPT, 2.0)
            path.write_text(text)
            backstories.append(text)
            time.sleep(0.15)
            print(f"Generated backstory {i}: {text[:80]}...")

    print(f"\n{N_BACKSTORIES} backstories ready.\n")

    # Stage 2: For each rep, for each backstory × item, predict
    all_rhos = []
    for rep in range(N_REPS):
        print(f"\n=== Rep {rep} ===")
        item_predictions = {row["QuestionId"]: [] for _, row in probe.iterrows()}

        for bi, backstory in enumerate(backstories):
            raw_dir = OUTPUT_DIR / f"rep{rep}" / f"backstory{bi}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    item_text = format_item_text(row)
                    prompt = PREDICTION_PROMPT.format(backstory=backstory, item_text=item_text)
                    try:
                        text = call_gemini(client, prompt, 1.5)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR rep{rep} bs{bi} qid={qid}: {e}")
                        time.sleep(2)
                        continue
                    time.sleep(0.15)

                p_inc = parse_predictions(text, correct)
                item_predictions[qid].append(p_inc)

        # Aggregate: mean p_incorrect across backstories
        results = []
        for _, row in probe.iterrows():
            qid = row["QuestionId"]
            preds = item_predictions[qid]
            if preds:
                results.append({
                    "QuestionId": qid,
                    "b_2pl": row["b_2pl"],
                    "mean_p_incorrect": np.mean(preds),
                    "std_p_incorrect": np.std(preds),
                    "n_backstories": len(preds),
                })

        df = pd.DataFrame(results)
        rho, p = stats.spearmanr(df["mean_p_incorrect"], df["b_2pl"])
        all_rhos.append(rho)
        print(f"  Rep {rep}: rho={rho:.3f} (p={p:.4f})")

    print(f"\n{'='*60}")
    print(f"TWO-STAGE RESULTS: mean rho = {np.mean(all_rhos):.3f} ± {np.std(all_rhos):.3f}")
    print(f"Individual reps: {[f'{r:.3f}' for r in all_rhos]}")
    print(f"{'='*60}")

    # Save
    summary = {"method": "two_stage", "n_backstories": N_BACKSTORIES,
               "n_reps": N_REPS, "mean_rho": float(np.mean(all_rhos)),
               "std_rho": float(np.std(all_rhos)), "rhos": all_rhos}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
