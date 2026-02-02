#!/usr/bin/env python3
"""Map the temperature transition zone for teacher_prediction prompt.

Tests t=0.6, 0.9, 1.2 to fill the gap between t=0.3 (ρ=0.119) and t=1.5 (ρ=0.673).
Uses the same 20 probe items and methodology as the RSM experiment.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROBE_CSV = Path("pilot/rsm_experiment/probe_items.csv")
OUTPUT_DIR = Path("pilot/rsm_experiment/transition_zone")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.6, 0.9, 1.2]
REPS = 3  # multiple reps per temperature for stability
STUDENTS_PER_ITEM = 20
MODEL_ID = "gemini-3-flash-preview"

PROFICIENCY_DISTRIBUTION = [
    ("below_basic", 0.25),
    ("basic", 0.35),
    ("proficient", 0.25),
    ("advanced", 0.15),
]


def build_prompt(row):
    """Teacher prediction prompt with hidden misconceptions (matching RSM center)."""
    item_text = (
        f"Question: {row['question_text']}\n"
        f"A) {row['AnswerAText']}\n"
        f"B) {row['AnswerBText']}\n"
        f"C) {row['AnswerCText']}\n"
        f"D) {row['AnswerDText']}"
    )
    return f"""You are an experienced UK maths teacher. For this question, predict the percentage of students at each ability level who would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""


def parse_teacher_prediction(text, correct_answer):
    """Parse into p_incorrect (proportion getting it wrong)."""
    total_correct = 0
    total_students = 0
    for level, weight in PROFICIENCY_DISTRIBUTION:
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total_pct = sum(pcts) or 1
            correct_idx = ord(correct_answer) - ord('A')
            p_correct = pcts[correct_idx] / total_pct
            n = max(1, round(STUDENTS_PER_ITEM * weight))
            np.random.seed(hash((level, text[:50])) % 2**31)
            correct_count = sum(1 for _ in range(n) if np.random.random() < p_correct)
            total_correct += correct_count
            total_students += n
        else:
            n = max(1, round(STUDENTS_PER_ITEM * weight))
            total_students += n
            total_correct += n // 2  # fallback 50%
    return 1 - (total_correct / total_students) if total_students > 0 else 0.5


def run():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    probe = pd.read_csv(PROBE_CSV)
    print(f"Loaded {len(probe)} probe items")

    results = {}

    for temp in TEMPERATURES:
        print(f"\n=== Temperature {temp} ===")
        rep_rhos = []

        for rep in range(REPS):
            cache_file = OUTPUT_DIR / f"t{temp}_rep{rep}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                predictions = cached["predictions"]
                print(f"  Rep {rep}: loaded from cache")
            else:
                predictions = []
                for _, row in probe.iterrows():
                    prompt = build_prompt(row)
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temp,
                            max_output_tokens=512,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    raw = response.text
                    p_inc = parse_teacher_prediction(raw, row["correct_answer_kaggle"])
                    predictions.append({
                        "item": row["target_key"],
                        "qid": int(row["QuestionId"]),
                        "b_2pl": float(row["b_2pl"]),
                        "p_incorrect": p_inc,
                        "raw": raw,
                    })
                cache_file.write_text(json.dumps({"temp": temp, "rep": rep, "predictions": predictions}, indent=2))
                print(f"  Rep {rep}: collected {len(predictions)} predictions")

            # Compute correlation
            b_vals = [p["b_2pl"] for p in predictions]
            p_vals = [p["p_incorrect"] for p in predictions]
            rho, pval = stats.spearmanr(b_vals, p_vals)
            rep_rhos.append(rho)
            print(f"  Rep {rep}: ρ={rho:.3f} (p={pval:.4f})")

        mean_rho = np.mean(rep_rhos)
        sd_rho = np.std(rep_rhos)
        results[str(temp)] = {
            "reps": rep_rhos,
            "mean_rho": mean_rho,
            "sd_rho": sd_rho,
        }
        print(f"  Mean ρ={mean_rho:.3f} ± {sd_rho:.3f}")

    # Add known endpoints for context
    results["known_endpoints"] = {
        "0.3": {"mean_rho": 0.119, "source": "RSM experiment"},
        "1.5": {"mean_rho": 0.673, "source": "RSM experiment"},
    }

    summary_file = OUTPUT_DIR / "transition_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\n=== Summary saved to {summary_file} ===")

    # Print full transition curve
    print("\nFull transition curve (teacher_prediction):")
    print(f"  t=0.3: ρ=0.119 (from RSM)")
    for t in TEMPERATURES:
        r = results[str(t)]
        print(f"  t={t}: ρ={r['mean_rho']:.3f} ± {r['sd_rho']:.3f}")
    print(f"  t=1.5: ρ=0.673 (from RSM)")


if __name__ == "__main__":
    run()
