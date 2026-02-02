#!/usr/bin/env python3
"""
Run contrastive prompt on all 105 items with b_2pl IRT parameters.
Gemini 3 Flash, 3 reps, t=1.5.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/b2pl_105_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3
TEMP = 1.5

CONTRASTIVE_PROMPT = """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""


def format_item_text(row):
    return "\n".join([
        f"Question: {row['QuestionText']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ])


def parse_predictions(text, correct_answer):
    weights = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}
    correct_idx = ord(correct_answer) - ord('A')
    weighted_p_correct = 0.0
    parsed = 0
    for level, w in weights.items():
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
            parsed += 1
        else:
            weighted_p_correct += 0.5 * w
    return 1 - weighted_p_correct, parsed


def call_gemini(prompt, temp):
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


def main():
    # Load items
    with open("results/irt_proper_statistics.json") as f:
        irt = json.load(f)["items"]

    eedi = pd.read_csv("data/eedi/eedi_with_student_data.csv")
    irt_ids = [int(k) for k in irt.keys()]
    items = eedi[eedi["QuestionId"].isin(irt_ids)].copy()
    items["b_2pl"] = items["QuestionId"].map(lambda x: irt[str(x)]["b_2pl"])

    items["correct_letter"] = items["CorrectAnswer"]  # Already A/B/C/D

    print(f"Items with b_2pl: {len(items)}")
    print(f"b_2pl range: {items['b_2pl'].min():.2f} to {items['b_2pl'].max():.2f}")
    print(f"Running Gemini 3 Flash, {N_REPS} reps, t={TEMP}\n")

    all_results = []
    rhos = []

    for rep in range(N_REPS):
        raw_dir = OUTPUT_DIR / f"gemini_flash/rep{rep}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        rep_preds = []
        n_parsed = 0
        n_total = 0

        for _, row in items.iterrows():
            qid = row["QuestionId"]
            correct = row["correct_letter"]
            if pd.isna(correct):
                continue

            raw_path = raw_dir / f"qid{qid}.txt"
            n_total += 1

            if raw_path.exists():
                text = raw_path.read_text()
            else:
                item_text = format_item_text(row)
                prompt = CONTRASTIVE_PROMPT.format(item_text=item_text)
                try:
                    text = call_gemini(prompt, TEMP)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"  ERROR rep{rep} qid={qid}: {e}")
                    time.sleep(2)
                    continue
                time.sleep(0.15)

            p_inc, parsed = parse_predictions(text, correct)
            if parsed >= 3:
                n_parsed += 1
            rep_preds.append({
                "QuestionId": qid, "b_2pl": row["b_2pl"],
                "p_incorrect": p_inc, "rep": rep, "levels_parsed": parsed
            })

        df = pd.DataFrame(rep_preds)
        valid = df[df["levels_parsed"] >= 3]
        rho, p = stats.spearmanr(valid["p_incorrect"], valid["b_2pl"])
        rhos.append(rho)
        all_results.extend(rep_preds)
        print(f"Rep {rep}: rho={rho:.3f} (p={p:.4f}), parsed={n_parsed}/{n_total} ({100*n_parsed/n_total:.0f}%)")

    # Averaged predictions
    adf = pd.DataFrame(all_results)
    avg = adf[adf["levels_parsed"] >= 3].groupby("QuestionId").agg(
        mean_p_inc=("p_incorrect", "mean"),
        b_2pl=("b_2pl", "first"),
        n_reps=("rep", "count")
    )
    avg_rho, avg_p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])

    print(f"\nPer-rep: mean rho={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    print(f"Averaged ({len(avg)} items, {N_REPS}-rep mean): rho={avg_rho:.3f} (p={avg_p:.6f})")

    # Bootstrap CI
    n_boot = 1000
    boot_rhos = []
    for _ in range(n_boot):
        sample = avg.sample(n=len(avg), replace=True)
        r, _ = stats.spearmanr(sample["mean_p_inc"], sample["b_2pl"])
        boot_rhos.append(r)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save
    summary = {
        "n_items": len(avg),
        "n_reps": N_REPS,
        "temperature": TEMP,
        "model": "gemini-3-flash-preview",
        "prompt": "contrastive",
        "per_rep_rhos": rhos,
        "per_rep_mean": float(np.mean(rhos)),
        "per_rep_std": float(np.std(rhos)),
        "averaged_rho": float(avg_rho),
        "averaged_p": float(avg_p),
        "ci_95": [float(ci_lo), float(ci_hi)],
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    avg.to_csv(OUTPUT_DIR / "averaged_predictions.csv")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
