#!/usr/bin/env python3
"""
Cognitive Modeling approach to difficulty estimation:
  - Prompt the model to generate 10 incomplete chains of thought
    (where students stop or go wrong at different points)
  - Aggregate into response distributions
  - Compare to direct teacher prediction (contrastive prompt)

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

OUTPUT_DIR = Path("pilot/rsm_experiment/cognitive_modeling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CHAINS = 10
N_REPS = 3

COGNITIVE_PROMPT = """You are simulating how {n_chains} different Year 9 students would think through this maths question. For each student, show their internal reasoning — but crucially, some students will make errors, get stuck, or give up partway through.

{item_text}

For each of the {n_chains} students, write:
- Student N [ability: below_basic/basic/proficient/advanced]: Their step-by-step thinking (including where they go wrong or get stuck), then their final answer choice (A/B/C/D).

Use this distribution of ability levels across the {n_chains} students: 2-3 below_basic, 3-4 basic, 2-3 proficient, 1-2 advanced.

After all students, summarize:
ANSWER_COUNTS: A=X B=X C=X D=X"""

# Also run contrastive baseline for direct comparison
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
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ])


def parse_answer_counts(text, correct_answer):
    """Parse ANSWER_COUNTS line or count individual student answers."""
    correct_idx = ord(correct_answer) - ord('A')
    letters = ['A', 'B', 'C', 'D']

    # Try ANSWER_COUNTS summary line
    pattern = r'ANSWER_COUNTS:\s*A\s*=\s*(\d+)\s*B\s*=\s*(\d+)\s*C\s*=\s*(\d+)\s*D\s*=\s*(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        counts = [int(match.group(i)) for i in range(1, 5)]
        total = sum(counts) or 1
        p_correct = counts[correct_idx] / total
        return 1 - p_correct

    # Fallback: count individual "final answer" mentions
    counts = [0, 0, 0, 0]
    # Look for patterns like "final answer: A" or "chooses A" or "answer choice (A)"
    for m in re.finditer(r'(?:final answer|chooses?|answer choice|selects?|picks?)[:\s]*\(?([ABCD])\)?', text, re.IGNORECASE):
        idx = ord(m.group(1).upper()) - ord('A')
        counts[idx] += 1

    total = sum(counts)
    if total >= 3:  # need at least 3 parseable answers
        p_correct = counts[correct_idx] / total
        return 1 - p_correct

    return None


def parse_contrastive(text, correct_answer):
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
            max_output_tokens=4096,  # longer for chain-of-thought
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    methods = {
        "cognitive": {"prompt": COGNITIVE_PROMPT, "parser": parse_answer_counts, "temp": 1.5},
        "contrastive_baseline": {"prompt": CONTRASTIVE_PROMPT, "parser": parse_contrastive, "temp": 1.5},
    }

    for method_name, config in methods.items():
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        template = config["prompt"]
        parser = config["parser"]
        temp = config["temp"]
        all_rhos = []

        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / method_name / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            results = []
            parse_failures = 0
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    item_text = format_item_text(row)
                    prompt = template.format(item_text=item_text, n_chains=N_CHAINS)
                    try:
                        text = call_gemini(client, prompt, temp)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {method_name} rep{rep} qid={qid}: {e}")
                        time.sleep(2)
                        continue
                    time.sleep(0.15)

                p_inc = parser(text, correct)
                if p_inc is None:
                    parse_failures += 1
                    p_inc = 0.5  # fallback
                results.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                "weighted_p_incorrect": p_inc})

            df = pd.DataFrame(results)
            if len(df) >= 5:
                rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                all_rhos.append(rho)
                print(f"  {method_name} rep{rep}: rho={rho:.3f} (p={p:.4f}, parse_fail={parse_failures})")

        if all_rhos:
            print(f"\n  {method_name} MEAN: {np.mean(all_rhos):.3f} ± {np.std(all_rhos):.3f}")
            print(f"  Individual: {[f'{r:.3f}' for r in all_rhos]}")

            summary = {"method": method_name, "mean_rho": float(np.mean(all_rhos)),
                        "std_rho": float(np.std(all_rhos)), "rhos": all_rhos}
            with open(OUTPUT_DIR / f"{method_name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
