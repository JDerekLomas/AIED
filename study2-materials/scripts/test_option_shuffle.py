#!/usr/bin/env python3
"""
Test whether shuffling MCQ option order across reps increases effective diversity.

Hypothesis: Presenting options in different orders forces different reasoning paths,
creating diversity orthogonal to temperature. When averaged, this should improve
calibration (higher Spearman ρ vs IRT b_2pl).

Design:
- Use v5_error_analysis prompt at temp=2.0 (our best config)
- 6 reps: rep0-2 = original order (cached), rep3-5 = shuffled orders
- For shuffled reps, permute A/B/C/D and map predictions back to original labels
- Compare: avg(reps 0-2) vs avg(reps 0-5) vs avg(shuffled only)
"""
import json, re, os, time, itertools, random
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/option_shuffle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHED_DIR = Path("pilot/rsm_experiment/metaprompt_sweep/v5_error_analysis_t2.0")

PROMPT_TEMPLATE = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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

# Fixed permutations for reproducibility (excluding identity ABCD)
SHUFFLES = [
    ['B', 'D', 'A', 'C'],  # rep3
    ['C', 'A', 'D', 'B'],  # rep4
    ['D', 'C', 'B', 'A'],  # rep5
]


def format_item_shuffled(row, perm):
    """Format item with shuffled option order. perm maps new position -> original label."""
    orig_texts = {
        'A': row['AnswerAText'],
        'B': row['AnswerBText'],
        'C': row['AnswerCText'],
        'D': row['AnswerDText'],
    }
    labels = ['A', 'B', 'C', 'D']
    lines = [f"Question: {row['question_text']}"]
    for i, new_label in enumerate(labels):
        orig_label = perm[i]
        lines.append(f"{new_label}) {orig_texts[orig_label]}")
    return "\n".join(lines)


def parse_predictions_shuffled(text, correct_answer_orig, perm):
    """Parse predictions and unmap back to original option order.
    perm[i] = original label at new position i.
    So if perm = [B,D,A,C], position 0 (A in output) = original B."""
    weights = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}

    # Build reverse map: original_label -> new_position
    labels = ['A', 'B', 'C', 'D']
    orig_to_new = {}
    for i, orig in enumerate(perm):
        orig_to_new[orig] = i  # original label -> index in shuffled output

    correct_idx_orig = ord(correct_answer_orig) - ord('A')
    weighted_p_correct = 0.0
    parsed_levels = 0

    for level, w in weights.items():
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # These are percentages for A,B,C,D in the SHUFFLED order
            shuffled_pcts = [int(match.group(i)) for i in range(1, 5)]
            # Unmap to original order
            orig_pcts = [0, 0, 0, 0]
            for new_pos in range(4):
                orig_label = perm[new_pos]
                orig_idx = ord(orig_label) - ord('A')
                orig_pcts[orig_idx] = shuffled_pcts[new_pos]

            total = sum(orig_pcts) or 1
            weighted_p_correct += (orig_pcts[correct_idx_orig] / total) * w
            parsed_levels += 1
        else:
            weighted_p_correct += 0.5 * w

    return 1 - weighted_p_correct, parsed_levels


def parse_predictions_orig(text, correct_answer):
    """Parse with original ABCD order (identity permutation)."""
    return parse_predictions_shuffled(text, correct_answer, ['A', 'B', 'C', 'D'])


def main():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    print("=== Option Shuffle Experiment ===", flush=True)
    print(f"Reps 0-2: cached (original order), Reps 3-5: shuffled orders", flush=True)
    print(f"Shuffles: {SHUFFLES}", flush=True)
    print(f"Items: {len(probe)}, API calls needed: {3 * len(probe)}\n", flush=True)

    # Collect all predictions: {qid: {rep: p_incorrect}}
    all_preds = []

    # Reps 0-2: load from cache
    for rep in range(3):
        parse_failures = 0
        for _, row in probe.iterrows():
            qid = row["QuestionId"]
            correct = row["correct_answer_kaggle"]
            raw_path = CACHED_DIR / f"rep{rep}" / f"qid{qid}.txt"

            if raw_path.exists():
                text = raw_path.read_text()
                p_inc, n_parsed = parse_predictions_orig(text, correct)
                if n_parsed < 3:
                    parse_failures += 1
                all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                  "rep": rep, "shuffled": False, "p_inc": p_inc})
            else:
                print(f"  WARNING: missing cache for rep{rep} qid{qid}", flush=True)

        # Compute per-rep rho
        df = pd.DataFrame([p for p in all_preds if p["rep"] == rep])
        if len(df) >= 5:
            rho, p = stats.spearmanr(df["p_inc"], df["b_2pl"])
            print(f"  rep{rep} (original): rho={rho:.3f} (parse_fail={parse_failures})", flush=True)

    # Reps 3-5: shuffled
    for shuffle_idx, perm in enumerate(SHUFFLES):
        rep = shuffle_idx + 3
        raw_dir = OUTPUT_DIR / f"rep{rep}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        parse_failures = 0
        for _, row in probe.iterrows():
            qid = row["QuestionId"]
            correct = row["correct_answer_kaggle"]
            raw_path = raw_dir / f"qid{qid}.txt"
            perm_path = raw_dir / f"qid{qid}_perm.json"

            if raw_path.exists():
                text = raw_path.read_text()
            else:
                item_text = format_item_shuffled(row, perm)
                prompt = PROMPT_TEMPLATE.format(item_text=item_text)
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=2.0,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    text = response.text
                    raw_path.write_text(text)
                    perm_path.write_text(json.dumps(perm))
                except Exception as e:
                    print(f"  ERROR rep{rep} qid={qid}: {e}", flush=True)
                    time.sleep(2)
                    continue
                time.sleep(0.15)

            p_inc, n_parsed = parse_predictions_shuffled(text, correct, perm)
            if n_parsed < 3:
                parse_failures += 1
            all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                              "rep": rep, "shuffled": True, "p_inc": p_inc})

        df = pd.DataFrame([p for p in all_preds if p["rep"] == rep])
        if len(df) >= 5:
            rho, p = stats.spearmanr(df["p_inc"], df["b_2pl"])
            print(f"  rep{rep} (shuffled {perm}): rho={rho:.3f} (parse_fail={parse_failures})", flush=True)

    # Analysis
    pdf = pd.DataFrame(all_preds)

    print(f"\n{'='*60}", flush=True)
    print("AGGREGATION COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)

    configs = {
        "Original only (reps 0-2)": pdf[pdf["rep"].isin([0, 1, 2])],
        "Shuffled only (reps 3-5)": pdf[pdf["rep"].isin([3, 4, 5])],
        "All 6 reps (0-5)": pdf,
        "Best 2 orig + 1 shuffled": pdf[pdf["rep"].isin([0, 1, 3])],
    }

    for label, subset in configs.items():
        avg = subset.groupby("QuestionId").agg(
            mean_p_inc=("p_inc", "mean"),
            b_2pl=("b_2pl", "first")
        ).dropna()
        if len(avg) >= 5:
            rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
            print(f"  {label}: rho={rho:.3f} (p={p:.4f})", flush=True)

    # Scaling: how does ρ change as we add reps?
    print(f"\n{'='*60}", flush=True)
    print("REP SCALING (cumulative)", flush=True)
    print(f"{'='*60}", flush=True)

    for n in range(1, 7):
        subset = pdf[pdf["rep"] < n]
        avg = subset.groupby("QuestionId").agg(
            mean_p_inc=("p_inc", "mean"),
            b_2pl=("b_2pl", "first")
        ).dropna()
        if len(avg) >= 5:
            rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
            shuffled_count = sum(1 for r in range(n) if r >= 3)
            print(f"  N={n} ({n - shuffled_count} orig + {shuffled_count} shuffled): rho={rho:.3f}", flush=True)

    # Save
    pdf.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
