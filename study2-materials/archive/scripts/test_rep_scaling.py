#!/usr/bin/env python3
"""
Test how averaged-prediction Spearman rho scales with number of reps (0..9).
Runs reps 3-9 for the two best configs, then computes scaling curve.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/metaprompt_sweep")

TOTAL_REPS = 10  # reps 0..9

# Two best configs from sweep
CONFIGS = {
    "v5_error_analysis": {
        "temp": 2.0,
        "dir": "v5_error_analysis_t2.0",
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
    },
    "v3_contrastive": {
        "temp": 1.5,
        "dir": "v3_contrastive_t1.5",
        "prompt": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",
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


def make_api_call(client, prompt, temperature):
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def compute_averaged_rho(probe, config_dir, n_reps):
    """Compute Spearman rho using averaged predictions across reps 0..n_reps-1."""
    # Collect predictions per item across reps
    item_preds = {}  # qid -> list of p_incorrect values
    for _, row in probe.iterrows():
        qid = row["QuestionId"]
        correct = row["correct_answer_kaggle"]
        preds = []
        for rep in range(n_reps):
            raw_path = config_dir / f"rep{rep}" / f"qid{qid}.txt"
            if raw_path.exists():
                text = raw_path.read_text()
                preds.append(parse_predictions(text, correct))
        if preds:
            item_preds[qid] = np.mean(preds)

    # Build dataframe and correlate
    rows = [{"QuestionId": qid, "avg_p_inc": p, "b_2pl": probe.loc[probe["QuestionId"] == qid, "b_2pl"].values[0]}
            for qid, p in item_preds.items()]
    df = pd.DataFrame(rows).dropna()
    if len(df) >= 5:
        rho, p = stats.spearmanr(df["avg_p_inc"], df["b_2pl"])
        return rho, p
    return None, None


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    n_items = len(probe)
    print(f"Loaded {n_items} probe items", flush=True)

    for cname, cfg in CONFIGS.items():
        config_dir = OUTPUT_DIR / cfg["dir"]
        temp = cfg["temp"]
        template = cfg["prompt"]

        print(f"\n{'='*60}", flush=True)
        print(f"CONFIG: {cname} (temp={temp})", flush=True)
        print(f"{'='*60}", flush=True)

        # Run reps 3..9 (reps 0-2 already cached)
        for rep in range(3, TOTAL_REPS):
            raw_dir = config_dir / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            cached = sum(1 for _, r in probe.iterrows()
                         if (raw_dir / f"qid{r['QuestionId']}.txt").exists())
            if cached == n_items:
                print(f"  rep{rep}: all {n_items} cached, skipping API calls", flush=True)
            else:
                print(f"  rep{rep}: running {n_items - cached} new calls...", flush=True)
                for _, row in probe.iterrows():
                    qid = row["QuestionId"]
                    raw_path = raw_dir / f"qid{qid}.txt"
                    if raw_path.exists():
                        continue
                    item_text = format_item_text(row)
                    prompt = template.format(item_text=item_text)
                    try:
                        text = make_api_call(client, prompt, temp)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"    ERROR rep{rep} qid={qid}: {e}", flush=True)
                        time.sleep(3)
                        continue
                    time.sleep(0.15)

            # After this rep, compute scaling curve so far
            rho, p = compute_averaged_rho(probe, config_dir, rep + 1)
            if rho is not None:
                print(f"  reps=0..{rep} (N={rep+1}): averaged rho = {rho:.4f} (p={p:.4f})", flush=True)

    # Final summary table
    print(f"\n{'='*60}", flush=True)
    print("SCALING TABLE: N_reps -> averaged rho", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"{'N_reps':<10}"
    for cname in CONFIGS:
        header += f"  {cname:<25}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for n in range(1, TOTAL_REPS + 1):
        line = f"{n:<10}"
        for cname, cfg in CONFIGS.items():
            config_dir = OUTPUT_DIR / cfg["dir"]
            rho, _ = compute_averaged_rho(probe, config_dir, n)
            if rho is not None:
                line += f"  {rho:<25.4f}"
            else:
                line += f"  {'N/A':<25}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
