#!/usr/bin/env python3
"""
Test whether enabling Gemini's thinking mode improves difficulty estimation.
Baseline: v5_error_analysis at temp=2.0 with thinking_budget=0 (rho=0.604+-0.062).
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

BASELINE_DIR = Path("pilot/rsm_experiment/metaprompt_sweep/v5_error_analysis_t2.0")
OUTPUT_DIR = Path("pilot/rsm_experiment/thinking_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3
THINKING_BUDGETS = [0, 1024, 4096, 8192]

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
    parsed_levels = 0
    for level, w in weights.items():
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
            parsed_levels += 1
        else:
            weighted_p_correct += 0.5 * w
    return 1 - weighted_p_correct, parsed_levels


def make_api_call(client, prompt, thinking_budget):
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=2.0,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=config,
    )
    # When thinking is enabled, extract non-thinking parts
    if thinking_budget > 0:
        text = ""
        for part in response.candidates[0].content.parts:
            if not hasattr(part, 'thought') or not part.thought:
                text += part.text
        return text
    return response.text


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    n_items = len(probe)
    # budget=0 is cached, rest need API calls
    n_calls = sum(1 for b in THINKING_BUDGETS if b > 0) * N_REPS * n_items
    print(f"Testing thinking_budgets={THINKING_BUDGETS}, {N_REPS} reps, {n_items} items", flush=True)
    print(f"API calls needed: {n_calls} (budget=0 cached from baseline)", flush=True)

    all_results = []

    for budget in THINKING_BUDGETS:
        print(f"\n=== thinking_budget={budget} ===", flush=True)
        rhos = []
        all_preds = []  # for averaged-prediction rho

        for rep in range(N_REPS):
            if budget == 0:
                raw_dir = BASELINE_DIR / f"rep{rep}"
            else:
                raw_dir = OUTPUT_DIR / f"budget_{budget}" / f"rep{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)

            items_pred = []
            parse_failures = 0
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    item_text = format_item_text(row)
                    prompt = PROMPT_TEMPLATE.format(item_text=item_text)
                    try:
                        text = make_api_call(client, prompt, budget)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR budget={budget} rep{rep} qid={qid}: {e}", flush=True)
                        time.sleep(2)
                        continue
                    time.sleep(0.15)

                p_inc, n_parsed = parse_predictions(text, correct)
                if n_parsed < 3:
                    parse_failures += 1
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})

            df = pd.DataFrame(items_pred)
            valid = df.dropna()
            if len(valid) >= 5:
                rho, p = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
                rhos.append(rho)
                all_preds.append(valid[["QuestionId", "weighted_p_incorrect"]].copy())
                print(f"  budget={budget} rep{rep}: rho={rho:.3f} (p={p:.3f}, parse_fail={parse_failures})", flush=True)

        # Averaged-prediction rho
        avg_rho = np.nan
        if len(all_preds) == N_REPS:
            merged = all_preds[0].rename(columns={"weighted_p_incorrect": "p0"})
            for i in range(1, N_REPS):
                merged = merged.merge(
                    all_preds[i].rename(columns={"weighted_p_incorrect": f"p{i}"}),
                    on="QuestionId"
                )
            merged["p_avg"] = merged[[f"p{i}" for i in range(N_REPS)]].mean(axis=1)
            b_vals = probe.set_index("QuestionId").loc[merged["QuestionId"], "b_2pl"].values
            avg_rho, _ = stats.spearmanr(merged["p_avg"], b_vals)

        mean_rho = np.mean(rhos) if rhos else np.nan
        std_rho = np.std(rhos) if rhos else np.nan
        all_results.append({
            "thinking_budget": budget,
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "avg_rho": avg_rho,
            "rhos": rhos,
        })
        print(f"  MEAN: {mean_rho:.3f} +/- {std_rho:.3f}, averaged-pred rho: {avg_rho:.3f}", flush=True)

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print("THINKING BUDGET RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Budget':<12} {'Mean rho':<12} {'SD':<10} {'Avg-pred rho':<14} {'Per-rep rhos'}", flush=True)
    print("-" * 70, flush=True)
    for r in all_results:
        rho_strs = [f"{x:.3f}" for x in r["rhos"]]
        print(f"{r['thinking_budget']:<12} {r['mean_rho']:<12.3f} {r['std_rho']:<10.3f} {r['avg_rho']:<14.3f} {rho_strs}", flush=True)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
