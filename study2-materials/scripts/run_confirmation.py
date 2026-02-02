#!/usr/bin/env python3
"""
Confirmation experiment: Run the two best prompt configs from the optimization
phase on held-out Eedi items that were never used during screening or tuning.

This follows a standard industrial DOE progression:
  1. Screening  — 5 methods × 6 models on full item set → all r≈0
  2. Optimization — RSM + metaprompt sweep on 20 probe items → ρ≈0.55-0.66
  3. Confirmation — Best 2 configs on 58 held-out items (this script)

Configs tested:
  - v3_contrastive @ temperature=1.5  (most stable: ρ=0.577±0.075 on probe)
  - v5_error_analysis @ temperature=2.0  (highest mean: ρ=0.604±0.062 on probe)

Each config is run 3 times (independent reps via high temperature) on 58 items
that have IRT parameters but were excluded from the 20-item probe set.

Items: 58 held-out from curated Eedi set (excludes INVERSE_OPS misconception
due to near-ceiling mastery). Covers ORDER_OPS (26), NEG_MULTIPLY (21),
FRAC_ADD (11).

Total API calls: 58 items × 2 configs × 3 reps = 348 calls.

Usage:
    python3 scripts/run_confirmation.py                # run everything
    python3 scripts/run_confirmation.py --config v3    # run only contrastive
    python3 scripts/run_confirmation.py --config v5    # run only error_analysis
    python3 scripts/run_confirmation.py --dry-run      # show items, no API calls
"""

import json, re, os, time, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/confirmation_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

# ---------------------------------------------------------------------------
# The two winning configs from the optimization phase
# ---------------------------------------------------------------------------
CONFIGS = {
    "v3_contrastive_t1.5": {
        "temperature": 1.5,
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
    "v5_error_analysis_t2.0": {
        "temperature": 2.0,
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
}


def load_held_out_items():
    """Load the 58 held-out items with IRT parameters.

    Selection criteria:
    - In curated_eedi_items.csv (expert-labeled misconception items)
    - Has IRT parameters in irt_4pl_statistics.json
    - NOT in the 20-item probe set used for optimization
    - NOT INVERSE_OPS misconception (near-ceiling mastery, excluded from study)

    Note on answer ordering: The LLM sees answer texts in Kaggle ordering
    (A={AnswerAText}, B={AnswerBText}, etc.), so we score against
    correct_answer_kaggle. The NeurIPS ordering (neurips_correct_pos) refers
    to screen position students saw, which is irrelevant for LLM scoring.
    The b_2pl parameter is a property of the item regardless of display order.
    """
    curated = pd.read_csv("data/eedi/curated_eedi_items.csv")
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")
    probe_ids = set(probe["QuestionId"].tolist())

    with open("results/irt_4pl_statistics.json") as f:
        irt_data = json.load(f)["items"]
    irt_ids = set(int(k) for k in irt_data.keys())

    held_out = curated[
        curated["QuestionId"].isin(irt_ids)
        & ~curated["QuestionId"].isin(probe_ids)
        & (curated["target_key"] != "INVERSE_OPS")
    ].copy()

    # Merge IRT parameters
    held_out["b_2pl"] = held_out["QuestionId"].apply(
        lambda qid: irt_data[str(qid)]["b_2pl"]
    )
    held_out["a_2pl"] = held_out["QuestionId"].apply(
        lambda qid: irt_data[str(qid)]["a_2pl"]
    )

    return held_out


def format_item_text(row):
    return "\n".join([
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ])


def parse_predictions(text, correct_answer):
    """Parse teacher-prediction response into weighted p(incorrect).

    Weights reflect a typical UK Year 9 class ability distribution:
    below_basic=25%, basic=35%, proficient=25%, advanced=15%.
    """
    weights = {
        "below_basic": 0.25,
        "basic": 0.35,
        "proficient": 0.25,
        "advanced": 0.15,
    }
    correct_idx = ord(correct_answer) - ord("A")
    weighted_p_correct = 0.0
    parsed_levels = 0
    for level, w in weights.items():
        pattern = (
            rf'{level.replace("_", "[_ ]")}:\s*'
            rf"A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?"
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
            parsed_levels += 1
        else:
            weighted_p_correct += 0.5 * w  # fallback: assume chance
    return 1 - weighted_p_correct, parsed_levels


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


def run_config(client, config_name, config, items, dry_run=False):
    """Run one config (prompt+temp) for N_REPS on all items. Returns per-rep results."""
    template = config["prompt"]
    temp = config["temperature"]
    rep_results = []

    for rep in range(N_REPS):
        raw_dir = OUTPUT_DIR / config_name / f"rep{rep}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        predictions = []
        parse_failures = 0

        for _, row in items.iterrows():
            qid = row["QuestionId"]
            correct = row["correct_answer_kaggle"]
            raw_path = raw_dir / f"qid{qid}.txt"

            if raw_path.exists():
                text = raw_path.read_text()
            elif dry_run:
                continue
            else:
                item_text = format_item_text(row)
                prompt = template.format(item_text=item_text)
                try:
                    text = make_api_call(client, prompt, temp)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"  ERROR {config_name} rep{rep} qid={qid}: {e}", flush=True)
                    time.sleep(2)
                    continue
                time.sleep(0.15)

            p_inc, n_parsed = parse_predictions(text, correct)
            if n_parsed < 3:
                parse_failures += 1
            predictions.append({
                "QuestionId": qid,
                "b_2pl": row["b_2pl"],
                "a_2pl": row["a_2pl"],
                "target_key": row["target_key"],
                "weighted_p_incorrect": p_inc,
                "levels_parsed": n_parsed,
            })

        df = pd.DataFrame(predictions)
        if dry_run:
            continue

        # Save per-rep predictions
        df.to_csv(raw_dir / "predictions.csv", index=False)

        valid = df.dropna(subset=["weighted_p_incorrect", "b_2pl"])
        if len(valid) >= 5:
            rho, p = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
            r, r_p = stats.pearsonr(valid["weighted_p_incorrect"], valid["b_2pl"])
            rep_results.append({
                "rep": rep,
                "n_items": len(valid),
                "parse_failures": parse_failures,
                "spearman_rho": rho,
                "spearman_p": p,
                "pearson_r": r,
                "pearson_p": r_p,
            })
            print(
                f"  {config_name} rep{rep}: ρ={rho:.3f} (p={p:.3f}), "
                f"r={r:.3f}, n={len(valid)}, parse_fail={parse_failures}",
                flush=True,
            )

    return rep_results


def compute_averaged_predictions(config_name, items):
    """Average predictions across reps for each item, then correlate."""
    all_preds = {}
    for rep in range(N_REPS):
        pred_path = OUTPUT_DIR / config_name / f"rep{rep}" / "predictions.csv"
        if not pred_path.exists():
            return None
        df = pd.read_csv(pred_path)
        for _, row in df.iterrows():
            qid = row["QuestionId"]
            if qid not in all_preds:
                all_preds[qid] = {"b_2pl": row["b_2pl"], "preds": []}
            all_preds[qid]["preds"].append(row["weighted_p_incorrect"])

    rows = []
    for qid, data in all_preds.items():
        if len(data["preds"]) == N_REPS:
            rows.append({
                "QuestionId": qid,
                "b_2pl": data["b_2pl"],
                "mean_p_incorrect": np.mean(data["preds"]),
                "std_p_incorrect": np.std(data["preds"]),
            })

    df = pd.DataFrame(rows)
    if len(df) < 5:
        return None
    rho, p = stats.spearmanr(df["mean_p_incorrect"], df["b_2pl"])
    r, r_p = stats.pearsonr(df["mean_p_incorrect"], df["b_2pl"])

    # Save averaged predictions
    df.to_csv(OUTPUT_DIR / config_name / "averaged_predictions.csv", index=False)

    return {
        "n_items": len(df),
        "spearman_rho": rho,
        "spearman_p": p,
        "pearson_r": r,
        "pearson_p": r_p,
    }


def main():
    parser = argparse.ArgumentParser(description="Confirmation experiment on held-out Eedi items")
    parser.add_argument("--config", choices=["v3", "v5", "both"], default="both",
                        help="Which config(s) to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show item counts and exit without API calls")
    args = parser.parse_args()

    items = load_held_out_items()
    print(f"Held-out items: {len(items)}")
    print(f"  ORDER_OPS: {(items['target_key'] == 'ORDER_OPS').sum()}")
    print(f"  NEG_MULTIPLY: {(items['target_key'] == 'NEG_MULTIPLY').sum()}")
    print(f"  FRAC_ADD: {(items['target_key'] == 'FRAC_ADD').sum()}")
    print(f"  Difficulty range: [{items['b_2pl'].min():.2f}, {items['b_2pl'].max():.2f}]")
    print()

    if args.dry_run:
        print("DRY RUN — no API calls")
        print(f"Would run: {len(items)} items × 2 configs × {N_REPS} reps = {len(items)*2*N_REPS} calls")
        return

    configs_to_run = {}
    if args.config in ("v3", "both"):
        configs_to_run["v3_contrastive_t1.5"] = CONFIGS["v3_contrastive_t1.5"]
    if args.config in ("v5", "both"):
        configs_to_run["v5_error_analysis_t2.0"] = CONFIGS["v5_error_analysis_t2.0"]

    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    all_results = {}
    for config_name, config in configs_to_run.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")

        rep_results = run_config(client, config_name, config, items)
        avg_result = compute_averaged_predictions(config_name, items)

        all_results[config_name] = {
            "per_rep": rep_results,
            "averaged": avg_result,
        }

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CONFIRMATION EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Items: {len(items)} held-out (never seen during optimization)")
    print()

    for config_name, results in all_results.items():
        reps = results["per_rep"]
        avg = results["averaged"]
        if not reps:
            print(f"{config_name}: NO RESULTS")
            continue

        rhos = [r["spearman_rho"] for r in reps]
        print(f"{config_name}:")
        print(f"  Per-rep ρ: {', '.join(f'{r:.3f}' for r in rhos)}")
        print(f"  Mean ρ: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        if avg:
            print(f"  Averaged predictions ρ: {avg['spearman_rho']:.3f} (p={avg['spearman_p']:.4f})")
            print(f"  Averaged predictions r: {avg['pearson_r']:.3f} (p={avg['pearson_p']:.4f})")
        print()

    # Compare to probe-set results from optimization phase
    print("COMPARISON TO OPTIMIZATION PHASE (20 probe items):")
    print("  v3_contrastive t=1.5:    ρ=0.577 ± 0.075 (avg ρ=0.660)")
    print("  v5_error_analysis t=2.0: ρ=0.604 ± 0.062 (avg ρ=0.666)")
    print()

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
