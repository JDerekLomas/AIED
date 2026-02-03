#!/usr/bin/env python3
"""
Phase 3: Cross-Dataset Confirmation — DBE-KT22
================================================
Tests top Phase 1 prompt framings on held-out dataset.

DBE-KT22: South African university database systems MCQs, 168 usable items.
Population: undergraduate CS students. Format: MCQ with 3-5 options.

Prompts adapted from Phase 1 SmartPaper winners:
  1. teacher — baseline direct estimation
  2. devil_advocate — bias correction ("experts overestimate")
  3. prerequisite_chain — KC Theory: count prerequisite failure points
  4. cognitive_load — Sweller CLT: count elements in working memory
  5. error_analysis — error-focused framing

No base-rate leakage. No population statistics.

Design: 5 prompts × 2 temps × 3 reps × 168 items = 5,040 calls on Gemini 3 Flash.

Pre-specified hypotheses (written before seeing results):
  H1: Top SmartPaper config (prerequisite_chain) produces ρ > 0.30 on DBE-KT22
  H2: SmartPaper ρ > DBE-KT22 ρ for every prompt tested
  H3: Prompt ranking is preserved across datasets
  H4: prerequisite_chain > teacher on DBE-KT22 (theory-driven framing transfers)

Usage:
  python3 scripts/run_phase3_dbe_kt22.py
  python3 scripts/run_phase3_dbe_kt22.py --framing teacher
  python3 scripts/run_phase3_dbe_kt22.py --analyze
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

DATA_DIR = Path("data/dbe-kt22")
OUTPUT_DIR = Path("pilot/phase3_dbe_kt22")
MODEL = "gemini-3-flash-preview"
N_REPS = 3
TEMPERATURES = [1.0, 2.0]

POPULATION = ("These are undergraduate computer science students at a South African university. "
              "Ability varies widely — some students have strong programming backgrounds, "
              "others are taking this as a required course with minimal prior exposure to databases.")


def load_items():
    """Load DBE-KT22 items with empirical p-correct."""
    questions = pd.read_csv(DATA_DIR / "Questions.csv")
    choices = pd.read_csv(DATA_DIR / "Question_Choices.csv")
    transactions = pd.read_csv(DATA_DIR / "Transaction.csv")

    empirical = transactions.groupby("question_id").agg(
        n_responses=("answer_state", "count"),
        n_correct=("answer_state", "sum"),
    ).reset_index()
    empirical["p_correct"] = empirical["n_correct"] / empirical["n_responses"]
    questions = questions.merge(empirical, left_on="id", right_on="question_id", how="left")

    # Build choice text
    choice_map = {}
    for qid, group in choices.groupby("question_id"):
        group = group.sort_values("id")
        opts = []
        correct_label = None
        for i, (_, row) in enumerate(group.iterrows()):
            label = chr(65 + i)
            opts.append(f"{label}. {row['choice_text']}")
            if row["is_correct"]:
                correct_label = label
        choice_map[qid] = {
            "options_text": "\n".join(opts),
            "correct_label": correct_label,
        }

    questions["options_text"] = questions["id"].map(lambda x: choice_map.get(x, {}).get("options_text", ""))
    questions["correct_label"] = questions["id"].map(lambda x: choice_map.get(x, {}).get("correct_label", ""))

    # Filter usable
    has_img = questions["question_rich_text"].str.contains("<img", na=False)
    usable = questions[
        (questions["question_text"].str.len() > 20) &
        (questions["n_responses"] >= 50) &
        (~has_img)
    ].copy()

    print(f"Items: {len(usable)} usable / {len(questions)} total")
    print(f"Empirical p_correct: {usable['p_correct'].mean():.2f} mean, "
          f"{usable['p_correct'].min():.2f}–{usable['p_correct'].max():.2f} range", flush=True)

    items = []
    for _, row in usable.iterrows():
        items.append({
            "id": int(row["id"]),
            "question_text": row["question_text"],
            "options_text": row["options_text"],
            "correct_label": row["correct_label"],
            "p_correct": float(row["p_correct"]),
        })
    return items


def call_llm(prompt, temperature=2.0):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=512,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            return response.text
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2 ** attempt)
    return ""


def parse_proportion(text):
    """Parse a proportion (0-1) from LLM output."""
    if not text:
        return None
    text = text.strip()

    # "estimate: 0.45" or "Your estimate: 0.45"
    m = re.search(r'(?:estimate|answer|proportion|correct)\s*:?\s*(0\.\d+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "PROPORTION CORRECT: XX%"
    m = re.search(r'PROPORTION CORRECT\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0

    # "CORRECT: XX%"
    m = re.search(r'CORRECT\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0

    # Bare decimal on own line
    m = re.search(r'^(0\.\d+)$', text, re.MULTILINE)
    if m:
        return float(m.group(1))

    # Last percentage
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        return float(matches[-1]) / 100.0

    # Last bare decimal
    matches = re.findall(r'\b(0\.\d+)\b', text)
    if matches:
        return float(matches[-1])

    return None


# ============================================================
# PROMPTS — adapted for DBE-KT22 (MCQ, university, CS)
# ============================================================

def prompt_teacher(item):
    """Direct teacher estimation — baseline."""
    return f"""You are an experienced university instructor in database systems.

{POPULATION}

For this multiple-choice question, estimate what proportion of students would answer correctly.

Question: {item['question_text']}

{item['options_text']}

Think about:
- What specific misconceptions would lead students to choose wrong options?
- How clearly does the question communicate what's being asked?
- What prerequisite knowledge is needed?
- How likely are students at this level to have that knowledge?

Respond with ONLY a number between 0 and 1 representing the proportion who would answer correctly.
For example: 0.45

Your estimate:"""


def prompt_devil_advocate(item):
    """Bias correction — challenges expert overconfidence."""
    return f"""You are an experienced university instructor in database systems.

{POPULATION}

IMPORTANT: Instructors and experts consistently OVERESTIMATE how well students will do because they forget:
- How many students have weak foundational skills
- That "basic" concepts aren't basic for struggling learners
- That exam anxiety causes random guessing
- That many students have never practiced this exact question type

Challenge your first instinct. If you think "most students could do this," ask yourself: could the weakest students in this course?

Question: {item['question_text']}

{item['options_text']}

After challenging your assumptions, estimate what proportion would answer correctly.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


def prompt_prerequisite_chain(item):
    """KC Theory — count prerequisite failure points."""
    return f"""You are an experienced university instructor in database systems.

{POPULATION}

For this question, identify the prerequisite knowledge and skills a student needs. Count how many independent things must ALL go right for a correct answer. Each prerequisite is a potential failure point.

Examples of prerequisites: understanding a SQL keyword, knowing a definition, multi-step reasoning, reading a schema diagram, distinguishing similar concepts, applying a rule to a novel case.

Question: {item['question_text']}

{item['options_text']}

List the prerequisites, then estimate what proportion would answer correctly.

PREREQUISITES: [list them]
COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_cognitive_load(item):
    """Sweller CLT — element interactivity."""
    return f"""You are an experienced university instructor in database systems.

{POPULATION}

Rate the element interactivity of this question: how many pieces of information must a student hold in working memory simultaneously to answer it?

Elements include: SQL syntax rules, table relationships, definition recall, logical conditions to evaluate, steps in a procedure, constraints to satisfy, distinctions between similar concepts.

Question: {item['question_text']}

{item['options_text']}

List the elements, then estimate what proportion would answer correctly.

ELEMENTS: [list them]
ELEMENT COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_error_analysis(item):
    """Error-focused framing."""
    return f"""You are an experienced university instructor in database systems.

{POPULATION}

For this question, think about what ERRORS would lead students to choose wrong options:
- What misconceptions about databases would cause incorrect answers?
- Which wrong option is the most tempting, and why?
- Would students guess randomly or make systematic errors?
- What would the most common wrong answer be?

Question: {item['question_text']}

{item['options_text']}

After analyzing likely errors, estimate what proportion would answer correctly.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


# ============================================================
# Registry
# ============================================================

FRAMINGS = {
    "teacher": prompt_teacher,
    "devil_advocate": prompt_devil_advocate,
    "prerequisite_chain": prompt_prerequisite_chain,
    "cognitive_load": prompt_cognitive_load,
    "error_analysis": prompt_error_analysis,
}


def run_experiment(only_framing=None):
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    framings = {only_framing: FRAMINGS[only_framing]} if only_framing else FRAMINGS
    temps = TEMPERATURES

    all_results = {}
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    total_configs = len(framings) * len(temps)
    config_idx = 0
    total_api = 0

    for framing_name, prompt_fn in framings.items():
        for temp in temps:
            config_idx += 1
            config_name = f"{framing_name}_t{temp}"
            config_dir = OUTPUT_DIR / config_name
            config_dir.mkdir(exist_ok=True)

            existing = list(config_dir.glob("rep*/*.txt"))
            expected = len(items) * N_REPS
            print(f"[{config_idx}/{total_configs}] {config_name} — {len(existing)}/{expected} files", flush=True)

            item_estimates = {}
            n_api = 0
            n_parse_fail = 0

            for i, item in enumerate(items):
                ik = f"qid{item['id']}"
                estimates = []

                for rep in range(N_REPS):
                    raw_path = config_dir / f"rep{rep}" / f"{ik}.txt"
                    raw_path.parent.mkdir(exist_ok=True)

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = prompt_fn(item)
                        text = call_llm(prompt, temperature=temp)
                        if text:
                            raw_path.parent.mkdir(parents=True, exist_ok=True)
                            raw_path.write_text(text)
                        n_api += 1
                        time.sleep(0.1)

                    p = parse_proportion(text)
                    if p is not None:
                        estimates.append(p)
                    else:
                        n_parse_fail += 1

                if estimates:
                    item_estimates[ik] = {
                        "mean": float(np.mean(estimates)),
                        "sd": float(np.std(estimates)),
                        "n": len(estimates),
                    }

                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

            total_api += n_api

            # Evaluate
            sim, actual = [], []
            for item in items:
                ik = f"qid{item['id']}"
                if ik in item_estimates:
                    sim.append(item_estimates[ik]["mean"])
                    actual.append(item["p_correct"])

            if len(sim) >= 10:
                rho, p_rho = stats.spearmanr(sim, actual)
                mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
                bias = float(np.mean(np.array(sim) - np.array(actual)))

                # Bootstrap CI
                rng = np.random.default_rng(42)
                boot_rhos = []
                for _ in range(2000):
                    idx = rng.integers(0, len(sim), len(sim))
                    br, _ = stats.spearmanr(np.array(sim)[idx], np.array(actual)[idx])
                    boot_rhos.append(br)
                ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

                all_results[config_name] = {
                    "framing": framing_name,
                    "temperature": temp,
                    "spearman_rho": round(float(rho), 3),
                    "spearman_p": float(f"{p_rho:.2e}"),
                    "ci_95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
                    "mae": round(mae, 3),
                    "bias": round(bias, 3),
                    "n_items": len(sim),
                    "n_api_calls": n_api,
                    "n_parse_failures": n_parse_fail,
                    "parse_rate": round(1 - n_parse_fail / (len(items) * N_REPS), 3),
                }
                print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | parse={1 - n_parse_fail / (len(items) * N_REPS):.1%}", flush=True)

                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

                pred_path = config_dir / "predictions.json"
                with open(pred_path, "w") as f:
                    json.dump(item_estimates, f, indent=2)

    print_summary(all_results)
    test_hypotheses(all_results)


def analyze_existing():
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print("No results.json found")
        return
    all_results = json.loads(results_path.read_text())
    print_summary(all_results)
    test_hypotheses(all_results)


def print_summary(all_results):
    print(f"\n{'=' * 80}")
    print(f"PHASE 3: DBE-KT22 CONFIRMATION — {MODEL}, {N_REPS} reps, no base-rate")
    print(f"{'=' * 80}\n")

    print(f"{'Config':<30} {'ρ':>7} {'95% CI':>16} {'MAE':>7} {'Bias':>7} {'Parse':>7}")
    print("-" * 80)
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["spearman_rho"]):
        ci = res.get("ci_95", [0, 0])
        pr = res.get("parse_rate", 0)
        print(f"{name:<30} {res['spearman_rho']:>7.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {res['mae']:>7.3f} {res['bias']:>+7.3f} {pr:>6.1%}")

    # Temperature comparison
    print(f"\n  Temperature effect (ρ):")
    print(f"  {'Framing':<25} {'t=1.0':>7} {'t=2.0':>7} {'Δ':>7}")
    for framing in FRAMINGS:
        k1 = f"{framing}_t1.0"
        k2 = f"{framing}_t2.0"
        r1 = all_results.get(k1, {}).get("spearman_rho")
        r2 = all_results.get(k2, {}).get("spearman_rho")
        if r1 is not None and r2 is not None:
            print(f"  {framing:<25} {r1:>7.3f} {r2:>7.3f} {r2-r1:>+7.3f}")

    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]["spearman_rho"])
        print(f"\n  Best: {best[0]} (ρ={best[1]['spearman_rho']:.3f})")


# Phase 1 SmartPaper results for cross-dataset comparison
SMARTPAPER_RESULTS = {
    "prerequisite_chain": 0.686,
    "cognitive_load": 0.673,
    "devil_advocate": 0.596,
    "error_analysis": 0.596,
    "teacher": 0.555,
}


def test_hypotheses(all_results):
    print(f"\n{'=' * 80}")
    print("PRE-SPECIFIED HYPOTHESIS TESTS")
    print(f"{'=' * 80}\n")

    # Use best temperature for each framing
    best_per_framing = {}
    for name, res in all_results.items():
        f = res["framing"]
        if f not in best_per_framing or res["spearman_rho"] > best_per_framing[f]["spearman_rho"]:
            best_per_framing[f] = res

    # H1: prerequisite_chain ρ > 0.30
    if "prerequisite_chain" in best_per_framing:
        rho = best_per_framing["prerequisite_chain"]["spearman_rho"]
        ci = best_per_framing["prerequisite_chain"].get("ci_95", [0, 0])
        passed = ci[0] > 0.0  # significantly > 0 at least
        strong = rho > 0.30
        print(f"  H1: prerequisite_chain ρ > 0.30 on DBE-KT22")
        print(f"      Result: ρ={rho:.3f} {ci} — {'PASS' if strong else 'FAIL'}")

    # H2: SmartPaper ρ > DBE-KT22 ρ for every prompt
    print(f"\n  H2: SmartPaper ρ > DBE-KT22 ρ for every prompt")
    all_higher = True
    for f, sp_rho in SMARTPAPER_RESULTS.items():
        if f in best_per_framing:
            dbe_rho = best_per_framing[f]["spearman_rho"]
            higher = sp_rho > dbe_rho
            if not higher:
                all_higher = False
            print(f"      {f:<25} SP={sp_rho:.3f}  DBE={dbe_rho:.3f}  Δ={sp_rho-dbe_rho:+.3f}  {'ok' if higher else 'REVERSED'}")
    print(f"      Overall: {'PASS' if all_higher else 'FAIL'}")

    # H3: Prompt ranking preserved
    print(f"\n  H3: Prompt ranking preserved across datasets")
    sp_rank = sorted(SMARTPAPER_RESULTS.keys(), key=lambda x: -SMARTPAPER_RESULTS[x])
    dbe_rank = sorted([f for f in best_per_framing if f in SMARTPAPER_RESULTS],
                      key=lambda x: -best_per_framing[x]["spearman_rho"])
    sp_order = [f for f in sp_rank if f in best_per_framing]
    if len(sp_order) >= 3:
        rho_rank, _ = stats.spearmanr(
            [sp_rank.index(f) for f in sp_order],
            [dbe_rank.index(f) for f in sp_order]
        )
        print(f"      SmartPaper rank: {sp_rank}")
        print(f"      DBE-KT22 rank:  {dbe_rank}")
        print(f"      Rank correlation: ρ={rho_rank:.3f}")

    # H4: prerequisite_chain > teacher on DBE-KT22
    if "prerequisite_chain" in best_per_framing and "teacher" in best_per_framing:
        pc = best_per_framing["prerequisite_chain"]["spearman_rho"]
        tc = best_per_framing["teacher"]["spearman_rho"]
        print(f"\n  H4: prerequisite_chain > teacher on DBE-KT22")
        print(f"      prerequisite_chain={pc:.3f}  teacher={tc:.3f}  Δ={pc-tc:+.3f} — {'PASS' if pc > tc else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framing", choices=list(FRAMINGS.keys()),
                        help="Run only this framing")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results only")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    if args.analyze:
        analyze_existing()
    else:
        run_experiment(only_framing=args.framing)
