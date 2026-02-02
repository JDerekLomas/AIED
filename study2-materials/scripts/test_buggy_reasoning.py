#!/usr/bin/env python3
"""
Buggy Reasoning difficulty estimation:
  - Give the model the known misconception for the target distractor
  - Ask it to infer what misconceptions drive the OTHER distractors too
  - Judge how "attractive" each buggy reasoning chain is at each ability level
  - Aggregate into predicted response distributions

Runs on multiple models (Gemini Flash, plus optionally Groq/OpenAI).

Objective: Spearman rho between predicted weighted_p_incorrect and IRT b_2pl
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/buggy_reasoning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

BUGGY_PROMPT = """You are an experienced UK maths teacher who specialises in diagnosing student errors.

For this question, one distractor is known to be caused by this misconception:
  Option {target_distractor}: "{misconception_name}"

{item_text}

Your task:
1. For EACH wrong answer option, identify the specific buggy reasoning or misconception that would lead a student to choose it. For option {target_distractor}, use the known misconception above. For the other wrong options, infer what error or shortcut would produce that answer.

2. For each buggy reasoning chain, judge how ATTRACTIVE it is — how naturally and easily would a student fall into this error? Consider:
   - Does the error feel like a "reasonable" next step?
   - Would the student even notice they went wrong?
   - At which ability levels is this error most likely?

3. Based on your analysis, predict the response distribution:

below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""

# Comparison: same error_analysis prompt but WITHOUT the misconception hint
ERROR_ANALYSIS_NO_HINT = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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

MODELS = {
    "gemini_flash": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 2.0},
    "groq_llama70b": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temp": 1.5},
    "groq_llama4scout": {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "temp": 1.5},
}

METHODS = {
    "buggy_with_hint": BUGGY_PROMPT,
    "error_analysis_no_hint": ERROR_ANALYSIS_NO_HINT,
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
        # Try standard format: below_basic: A=XX% B=XX% C=XX% D=XX%
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            weighted_p_correct += (pcts[correct_idx] / total) * w
            continue
        # Try markdown table format: | **Below-Basic** | 10% | 20% | 15% | 55% |
        level_variants = [level, level.replace("_", "-"), level.replace("_", " ")]
        if level == "below_basic":
            level_variants.extend(["below basic", "below-basic", "below_basic"])
        for lv in level_variants:
            tbl_pat = rf'\*?\*?{re.escape(lv)}\*?\*?\s*\|\s*\*?\*?(\d+)%?\*?\*?\s*\|\s*\*?\*?(\d+)%?\*?\*?\s*\|\s*\*?\*?(\d+)%?\*?\*?\s*\|\s*\*?\*?(\d+)%?\*?\*?'
            match = re.search(tbl_pat, text, re.IGNORECASE)
            if match:
                pcts = [int(match.group(i)) for i in range(1, 5)]
                total = sum(pcts) or 1
                weighted_p_correct += (pcts[correct_idx] / total) * w
                break
        else:
            weighted_p_correct += 0.5 * w
    return 1 - weighted_p_correct


def call_gemini(prompt, model, temp):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp, max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_groq(prompt, model, temp):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temp, max_tokens=2048,
    )
    return resp.choices[0].message.content


def make_call(prompt, provider, model, temp):
    if provider == "gemini":
        return call_gemini(prompt, model, temp)
    elif provider == "groq":
        return call_groq(prompt, model, temp)


def main():
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    # Select models via CLI args, default to all
    selected_models = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    models_to_run = {k: v for k, v in MODELS.items() if k in selected_models}

    for method_name, template in METHODS.items():
        for mname, mconfig in models_to_run.items():
            provider = mconfig["provider"]
            model = mconfig["model"]
            temp = mconfig["temp"]
            config_key = f"{method_name}__{mname}"

            print(f"\n{'='*60}", flush=True)
            print(f"{config_key}", flush=True)
            print(f"{'='*60}", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
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
                        prompt = template.format(
                            item_text=item_text,
                            target_distractor=row["target_distractor_kaggle"],
                            misconception_name=row["misconception_name"],
                        )
                        try:
                            text = make_call(prompt, provider, model, temp)
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} qid={qid}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        sleep = 0.5 if provider == "groq" else 0.15
                        time.sleep(sleep)

                    p_inc = parse_predictions(text, correct)
                    results.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                    "weighted_p_incorrect": p_inc})

                df = pd.DataFrame(results)
                if len(df) >= 5:
                    rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                    rhos.append(rho)
                    print(f"  {config_key} rep{rep}: rho={rho:.3f} (p={p:.4f})", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"  {config_key} MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)

                summary = {"method": method_name, "model": mname, "provider": provider,
                           "mean_rho": float(mean_rho), "std_rho": float(std_rho),
                           "rhos": [float(r) for r in rhos]}
                with open(OUTPUT_DIR / f"{config_key}_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

    # Final comparison
    print(f"\n{'='*60}", flush=True)
    print("ALL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for p in sorted(OUTPUT_DIR.glob("*_summary.json")):
        d = json.loads(p.read_text())
        print(f"  {d['method']}__{d['model']}: {d['mean_rho']:.3f} ± {d['std_rho']:.3f}", flush=True)


if __name__ == "__main__":
    main()
