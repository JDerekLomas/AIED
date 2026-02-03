#!/usr/bin/env python3
"""
Test difficulty estimation across Gemini model sizes.
Uses the best prompt (v5_error_analysis) at temp=2.0, 3 reps each.
Reuses cached gemini-3-flash results.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/gemini_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

ERROR_ANALYSIS_PROMPT = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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
    "gemini_3_flash": {
        "model": "gemini-3-flash-preview",
        "cached_dir": "pilot/rsm_experiment/metaprompt_sweep/v5_error_analysis_t2.0",
    },
    "gemini_3_pro": {
        "model": "gemini-3-pro-preview",
    },
    "gemini_2.5_pro": {
        "model": "gemini-2.5-pro",
    },
    "gemini_2.5_flash": {
        "model": "gemini-2.5-flash",
    },
    "gemini_2.0_flash": {
        "model": "gemini-2.0-flash",
    },
}

TEMP = 2.0


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


def make_api_call(client, model, prompt, temperature):
    from google.genai import types
    # 2.5-pro and 3-pro need thinking_budget >= 1 (can't disable thinking)
    # Use minimal thinking for pro models to keep output comparable
    if "pro" in model:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
        )
    else:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    # Extract text, handling thinking models that may split parts
    text = response.text
    if text is None:
        # Fallback: iterate parts for non-thought content
        for part in (response.candidates[0].content.parts or []):
            if hasattr(part, 'thought') and part.thought:
                continue
            if part.text:
                return part.text
        return ""
    return text


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    models_to_run = {k: v for k, v in MODELS.items() if k in selected}

    print(f"Testing {len(models_to_run)} Gemini models × {N_REPS} reps × {len(probe)} items", flush=True)
    print(f"Prompt: error_analysis, temp={TEMP}\n", flush=True)

    summary = {}
    all_preds = []

    for mname, mcfg in models_to_run.items():
        model = mcfg["model"]
        cached_dir = mcfg.get("cached_dir")
        print(f"\n{'='*50}", flush=True)
        print(f"{mname} ({model})", flush=True)
        print(f"{'='*50}", flush=True)

        rhos = []
        for rep in range(N_REPS):
            # Check for cached results
            if cached_dir:
                raw_dir = Path(cached_dir) / f"rep{rep}"
            else:
                raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
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
                    prompt = ERROR_ANALYSIS_PROMPT.format(item_text=item_text)
                    try:
                        text = make_api_call(client, model, prompt, TEMP)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {mname} rep{rep} qid={qid}: {e}", flush=True)
                        time.sleep(3)
                        continue
                    # Rate limit: pro models are slower/more limited
                    sleep = 1.0 if "pro" in model else 0.15
                    time.sleep(sleep)

                p_inc, n_parsed = parse_predictions(text, correct)
                if n_parsed < 3:
                    parse_failures += 1
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})
                all_preds.append({"model": mname, "rep": rep, "QuestionId": qid,
                                  "b_2pl": row["b_2pl"], "p_inc": p_inc})

            df = pd.DataFrame(items_pred)
            valid = df.dropna()
            if len(valid) >= 5:
                rho, p = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
                rhos.append(rho)
                print(f"  rep{rep}: rho={rho:.3f} (p={p:.4f}, parse_fail={parse_failures})", flush=True)

        if rhos:
            mean_rho = np.mean(rhos)
            std_rho = np.std(rhos)
            summary[mname] = {"mean": mean_rho, "std": std_rho, "rhos": rhos, "model": model}
            print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)

    # Averaged-prediction rho per model
    print(f"\n{'='*60}", flush=True)
    print("AVERAGED PREDICTIONS (mean p_incorrect across reps)", flush=True)
    print(f"{'='*60}", flush=True)
    pdf = pd.DataFrame(all_preds)
    for mname in models_to_run:
        sub = pdf[pdf["model"] == mname]
        if len(sub) == 0:
            continue
        avg = sub.groupby("QuestionId").agg(
            mean_p_inc=("p_inc", "mean"),
            b_2pl=("b_2pl", "first")
        ).dropna()
        if len(avg) >= 5:
            rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
            print(f"  {mname}: rho={rho:.3f} (p={p:.4f})", flush=True)

    # Final ranking
    print(f"\n{'='*60}", flush=True)
    print("RANKING (by mean single-rep rho)", flush=True)
    print(f"{'='*60}", flush=True)
    for name, data in sorted(summary.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name} ({data['model']}): {data['mean']:.3f} ± {data['std']:.3f}  {[f'{r:.3f}' for r in data['rhos']]}", flush=True)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
