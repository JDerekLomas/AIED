#!/usr/bin/env python3
"""
Frontier model comparison: Test the best prompt (error_analysis) at optimal
temperature across frontier-tier models.

Models:
  - Gemini Flash (baseline, already tested)
  - Gemini Pro (gemini-2.5-pro-preview-06-05)
  - Claude Sonnet 4 (via Anthropic API)
  - GPT-4o (via OpenAI API)
  - Claude Haiku 3.5 (cheaper frontier comparison)

3 reps each. Uses the error_analysis prompt.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/frontier_models")
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
    "gemini_flash": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 2.0},
    "gemini_pro": {"provider": "gemini", "model": "gemini-2.5-pro-preview-06-05", "temp": 2.0},
    "claude_sonnet4": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temp": 1.0},
    "claude_haiku35": {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "temp": 1.0},
    "gpt_4o": {"provider": "openai", "model": "gpt-4o", "temp": 1.5},
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


def call_gemini(prompt, model, temp):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temp, max_output_tokens=1024,
    )
    # Only disable thinking for Flash (Pro needs it for quality)
    if "flash" in model:
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return resp.text


def call_anthropic(prompt, model, temp):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(prompt, model, temp):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp, max_tokens=1024,
    )
    return resp.choices[0].message.content


def make_call(prompt, provider, model, temp):
    if provider == "gemini":
        return call_gemini(prompt, model, temp)
    elif provider == "anthropic":
        return call_anthropic(prompt, model, temp)
    elif provider == "openai":
        return call_openai(prompt, model, temp)


def main():
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    models_to_run = {k: v for k, v in MODELS.items() if k in selected}

    total = len(models_to_run) * N_REPS * len(probe)
    print(f"Running {len(models_to_run)} models x {N_REPS} reps x {len(probe)} items = {total} calls\n", flush=True)

    summary = {}
    for mname, mconfig in models_to_run.items():
        provider = mconfig["provider"]
        model = mconfig["model"]
        temp = mconfig["temp"]
        print(f"\n--- {mname} ({model}, t={temp}) ---", flush=True)

        rhos = []
        all_preds = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Reuse Gemini Flash from metaprompt_sweep
            sweep_dir = Path("pilot/rsm_experiment/metaprompt_sweep/v5_error_analysis_t2.0") / f"rep{rep}"

            items_pred = []
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                elif mname == "gemini_flash" and sweep_dir.exists() and (sweep_dir / f"qid{qid}.txt").exists():
                    text = (sweep_dir / f"qid{qid}.txt").read_text()
                    raw_path.write_text(text)
                else:
                    item_text = format_item_text(row)
                    prompt = ERROR_ANALYSIS_PROMPT.format(item_text=item_text)
                    try:
                        text = make_call(prompt, provider, model, temp)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {mname} rep{rep} qid={qid}: {e}", flush=True)
                        time.sleep(3)
                        continue
                    # Rate limiting
                    sleep = {"gemini": 0.15, "anthropic": 0.5, "openai": 0.3}.get(provider, 0.5)
                    time.sleep(sleep)

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})
                all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                  "rep": rep, "p_inc": p_inc})

            df = pd.DataFrame(items_pred)
            if len(df) >= 5:
                rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                rhos.append(rho)
                print(f"  rep{rep}: rho={rho:.3f} (p={p:.4f})", flush=True)

        if rhos:
            mean_rho = np.mean(rhos)
            std_rho = np.std(rhos)
            summary[mname] = {"mean": float(mean_rho), "std": float(std_rho),
                              "rhos": [float(r) for r in rhos],
                              "provider": provider, "model": model, "temp": temp}
            print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)

        # Averaged-prediction rho
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("p_inc", "mean"), b_2pl=("b_2pl", "first")
            ).dropna()
            if len(avg) >= 5:
                rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
                print(f"  Averaged: rho={rho:.3f} (p={p:.4f})", flush=True)

    # Ranking
    print(f"\n{'='*60}", flush=True)
    print("FRONTIER MODEL RANKING", flush=True)
    print(f"{'='*60}", flush=True)
    for name, data in sorted(summary.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name} ({data['model']}, t={data['temp']}): {data['mean']:.3f} ± {data['std']:.3f}", flush=True)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
