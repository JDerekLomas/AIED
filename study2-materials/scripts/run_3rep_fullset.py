#!/usr/bin/env python3
"""
Run 3-rep averaged difficulty estimation on the full 1,869 Eedi item set
for top-tier models (groq_llama4scout, deepseek_chat, gemini_flash).
Correlates against p_value. Includes cost tracking and bootstrap CIs.
"""
import json, re, os, time, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/replications/3rep_fullset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

CONTRASTIVE_PROMPT = """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""

MODELS = {
    "groq_llama4scout": {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "temp": 1.5},
    "deepseek_chat": {"provider": "deepseek", "model": "deepseek-chat", "temp": 1.5},
    "gemini_flash": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 1.5},
}

# Per-million-token pricing (input, output)
PRICING = {
    "groq_llama4scout": (0.11, 0.34),
    "deepseek_chat": (0.27, 1.10),
    "gemini_flash": (0.10, 0.40),
}

RATE_LIMITS = {
    "groq": 0.5,
    "deepseek": 0.2,
    "gemini": 0.2,
}


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


def call_groq(prompt, model, temp):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def call_gemini(prompt, model, temp):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_deepseek(prompt, model, temp):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def make_call(prompt, provider, model, temp):
    if provider == "groq":
        return call_groq(prompt, model, temp)
    elif provider == "gemini":
        return call_gemini(prompt, model, temp)
    elif provider == "deepseek":
        return call_deepseek(prompt, model, temp)


def bootstrap_ci(x, y, n_boot=1000, alpha=0.05):
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(rho)
    return np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def estimate_cost(mname, input_chars, output_chars):
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    price_in, price_out = PRICING[mname]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def main():
    parser = argparse.ArgumentParser(description="3-rep fullset difficulty estimation")
    parser.add_argument("models", nargs="*", default=list(MODELS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("--reps", type=str, default=None,
                        help="Comma-separated rep indices to run (e.g. '0' or '1,2'). Default: all")
    args = parser.parse_args()

    models_to_run = {k: v for k, v in MODELS.items() if k in args.models}
    if not models_to_run:
        print(f"No valid models. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    # Load full dataset
    df_all = pd.read_csv("data/eedi/eedi_with_student_data.csv")
    print(f"Loaded {len(df_all)} items")

    # Compute p_value from neurips correct position
    # neurips_correct_pos is a letter (A/B/C/D), pct_{letter} columns have the percentages
    def get_p_value(row):
        letter = row["neurips_correct_pos"]
        col = f"pct_{letter}"
        if col in row.index and pd.notna(row[col]):
            return row[col] / 100.0
        return np.nan

    df_all["p_value"] = df_all.apply(get_p_value, axis=1)
    valid_items = df_all.dropna(subset=["p_value", "CorrectAnswer", "QuestionText"])
    print(f"Items with valid p_value: {len(valid_items)}")

    total_calls = len(models_to_run) * N_REPS * len(valid_items)
    print(f"Running {len(models_to_run)} models x {N_REPS} reps x {len(valid_items)} items = {total_calls} calls\n")

    summary = {}

    for mname, mconfig in models_to_run.items():
        provider = mconfig["provider"]
        model = mconfig["model"]
        temp = mconfig["temp"]
        sleep_time = RATE_LIMITS.get(provider, 0.2)
        print(f"\n{'='*60}")
        print(f"--- {mname} ({model}) ---")
        print(f"{'='*60}")

        total_input_chars = 0
        total_output_chars = 0
        per_rep_rhos = []
        reps_to_run = [int(r) for r in args.reps.split(",")] if args.reps else list(range(N_REPS))

        for rep in reps_to_run:
            raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            items_pred = []
            n_cached = 0
            n_called = 0
            n_errors = 0

            for i, (_, row) in enumerate(valid_items.iterrows()):
                qid = row["QuestionId"]
                correct = row["CorrectAnswer"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                    n_cached += 1
                else:
                    item_text = format_item_text(row)
                    prompt = CONTRASTIVE_PROMPT.format(item_text=item_text)
                    total_input_chars += len(prompt)
                    try:
                        text = make_call(prompt, provider, model, temp)
                        raw_path.write_text(text)
                        total_output_chars += len(text)
                        n_called += 1
                    except Exception as e:
                        print(f"  ERROR {mname} rep{rep} qid={qid}: {e}")
                        n_errors += 1
                        time.sleep(2)
                        continue
                    time.sleep(sleep_time)

                    if n_called % 100 == 0:
                        print(f"  rep{rep}: {n_called} called, {n_cached} cached, {n_errors} errors ({i+1}/{len(valid_items)})")

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "p_value": row["p_value"],
                                   "weighted_p_incorrect": p_inc})

            df_rep = pd.DataFrame(items_pred).dropna()
            if len(df_rep) >= 5:
                rho, p = stats.spearmanr(df_rep["weighted_p_incorrect"], df_rep["p_value"])
                per_rep_rhos.append(rho)
                print(f"  {mname} rep{rep}: rho={rho:.3f} (p={p:.2e}) [n={len(df_rep)}, cached={n_cached}, called={n_called}, errors={n_errors}]")

        # Averaged predictions across reps
        all_preds = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
            for _, row in valid_items.iterrows():
                qid = row["QuestionId"]
                correct = row["CorrectAnswer"]
                raw_path = raw_dir / f"qid{qid}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                    p_inc = parse_predictions(text, correct)
                    all_preds.append({"QuestionId": qid, "p_value": row["p_value"],
                                      "rep": rep, "p_inc": p_inc})

        avg_rho = None
        ci_low = ci_high = None
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("p_inc", "mean"),
                p_value=("p_value", "first")
            ).dropna()
            if len(avg) >= 5:
                avg_rho, avg_p = stats.spearmanr(avg["mean_p_inc"], avg["p_value"])
                x = avg["mean_p_inc"].values
                y = avg["p_value"].values
                ci_low, ci_high = bootstrap_ci(x, y)
                print(f"\n  {mname} AVERAGED: rho={avg_rho:.3f} (p={avg_p:.2e}) 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")

        # Cost estimate
        cost = estimate_cost(mname, total_input_chars, total_output_chars)

        model_summary = {
            "model": model,
            "provider": provider,
            "n_items": len(valid_items),
            "per_rep_rhos": per_rep_rhos,
            "mean_rep_rho": float(np.mean(per_rep_rhos)) if per_rep_rhos else None,
            "std_rep_rho": float(np.std(per_rep_rhos)) if per_rep_rhos else None,
            "averaged_rho": float(avg_rho) if avg_rho is not None else None,
            "ci_95_low": float(ci_low) if ci_low is not None else None,
            "ci_95_high": float(ci_high) if ci_high is not None else None,
            "est_cost_usd": round(cost, 4),
            "total_input_chars": total_input_chars,
            "total_output_chars": total_output_chars,
        }
        summary[mname] = model_summary

        if per_rep_rhos:
            print(f"  {mname} MEAN REP: {np.mean(per_rep_rhos):.3f} +/- {np.std(per_rep_rhos):.3f}")
        print(f"  Est cost (new calls only): ${cost:.4f}")

    # Final ranking
    print(f"\n{'='*60}")
    print("FINAL RANKING (by averaged-prediction rho)")
    print(f"{'='*60}")
    ranked = sorted(summary.items(),
                    key=lambda x: x[1].get("averaged_rho") or 0, reverse=True)
    for name, data in ranked:
        avg = data.get("averaged_rho")
        ci_l = data.get("ci_95_low")
        ci_h = data.get("ci_95_high")
        if avg is not None:
            print(f"  {name}: rho={avg:.3f} 95%CI=[{ci_l:.3f},{ci_h:.3f}] cost=${data['est_cost_usd']:.4f}")

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
