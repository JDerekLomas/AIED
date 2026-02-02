#!/usr/bin/env python3
"""
Test the winning contrastive teacher_prediction prompt across multiple models.
Uses Groq (fast), OpenAI (4o-mini), and Gemini for comparison.
3 reps each to assess stability.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/cross_model")
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
    "groq_llama70b": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temp": 1.5},
    "groq_llama4scout": {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "temp": 1.5},
    "groq_qwen32b": {"provider": "groq", "model": "qwen/qwen3-32b", "temp": 1.5},
    "groq_gptoss120b": {"provider": "groq", "model": "openai/gpt-oss-120b", "temp": 1.5},
    "groq_kimik2": {"provider": "groq", "model": "moonshotai/kimi-k2-instruct-0905", "temp": 1.5},
    "groq_llama8b": {"provider": "groq", "model": "llama-3.1-8b-instant", "temp": 1.5},
    "openai_4omini": {"provider": "openai", "model": "gpt-4o-mini", "temp": 1.5},
    "openai_4o": {"provider": "openai", "model": "gpt-4o", "temp": 1.5},
    "gemini_flash": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 1.5},
    "deepseek_chat": {"provider": "deepseek", "model": "deepseek-chat", "temp": 1.5},
    "deepseek_reasoner": {"provider": "deepseek", "model": "deepseek-reasoner", "temp": 1.0},  # reasoner doesn't support high temp
    "openrouter_gemini_flash": {"provider": "openrouter", "model": "google/gemini-2.0-flash-001", "temp": 1.5},
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


def call_openai(prompt, model, temp):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024}
    if model != "deepseek-reasoner":
        kwargs["temperature"] = temp
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def call_openrouter(prompt, model, temp):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
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
    elif provider == "openai":
        return call_openai(prompt, model, temp)
    elif provider == "gemini":
        return call_gemini(prompt, model, temp)
    elif provider == "deepseek":
        return call_deepseek(prompt, model, temp)
    elif provider == "openrouter":
        return call_openrouter(prompt, model, temp)


def main():
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    # Allow selecting specific models via CLI
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    models_to_run = {k: v for k, v in MODELS.items() if k in selected}

    total_calls = len(models_to_run) * N_REPS * len(probe)
    print(f"Running {len(models_to_run)} models x {N_REPS} reps x {len(probe)} items = {total_calls} calls\n")

    summary = {}
    for mname, mconfig in models_to_run.items():
        provider = mconfig["provider"]
        model = mconfig["model"]
        temp = mconfig["temp"]
        print(f"\n--- {mname} ({model}) ---")

        rhos = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
            raw_dir.mkdir(parents=True, exist_ok=True)

            items_pred = []
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"

                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    item_text = format_item_text(row)
                    prompt = CONTRASTIVE_PROMPT.format(item_text=item_text)
                    try:
                        text = make_call(prompt, provider, model, temp)
                        raw_path.write_text(text)
                    except Exception as e:
                        print(f"  ERROR {mname} rep{rep} qid={qid}: {e}")
                        time.sleep(2)
                        continue
                    # Groq paid tier: higher limits
                    sleep = 0.5 if provider == "groq" else 0.2
                    time.sleep(sleep)

                p_inc = parse_predictions(text, correct)
                items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                   "weighted_p_incorrect": p_inc})

            df = pd.DataFrame(items_pred)
            valid = df.dropna()
            if len(valid) >= 5:
                rho, _ = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
                rhos.append(rho)
                print(f"  {mname} rep{rep}: rho={rho:.3f}")

        if rhos:
            mean_rho = np.mean(rhos)
            std_rho = np.std(rhos)
            summary[mname] = {"mean": mean_rho, "std": std_rho, "rhos": rhos,
                              "provider": provider, "model": model}
            print(f"  {mname} MEAN: {mean_rho:.3f} ± {std_rho:.3f}")

    # Also compute averaged-prediction rho
    print(f"\n{'='*60}")
    print("AVERAGED PREDICTIONS (mean p_incorrect across reps)")
    print(f"{'='*60}")
    for mname in models_to_run:
        all_preds = []
        for rep in range(N_REPS):
            raw_dir = OUTPUT_DIR / mname / f"rep{rep}"
            for _, row in probe.iterrows():
                qid = row["QuestionId"]
                correct = row["correct_answer_kaggle"]
                raw_path = raw_dir / f"qid{qid}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                    p_inc = parse_predictions(text, correct)
                    all_preds.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                      "rep": rep, "p_inc": p_inc})
        if all_preds:
            pdf = pd.DataFrame(all_preds)
            avg = pdf.groupby("QuestionId").agg(
                mean_p_inc=("p_inc", "mean"),
                b_2pl=("b_2pl", "first")
            ).dropna()
            if len(avg) >= 5:
                rho, p = stats.spearmanr(avg["mean_p_inc"], avg["b_2pl"])
                print(f"  {mname}: rho={rho:.3f} (p={p:.4f})")

    print(f"\n{'='*60}")
    print("FINAL RANKING (by mean single-rep rho)")
    print(f"{'='*60}")
    for name, data in sorted(summary.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name} ({data['model']}): {data['mean']:.3f} ± {data['std']:.3f}")

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
