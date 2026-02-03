#!/usr/bin/env python3
"""
Two-stage diversity injection on SmartPaper items (full 140).
Tests whether the cognitive chains approach generalizes beyond Eedi MCQs.

SmartPaper: open-ended items, Indian govt schools, Grades 6-8, 4 subjects.
Ground truth: classical_difficulty (proportion scoring full marks).

Stage 1 (t=2.0): Generate 5 diverse student attempts per item
Stage 2 (t=1.5): Predict difficulty using attempts as context

Conditions:
  A. cognitive_chains — 5 diverse student attempts (adapted for open-ended)
  B. error_perspectives — 5 "why this is hard" analyses
  C. direct_baseline — predict difficulty directly (no two-stage)
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/smartpaper_two_stage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
N_REPS = 3

VISUAL_KEYWORDS = ["given figure", "in this picture", "based on images",
                   "observe the picture", "following figures", "in the given"]

# ============================================================
# Stage 1 prompts (adapted for open-ended Indian school context)
# ============================================================

STAGE1_COGNITIVE = """You are simulating a real Class {grade} student in an Indian government school attempting this question. You are NOT a strong student — you have gaps in your knowledge and sometimes make careless errors. Your English is limited.

Question: {question_text}
Rubric (what a correct answer looks like): {rubric}
Maximum score: {max_score}

Write your answer attempt exactly as this student would write it — with spelling errors, incomplete reasoning, and mistakes typical of this population. You may or may not get it right.

Your answer attempt:"""

STAGE1_ERROR_PERSPECTIVE = """You are an education researcher studying assessment items for Indian government school students (Class {grade}).

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Identify ONE specific reason why students in this context might find this question difficult or easy. Consider:
- Language barriers (Hindi-medium students answering in English)
- Conceptual gaps common at this grade level
- Whether the question requires recall vs application vs reasoning
- Specific procedural errors students might make

Be specific and concrete, not generic."""

# ============================================================
# Stage 2 prompt
# ============================================================

STAGE2_TEMPLATE = """You are an experienced teacher in an Indian government school who has just marked papers from 500 Class {grade} students.

Here are {n_seeds} different examples of how students might attempt this question:

{seed_content}

---

Now, considering ALL of the above attempts and your experience, estimate what proportion of your 500 students would score full marks on this question.

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Think about the distribution of ability in your classroom — many students struggle with English, some are competent, few are advanced.

Respond with ONLY a number between 0 and 100 representing the percentage who would score full marks.

Percentage scoring full marks:"""

DIRECT_BASELINE = """You are an experienced teacher in an Indian government school who has just marked papers from 500 Class {grade} students.

Before predicting, think about:
- What specific errors would students make on this question?
- How many students would understand what is being asked?
- How many could formulate a correct response?
- What proportion of your class has the prerequisite knowledge?

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Respond with ONLY a number between 0 and 100 representing the percentage who would score full marks.

Percentage scoring full marks:"""


STAGE1_PROMPTS = {
    "cognitive_chains": STAGE1_COGNITIVE,
    "error_perspectives": STAGE1_ERROR_PERSPECTIVE,
}

STAGE2_MODELS = {
    "gemini": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 1.5},
    "scout": {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "temp": 2.0},
}


def call_gemini(prompt, model, temp):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp, max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_groq(prompt, model, temp):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model, max_tokens=1024, temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def make_call(prompt, provider, model, temp):
    if provider == "gemini":
        return call_gemini(prompt, model, temp)
    elif provider == "groq":
        return call_groq(prompt, model, temp)


def parse_percentage(text):
    """Extract a percentage from model output."""
    # Try to find a number
    nums = re.findall(r'(\d+(?:\.\d+)?)', text.strip()[:200])
    if nums:
        val = float(nums[0])
        if val > 1 and val <= 100:
            return val / 100.0
        elif val <= 1:
            return val
    return None


def generate_seeds(item, stage1_prompt, seed_dir, seed_provider="gemini",
                   seed_model="gemini-3-flash-preview", seed_temp=2.0):
    seeds = []
    for i in range(N_SEEDS):
        seed_path = seed_dir / f"seed{i}.txt"
        if seed_path.exists():
            seeds.append(seed_path.read_text())
        else:
            prompt = stage1_prompt.format(
                grade=item["grade"],
                question_text=item["question_text"],
                rubric=item["rubric"],
                max_score=item["max_score"],
            )
            try:
                text = make_call(prompt, seed_provider, seed_model, seed_temp)
                seed_path.write_text(text)
                seeds.append(text)
            except Exception as e:
                print(f"    SEED ERROR q{item['question_number']}: {e}", flush=True)
                time.sleep(2)
                continue
            time.sleep(0.15)
    return seeds


def load_items():
    """Load all 140 SmartPaper items, excluding visual-dependent ones."""
    with open("data/smartpaper/item_statistics.json") as f:
        items = json.load(f)
    # Filter out visual items
    filtered = []
    for item in items:
        q = item["question_text"].lower()
        if any(kw in q for kw in VISUAL_KEYWORDS):
            continue
        filtered.append(item)
    print(f"Loaded {len(filtered)} items (excluded {len(items)-len(filtered)} visual items)", flush=True)
    return filtered


def main():
    items = load_items()

    # Parse CLI args
    selected_conditions = []
    selected_models = []
    use_scout_seeds = "--scout-seeds" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        if arg in STAGE1_PROMPTS or arg == "direct_baseline":
            selected_conditions.append(arg)
        elif arg in STAGE2_MODELS:
            selected_models.append(arg)

    conditions = selected_conditions or list(STAGE1_PROMPTS.keys()) + ["direct_baseline"]
    models = selected_models or ["gemini"]

    if use_scout_seeds:
        seed_provider, seed_model, seed_temp = "groq", "meta-llama/llama-4-scout-17b-16e-instruct", 2.0
        seed_prefix = "seeds_scout_"
    else:
        seed_provider, seed_model, seed_temp = "gemini", "gemini-3-flash-preview", 2.0
        seed_prefix = "seeds_"

    all_results = []

    # === Two-stage conditions ===
    for cond_name in [c for c in conditions if c != "direct_baseline"]:
        stage1_prompt = STAGE1_PROMPTS[cond_name]

        for mname in models:
            mconfig = STAGE2_MODELS[mname]
            seed_tag = "scoutseeds_" if use_scout_seeds else ""
            config_key = f"{seed_tag}{cond_name}__{mname}"
            print(f"\n{'='*60}", flush=True)
            print(f"{config_key} ({len(items)} items)", flush=True)
            print(f"{'='*60}", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                seed_dir = OUTPUT_DIR / f"{seed_prefix}{cond_name}" / f"rep{rep}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                preds = []
                for item in items:
                    qkey = f"{item['assessment']}_q{item['question_number']}"
                    qkey_safe = qkey.replace(" ", "_").replace("—", "-")
                    raw_path = raw_dir / f"{qkey_safe}.txt"

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        qseed_dir = seed_dir / qkey_safe
                        qseed_dir.mkdir(parents=True, exist_ok=True)
                        seeds = generate_seeds(item, stage1_prompt, qseed_dir,
                                               seed_provider, seed_model, seed_temp)
                        if not seeds:
                            continue

                        seed_content = "\n\n".join(
                            f"--- Student Attempt {i+1} ---\n{s}" for i, s in enumerate(seeds)
                        )
                        prompt = STAGE2_TEMPLATE.format(
                            grade=item["grade"],
                            n_seeds=len(seeds),
                            seed_content=seed_content,
                            question_text=item["question_text"],
                            rubric=item["rubric"],
                            max_score=item["max_score"],
                        )
                        try:
                            text = make_call(prompt, mconfig["provider"], mconfig["model"], mconfig["temp"])
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} {qkey_safe}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        time.sleep(0.15 if mconfig["provider"] != "groq" else 0.05)

                    p = parse_percentage(text)
                    if p is not None:
                        # predicted_difficulty = 1 - p (proportion scoring full marks)
                        # but we correlate predicted easiness (p) with actual easiness (classical_difficulty)
                        preds.append({
                            "item": qkey,
                            "predicted_easiness": p,
                            "actual_easiness": item["classical_difficulty"],
                        })

                df = pd.DataFrame(preds)
                if len(df) >= 10:
                    rho, pval = stats.spearmanr(df["predicted_easiness"], df["actual_easiness"])
                    rhos.append(rho)
                    print(f"  rep{rep}: rho={rho:.3f} (p={pval:.4f}, n={len(df)})", flush=True)
                else:
                    print(f"  rep{rep}: only {len(df)} parseable predictions, skipping", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)
                all_results.append({
                    "config": config_key, "condition": cond_name, "stage2_model": mname,
                    "mean_rho": float(mean_rho), "std_rho": float(std_rho),
                    "rhos": [float(r) for r in rhos],
                })

    # === Direct baseline ===
    if "direct_baseline" in conditions:
        for mname in models:
            mconfig = STAGE2_MODELS[mname]
            config_key = f"direct_baseline__{mname}"
            print(f"\n{'='*60}", flush=True)
            print(f"{config_key} ({len(items)} items)", flush=True)
            print(f"{'='*60}", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)

                preds = []
                for item in items:
                    qkey = f"{item['assessment']}_q{item['question_number']}"
                    qkey_safe = qkey.replace(" ", "_").replace("—", "-")
                    raw_path = raw_dir / f"{qkey_safe}.txt"

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = DIRECT_BASELINE.format(
                            grade=item["grade"],
                            question_text=item["question_text"],
                            rubric=item["rubric"],
                            max_score=item["max_score"],
                        )
                        try:
                            text = make_call(prompt, mconfig["provider"], mconfig["model"], mconfig["temp"])
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} {qkey_safe}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        time.sleep(0.15 if mconfig["provider"] != "groq" else 0.05)

                    p = parse_percentage(text)
                    if p is not None:
                        preds.append({
                            "item": qkey,
                            "predicted_easiness": p,
                            "actual_easiness": item["classical_difficulty"],
                        })

                df = pd.DataFrame(preds)
                if len(df) >= 10:
                    rho, pval = stats.spearmanr(df["predicted_easiness"], df["actual_easiness"])
                    rhos.append(rho)
                    print(f"  rep{rep}: rho={rho:.3f} (p={pval:.4f}, n={len(df)})", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)
                all_results.append({
                    "config": config_key, "condition": "direct_baseline", "stage2_model": mname,
                    "mean_rho": float(mean_rho), "std_rho": float(std_rho),
                    "rhos": [float(r) for r in rhos],
                })

    # === Summary ===
    print(f"\n{'='*60}", flush=True)
    print(f"ALL RESULTS ({len(items)} items)", flush=True)
    print(f"{'='*60}", flush=True)
    for r in sorted(all_results, key=lambda x: x["mean_rho"], reverse=True):
        print(f"  {r['config']}: {r['mean_rho']:.3f} ± {r['std_rho']:.3f}", flush=True)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
