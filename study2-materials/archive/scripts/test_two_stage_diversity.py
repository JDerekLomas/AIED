#!/usr/bin/env python3
"""
Two-stage diversity injection: Generate diverse content at high temperature,
then use it as context for prediction at lower temperature.

Tests whether injecting specific TYPES of diversity improves predictions,
and whether the two-stage approach can exceed single-stage high-temp.

Stage 1 (t=2.0, Gemini Flash): Generate N diverse seeds per item
Stage 2 (t=1.5, Gemini Flash or t=1.0 Claude): Predict using seeds as context

Conditions:
  A. cognitive_chains — 5 diverse student reasoning chains (high-temp rethink of
     cognitive modeling: generate chains, but use them as CONTEXT for prediction
     rather than counting answers directly)
  B. buggy_analyses — 5 diverse error/misconception analyses per item
  C. error_perspectives — 5 diverse "what makes this hard" analyses
  D. direct_baseline — standard error_analysis at t=2.0 (no two-stage)

For each condition, test:
  - gemini_gemini: stage1=Gemini t=2.0, stage2=Gemini t=1.5
  - gemini_claude: stage1=Gemini t=2.0, stage2=Claude t=1.0

3 reps each.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/two_stage_diversity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
N_REPS = 3

# ============================================================
# Stage 1 prompts: generate diverse content at high temp
# ============================================================

STAGE1_COGNITIVE = """You are simulating a real UK Year 9 student attempting this maths question. You are NOT a strong student — you have gaps in your knowledge and sometimes make careless errors.

{item_text}

Show your working step by step, exactly as a real student would write it. Include hesitations, crossing-out, and mistakes. You may or may not get the final answer right. Choose your answer from A/B/C/D at the end.

Your working:"""

STAGE1_BUGGY = """You are an expert in mathematical misconceptions. For this question, identify one specific misconception or buggy procedure that a UK Year 9 student might apply.

{item_text}

Describe:
1. The specific misconception or procedural error (be precise — name the bug)
2. How a student with this misconception would work through this question step by step
3. Which answer option (A/B/C/D) this misconception leads to
4. How natural/attractive this error is — would the student even realise they went wrong?"""

STAGE1_ERROR_PERSPECTIVE = """You are a maths education researcher studying this question. Identify one specific reason why students might find this question difficult or easy.

{item_text}

Focus on ONE specific aspect:
- A particular step where students commonly go wrong
- A feature of the numbers/context that makes the error more or less likely
- A visual or notational trap
- A common shortcut that happens to work or fail here

Be specific and concrete, not generic."""

# ============================================================
# Stage 2 prompts: predict using seeds as context
# ============================================================

STAGE2_TEMPLATE = """You are an experienced UK maths teacher marking a set of Year 9 mock exams.

Here are {n_seeds} different perspectives on how students might approach this question:

{seed_content}

---

Now, considering ALL of the above perspectives, predict what percentage of students at each ability level would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%"""

# Direct baseline (no two-stage)
DIRECT_BASELINE = """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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


STAGE1_PROMPTS = {
    "cognitive_chains": STAGE1_COGNITIVE,
    "buggy_analyses": STAGE1_BUGGY,
    "error_perspectives": STAGE1_ERROR_PERSPECTIVE,
}

STAGE2_MODELS = {
    "gemini": {"provider": "gemini", "model": "gemini-3-flash-preview", "temp": 1.5},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temp": 1.0},
    "scout": {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "temp": 2.0},
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
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp, max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_anthropic(prompt, model, temp):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


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
    elif provider == "anthropic":
        return call_anthropic(prompt, model, temp)
    elif provider == "groq":
        return call_groq(prompt, model, temp)


def generate_seeds(item_text, stage1_prompt, qid, seed_dir, seed_provider="gemini", seed_model="gemini-3-flash-preview", seed_temp=2.0):
    """Generate N_SEEDS diverse stage-1 outputs at high temp."""
    seeds = []
    for i in range(N_SEEDS):
        seed_path = seed_dir / f"seed{i}.txt"
        if seed_path.exists():
            seeds.append(seed_path.read_text())
        else:
            prompt = stage1_prompt.format(item_text=item_text)
            try:
                text = make_call(prompt, seed_provider, seed_model, seed_temp)
                seed_path.write_text(text)
                seeds.append(text)
            except Exception as e:
                print(f"    SEED ERROR qid={qid} seed{i}: {e}", flush=True)
                time.sleep(2)
                continue
            time.sleep(0.15)
    return seeds


def main():
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    # Parse CLI args for selecting conditions/models
    # Use --scout-seeds to generate seeds with Scout instead of Gemini
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
    models = selected_models or list(STAGE2_MODELS.keys())

    # Seed generation config
    if use_scout_seeds:
        seed_provider = "groq"
        seed_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        seed_temp = 2.0
        seed_prefix = "seeds_scout_"
    else:
        seed_provider = "gemini"
        seed_model = "gemini-3-flash-preview"
        seed_temp = 2.0
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
            print(f"{config_key}", flush=True)
            print(f"{'='*60}", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                seed_dir = OUTPUT_DIR / f"{seed_prefix}{cond_name}" / f"rep{rep}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                items_pred = []
                for _, row in probe.iterrows():
                    qid = row["QuestionId"]
                    correct = row["correct_answer_kaggle"]
                    raw_path = raw_dir / f"qid{qid}.txt"

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        item_text = format_item_text(row)
                        # Generate seeds (cached per condition+rep)
                        qseed_dir = seed_dir / f"qid{qid}"
                        qseed_dir.mkdir(parents=True, exist_ok=True)
                        seeds = generate_seeds(item_text, stage1_prompt, qid, qseed_dir,
                                               seed_provider, seed_model, seed_temp)

                        if not seeds:
                            continue

                        # Format seeds as numbered perspectives
                        seed_content = "\n\n".join(
                            f"--- Perspective {i+1} ---\n{s}" for i, s in enumerate(seeds)
                        )

                        # Stage 2: predict using seeds
                        stage2_prompt = STAGE2_TEMPLATE.format(
                            n_seeds=len(seeds),
                            seed_content=seed_content,
                            item_text=item_text,
                        )
                        try:
                            text = make_call(
                                stage2_prompt,
                                mconfig["provider"], mconfig["model"], mconfig["temp"]
                            )
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} qid={qid}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        sleep = 0.5 if mconfig["provider"] == "anthropic" else 0.15
                        time.sleep(sleep)

                    p_inc = parse_predictions(text, correct)
                    items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                       "weighted_p_incorrect": p_inc})

                df = pd.DataFrame(items_pred)
                if len(df) >= 5:
                    rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                    rhos.append(rho)
                    print(f"  rep{rep}: rho={rho:.3f} (p={p:.4f})", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)
                result = {"config": config_key, "condition": cond_name, "stage2_model": mname,
                          "mean_rho": float(mean_rho), "std_rho": float(std_rho),
                          "rhos": [float(r) for r in rhos]}
                all_results.append(result)

    # === Direct baseline ===
    if "direct_baseline" in conditions:
        for mname in models:
            mconfig = STAGE2_MODELS[mname]
            config_key = f"direct_baseline__{mname}"
            print(f"\n{'='*60}", flush=True)
            print(f"{config_key}", flush=True)
            print(f"{'='*60}", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
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
                        prompt = DIRECT_BASELINE.format(item_text=item_text)
                        try:
                            text = make_call(
                                prompt,
                                mconfig["provider"], mconfig["model"], mconfig["temp"]
                            )
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} qid={qid}: {e}", flush=True)
                            time.sleep(2)
                            continue
                        sleep = 0.5 if mconfig["provider"] == "anthropic" else 0.15
                        time.sleep(sleep)

                    p_inc = parse_predictions(text, correct)
                    items_pred.append({"QuestionId": qid, "b_2pl": row["b_2pl"],
                                       "weighted_p_incorrect": p_inc})

                df = pd.DataFrame(items_pred)
                if len(df) >= 5:
                    rho, p = stats.spearmanr(df["weighted_p_incorrect"], df["b_2pl"])
                    rhos.append(rho)
                    print(f"  rep{rep}: rho={rho:.3f} (p={p:.4f})", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"  MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)
                result = {"config": config_key, "condition": "direct_baseline",
                          "stage2_model": mname,
                          "mean_rho": float(mean_rho), "std_rho": float(std_rho),
                          "rhos": [float(r) for r in rhos]}
                all_results.append(result)

    # === Summary ===
    print(f"\n{'='*60}", flush=True)
    print("ALL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for r in sorted(all_results, key=lambda x: x["mean_rho"], reverse=True):
        print(f"  {r['config']}: {r['mean_rho']:.3f} ± {r['std_rho']:.3f}", flush=True)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
