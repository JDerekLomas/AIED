#!/usr/bin/env python3
"""
Sweep temperature (1.5, 1.8, 2.0) × prompt variants (metaprompt-generated).
3 reps each for stability.
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OUTPUT_DIR = Path("pilot/rsm_experiment/metaprompt_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 3

# --- Prompt variants ---
# v3 = our current best (contrastive)
# v5-v10 = metaprompt-generated variations exploring different cognitive framings

PROMPTS = {
    "v3_contrastive": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v5_error_analysis": """You are an experienced UK maths teacher marking a set of Year 9 mock exams. You've just marked 200 papers.

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

    "v6_distractor_first": """You are an experienced UK maths teacher. First, analyze each answer option to understand what error or correct reasoning produces it. Then predict the percentage of students at each ability level who would choose each option.

{item_text}

Step 1 — For each option, briefly state what reasoning leads to it:
A) [reasoning]
B) [reasoning]
C) [reasoning]
D) [reasoning]

Step 2 — Now predict distributions:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v7_comparative_difficulty": """You are an experienced UK maths teacher. Rate this question's difficulty, then predict response distributions.

Consider: On a scale where "What is 2+3?" is difficulty 1 and "Solve a quadratic by completing the square" is difficulty 10, where does this question fall for a typical Year 9 class?

Think about: Is this the kind of question where students confidently get it wrong (high misconception activation) or where they're unsure and guess randomly?

{item_text}

DIFFICULTY: X/10
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v8_imagine_classroom": """You are a UK maths teacher about to give this question to your Year 9 class of 30 students. You know your students well.

Picture the moment after they submit their answers. Before looking at results, predict: what will the answer distribution look like? Which students will get it wrong and WHY — not just that they'll get it wrong, but what will go through their minds?

Remember: some questions that seem easy to you are actually tricky for students, and vice versa. Trust your classroom instinct over mathematical analysis.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v9_devil_advocate": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Important: Your first instinct is probably that this question is easier than it actually is (or harder). Teachers consistently misjudge difficulty because they think like mathematicians, not like students. Challenge your assumptions.

Ask yourself: "What if I'm wrong about how easy/hard this is?" If you think most students would get it right, consider what could trip them up. If you think it's hard, consider whether students might use a shortcut you didn't think of.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v10_sparse": """UK maths teacher. Predict student responses by ability level.

{item_text}

below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",
}

TEMPERATURES = [1.5, 1.8, 2.0]


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


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")

    # Allow selecting specific variants/temps via CLI
    selected_variants = None
    selected_temps = None
    for arg in sys.argv[1:]:
        if arg.startswith("v"):
            selected_variants = selected_variants or []
            selected_variants.append(arg)
        elif arg.replace(".","").isdigit():
            selected_temps = selected_temps or []
            selected_temps.append(float(arg))

    prompts_to_run = {k: v for k, v in PROMPTS.items() if not selected_variants or k in selected_variants}
    temps_to_run = selected_temps or TEMPERATURES

    total = len(prompts_to_run) * len(temps_to_run) * N_REPS * len(probe)
    print(f"Running {len(prompts_to_run)} prompts x {len(temps_to_run)} temps x {N_REPS} reps x {len(probe)} items = {total} calls", flush=True)

    all_results = []
    for vname, template in prompts_to_run.items():
        for temp in temps_to_run:
            config_key = f"{vname}_t{temp}"
            print(f"\n--- {config_key} ---", flush=True)

            rhos = []
            for rep in range(N_REPS):
                raw_dir = OUTPUT_DIR / config_key / f"rep{rep}"
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
                        prompt = template.format(item_text=item_text)
                        try:
                            text = make_api_call(client, prompt, temp)
                            raw_path.write_text(text)
                        except Exception as e:
                            print(f"  ERROR {config_key} rep{rep} qid={qid}: {e}", flush=True)
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
                    print(f"  {config_key} rep{rep}: rho={rho:.3f} (p={p:.3f}, parse_fail={parse_failures})", flush=True)

            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                result = {"variant": vname, "temperature": temp, "mean_rho": mean_rho,
                          "std_rho": std_rho, "rhos": rhos}
                all_results.append(result)
                print(f"  {config_key} MEAN: {mean_rho:.3f} ± {std_rho:.3f}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS MATRIX (mean_rho by variant × temperature)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Variant':<30}", end="", flush=True)
    for t in temps_to_run:
        print(f"  t={t:<6}", end="")
    print()
    for vname in prompts_to_run:
        print(f"{vname:<30}", end="")
        for t in temps_to_run:
            match = [r for r in all_results if r["variant"] == vname and r["temperature"] == t]
            if match:
                m = match[0]
                print(f"  {m['mean_rho']:.3f}±{m['std_rho']:.3f}", end="")
            else:
                print(f"  {'N/A':<12}", end="")
        print(flush=True)

    print(f"\n{'='*70}", flush=True)
    print("TOP 5 CONFIGS", flush=True)
    print(f"{'='*70}", flush=True)
    sorted_results = sorted(all_results, key=lambda x: x["mean_rho"], reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r['variant']}_t{r['temperature']}: {r['mean_rho']:.3f} ± {r['std_rho']:.3f}  {r['rhos']}", flush=True)

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
