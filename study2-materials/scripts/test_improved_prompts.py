#!/usr/bin/env python3
"""
Test improved teacher_prediction prompts against config 7 baseline.
Runs each variant on the same 20 probe items with temp=1.5, gemini-3-flash.
"""

import json, re, os, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)
np.random.seed(42)

OUTPUT_DIR = Path("pilot/rsm_experiment/prompt_variants")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Prompt variants ---

PROMPTS = {
    "v0_baseline": """You are an experienced UK maths teacher. For this question, predict the percentage of students at each ability level who would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v1_anchored": """You are an experienced UK maths teacher with 20 years experience across Years 7-11.

For this question, predict what percentage of students at each ability level would choose each option.

Important calibration notes:
- Below-basic students (bottom 25%) often guess or apply incorrect procedures. On hard topics they perform near chance (25% per option). On familiar topics they may reach 40-50% correct.
- Basic students (next 35%) can handle routine problems but struggle with multi-step or unfamiliar formats. Typical correct rates: 30-60%.
- Proficient students (next 25%) usually get questions right but can be caught by subtle distractors. Typical correct rates: 60-85%.
- Advanced students (top 15%) rarely make errors. Typical correct rates: 85-98%.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v2_rank_then_predict": """You are an experienced UK maths teacher. 

Step 1: First, assess this question's overall difficulty on a scale from 1 (very easy, >80% of students correct) to 10 (very hard, <20% correct). Consider what specific errors students would make and how common those errors are in practice.

Step 2: Then predict the percentage of students at each ability level who would choose each option. Your predictions should be consistent with your difficulty rating.

{item_text}

Respond in this exact format:
DIFFICULTY: X/10
REASONING: [one sentence on what makes this easy or hard for real students]
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",

    "v3_contrastive": """You are an experienced UK maths teacher. For this question, predict what percentage of students at each ability level would choose each option.

Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students.

Focus on: Would real students actually make errors on THIS specific question, or would most get it right despite the misconception being testable?

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%""",
}


def format_item_text(row):
    """Format item — hidden hints (no correct answer shown)."""
    lines = [
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
    ]
    return "\n".join(lines)


def parse_predictions(text, correct_answer):
    """Parse teacher predictions, return weighted p_incorrect."""
    weights = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}
    correct_idx = ord(correct_answer) - ord('A')
    
    weighted_p_correct = 0.0
    level_data = {}
    for level, w in weights.items():
        pattern = rf'{level.replace("_", "[_ ]")}:\s*A\s*=\s*(\d+)%?\s*B\s*=\s*(\d+)%?\s*C\s*=\s*(\d+)%?\s*D\s*=\s*(\d+)%?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pcts = [int(match.group(i)) for i in range(1, 5)]
            total = sum(pcts) or 1
            p_correct = pcts[correct_idx] / total
            weighted_p_correct += p_correct * w
            level_data[level] = p_correct
        else:
            weighted_p_correct += 0.5 * w  # fallback
            level_data[level] = None
    
    # Also extract difficulty rating for v2
    diff_match = re.search(r'DIFFICULTY:\s*(\d+)', text)
    difficulty_rating = int(diff_match.group(1)) if diff_match else None
    
    return {
        "weighted_p_incorrect": 1 - weighted_p_correct,
        "difficulty_rating": difficulty_rating,
        **{f"{l}_p_correct": v for l, v in level_data.items()},
    }


def make_api_call(client, prompt, temperature=1.5):
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


def run_variant(variant_name, prompt_template, probe, client):
    """Run a prompt variant on all probe items."""
    raw_dir = OUTPUT_DIR / variant_name
    raw_dir.mkdir(exist_ok=True)
    
    results = []
    for _, row in probe.iterrows():
        qid = row["QuestionId"]
        correct = row["correct_answer_kaggle"]
        
        raw_path = raw_dir / f"qid{qid}.txt"
        if raw_path.exists():
            text = raw_path.read_text()
        else:
            item_text = format_item_text(row)
            prompt = prompt_template.format(item_text=item_text)
            try:
                text = make_api_call(client, prompt)
                raw_path.write_text(text)
            except Exception as e:
                print(f"  ERROR {variant_name} qid={qid}: {e}")
                time.sleep(1)
                continue
            time.sleep(0.1)
        
        parsed = parse_predictions(text, correct)
        parsed["QuestionId"] = qid
        parsed["b_2pl"] = row["b_2pl"]
        results.append(parsed)
    
    return pd.DataFrame(results)


def evaluate(df, label):
    """Compute and print correlations."""
    valid = df[["weighted_p_incorrect", "b_2pl"]].dropna()
    if len(valid) < 5:
        print(f"  {label}: insufficient data")
        return None
    
    rho, p_rho = stats.spearmanr(valid["weighted_p_incorrect"], valid["b_2pl"])
    r, p_r = stats.pearsonr(valid["weighted_p_incorrect"], valid["b_2pl"])
    
    # Range of predictions
    pred_range = valid["weighted_p_incorrect"].max() - valid["weighted_p_incorrect"].min()
    pred_std = valid["weighted_p_incorrect"].std()
    
    print(f"  {label}: Spearman rho={rho:.3f} (p={p_rho:.4f}), Pearson r={r:.3f}, "
          f"pred_range={pred_range:.2f}, pred_std={pred_std:.3f}")
    
    # Per-level correlations
    for level in ["below_basic", "basic", "proficient", "advanced"]:
        col = f"{level}_p_correct"
        if col in df.columns:
            lv = df[[col, "b_2pl"]].dropna()
            if len(lv) >= 5:
                lr, lp = stats.spearmanr(lv[col], lv["b_2pl"])
                print(f"    {level}: rho={lr:.3f}, mean={lv[col].mean():.2f}")
    
    if "difficulty_rating" in df.columns:
        dr = df[["difficulty_rating", "b_2pl"]].dropna()
        if len(dr) >= 5:
            dr_rho, dr_p = stats.spearmanr(dr["difficulty_rating"], dr["b_2pl"])
            print(f"    difficulty_rating: rho={dr_rho:.3f} (p={dr_p:.4f})")
    
    return rho


def main():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    probe = pd.read_csv("pilot/rsm_experiment/probe_items.csv")
    print(f"Testing {len(PROMPTS)} prompt variants on {len(probe)} probe items\n")
    
    all_rhos = {}
    for variant_name, template in PROMPTS.items():
        print(f"\n--- {variant_name} ---")
        df = run_variant(variant_name, template, probe, client)
        rho = evaluate(df, variant_name)
        all_rhos[variant_name] = rho
        df.to_csv(OUTPUT_DIR / f"{variant_name}_results.csv", index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, rho in sorted(all_rhos.items(), key=lambda x: x[1] or -999, reverse=True):
        print(f"  {name}: rho={rho:.3f}" if rho else f"  {name}: FAILED")


if __name__ == "__main__":
    main()
