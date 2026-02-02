#!/usr/bin/env python3
"""
Run difficulty estimation on ALL 140 SmartPaper items.
Tests whether the ρ≈0.66-0.83 on 20 probe items generalizes to full set.

Usage:
    python3 scripts/run_smartpaper_expansion.py [gemini|scout|all]
"""
import json, re, os, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).parent.parent)

# Adapted for open-ended items with rubrics (not MCQ)
DIFFICULTY_PROMPT = """You are an experienced teacher in {subject} for Grade {grade} students in India.

For this open-ended question, estimate what proportion of students would score full marks.

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Think about:
- What specific errors or misunderstandings would cause students to lose marks?
- How clearly does the question communicate what's expected?
- What prerequisite knowledge is needed?
- How likely are students at this grade level to have that knowledge?

Respond with ONLY a number between 0 and 1 representing the proportion of students who would get full marks.
For example: 0.45

Your estimate:"""


def call_gemini(prompt, temp=2.0):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def call_scout(prompt, temp=2.0):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=256,
    )
    return resp.choices[0].message.content


def parse_proportion(text):
    """Extract a proportion (0-1) from model response."""
    # Try to find a decimal number
    matches = re.findall(r'(?:^|\s)(0?\.\d+|1\.0|0|1)(?:\s|$|\.)', text.strip())
    if matches:
        val = float(matches[0])
        if 0 <= val <= 1:
            return val
    # Try harder — any number that could be a proportion
    matches = re.findall(r'(\d+\.?\d*)', text.strip())
    for m in matches:
        val = float(m)
        if 0 <= val <= 1:
            return val
        if 1 < val <= 100:  # percentage
            return val / 100
    return None


def run_experiment(name, items, call_fn, n_reps, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []

    for rep in range(n_reps):
        raw_dir = output_dir / f"rep{rep}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(items):
            key = f"{item['assessment']}_q{item['question_number']}"
            safe_key = re.sub(r'[^\w]', '_', key)
            raw_path = raw_dir / f"{safe_key}.txt"

            if raw_path.exists() and raw_path.stat().st_size > 0:
                text = raw_path.read_text()
            else:
                prompt = DIFFICULTY_PROMPT.format(
                    subject=item['subject'],
                    grade=item['grade'],
                    question_text=item['question_text'],
                    rubric=item['rubric'],
                    max_score=item['max_score'],
                )
                try:
                    text = call_fn(prompt)
                    raw_path.write_text(text)
                except Exception as e:
                    print(f"  ERROR rep{rep} {safe_key}: {e}")
                    time.sleep(3)
                    try:
                        text = call_fn(prompt)
                        raw_path.write_text(text)
                    except Exception as e2:
                        print(f"  RETRY FAILED: {e2}")
                        continue
                time.sleep(0.3)

            p_est = parse_proportion(text)
            if p_est is not None:
                all_preds.append({
                    'assessment': item['assessment'],
                    'question_number': item['question_number'],
                    'subject': item['subject'],
                    'grade': item['grade'],
                    'classical_difficulty': item['classical_difficulty'],
                    'rep': rep,
                    'p_estimated': p_est,
                })

    pdf = pd.DataFrame(all_preds)
    if len(pdf) == 0:
        print(f"  {name}: No valid predictions!")
        return {}

    # Per-rep correlations
    rhos = []
    for rep in range(n_reps):
        sub = pdf[pdf['rep'] == rep]
        if len(sub) >= 10:
            rho, p = stats.spearmanr(sub['p_estimated'], sub['classical_difficulty'])
            rhos.append(rho)
            print(f"  {name} rep{rep}: rho={rho:.3f} (n={len(sub)}, p={p:.4f})")

    # Averaged predictions
    avg = pdf.groupby(['assessment', 'question_number']).agg(
        mean_p_est=('p_estimated', 'mean'),
        classical_difficulty=('classical_difficulty', 'first'),
        subject=('subject', 'first'),
    ).dropna()

    avg_rho, avg_p = stats.spearmanr(avg['mean_p_est'], avg['classical_difficulty'])
    print(f"\n  {name} avg-pred: rho={avg_rho:.3f} (p={avg_p:.6f}, n={len(avg)})")

    if rhos:
        print(f"  {name} mean per-rep: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")

    # By subject
    print(f"\n  By subject:")
    for subj in sorted(avg['subject'].unique()):
        sub = avg[avg['subject'] == subj]
        if len(sub) >= 5:
            rho, p = stats.spearmanr(sub['mean_p_est'], sub['classical_difficulty'])
            print(f"    {subj}: rho={rho:.3f} (n={len(sub)}, p={p:.4f})")

    # Check probe vs non-probe
    probe_items = json.load(open('pilot/smartpaper_rsm_v2/probe_items.json'))
    probe_keys = set((p['assessment'], p['question_number']) for p in probe_items)
    avg['is_probe'] = avg.apply(lambda r: (r.name[0], r.name[1]) in probe_keys, axis=1)

    probe_sub = avg[avg['is_probe']]
    nonprobe_sub = avg[~avg['is_probe']]

    if len(probe_sub) >= 5:
        rho, p = stats.spearmanr(probe_sub['mean_p_est'], probe_sub['classical_difficulty'])
        print(f"\n  Original 20 probe items: rho={rho:.3f} (p={p:.4f})")
    if len(nonprobe_sub) >= 5:
        rho, p = stats.spearmanr(nonprobe_sub['mean_p_est'], nonprobe_sub['classical_difficulty'])
        print(f"  Remaining {len(nonprobe_sub)} items: rho={rho:.3f} (p={p:.4f})")

    results = {
        'name': name,
        'n_items': len(items),
        'n_reps': n_reps,
        'avg_pred_rho': float(avg_rho),
        'avg_pred_p': float(avg_p),
        'mean_rho': float(np.mean(rhos)) if rhos else None,
        'std_rho': float(np.std(rhos)) if rhos else None,
        'rhos': [float(r) for r in rhos],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    avg.to_csv(output_dir / 'averaged_predictions.csv')

    return results


def main():
    selected = sys.argv[1] if len(sys.argv) > 1 else "all"

    items = json.load(open('data/smartpaper/item_statistics.json'))
    print(f"SmartPaper: {len(items)} items across {len(set(i['subject'] for i in items))} subjects\n")

    if selected in ("gemini", "all"):
        print("=" * 60)
        print("Gemini 3 Flash on ALL 140 SmartPaper items (t=2.0, 3 reps)")
        print("=" * 60)
        run_experiment("gemini_flash_140", items, call_gemini, 3,
                      "pilot/smartpaper_expansion/gemini_flash")

    if selected in ("scout", "all"):
        print("\n" + "=" * 60)
        print("Llama-4-Scout on ALL 140 SmartPaper items (t=2.0, 3 reps)")
        print("=" * 60)
        run_experiment("scout_140", items, call_scout, 3,
                      "pilot/smartpaper_expansion/llama4_scout")


if __name__ == "__main__":
    main()
