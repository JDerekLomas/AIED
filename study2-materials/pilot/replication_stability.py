#!/usr/bin/env python3
"""
Replication stability analysis for AIED paper revision.

Answers Reviewer 4: "How stable is the rank order of items across three replications?"

For each prompt condition we compute:
  - Spearman rho between every pair of replications (inter-rep stability)
  - Each individual rep's correlation with ground-truth difficulty
  - The averaged-across-reps prediction's correlation with ground truth
  - Fraction of items whose rank changes by >20 positions across any rep pair
"""

import json, os, re, glob, sys
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent
GT_PATH = BASE.parent / "data" / "smartpaper" / "item_statistics.json"

# ── Ground truth ──────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_items = json.load(f)
gt = {f'{it["assessment"]}_q{it["question_number"]}': it["classical_difficulty"]
      for it in gt_items}

# ── Data sources ──────────────────────────────────────────────────────────
# Screening: prompt_framing_experiment/{prompt_name}/rep{0,1,2}/
# Model survey: model_survey/gemini/{prompt}/rep{0,1,2}/

SCREENING_PROMPTS = {
    "prerequisite_chain_t0.5": ("screening", "prerequisite_chain_t0.5"),
    "cognitive_load_t2.0":     ("screening", "cognitive_load_t2.0"),
    "buggy_rules_t1.0":       ("screening", "buggy_rules_t1.0"),
    "teacher_t1.0":            ("screening", "teacher_t1.0"),
}

MODEL_SURVEY_PROMPTS = {
    "gemini/prerequisite_chain": ("model_survey", "prerequisite_chain"),
    "gemini/cognitive_load":     ("model_survey", "cognitive_load"),
    "gemini/teacher":            ("model_survey", "teacher"),
    "gemini/contrastive":        ("model_survey", "contrastive"),
}

def extract_prediction(filepath):
    """Extract predicted p-correct from the last line containing a bare float."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Walk backwards to find the last line that is (or contains) a bare float
    for line in reversed(lines):
        line = line.strip()
        # Try the whole line as a float
        try:
            val = float(line)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
        # Try to find a bare float at the end of the line
        m = re.search(r'(\d+\.\d+)\s*$', line)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
    return None

def normalize_screening_key(filename):
    """Grade 6 — English_q1.txt -> Grade 6 — English_q1"""
    return filename.replace('.txt', '')

def normalize_model_survey_key(filename):
    """Grade_6___English_q1.txt -> Grade 6 — English_q1"""
    k = filename.replace('.txt', '')
    k = k.replace('___', ' \u2014 ')  # triple underscore -> em dash
    k = re.sub(r' (q\d+)$', r'_\1', k.replace('_', ' '))
    return k

def load_reps(source_type, prompt_name):
    """Load predictions for rep0, rep1, rep2. Returns {item_key: [pred0, pred1, pred2]}."""
    reps = {}
    for rep_idx in range(3):
        if source_type == "screening":
            rep_dir = BASE / "prompt_framing_experiment" / prompt_name / f"rep{rep_idx}"
            norm_fn = normalize_screening_key
        else:
            rep_dir = BASE / "model_survey" / "gemini" / prompt_name / f"rep{rep_idx}"
            norm_fn = normalize_model_survey_key

        if not rep_dir.exists():
            print(f"  WARNING: {rep_dir} does not exist", file=sys.stderr)
            continue

        for fp in sorted(rep_dir.glob("*.txt")):
            key = norm_fn(fp.name)
            pred = extract_prediction(fp)
            if pred is not None:
                reps.setdefault(key, [None, None, None])
                reps[key][rep_idx] = pred

    return reps

def analyze_condition(label, source_type, prompt_name):
    """Run full stability analysis for one condition."""
    reps = load_reps(source_type, prompt_name)

    # Filter to items present in all 3 reps AND in ground truth
    valid_keys = sorted(k for k, v in reps.items()
                        if all(x is not None for x in v) and k in gt)

    n = len(valid_keys)
    if n == 0:
        return None

    preds = np.array([[reps[k][r] for k in valid_keys] for r in range(3)])  # (3, n_items)
    truth = np.array([gt[k] for k in valid_keys])

    # Spearman correlations: inter-rep pairs
    pairs = [(0, 1), (0, 2), (1, 2)]
    inter_rhos = []
    for i, j in pairs:
        rho, _ = stats.spearmanr(preds[i], preds[j])
        inter_rhos.append(rho)
    mean_inter = np.mean(inter_rhos)

    # Each rep vs ground truth
    rep_gt_rhos = []
    for r in range(3):
        rho, _ = stats.spearmanr(preds[r], truth)
        rep_gt_rhos.append(rho)

    # Averaged predictions vs ground truth
    avg_preds = preds.mean(axis=0)
    avg_rho, _ = stats.spearmanr(avg_preds, truth)

    # Rank displacement: fraction of items with rank change > 20 across any rep pair
    ranks = np.array([stats.rankdata(preds[r]) for r in range(3)])  # (3, n)
    max_displacements = np.zeros(n)
    for i, j in pairs:
        disp = np.abs(ranks[i] - ranks[j])
        max_displacements = np.maximum(max_displacements, disp)
    frac_large_shift = np.mean(max_displacements > 20)

    return {
        'label': label,
        'n_items': n,
        'rep_gt_rhos': rep_gt_rhos,
        'inter_rhos': inter_rhos,
        'mean_inter': mean_inter,
        'avg_rho': avg_rho,
        'frac_large_shift': frac_large_shift,
        'mean_max_disp': np.mean(max_displacements),
        'median_max_disp': np.median(max_displacements),
    }


# ── Run analysis ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_conditions = {}
    all_conditions.update(SCREENING_PROMPTS)
    all_conditions.update(MODEL_SURVEY_PROMPTS)

    results = []
    for label, (src, name) in all_conditions.items():
        r = analyze_condition(label, src, name)
        if r:
            results.append(r)

    # ── Print table ───────────────────────────────────────────────────────
    print("=" * 110)
    print("REPLICATION STABILITY ANALYSIS")
    print("=" * 110)
    print()

    header = f"{'Condition':<32} {'N':>4} {'rep0→GT':>8} {'rep1→GT':>8} {'rep2→GT':>8} {'Inter-rep':>10} {'Avg→GT':>8} {'%Shift>20':>10}"
    print(header)
    print("-" * 110)

    for r in results:
        print(f"{r['label']:<32} {r['n_items']:>4} "
              f"{r['rep_gt_rhos'][0]:>8.3f} {r['rep_gt_rhos'][1]:>8.3f} {r['rep_gt_rhos'][2]:>8.3f} "
              f"{r['mean_inter']:>10.3f} {r['avg_rho']:>8.3f} {r['frac_large_shift']*100:>9.1f}%")

    print()
    print("=" * 110)
    print("SUMMARY STATISTICS")
    print("=" * 110)

    # Aggregate across screening prompts
    screening = [r for r in results if r['label'] in SCREENING_PROMPTS]
    model_survey = [r for r in results if r['label'] in MODEL_SURVEY_PROMPTS]

    for group_name, group in [("Screening (Phase 1)", screening), ("Model Survey (Phase 2)", model_survey)]:
        if not group:
            continue
        inter_rhos = [r['mean_inter'] for r in group]
        avg_gt = [r['avg_rho'] for r in group]
        shifts = [r['frac_large_shift'] for r in group]
        print(f"\n{group_name}:")
        print(f"  Mean inter-rep ρ:        {np.mean(inter_rhos):.3f} (range {np.min(inter_rhos):.3f}–{np.max(inter_rhos):.3f})")
        print(f"  Mean averaged→GT ρ:      {np.mean(avg_gt):.3f} (range {np.min(avg_gt):.3f}–{np.max(avg_gt):.3f})")
        print(f"  Mean % items shift >20:  {np.mean(shifts)*100:.1f}%")

    # Overall
    all_inter = [r['mean_inter'] for r in results]
    all_shifts = [r['frac_large_shift'] for r in results]
    print(f"\nOverall across all {len(results)} conditions:")
    print(f"  Mean inter-rep ρ:        {np.mean(all_inter):.3f} (range {np.min(all_inter):.3f}–{np.max(all_inter):.3f})")
    print(f"  Mean % items shift >20:  {np.mean(all_shifts)*100:.1f}%")

    # Detailed inter-rep breakdown
    print()
    print("=" * 110)
    print("DETAILED INTER-REP CORRELATIONS")
    print("=" * 110)
    header2 = f"{'Condition':<32} {'ρ(0,1)':>8} {'ρ(0,2)':>8} {'ρ(1,2)':>8} {'Mean disp':>10} {'Med disp':>10}"
    print(header2)
    print("-" * 110)
    for r in results:
        print(f"{r['label']:<32} "
              f"{r['inter_rhos'][0]:>8.3f} {r['inter_rhos'][1]:>8.3f} {r['inter_rhos'][2]:>8.3f} "
              f"{r['mean_max_disp']:>10.1f} {r['median_max_disp']:>10.1f}")
