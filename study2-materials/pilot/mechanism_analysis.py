#!/usr/bin/env python3
"""
Mechanism Analysis: Why do prerequisite_chain and cognitive_load prompts
work for item difficulty estimation?

Analyzes raw model outputs to extract enumerated counts, correlate them
with ground truth difficulty, and test mediation hypotheses.
"""

import json
import os
import re
import glob
import random
from collections import defaultdict

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BASE = "/Users/dereklomas/AIED/study2-materials"
GROUND_TRUTH_PATH = f"{BASE}/data/smartpaper/item_statistics.json"

SCREENING_DIRS = {
    "prerequisite_chain": f"{BASE}/pilot/prompt_framing_experiment/prerequisite_chain_t0.5",
    "cognitive_load": f"{BASE}/pilot/prompt_framing_experiment/cognitive_load_t2.0",
    "buggy_rules": f"{BASE}/pilot/prompt_framing_experiment/buggy_rules_t1.0",
    "teacher": f"{BASE}/pilot/prompt_framing_experiment/teacher_t1.0",
}

MODEL_SURVEY_DIR = f"{BASE}/pilot/model_survey"

N_BOOTSTRAP = 1000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# Load ground truth
# ─────────────────────────────────────────────────────────────
def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        items = json.load(f)
    gt = {}
    for item in items:
        key = f"{item['assessment']}_q{item['question_number']}"
        gt[key] = item["classical_difficulty"]
    return gt


# ─────────────────────────────────────────────────────────────
# Filename normalization
# ─────────────────────────────────────────────────────────────
def normalize_screening_filename(fname):
    """Strip .txt from screening experiment files."""
    return fname.replace(".txt", "")


def normalize_model_survey_filename(fname):
    """Convert model_survey filenames to ground truth keys."""
    k = fname.replace(".txt", "")
    k = k.replace("___", " \u2014 ")  # triple underscore -> em dash
    k = re.sub(r"_(?!q\d)", " ", k)   # remaining _ -> space, except _q
    # Fix: the regex above might not catch _q at end properly
    # Actually let's be more careful:
    k = fname.replace(".txt", "")
    k = k.replace("___", " \u2014 ")
    # Replace _ with space, but preserve _q suffix
    k = re.sub(r" (q\d+)$", r"_\1", k.replace("_", " ").replace(" \u2014 ", "___"))
    # Undo the triple underscore hack
    k = k.replace("___", " \u2014 ")
    return k


# ─────────────────────────────────────────────────────────────
# Parsing functions
# ─────────────────────────────────────────────────────────────
def extract_final_float(text):
    """Extract the last float (0-1) from text."""
    # Look for floats on their own line, searching from the end
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        # Match lines that are just a float, possibly with prefix like "Your estimate:"
        m = re.search(r'(?:^|\s)(0?\.\d+)\s*$', line)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 1:
                return val
        # Also match "0.XX" at start of line
        m = re.match(r'^(0?\.\d+)\s*$', line)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 1:
                return val
    return None


def parse_prerequisite_chain(text):
    """Parse prerequisite chain output: COUNT field, numbered items, final float."""
    # COUNT field
    count_match = re.search(r'COUNT:\s*\[?(\d+)\]?', text)
    count_from_field = int(count_match.group(1)) if count_match else None

    # Count numbered list items as backup
    numbered_items = re.findall(r'^\d+[\.\)]\s+\S', text, re.MULTILINE)
    count_from_list = len(numbered_items)

    # Extract prerequisite text (everything between PREREQUISITES: and COUNT:)
    prereq_text = ""
    m = re.search(r'PREREQUISITES:\s*\n(.*?)(?=\nCOUNT:|\Z)', text, re.DOTALL)
    if m:
        prereq_text = m.group(1).strip()

    count = count_from_field if count_from_field is not None else count_from_list
    prediction = extract_final_float(text)

    return {
        "count": count,
        "count_from_field": count_from_field,
        "count_from_list": count_from_list,
        "prediction": prediction,
        "prereq_text": prereq_text,
        "raw": text,
    }


def parse_cognitive_load(text):
    """Parse cognitive load output: ELEMENT COUNT field, numbered items, final float."""
    # Try multiple patterns for count field
    count_from_field = None
    for pattern in [
        r'ELEMENT.?COUNT:\s*\[?(\d+)\]?',
        r'ELEMENTS:\s*\[?(\d+)\]?',
        r'ELEMENT_COUNT:\s*\[?(\d+)\]?',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            count_from_field = int(m.group(1))
            break

    numbered_items = re.findall(r'^\d+[\.\)]\s+\S', text, re.MULTILINE)
    count_from_list = len(numbered_items)

    count = count_from_field if count_from_field is not None else count_from_list
    prediction = extract_final_float(text)

    return {
        "count": count,
        "count_from_field": count_from_field,
        "count_from_list": count_from_list,
        "prediction": prediction,
        "raw": text,
    }


def parse_buggy_rules(text):
    """Parse buggy rules output: Step 1 count, Step 2 count, final float."""
    # Find Step 1 and Step 2 sections
    step1_items = 0
    step2_items = 0

    # Split by Step markers
    step1_match = re.search(r'Step 1.*?\n(.*?)(?=Step 2|\Z)', text, re.DOTALL | re.IGNORECASE)
    step2_match = re.search(r'Step 2.*?\n(.*?)(?=Step 3|\Z)', text, re.DOTALL | re.IGNORECASE)

    if step1_match:
        step1_items = len(re.findall(r'^\d+[\.\)]\s+\S', step1_match.group(1), re.MULTILINE))
        # Also count **bold** items
        if step1_items == 0:
            step1_items = len(re.findall(r'^\d+[\.\)]\s+\*\*', step1_match.group(1), re.MULTILINE))

    if step2_match:
        step2_items = len(re.findall(r'^\d+[\.\)]\s+\S', step2_match.group(1), re.MULTILINE))
        if step2_items == 0:
            step2_items = len(re.findall(r'^\d+[\.\)]\s+\*\*', step2_match.group(1), re.MULTILINE))

    prediction = extract_final_float(text)

    return {
        "step1_count": step1_items,
        "step2_count": step2_items,
        "total_count": step1_items + step2_items,
        "prediction": prediction,
        "raw": text,
    }


def parse_teacher(text):
    """Parse teacher output: just a float."""
    prediction = extract_final_float(text)
    return {"prediction": prediction, "raw": text}


# ─────────────────────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────────────────────
def load_screening_outputs(prompt_type, parse_fn, normalize_fn=normalize_screening_filename):
    """Load outputs from screening experiment across 3 reps."""
    base_dir = SCREENING_DIRS[prompt_type]
    results = defaultdict(list)  # key -> [rep0_parsed, rep1_parsed, rep2_parsed]

    for rep in range(3):
        rep_dir = os.path.join(base_dir, f"rep{rep}")
        if not os.path.isdir(rep_dir):
            continue
        for fname in os.listdir(rep_dir):
            if not fname.endswith(".txt"):
                continue
            key = normalize_fn(fname)
            fpath = os.path.join(rep_dir, fname)
            try:
                with open(fpath) as f:
                    text = f.read()
                parsed = parse_fn(text)
                results[key].append(parsed)
            except Exception as e:
                pass  # skip corrupt files

    return dict(results)


def load_model_survey_outputs(model, prompt_type, parse_fn):
    """Load outputs from model survey for a specific model and prompt type."""
    base_dir = os.path.join(MODEL_SURVEY_DIR, model, prompt_type)
    results = defaultdict(list)

    for rep in range(3):
        rep_dir = os.path.join(base_dir, f"rep{rep}")
        if not os.path.isdir(rep_dir):
            continue
        for fname in os.listdir(rep_dir):
            if not fname.endswith(".txt"):
                continue
            key = normalize_model_survey_filename(fname)
            fpath = os.path.join(rep_dir, fname)
            try:
                with open(fpath) as f:
                    text = f.read()
                parsed = parse_fn(text)
                results[key].append(parsed)
            except Exception:
                pass

    return dict(results)


# ─────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────
def bootstrap_spearman_ci(x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 95% CI for Spearman correlation."""
    x, y = np.array(x), np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, (np.nan, np.nan), len(x)

    rho, p = stats.spearmanr(x, y)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        try:
            r, _ = stats.spearmanr(x[idx], y[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        except:
            pass

    if len(boot_rhos) < 10:
        return rho, (np.nan, np.nan), n

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_rhos, 100 * alpha)
    hi = np.percentile(boot_rhos, 100 * (1 - alpha))
    return rho, (lo, hi), n


def partial_correlation(x, y, z):
    """Partial correlation of x and y, controlling for z (Spearman-based)."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return np.nan, len(x)

    # Rank-based partial correlation
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    # Residualize x and y on z using OLS on ranks
    def residualize(v, c):
        c = c.reshape(-1, 1)
        c = np.hstack([c, np.ones((len(c), 1))])
        beta = np.linalg.lstsq(c, v, rcond=None)[0]
        return v - c @ beta

    rx_res = residualize(rx, rz)
    ry_res = residualize(ry, rz)

    r, _ = stats.pearsonr(rx_res, ry_res)
    return r, len(x)


# ─────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────
def average_across_reps(results, field):
    """Average a numeric field across reps for each item."""
    out = {}
    for key, reps in results.items():
        vals = [r[field] for r in reps if r.get(field) is not None]
        if vals:
            out[key] = np.mean(vals)
    return out


def align_with_gt(values_dict, gt):
    """Align a dict of item->value with ground truth, returning matched arrays."""
    keys = sorted(set(values_dict.keys()) & set(gt.keys()))
    x = np.array([values_dict[k] for k in keys])
    y = np.array([gt[k] for k in keys])
    return x, y, keys


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────
def main():
    gt = load_ground_truth()
    print(f"Loaded ground truth for {len(gt)} items")
    print()

    # =========================================================
    # Load all screening data
    # =========================================================
    prereq_data = load_screening_outputs("prerequisite_chain", parse_prerequisite_chain)
    cogload_data = load_screening_outputs("cognitive_load", parse_cognitive_load)
    buggy_data = load_screening_outputs("buggy_rules", parse_buggy_rules)
    teacher_data = load_screening_outputs("teacher", parse_teacher)

    print(f"Loaded screening outputs:")
    print(f"  prerequisite_chain: {len(prereq_data)} items")
    print(f"  cognitive_load:     {len(cogload_data)} items")
    print(f"  buggy_rules:        {len(buggy_data)} items")
    print(f"  teacher:            {len(teacher_data)} items")
    print()

    # =========================================================
    # SECTION A: Count → Difficulty Correlations
    # =========================================================
    print("=" * 70)
    print("SECTION A: Count → Difficulty Correlations (Screening Experiment)")
    print("=" * 70)
    print()

    for prompt_type, data, count_field in [
        ("prerequisite_chain", prereq_data, "count"),
        ("cognitive_load", cogload_data, "count"),
        ("buggy_rules (step1)", buggy_data, "step1_count"),
        ("buggy_rules (step2)", buggy_data, "step2_count"),
        ("buggy_rules (total)", buggy_data, "total_count"),
    ]:
        avg_counts = average_across_reps(data, count_field)
        avg_preds = average_across_reps(data, "prediction")

        counts_arr, gt_arr, keys = align_with_gt(avg_counts, gt)
        preds_arr, gt_arr2, keys2 = align_with_gt(avg_preds, gt)

        print(f"--- {prompt_type} ---")

        if len(counts_arr) >= 5:
            rho, ci, n = bootstrap_spearman_ci(counts_arr, gt_arr)
            print(f"  count vs ground_truth:      rho={rho:+.3f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  n={n}")

        if len(preds_arr) >= 5:
            rho, ci, n = bootstrap_spearman_ci(preds_arr, gt_arr2)
            print(f"  prediction vs ground_truth:  rho={rho:+.3f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  n={n}")

        # Count vs prediction
        if len(counts_arr) >= 5 and len(preds_arr) >= 5:
            # Align count and prediction on same keys
            shared_keys = sorted(set(avg_counts.keys()) & set(avg_preds.keys()))
            c = np.array([avg_counts[k] for k in shared_keys])
            p = np.array([avg_preds[k] for k in shared_keys])
            rho, ci, n = bootstrap_spearman_ci(c, p)
            print(f"  count vs prediction:         rho={rho:+.3f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  n={n}")

        print()

    # Teacher (no count, just prediction)
    avg_teacher_preds = average_across_reps(teacher_data, "prediction")
    preds_arr, gt_arr, _ = align_with_gt(avg_teacher_preds, gt)
    rho, ci, n = bootstrap_spearman_ci(preds_arr, gt_arr)
    print(f"--- teacher (baseline, no enumeration) ---")
    print(f"  prediction vs ground_truth:  rho={rho:+.3f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  n={n}")
    print()

    # =========================================================
    # SECTION B: Mediation Analysis
    # =========================================================
    print("=" * 70)
    print("SECTION B: Mediation Analysis")
    print("=" * 70)
    print()
    print("Does the enumerated count mediate the prediction→ground_truth relationship?")
    print("If count fully mediates: partial_corr(pred, gt | count) ≈ 0")
    print("If model adds info beyond count: partial_corr(pred, gt | count) remains significant")
    print()

    for prompt_type, data, count_field in [
        ("prerequisite_chain", prereq_data, "count"),
        ("cognitive_load", cogload_data, "count"),
        ("buggy_rules", buggy_data, "total_count"),
    ]:
        avg_counts = average_across_reps(data, count_field)
        avg_preds = average_across_reps(data, "prediction")

        # Align all three on shared keys
        shared_keys = sorted(set(avg_counts.keys()) & set(avg_preds.keys()) & set(gt.keys()))
        if len(shared_keys) < 5:
            print(f"  {prompt_type}: insufficient data ({len(shared_keys)} items)")
            continue

        c = np.array([avg_counts[k] for k in shared_keys])
        p = np.array([avg_preds[k] for k in shared_keys])
        g = np.array([gt[k] for k in shared_keys])

        r_pred_gt_ctrl_count, n1 = partial_correlation(p, g, c)
        r_count_gt_ctrl_pred, n2 = partial_correlation(c, g, p)

        print(f"--- {prompt_type} (n={n1}) ---")
        print(f"  partial_corr(prediction, gt | count):  r={r_pred_gt_ctrl_count:+.3f}")
        print(f"  partial_corr(count, gt | prediction):  r={r_count_gt_ctrl_pred:+.3f}")

        # Interpretation
        if abs(r_pred_gt_ctrl_count) > abs(r_count_gt_ctrl_pred):
            print(f"  → Model prediction adds information BEYOND the count")
        else:
            print(f"  → Count carries most of the predictive signal")
        print()

    # =========================================================
    # SECTION C: Count Calibration (Binned Analysis)
    # =========================================================
    print("=" * 70)
    print("SECTION C: Count Calibration — More Prerequisites → Lower p-correct?")
    print("=" * 70)
    print()

    for prompt_type, data, count_field, bins, labels in [
        ("prerequisite_chain", prereq_data, "count",
         [0, 3.5, 5.5, 7.5, 100], ["1-3", "4-5", "6-7", "8+"]),
        ("cognitive_load", cogload_data, "count",
         [0, 3.5, 5.5, 7.5, 100], ["1-3", "4-5", "6-7", "8+"]),
    ]:
        avg_counts = average_across_reps(data, count_field)
        shared_keys = sorted(set(avg_counts.keys()) & set(gt.keys()))

        counts = np.array([avg_counts[k] for k in shared_keys])
        gts = np.array([gt[k] for k in shared_keys])

        print(f"--- {prompt_type} ---")
        print(f"  {'Bin':<8} {'N':>4} {'Mean p-correct':>16} {'Mean count':>12} {'SD p-correct':>14}")

        bin_indices = np.digitize(counts, bins) - 1
        for i, label in enumerate(labels):
            mask = bin_indices == i
            if mask.sum() == 0:
                print(f"  {label:<8} {0:>4}     {'—':>12}")
                continue
            mean_p = np.mean(gts[mask])
            sd_p = np.std(gts[mask])
            mean_c = np.mean(counts[mask])
            print(f"  {label:<8} {mask.sum():>4} {mean_p:>16.3f} {mean_c:>12.1f} {sd_p:>14.3f}")

        # Test monotonicity
        bin_means = []
        for i in range(len(labels)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(np.mean(gts[mask]))
        is_monotonic = all(bin_means[i] >= bin_means[i+1] for i in range(len(bin_means)-1))
        print(f"  Monotonically decreasing: {is_monotonic}")
        print()

    # =========================================================
    # SECTION D: Worked Examples
    # =========================================================
    print("=" * 70)
    print("SECTION D: Worked Examples")
    print("=" * 70)
    print()

    # Use prerequisite_chain screening data
    avg_counts_prereq = average_across_reps(prereq_data, "count")
    avg_preds_prereq = average_across_reps(prereq_data, "prediction")
    shared_keys = sorted(set(avg_counts_prereq.keys()) & set(avg_preds_prereq.keys()) & set(gt.keys()))

    # Sort by ground truth difficulty
    by_difficulty = sorted(shared_keys, key=lambda k: gt[k], reverse=True)

    # 2 easiest
    easy = by_difficulty[:2]
    # 2 hardest
    hard = by_difficulty[-2:]

    # 2 "interesting failures" — biggest residual between count rank and difficulty rank
    count_ranks = {k: r for r, k in enumerate(sorted(shared_keys, key=lambda k: avg_counts_prereq[k]))}
    diff_ranks = {k: r for r, k in enumerate(sorted(shared_keys, key=lambda k: gt[k]))}
    residuals = {k: abs(count_ranks[k] - diff_ranks[k]) for k in shared_keys}
    by_residual = sorted(shared_keys, key=lambda k: residuals[k], reverse=True)

    # Get failures that are specifically interesting: high count but easy, or low count but hard
    high_count_easy = [k for k in shared_keys if avg_counts_prereq[k] >= np.percentile(
        [avg_counts_prereq[kk] for kk in shared_keys], 75) and gt[k] >= np.percentile(
        [gt[kk] for kk in shared_keys], 75)]
    low_count_hard = [k for k in shared_keys if avg_counts_prereq[k] <= np.percentile(
        [avg_counts_prereq[kk] for kk in shared_keys], 25) and gt[k] <= np.percentile(
        [gt[kk] for kk in shared_keys], 25)]

    failures = []
    if high_count_easy:
        failures.append(sorted(high_count_easy, key=lambda k: residuals[k], reverse=True)[0])
    if low_count_hard:
        failures.append(sorted(low_count_hard, key=lambda k: residuals[k], reverse=True)[0])
    # Fill with top residuals if needed
    for k in by_residual:
        if len(failures) >= 2:
            break
        if k not in failures and k not in easy and k not in hard:
            failures.append(k)

    def print_item(key, category):
        print(f"  [{category}] {key}")
        print(f"  Ground truth p-correct: {gt[key]:.3f}")
        print(f"  Avg prerequisite count: {avg_counts_prereq[key]:.1f}")
        print(f"  Avg model prediction:   {avg_preds_prereq[key]:.3f}")
        # Print prerequisite list from rep0
        if key in prereq_data and prereq_data[key]:
            text = prereq_data[key][0].get("prereq_text", "")
            if text:
                print(f"  Prerequisites (rep0):")
                for line in text.split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
        print()

    for k in easy:
        print_item(k, "EASY")
    for k in hard:
        print_item(k, "HARD")
    for k in failures[:2]:
        is_hc_easy = k in high_count_easy if high_count_easy else False
        label = "HIGH COUNT BUT EASY" if is_hc_easy else "LOW COUNT BUT HARD"
        print_item(k, label)

    # =========================================================
    # SECTION E: Cross-Model Count Comparison (Model Survey)
    # =========================================================
    print("=" * 70)
    print("SECTION E: Cross-Model Count Comparison (prerequisite_chain)")
    print("=" * 70)
    print()

    # Find all models with prerequisite_chain
    models_with_prereq = []
    for entry in os.listdir(MODEL_SURVEY_DIR):
        model_dir = os.path.join(MODEL_SURVEY_DIR, entry, "prerequisite_chain")
        if os.path.isdir(model_dir):
            models_with_prereq.append(entry)

    models_with_prereq.sort()
    print(f"Models with prerequisite_chain data: {', '.join(models_with_prereq)}")
    print()

    print(f"  {'Model':<20} {'N items':>8} {'Mean count':>12} {'SD count':>10} {'count→GT rho':>14} {'95% CI':>22} {'pred→GT rho':>14}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*10} {'-'*14} {'-'*22} {'-'*14}")

    for model in models_with_prereq:
        data = load_model_survey_outputs(model, "prerequisite_chain", parse_prerequisite_chain)
        if not data:
            continue

        avg_counts = average_across_reps(data, "count")
        avg_preds = average_across_reps(data, "prediction")

        counts_arr, gt_arr, keys = align_with_gt(avg_counts, gt)
        preds_arr, gt_arr2, keys2 = align_with_gt(avg_preds, gt)

        if len(counts_arr) < 5:
            continue

        mean_c = np.mean(counts_arr)
        sd_c = np.std(counts_arr)

        rho_c, ci_c, n_c = bootstrap_spearman_ci(counts_arr, gt_arr)

        if len(preds_arr) >= 5:
            rho_p, _, _ = bootstrap_spearman_ci(preds_arr, gt_arr2)
            pred_str = f"{rho_p:+.3f}"
        else:
            pred_str = "N/A"

        print(f"  {model:<20} {n_c:>8} {mean_c:>12.1f} {sd_c:>10.1f} {rho_c:>+14.3f} [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]   {pred_str:>14}")

    print()

    # Also do cognitive_load comparison for models that have it
    print(f"  --- cognitive_load comparison ---")
    models_with_cogload = []
    for entry in os.listdir(MODEL_SURVEY_DIR):
        model_dir = os.path.join(MODEL_SURVEY_DIR, entry, "cognitive_load")
        if os.path.isdir(model_dir):
            models_with_cogload.append(entry)
    models_with_cogload.sort()

    print(f"  {'Model':<20} {'N items':>8} {'Mean count':>12} {'count→GT rho':>14} {'95% CI':>22}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*14} {'-'*22}")

    for model in models_with_cogload:
        data = load_model_survey_outputs(model, "cognitive_load", parse_cognitive_load)
        if not data:
            continue

        avg_counts = average_across_reps(data, "count")
        counts_arr, gt_arr, keys = align_with_gt(avg_counts, gt)

        if len(counts_arr) < 5:
            continue

        mean_c = np.mean(counts_arr)
        rho_c, ci_c, n_c = bootstrap_spearman_ci(counts_arr, gt_arr)
        print(f"  {model:<20} {n_c:>8} {mean_c:>12.1f} {rho_c:>+14.3f} [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]")

    print()

    # =========================================================
    # SECTION F: Qualitative Validity Sample
    # =========================================================
    print("=" * 70)
    print("SECTION F: Qualitative Validity Sample (10 random items, Gemini)")
    print("=" * 70)
    print()
    print("Review these prerequisite lists for pedagogical sensibility.")
    print()

    gemini_prereq = load_model_survey_outputs("gemini", "prerequisite_chain", parse_prerequisite_chain)
    gemini_keys = sorted(set(gemini_prereq.keys()) & set(gt.keys()))

    rng = random.Random(SEED)
    sample = rng.sample(gemini_keys, min(10, len(gemini_keys)))

    for i, key in enumerate(sample, 1):
        reps = gemini_prereq[key]
        avg_count = np.mean([r["count"] for r in reps if r["count"] is not None])
        avg_pred = np.mean([r["prediction"] for r in reps if r["prediction"] is not None])
        print(f"  [{i}] {key}")
        print(f"      Ground truth p-correct: {gt[key]:.3f}")
        print(f"      Avg count: {avg_count:.1f}, Avg prediction: {avg_pred:.3f}")
        # Print prereq text from rep0
        if reps and reps[0].get("prereq_text"):
            print(f"      Prerequisites (rep0):")
            for line in reps[0]["prereq_text"].split("\n"):
                if line.strip():
                    print(f"        {line.strip()}")
        print()

    # =========================================================
    # SUMMARY
    # =========================================================
    print("=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    print()

    # Recompute key numbers for summary
    avg_prereq_counts = average_across_reps(prereq_data, "count")
    avg_prereq_preds = average_across_reps(prereq_data, "prediction")
    avg_cogload_counts = average_across_reps(cogload_data, "count")
    avg_cogload_preds = average_across_reps(cogload_data, "prediction")

    pc, pg, _ = align_with_gt(avg_prereq_counts, gt)
    rho_prereq_count, _, _ = bootstrap_spearman_ci(pc, pg)

    pp, pg2, _ = align_with_gt(avg_prereq_preds, gt)
    rho_prereq_pred, _, _ = bootstrap_spearman_ci(pp, pg2)

    cc, cg, _ = align_with_gt(avg_cogload_counts, gt)
    rho_cogload_count, _, _ = bootstrap_spearman_ci(cc, cg)

    cp, cg2, _ = align_with_gt(avg_cogload_preds, gt)
    rho_cogload_pred, _, _ = bootstrap_spearman_ci(cp, cg2)

    tp, tg, _ = align_with_gt(avg_teacher_preds, gt)
    rho_teacher, _, _ = bootstrap_spearman_ci(tp, tg)

    print(f"  1. Enumerated counts alone predict difficulty:")
    print(f"     - prerequisite count → GT: rho={rho_prereq_count:+.3f}")
    print(f"     - cognitive load count → GT: rho={rho_cogload_count:+.3f}")
    print()
    print(f"  2. Structured prompts improve over baseline:")
    print(f"     - prerequisite_chain prediction → GT: rho={rho_prereq_pred:+.3f}")
    print(f"     - cognitive_load prediction → GT:     rho={rho_cogload_pred:+.3f}")
    print(f"     - teacher (no enumeration) → GT:      rho={rho_teacher:+.3f}")
    print()
    print(f"  3. The mechanism: forcing enumeration creates a structured")
    print(f"     decomposition that grounds the final estimate in countable")
    print(f"     pedagogical features, rather than a holistic guess.")


if __name__ == "__main__":
    main()
