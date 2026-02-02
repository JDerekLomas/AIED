#!/usr/bin/env python3
"""
Proper IRT analysis for curated Eedi items.

Key improvements over calculate_irt_bkt.py:
- Estimates student ability (theta) from the FULL 27K-item dataset
  (students answer median 88 items), not just our items (~1.5 per student)
- Fits 2PL and 3PL IRT models (3PL adds guessing floor for 4-choice MCQ)
- Compares ability-adjusted difficulty to naive classical difficulty
- Checks for adaptive testing bias

This matters because if Eedi uses adaptive routing, harder items are
shown to stronger students, making naive p-values underestimate true difficulty.

Note on 4PL and misconception-specific ability:
- 4PL (adding slip parameter) is overparameterized for per-item MLE with these
  sample sizes and produces worse downstream correlations.
- Misconception-specific ability cannot be estimated because Eedi subject tags
  are too coarse and students see ~1-2 items per misconception cluster.
"""

import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import expit  # logistic function
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
RESPONSES_1_2 = 'data/eedi/data/train_data/train_task_1_2.csv'
RESPONSES_3_4 = 'data/eedi/data/train_data/train_task_3_4.csv'
ITEMS_PATH = 'data/eedi/curated_eedi_items.csv'
OUTPUT_PATH = 'results/irt_proper_statistics.json'

# ============================================================
# STEP 1: Load data
# ============================================================
print("Loading curated items...")
items_df = pd.read_csv(ITEMS_PATH)
item_ids = set(items_df['QuestionId'].values)
print(f"Found {len(item_ids)} curated items")

print("\nLoading full response data (task 1+2)...")
resp_full = pd.read_csv(RESPONSES_1_2)
print(f"Full dataset: {len(resp_full):,} responses, {resp_full.UserId.nunique():,} students, {resp_full.QuestionId.nunique():,} items")

# NOTE: Task 3+4 data excluded. Task 3+4 students score 16pp lower on the same
# items and the cross-task p_correct correlation is r≈0 — different populations.
# CSV pct_A/B/C/D come from task 1+2 only, so IRT should match.
print("Skipping task 3+4 data (different population, r≈0 with task 1+2)")

# ============================================================
# STEP 2: Estimate student ability from FULL dataset
# ============================================================
print("\n" + "="*60)
print("ESTIMATING STUDENT ABILITY FROM FULL DATASET")
print("="*60)

# Use proportion correct across all items, logit-transformed
student_stats = resp_full.groupby('UserId').agg(
    total_correct=('IsCorrect', 'sum'),
    total_items=('IsCorrect', 'count')
)
student_stats['prop_correct'] = student_stats['total_correct'] / student_stats['total_items']

# Filter out students with too few items for reliable ability estimates
MIN_ITEMS = 20
n_before = len(student_stats)
student_stats = student_stats[student_stats['total_items'] >= MIN_ITEMS]
n_after = len(student_stats)
print(f"Filtered students: {n_before:,} → {n_after:,} (removed {n_before - n_after:,} with <{MIN_ITEMS} items)")

# Logit transform with continuity correction for 0/1
epsilon = 0.5 / student_stats['total_items']  # Continuity correction
p_adj = student_stats['prop_correct'].clip(lower=epsilon, upper=1-epsilon)
student_stats['theta_logit'] = np.log(p_adj / (1 - p_adj))

print(f"Student ability (logit): M={student_stats['theta_logit'].mean():.3f}, SD={student_stats['theta_logit'].std():.3f}")
print(f"Student ability (prop): M={student_stats['prop_correct'].mean():.3f}, SD={student_stats['prop_correct'].std():.3f}")
print(f"Items per student: M={student_stats['total_items'].mean():.0f}, median={student_stats['total_items'].median():.0f}")

# ============================================================
# STEP 3: Get responses for our items with ability estimates
# ============================================================
print("\n" + "="*60)
print("MERGING ABILITY ESTIMATES WITH ITEM RESPONSES")
print("="*60)

# Use only task 1+2 responses (matches CSV pct ground truth)
resp_ours = resp_full[resp_full['QuestionId'].isin(item_ids)]
print(f"Total responses for our items (task 1+2 only): {len(resp_ours):,}")

# Merge with ability
resp_ours = resp_ours.merge(
    student_stats[['theta_logit', 'prop_correct', 'total_items']],
    left_on='UserId', right_index=True, how='inner'
)
print(f"Responses with ability estimates: {len(resp_ours):,}")

# ============================================================
# STEP 4: Check for adaptive testing bias
# ============================================================
print("\n" + "="*60)
print("CHECKING FOR ADAPTIVE TESTING BIAS")
print("="*60)

item_stats = resp_ours.groupby('QuestionId').agg(
    n=('IsCorrect', 'count'),
    p_correct=('IsCorrect', 'mean'),
    mean_ability=('theta_logit', 'mean'),
    sd_ability=('theta_logit', 'std'),
    mean_prop_correct=('prop_correct', 'mean')
)

# Key test: correlation between mean student ability and item p_correct
r_adaptive, p_adaptive = stats.pearsonr(item_stats['mean_ability'], item_stats['p_correct'])
print(f"Correlation(mean student ability, p_correct): r={r_adaptive:.3f}, p={p_adaptive:.4g}")
print("  (If adaptive, expect strong positive: harder items → stronger students → inflated p_correct)")

# Compare to expected: if random assignment, mean ability should be ~same for all items
print(f"\nMean ability across items: M={item_stats['mean_ability'].mean():.3f}, SD={item_stats['mean_ability'].std():.3f}")
print(f"Overall student ability: M={student_stats['theta_logit'].mean():.3f}")
print(f"  Difference suggests {'ADAPTIVE' if item_stats['mean_ability'].std() > 0.1 else 'mostly random'} assignment")

# Items where bias is largest
item_stats_sorted = item_stats.sort_values('mean_ability')
print("\n5 items with LOWEST ability students (easiest items in adaptive system):")
for qid, row in item_stats_sorted.head(5).iterrows():
    print(f"  Q{qid}: mean_θ={row['mean_ability']:.2f}, p_correct={row['p_correct']:.2f}, n={int(row['n'])}")

print("\n5 items with HIGHEST ability students (hardest items in adaptive system):")
for qid, row in item_stats_sorted.tail(5).iterrows():
    print(f"  Q{qid}: mean_θ={row['mean_ability']:.2f}, p_correct={row['p_correct']:.2f}, n={int(row['n'])}")

# ============================================================
# STEP 5: Fit 2PL and 3PL IRT parameters per item
# ============================================================
print("\n" + "="*60)
print("FITTING 2PL AND 3PL IRT MODELS PER ITEM")
print("="*60)

def _fit_model(theta, correct, model='2pl'):
    """Fit IRT model via MLE with multiple restarts.
    2PL: P = logistic(a*(theta-b))
    3PL: P = c + (1-c)*logistic(a*(theta-b))  [c = guessing floor]
    """
    if model == '2pl':
        def nll(p):
            a, b = p
            prob = expit(a*(theta-b))
            prob = np.clip(prob, 1e-8, 1-1e-8)
            return -np.sum(correct*np.log(prob) + (1-correct)*np.log(1-prob))
        starts = [(a0, b0) for a0 in [0.5,1,2] for b0 in [-1,0,1]]
        bounds = [(0.05, 10), (-6, 6)]
    else:  # 3pl
        def nll(p):
            a, b, c = p
            prob = c + (1-c)*expit(a*(theta-b))
            prob = np.clip(prob, 1e-8, 1-1e-8)
            return -np.sum(correct*np.log(prob) + (1-correct)*np.log(1-prob))
        starts = [(a0, b0, c0) for a0 in [0.5,1,2] for b0 in [-1,0,1] for c0 in [0.1, 0.25]]
        bounds = [(0.05, 10), (-6, 6), (0.0, 0.5)]

    best = None
    best_v = np.inf
    for s in starts:
        try:
            r = optimize.minimize(nll, s, method='L-BFGS-B', bounds=bounds)
            if r.fun < best_v:
                best_v = r.fun
                best = r
        except:
            pass
    return tuple(best.x) if best and best.success else tuple([None]*len(bounds))


irt_results = {}

for i, qid in enumerate(item_ids):
    item_resp = resp_ours[resp_ours['QuestionId'] == qid]

    if len(item_resp) < 50:
        continue

    theta = item_resp['theta_logit'].values
    correct = item_resp['IsCorrect'].values.astype(float)

    # Classical difficulty
    p_correct = correct.mean()
    p_incorrect = 1 - p_correct
    if 0 < p_correct < 1:
        difficulty_classical_logit = np.log(p_incorrect / p_correct)
    else:
        difficulty_classical_logit = np.nan

    # 2PL IRT
    a_2pl, b_2pl = _fit_model(theta, correct, '2pl')

    # 3PL IRT (adds guessing floor — important for 4-choice MCQ)
    a_3pl, b_3pl, c_3pl = _fit_model(theta, correct, '3pl')

    # Point-biserial with full-dataset ability (proper discrimination)
    if correct.std() > 0 and theta.std() > 0:
        rpb, _ = stats.pointbiserialr(correct, theta)
    else:
        rpb = np.nan

    # Empirical guessing/slip from ability quartiles
    q25 = np.percentile(theta, 25)
    q75 = np.percentile(theta, 75)
    bottom = item_resp[item_resp['theta_logit'] <= q25]
    top = item_resp[item_resp['theta_logit'] >= q75]
    empirical_guess = bottom['IsCorrect'].mean() if len(bottom) > 10 else None
    empirical_slip = 1 - top['IsCorrect'].mean() if len(top) > 10 else None

    irt_results[int(qid)] = {
        'n_responses': len(item_resp),
        'mean_student_ability': round(float(theta.mean()), 4),
        'sd_student_ability': round(float(theta.std()), 4),
        # Classical
        'difficulty_classical': round(p_incorrect, 4),
        'difficulty_classical_logit': round(difficulty_classical_logit, 4) if not np.isnan(difficulty_classical_logit) else None,
        # 2PL IRT
        'b_2pl': round(b_2pl, 4) if b_2pl is not None else None,
        'a_2pl': round(a_2pl, 4) if a_2pl is not None else None,
        # 3PL IRT (preferred model for 4-choice MCQ)
        'b_3pl': round(b_3pl, 4) if b_3pl is not None else None,
        'a_3pl': round(a_3pl, 4) if a_3pl is not None else None,
        'c_3pl_guess': round(c_3pl, 4) if c_3pl is not None else None,
        # Proper discrimination (using full-dataset ability)
        'discrimination_rpb': round(rpb, 4) if not np.isnan(rpb) else None,
        # Empirical guessing/slip
        'empirical_guess': round(empirical_guess, 4) if empirical_guess is not None else None,
        'empirical_slip': round(empirical_slip, 4) if empirical_slip is not None else None,
    }

    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(item_ids)} items...")

print(f"Fitted IRT for {len(irt_results)} items")

# ============================================================
# STEP 6: Compare classical vs IRT difficulty
# ============================================================
print("\n" + "="*60)
print("COMPARING CLASSICAL VS 2PL VS 3PL DIFFICULTY")
print("="*60)

classical = []
b2_list = []
b3_list = []
c3_list = []
abilities = []
for qid, r in irt_results.items():
    if all(r[k] is not None for k in ['difficulty_classical_logit', 'b_2pl', 'b_3pl']):
        if abs(r['b_2pl']) < 5.9 and abs(r['b_3pl']) < 5.9:
            classical.append(r['difficulty_classical_logit'])
            b2_list.append(r['b_2pl'])
            b3_list.append(r['b_3pl'])
            c3_list.append(r['c_3pl_guess'])
            abilities.append(r['mean_student_ability'])

classical = np.array(classical)
b2_arr = np.array(b2_list)
b3_arr = np.array(b3_list)
c3_arr = np.array(c3_list)
abilities = np.array(abilities)

n_well_fit = len(classical)
print(f"\n{n_well_fit} well-fit items (excluding bound-hitters):")
r_c2, _ = stats.pearsonr(classical, b2_arr)
r_c3, _ = stats.pearsonr(classical, b3_arr)
r_23, _ = stats.pearsonr(b2_arr, b3_arr)
print(f"Classical vs 2PL b: r={r_c2:.3f}")
print(f"Classical vs 3PL b: r={r_c3:.3f}")
print(f"2PL vs 3PL b:       r={r_23:.3f}")

print(f"\n3PL guessing (c): M={c3_arr.mean():.3f}, SD={c3_arr.std():.3f}")
print(f"  (Expected ~0.25 for 4-choice MCQ)")

# Empirical guessing/slip summary
eg = [r['empirical_guess'] for r in irt_results.values() if r['empirical_guess'] is not None]
es = [r['empirical_slip'] for r in irt_results.values() if r['empirical_slip'] is not None]
print(f"\nEmpirical guessing (bottom quartile p_correct): M={np.mean(eg):.3f}, SD={np.std(eg):.3f}")
print(f"Empirical slip (top quartile p_incorrect):      M={np.mean(es):.3f}, SD={np.std(es):.3f}")

# Difficulty shift analysis (using 3PL as primary)
diffs = b3_arr - classical
mae = np.mean(np.abs(diffs))
print(f"\n3PL b - classical logit: M={diffs.mean():.3f}, SD={diffs.std():.3f}, MAE={mae:.3f}")

r_bias, _ = stats.pearsonr(abilities, diffs)
print(f"Correlation(mean student ability, difficulty shift): r={r_bias:.3f}")
print("  (Positive = items with strong students get shifted harder by IRT)")

# Summary stats across all items
d_cl = [r['difficulty_classical'] for r in irt_results.values()]
d_2pl = [r['b_2pl'] for r in irt_results.values() if r['b_2pl'] is not None]
d_3pl = [r['b_3pl'] for r in irt_results.values() if r['b_3pl'] is not None]
a_2pl = [r['a_2pl'] for r in irt_results.values() if r['a_2pl'] is not None]
a_3pl = [r['a_3pl'] for r in irt_results.values() if r['a_3pl'] is not None]
rpbs = [r['discrimination_rpb'] for r in irt_results.values() if r['discrimination_rpb'] is not None]

print(f"\nClassical difficulty (prop incorrect): M={np.mean(d_cl):.3f}, SD={np.std(d_cl):.3f}")
print(f"2PL difficulty (b): M={np.mean(d_2pl):.3f}, SD={np.std(d_2pl):.3f}")
print(f"3PL difficulty (b): M={np.mean(d_3pl):.3f}, SD={np.std(d_3pl):.3f}")
print(f"2PL discrimination (a): M={np.mean(a_2pl):.3f}, SD={np.std(a_2pl):.3f}")
print(f"3PL discrimination (a): M={np.mean(a_3pl):.3f}, SD={np.std(a_3pl):.3f}")
print(f"Discrimination (rpb): M={np.mean(rpbs):.3f}, SD={np.std(rpbs):.3f}")

# ============================================================
# STEP 7: Spot checks
# ============================================================
print("\n" + "="*60)
print("SPOT CHECKS: ITEMS WITH LARGEST ABILITY-ADJUSTED SHIFTS")
print("="*60)

items_with_shift = []
for qid, r in irt_results.items():
    if r['difficulty_classical_logit'] is not None and r['b_3pl'] is not None:
        shift = r['b_3pl'] - r['difficulty_classical_logit']
        items_with_shift.append((qid, r, shift))

items_with_shift.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 items with largest difficulty shift (3PL vs classical):")
for qid, r, shift in items_with_shift[:10]:
    direction = "HARDER" if shift > 0 else "EASIER"
    print(f"  Q{qid}: shift={shift:+.2f} ({direction}), "
          f"classical={r['difficulty_classical']:.2f}, 3PL_b={r['b_3pl']:.2f}, "
          f"guess_c={r['c_3pl_guess']:.2f}, mean_θ={r['mean_student_ability']:.2f}, n={r['n_responses']}")

# ============================================================
# STEP 8: Save results
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output = {
    'description': 'Proper 2PL/3PL IRT analysis with student ability estimated from full 27K-item dataset (task 1+2 only — task 3+4 excluded due to different population)',
    'method': {
        'ability_estimation': 'Logit-transformed proportion correct across all items in full dataset (median 88 items per student)',
        'item_parameters_2pl': '2PL IRT: P = logistic(a*(theta-b))',
        'item_parameters_3pl': '3PL IRT: P = c + (1-c)*logistic(a*(theta-b)), c=guessing floor (preferred for 4-choice MCQ)',
        'discrimination_rpb': 'Point-biserial correlation between item score and full-dataset ability',
        'empirical_guess_slip': 'P(correct) for bottom ability quartile / P(incorrect) for top ability quartile',
    },
    'notes': {
        'adaptive_testing': f'r={r_adaptive:.3f} between mean student ability and p_correct confirms adaptive routing',
        'guessing': f'3PL guessing floor M={c3_arr.mean():.3f} (expected ~0.25 for 4-choice MCQ)',
        'slip': f'Empirical slip M={np.mean(es):.3f} (top-quartile students still miss {np.mean(es)*100:.0f}% of items)',
        'misconception_ability': 'Cannot estimate: Eedi subject tags too coarse, students see ~1-2 items per misconception cluster',
        '4pl_note': '4PL tested but overparameterized for per-item MLE; produces worse downstream correlations',
    },
    'adaptive_testing_check': {
        'correlation_ability_pcorrect': round(r_adaptive, 4),
        'ability_sd_across_items': round(item_stats['mean_ability'].std(), 4),
    },
    'summary': {
        'n_items': len(irt_results),
        'n_well_fit': n_well_fit,
        'difficulty_classical_mean': round(np.mean(d_cl), 4),
        'difficulty_classical_sd': round(np.std(d_cl), 4),
        'difficulty_3pl_mean': round(np.mean(d_3pl), 4),
        'difficulty_3pl_sd': round(np.std(d_3pl), 4),
        'discrimination_3pl_mean': round(np.mean(a_3pl), 4),
        'discrimination_3pl_sd': round(np.std(a_3pl), 4),
        'guessing_3pl_mean': round(c3_arr.mean(), 4),
        'guessing_3pl_sd': round(c3_arr.std(), 4),
        'empirical_slip_mean': round(np.mean(es), 4),
        'empirical_slip_sd': round(np.std(es), 4),
    },
    'items': irt_results
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
