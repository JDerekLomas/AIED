#!/usr/bin/env python3
"""
Calculate IRT and BKT statistics for the 113 curated Eedi items.

IRT (Item Response Theory):
- Difficulty (b): proportion incorrect (classical) or logit-transformed
- Discrimination (a): point-biserial correlation with total score
- Pseudo-guessing (c): estimated from low-ability student performance

BKT (Bayesian Knowledge Tracing) per misconception:
- P(L0): initial probability of mastery
- P(T): probability of learning (transition)
- P(G): probability of guess
- P(S): probability of slip
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
RESPONSES_PATH = 'data/eedi/data/train_data/train_task_1_2.csv'
ITEMS_PATH = 'data/eedi/curated_eedi_items.csv'
OUTPUT_PATH = 'results/irt_bkt_statistics.json'

print("Loading curated items...")
items_df = pd.read_csv(ITEMS_PATH)
item_ids = set(items_df['QuestionId'].values)
print(f"Found {len(item_ids)} curated items")

print("\nLoading response data (this may take a moment)...")
# Read in chunks to handle large file
chunks = []
for chunk in pd.read_csv(RESPONSES_PATH, chunksize=1000000):
    # Filter to our items
    filtered = chunk[chunk['QuestionId'].isin(item_ids)]
    chunks.append(filtered)
    print(f"  Processed chunk, found {len(filtered)} relevant responses")

responses = pd.concat(chunks, ignore_index=True)
print(f"\nTotal relevant responses: {len(responses):,}")
print(f"Unique students: {responses['UserId'].nunique():,}")
print(f"Unique items: {responses['QuestionId'].nunique()}")

# Merge with item metadata
responses = responses.merge(
    items_df[['QuestionId', 'misconception_id', 'target_distractor', 'correct_answer']],
    on='QuestionId',
    how='left'
)

# Calculate student total scores for discrimination
print("\nCalculating student total scores...")
student_scores = responses.groupby('UserId')['IsCorrect'].agg(['sum', 'count'])
student_scores.columns = ['total_correct', 'total_attempts']
student_scores['accuracy'] = student_scores['total_correct'] / student_scores['total_attempts']

# For discrimination, prefer students with more responses but don't exclude items
reliable_students = student_scores[student_scores['total_attempts'] >= 3].index
print(f"Students with 3+ responses: {len(reliable_students):,}")

responses_reliable = responses[responses['UserId'].isin(reliable_students)].copy()
responses_reliable = responses_reliable.merge(
    student_scores[['accuracy']],
    left_on='UserId',
    right_index=True
)

# Also calculate for ALL responses (for items with few reliable-student responses)
all_responses_with_acc = responses.merge(
    student_scores[['accuracy']],
    left_on='UserId',
    right_index=True,
    how='left'
)
all_responses_with_acc['accuracy'] = all_responses_with_acc['accuracy'].fillna(
    all_responses_with_acc.groupby('UserId')['IsCorrect'].transform('mean')
)

# ============================================================
# IRT CALCULATIONS
# ============================================================
print("\n" + "="*60)
print("CALCULATING IRT PARAMETERS")
print("="*60)

irt_results = {}

for qid in item_ids:
    item_resp = responses_reliable[responses_reliable['QuestionId'] == qid]

    # Fall back to all responses if not enough reliable ones
    if len(item_resp) < 30:
        item_resp = all_responses_with_acc[all_responses_with_acc['QuestionId'] == qid]

    if len(item_resp) < 10:
        continue

    # Difficulty (proportion incorrect, then logit transform)
    p_correct = item_resp['IsCorrect'].mean()
    p_incorrect = 1 - p_correct

    # Logit difficulty (higher = harder)
    if 0 < p_correct < 1:
        difficulty_logit = np.log(p_incorrect / p_correct)
    else:
        difficulty_logit = np.nan

    # Discrimination (point-biserial correlation)
    # Correlation between item score and total score
    if item_resp['IsCorrect'].std() > 0 and item_resp['accuracy'].std() > 0:
        discrimination, _ = stats.pointbiserialr(
            item_resp['IsCorrect'],
            item_resp['accuracy']
        )
    else:
        discrimination = np.nan

    # Pseudo-guessing estimate (from bottom quartile students)
    bottom_quartile = item_resp[item_resp['accuracy'] <= item_resp['accuracy'].quantile(0.25)]
    if len(bottom_quartile) > 10:
        guessing = bottom_quartile['IsCorrect'].mean()
    else:
        guessing = 0.25  # Default for 4-choice

    irt_results[int(qid)] = {
        'n_responses': len(item_resp),
        'difficulty_classical': round(p_incorrect, 4),
        'difficulty_logit': round(difficulty_logit, 4) if not np.isnan(difficulty_logit) else None,
        'discrimination': round(discrimination, 4) if not np.isnan(discrimination) else None,
        'guessing_estimate': round(guessing, 4)
    }

print(f"Calculated IRT for {len(irt_results)} items")

# Summary statistics
difficulties = [r['difficulty_classical'] for r in irt_results.values()]
discriminations = [r['discrimination'] for r in irt_results.values() if r['discrimination']]
print(f"\nDifficulty: M={np.mean(difficulties):.3f}, SD={np.std(difficulties):.3f}")
print(f"Discrimination: M={np.mean(discriminations):.3f}, SD={np.std(discriminations):.3f}")

# ============================================================
# BKT CALCULATIONS
# ============================================================
print("\n" + "="*60)
print("CALCULATING BKT PARAMETERS")
print("="*60)

# BKT requires sequential data per student per skill
# Group items by misconception and create sequences

# Create a mapping of question to misconception
q_to_misc = dict(zip(items_df['QuestionId'], items_df['misconception_id']))

# Add misconception to responses
responses['misconception'] = responses['QuestionId'].map(q_to_misc)

# Sort by user and create sequences per misconception
print("Building response sequences per student-misconception...")

def fit_bkt(sequences, max_iter=100):
    """
    Fit BKT parameters using EM algorithm.

    Parameters:
    - sequences: list of lists, each inner list is [0,1,1,0,...] for incorrect/correct

    Returns:
    - P(L0): initial knowledge probability
    - P(T): learning/transition probability
    - P(G): guess probability
    - P(S): slip probability
    """
    # Initialize parameters
    pL0 = 0.3
    pT = 0.1
    pG = 0.25
    pS = 0.1

    for iteration in range(max_iter):
        # E-step: compute expected values
        total_L0 = 0
        total_not_L0 = 0
        total_T_given_not_L = 0
        total_not_L = 0
        total_G = 0
        total_not_L_obs = 0
        total_S = 0
        total_L_obs = 0

        for seq in sequences:
            if len(seq) < 2:
                continue

            # Forward-backward for this sequence
            n = len(seq)
            alpha = np.zeros(n)  # P(L_t | O_1:t)

            # Forward pass
            for t in range(n):
                obs = seq[t]
                if t == 0:
                    # P(L_0 | O_0)
                    pO_given_L = (1 - pS) if obs == 1 else pS
                    pO_given_not_L = pG if obs == 1 else (1 - pG)
                    pL_prior = pL0
                else:
                    # P(L_t | O_1:t-1) = P(L_t-1|...) + P(not L_t-1|...) * P(T)
                    pL_prior = alpha[t-1] + (1 - alpha[t-1]) * pT
                    pO_given_L = (1 - pS) if obs == 1 else pS
                    pO_given_not_L = pG if obs == 1 else (1 - pG)

                # Bayes update
                pO = pL_prior * pO_given_L + (1 - pL_prior) * pO_given_not_L
                if pO > 0:
                    alpha[t] = (pL_prior * pO_given_L) / pO
                else:
                    alpha[t] = pL_prior

            # Accumulate statistics
            total_L0 += alpha[0] if len(alpha) > 0 else pL0
            total_not_L0 += (1 - alpha[0]) if len(alpha) > 0 else (1 - pL0)

            for t in range(1, n):
                p_not_L_prev = 1 - alpha[t-1]
                # P(transition | not learned before)
                if p_not_L_prev > 0:
                    p_learned_now = (1 - alpha[t-1]) * pT
                    total_T_given_not_L += p_learned_now
                    total_not_L += p_not_L_prev

            for t in range(n):
                obs = seq[t]
                pL_t = alpha[t]
                if obs == 1:  # Correct
                    total_G += (1 - pL_t)  # Guessed correctly
                    total_L_obs += pL_t     # Knew and got right
                else:  # Incorrect
                    total_S += pL_t         # Slipped
                    total_not_L_obs += (1 - pL_t)  # Didn't know

        # M-step: update parameters
        n_seqs = len([s for s in sequences if len(s) >= 2])
        if n_seqs == 0:
            break

        new_pL0 = total_L0 / (total_L0 + total_not_L0) if (total_L0 + total_not_L0) > 0 else pL0
        new_pT = total_T_given_not_L / total_not_L if total_not_L > 0 else pT
        new_pG = total_G / (total_G + total_not_L_obs) if (total_G + total_not_L_obs) > 0 else pG
        new_pS = total_S / (total_S + total_L_obs) if (total_S + total_L_obs) > 0 else pS

        # Clamp to valid ranges
        new_pL0 = np.clip(new_pL0, 0.01, 0.99)
        new_pT = np.clip(new_pT, 0.01, 0.99)
        new_pG = np.clip(new_pG, 0.01, 0.5)  # Guess shouldn't exceed chance
        new_pS = np.clip(new_pS, 0.01, 0.5)  # Slip shouldn't be too high

        # Check convergence
        if abs(new_pL0 - pL0) < 0.001 and abs(new_pT - pT) < 0.001:
            break

        pL0, pT, pG, pS = new_pL0, new_pT, new_pG, new_pS

    return pL0, pT, pG, pS

# Build sequences per misconception
bkt_results = {}
misconception_ids = items_df['misconception_id'].unique()

for misc_id in misconception_ids:
    print(f"\nProcessing misconception {misc_id}...")

    # Get items for this misconception
    misc_items = items_df[items_df['misconception_id'] == misc_id]['QuestionId'].values

    # Get responses for these items
    misc_responses = responses[responses['QuestionId'].isin(misc_items)].copy()

    if len(misc_responses) < 100:
        print(f"  Skipping - only {len(misc_responses)} responses")
        continue

    # Build sequences per student
    # Sort by user (we don't have timestamps, so order within user is arbitrary)
    sequences = []
    for user_id, user_data in misc_responses.groupby('UserId'):
        if len(user_data) >= 2:
            seq = user_data['IsCorrect'].tolist()
            sequences.append(seq)

    print(f"  {len(sequences)} students with 2+ responses")

    if len(sequences) < 50:
        print(f"  Skipping - not enough sequences")
        continue

    # Fit BKT
    try:
        pL0, pT, pG, pS = fit_bkt(sequences)

        bkt_results[int(misc_id)] = {
            'misconception_name': items_df[items_df['misconception_id'] == misc_id]['misconception_name'].iloc[0],
            'n_items': len(misc_items),
            'n_responses': len(misc_responses),
            'n_sequences': len(sequences),
            'p_L0': round(pL0, 4),  # Initial knowledge
            'p_T': round(pT, 4),    # Learning rate
            'p_G': round(pG, 4),    # Guess rate
            'p_S': round(pS, 4),    # Slip rate
        }

        print(f"  P(L0)={pL0:.3f}, P(T)={pT:.3f}, P(G)={pG:.3f}, P(S)={pS:.3f}")

    except Exception as e:
        print(f"  Error fitting BKT: {e}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output = {
    'irt': {
        'description': 'Item Response Theory parameters per item',
        'parameters': {
            'difficulty_classical': 'Proportion incorrect (0-1, higher = harder)',
            'difficulty_logit': 'Log-odds of incorrect (logit scale)',
            'discrimination': 'Point-biserial correlation with total score',
            'guessing_estimate': 'P(correct) for bottom quartile students'
        },
        'items': irt_results
    },
    'bkt': {
        'description': 'Bayesian Knowledge Tracing parameters per misconception',
        'parameters': {
            'p_L0': 'Initial probability of mastery',
            'p_T': 'Probability of learning (transition from not-known to known)',
            'p_G': 'Probability of guessing correctly when not mastered',
            'p_S': 'Probability of slipping (error when mastered)'
        },
        'misconceptions': bkt_results
    },
    'summary': {
        'n_items_with_irt': len(irt_results),
        'n_misconceptions_with_bkt': len(bkt_results),
        'irt_difficulty_mean': round(np.mean(difficulties), 4),
        'irt_difficulty_sd': round(np.std(difficulties), 4),
        'irt_discrimination_mean': round(np.mean(discriminations), 4),
        'irt_discrimination_sd': round(np.std(discriminations), 4)
    }
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
print(f"\nSummary:")
print(f"  IRT: {len(irt_results)} items")
print(f"  BKT: {len(bkt_results)} misconceptions")
