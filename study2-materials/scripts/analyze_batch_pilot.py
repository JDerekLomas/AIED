#!/usr/bin/env python3
"""
Batch Pilot Analysis for Study 2: Misconception Alignment

Analyzes batch responses to test the capability hypothesis:
Do weaker models show better alignment with human misconceptions?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def load_responses(path: Path) -> pd.DataFrame:
    """Load JSONL responses."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def analyze_pilot():
    """Run comprehensive pilot analysis."""
    # Load data
    data_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/batch/all_responses.jsonl")
    df = load_responses(data_path)

    print("=" * 70)
    print("STUDY 2 PILOT ANALYSIS: Misconception Alignment in Batch Mode")
    print("=" * 70)

    # Basic stats
    print(f"\n{'='*70}")
    print("1. DATA OVERVIEW")
    print("=" * 70)
    print(f"Total item-responses: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Model tiers: {df['model_tier'].unique().tolist()}")
    print(f"Parse success rate: {df['parsed_answer'].notna().mean():.1%}")

    # Filter to valid responses
    df_valid = df[df['parsed_answer'].notna()].copy()
    # Fix type issues
    df_valid['is_correct'] = df_valid['is_correct'].astype(float)
    df_valid['gsm8k_score'] = df_valid['gsm8k_score'].astype(float)
    print(f"Valid responses: {len(df_valid)}")

    # Overall accuracy
    print(f"\n{'='*70}")
    print("2. ACCURACY BY MODEL")
    print("=" * 70)
    model_acc = df_valid.groupby(['model', 'model_tier', 'gsm8k_score']).agg({
        'is_correct': ['mean', 'count']
    }).round(3)
    model_acc.columns = ['accuracy', 'n']
    model_acc = model_acc.sort_values('accuracy', ascending=False)
    print(model_acc.to_string())

    # Accuracy by tier
    print(f"\n{'='*70}")
    print("3. ACCURACY BY CAPABILITY TIER")
    print("=" * 70)
    tier_acc = df_valid.groupby('model_tier').agg({
        'is_correct': ['mean', 'std', 'count']
    }).round(3)
    tier_acc.columns = ['accuracy', 'std', 'n']
    print(tier_acc.to_string())

    # Accuracy by condition
    print(f"\n{'='*70}")
    print("4. ACCURACY BY CONDITION (explain vs persona)")
    print("=" * 70)
    cond_acc = df_valid.groupby(['condition', 'model_tier']).agg({
        'is_correct': ['mean', 'count']
    }).round(3)
    cond_acc.columns = ['accuracy', 'n']
    print(cond_acc.to_string())

    # Position effects
    print(f"\n{'='*70}")
    print("5. POSITION EFFECTS IN BATCH")
    print("=" * 70)
    pos_acc = df_valid.groupby('position_in_batch')['is_correct'].agg(['mean', 'count'])
    pos_acc.columns = ['accuracy', 'n']
    print(pos_acc.to_string())

    # Test for position effect
    first_half = df_valid[df_valid['position_in_batch'] <= 5]['is_correct']
    second_half = df_valid[df_valid['position_in_batch'] > 5]['is_correct']
    if len(first_half) > 0 and len(second_half) > 0:
        t_stat, p_val = stats.ttest_ind(first_half, second_half)
        print(f"\nFirst half (pos 1-5) vs Second half (pos 6-10):")
        print(f"  First half accuracy: {first_half.mean():.3f}")
        print(f"  Second half accuracy: {second_half.mean():.3f}")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")

    # CAPABILITY HYPOTHESIS TEST
    print(f"\n{'='*70}")
    print("6. CAPABILITY HYPOTHESIS: GSM8K vs Accuracy")
    print("=" * 70)

    model_stats = df_valid.groupby(['model', 'gsm8k_score']).agg({
        'is_correct': 'mean'
    }).reset_index()
    model_stats.columns = ['model', 'gsm8k_score', 'accuracy']

    if len(model_stats) > 2:
        r, p = stats.pearsonr(model_stats['gsm8k_score'], model_stats['accuracy'])
        print(f"Correlation: GSM8K score vs Pilot accuracy")
        print(f"  r = {r:.3f}, p = {p:.3f}")
        print(f"  Interpretation: {'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'} "
              f"{'positive' if r > 0 else 'negative'} correlation")
        print(f"\nModel-level data:")
        print(model_stats.sort_values('gsm8k_score').to_string(index=False))

    # Misconception analysis
    print(f"\n{'='*70}")
    print("7. MISCONCEPTION-LEVEL ANALYSIS")
    print("=" * 70)

    if 'misconception_id' in df_valid.columns:
        misc_acc = df_valid.groupby(['misconception_id', 'misconception_category']).agg({
            'is_correct': ['mean', 'count']
        }).round(3)
        misc_acc.columns = ['accuracy', 'n']
        misc_acc = misc_acc.sort_values('accuracy')
        print("Accuracy by misconception (sorted by difficulty):")
        print(misc_acc.to_string())

        # By category
        print(f"\nBy misconception category:")
        cat_acc = df_valid.groupby('misconception_category')['is_correct'].agg(['mean', 'count'])
        cat_acc.columns = ['accuracy', 'n']
        print(cat_acc.to_string())

    # Distractor selection analysis
    print(f"\n{'='*70}")
    print("8. DISTRACTOR SELECTION (Key for Misconception Alignment)")
    print("=" * 70)

    if 'target_distractor' in df_valid.columns:
        # When wrong, did model select the misconception distractor?
        df_wrong = df_valid[df_valid['is_correct'] == False].copy()
        print(f"Total incorrect responses: {len(df_wrong)}")

        if len(df_wrong) > 0:
            df_wrong['selected_target'] = df_wrong['parsed_answer'] == df_wrong['target_distractor']

            target_rate = df_wrong.groupby(['model', 'model_tier']).agg({
                'selected_target': ['mean', 'count']
            }).round(3)
            target_rate.columns = ['target_distractor_rate', 'n_wrong']
            target_rate = target_rate.sort_values('target_distractor_rate', ascending=False)

            print("\nWhen wrong, rate of selecting the TARGET (misconception) distractor:")
            print(target_rate.to_string())

            # By tier
            tier_target = df_wrong.groupby('model_tier')['selected_target'].agg(['mean', 'count'])
            tier_target.columns = ['target_rate', 'n']
            print(f"\nBy capability tier:")
            print(tier_target.to_string())

            # This is the KEY metric: does tier predict misconception alignment?
            print(f"\n*** CAPABILITY HYPOTHESIS TEST ***")
            print("If weaker models show 'authentic' misconceptions:")
            print("  - They should have HIGHER target distractor selection when wrong")
            print("  - Frontier models 'performing' errors would select randomly")

    # Summary
    print(f"\n{'='*70}")
    print("9. PILOT SUMMARY")
    print("=" * 70)

    overall_acc = df_valid['is_correct'].mean()
    print(f"Overall accuracy: {overall_acc:.1%}")

    if len(model_stats) > 2:
        r, _ = stats.pearsonr(model_stats['gsm8k_score'], model_stats['accuracy'])
        print(f"GSM8K correlation with pilot: r = {r:.3f}")

    print(f"\nKey findings for full study design:")
    print("1. Parse rate and data quality sufficient for full study")
    print("2. Position effects in batch mode should be controlled for")
    print("3. Misconception-level difficulty varies - may need item-level analysis")

    return df_valid

if __name__ == "__main__":
    df = analyze_pilot()
