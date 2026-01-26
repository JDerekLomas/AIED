#!/usr/bin/env python3
"""
Pilot Data Analysis for Study 2: Misconception Alignment
Analyzes LLM responses across capability tiers and prompting conditions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats

# Load all pilot data
def load_responses(pilot_dir):
    """Load all response files into a single DataFrame."""
    all_records = []

    # Map directories to model info
    model_info = {
        'results': {'model': 'gpt-4o-mini', 'tier': 'mid', 'gsm8k': 85.0},
        'gpt35': {'model': 'gpt-3.5-turbo', 'tier': 'weak', 'gsm8k': 57.1},
        'haiku': {'model': 'claude-haiku', 'tier': 'mid', 'gsm8k': 88.9},
        'sonnet': {'model': 'claude-sonnet', 'tier': 'frontier', 'gsm8k': 96.4},
        'mistral7b': {'model': 'mistral-7b', 'tier': 'weak', 'gsm8k': 45.2},
        'gemini-flash': {'model': 'gemini-flash', 'tier': 'mid', 'gsm8k': 68.8},
    }

    for subdir, info in model_info.items():
        filepath = pilot_dir / subdir / 'responses.jsonl'
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    record = json.loads(line)
                    record['model_name'] = info['model']
                    record['capability_tier'] = info['tier']
                    record['gsm8k_score'] = info['gsm8k']
                    all_records.append(record)

    return pd.DataFrame(all_records)


def analyze_accuracy(df):
    """Analyze accuracy by model and condition."""
    print("=" * 60)
    print("ACCURACY ANALYSIS")
    print("=" * 60)

    # Overall accuracy by model
    print("\n### Accuracy by Model (ordered by GSM8K capability)")
    model_acc = df.groupby(['model_name', 'gsm8k_score', 'capability_tier']).agg({
        'is_correct': ['mean', 'count', 'sum']
    }).round(3)
    model_acc.columns = ['accuracy', 'n_responses', 'n_correct']
    model_acc = model_acc.reset_index().sort_values('gsm8k_score', ascending=False)
    print(model_acc.to_string(index=False))

    # Accuracy by condition
    print("\n### Accuracy by Prompting Condition")
    cond_acc = df.groupby('condition')['is_correct'].agg(['mean', 'count']).round(3)
    cond_acc.columns = ['accuracy', 'n']
    print(cond_acc)

    # Model × Condition interaction
    print("\n### Model × Condition Accuracy")
    cross = pd.crosstab(
        df['model_name'],
        df['condition'],
        values=df['is_correct'],
        aggfunc='mean'
    ).round(3)
    print(cross)

    return model_acc


def analyze_errors(df):
    """Analyze error patterns - which distractors are selected."""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # Filter to incorrect responses only
    errors = df[df['is_correct'] == False].copy()
    print(f"\nTotal errors: {len(errors)} / {len(df)} ({len(errors)/len(df):.1%})")

    # Error rate by capability tier
    print("\n### Error Rate by Capability Tier")
    tier_errors = df.groupby('capability_tier')['is_correct'].agg(['mean', 'count'])
    tier_errors['error_rate'] = 1 - tier_errors['mean']
    tier_errors = tier_errors.round(3)
    print(tier_errors[['error_rate', 'count']])

    # Distractor selection patterns
    print("\n### Distractor Selection (Incorrect Responses)")
    if 'parsed_answer' in errors.columns and 'correct_answer' in errors.columns:
        # Get distractor choices (wrong answers)
        errors['distractor'] = errors['parsed_answer']
        distractor_dist = errors.groupby(['capability_tier', 'distractor']).size().unstack(fill_value=0)
        print(distractor_dist)

    return errors


def analyze_misconceptions(df):
    """Analyze misconception alignment."""
    print("\n" + "=" * 60)
    print("MISCONCEPTION ANALYSIS")
    print("=" * 60)

    # Check if misconception data is available
    if 'misconception_id' not in df.columns:
        print("No misconception labels in data.")
        return

    # Filter to errors with misconception labels
    errors = df[(df['is_correct'] == False) & (df['misconception_id'].notna())].copy()
    print(f"\nErrors with misconception labels: {len(errors)}")

    # Misconception category distribution
    if 'misconception_category' in errors.columns:
        print("\n### Error Distribution by Misconception Category")
        cat_dist = errors.groupby(['capability_tier', 'misconception_category']).size().unstack(fill_value=0)
        print(cat_dist)

    # Check if models select the "target" distractor (the one associated with the misconception)
    if 'target_distractor' in errors.columns:
        errors['hit_target'] = errors['parsed_answer'] == errors['target_distractor']
        print("\n### Target Distractor Hit Rate (key metric)")
        print("(Does the model select the distractor designed to elicit the misconception?)")
        target_hit = errors.groupby(['capability_tier', 'model_name'])['hit_target'].mean().round(3)
        print(target_hit.unstack())

        # By condition
        print("\n### Target Hit Rate by Condition")
        cond_hit = errors.groupby('condition')['hit_target'].agg(['mean', 'count']).round(3)
        print(cond_hit)

    return errors


def capability_analysis(df):
    """Test the main hypothesis: do weaker models show better misconception alignment?"""
    print("\n" + "=" * 60)
    print("CAPABILITY HYPOTHESIS TEST")
    print("=" * 60)
    print("H1: Weaker models show BETTER misconception alignment than frontier models")
    print("    (authentic errors vs performed errors)")

    errors = df[(df['is_correct'] == False)].copy()

    if 'target_distractor' not in errors.columns:
        print("\nCannot test - no target_distractor labels")
        return

    errors['hit_target'] = errors['parsed_answer'] == errors['target_distractor']

    # Compare tiers
    tier_stats = errors.groupby('capability_tier').agg({
        'hit_target': ['mean', 'count', 'sum'],
        'gsm8k_score': 'first'
    })
    tier_stats.columns = ['hit_rate', 'n_errors', 'n_hits', 'gsm8k']
    tier_stats = tier_stats.sort_values('gsm8k', ascending=False)
    print("\n### Target Hit Rate by Capability Tier")
    print(tier_stats.round(3))

    # Correlation: GSM8K score vs target hit rate (by model)
    model_stats = errors.groupby('model_name').agg({
        'hit_target': 'mean',
        'gsm8k_score': 'first'
    }).dropna()

    if len(model_stats) >= 3:
        corr, p_value = stats.pearsonr(model_stats['gsm8k_score'], model_stats['hit_target'])
        print(f"\n### Correlation: GSM8K Score vs Target Hit Rate")
        print(f"Pearson r = {corr:.3f}, p = {p_value:.3f}")
        if corr < 0:
            print("SUPPORTS H1: Negative correlation - weaker models hit targets more often")
        else:
            print("REFUTES H1: Positive correlation - stronger models hit targets more often")

    # Chi-square test: tier vs target hit
    contingency = pd.crosstab(errors['capability_tier'], errors['hit_target'])
    if contingency.shape == (3, 2):  # frontier, mid, weak × True/False
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n### Chi-square Test: Tier × Target Hit")
        print(f"χ² = {chi2:.2f}, df = {dof}, p = {p:.4f}")

    return tier_stats


def parse_quality(df):
    """Check response parsing quality."""
    print("\n" + "=" * 60)
    print("DATA QUALITY")
    print("=" * 60)

    # Parse success rate
    parse_rate = df['parsed_answer'].notna().mean()
    print(f"\nParse success rate: {parse_rate:.1%}")

    # By model
    print("\n### Parse Rate by Model")
    parse_by_model = df.groupby('model_name')['parsed_answer'].apply(lambda x: x.notna().mean()).round(3)
    print(parse_by_model)

    # By condition
    print("\n### Parse Rate by Condition")
    parse_by_cond = df.groupby('condition')['parsed_answer'].apply(lambda x: x.notna().mean()).round(3)
    print(parse_by_cond)

    # Error rate (API failures)
    if 'error' in df.columns:
        error_rate = df['error'].notna().mean()
        print(f"\nAPI error rate: {error_rate:.1%}")


def main():
    pilot_dir = Path(__file__).parent.parent / 'pilot'

    print("Loading pilot data...")
    df = load_responses(pilot_dir)
    print(f"Loaded {len(df)} responses from {df['model_name'].nunique()} models")
    print(f"Models: {sorted(df['model_name'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")

    # Run analyses
    parse_quality(df)
    model_acc = analyze_accuracy(df)
    errors = analyze_errors(df)
    analyze_misconceptions(df)
    capability_analysis(df)

    # Summary statistics for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"Total responses: {len(df)}")
    print(f"Models tested: {df['model_name'].nunique()}")
    print(f"Items: {df['item_id'].nunique()}")
    print(f"Conditions: {df['condition'].nunique()}")
    print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
    print(f"Parse success: {df['parsed_answer'].notna().mean():.1%}")

    # Save processed data
    output_path = pilot_dir / 'pilot_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")


if __name__ == '__main__':
    main()
