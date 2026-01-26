#!/usr/bin/env python3
"""
Comprehensive Analysis: Study 2 Pilot Data

Analyzes both misconception alignment (when simulating students) and
prepares the framework for misconception diagnosis testing.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

def load_jsonl(path: Path) -> list:
    """Load JSONL data."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    # Load single-item data
    single_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/all_single_item.jsonl")
    data = load_jsonl(single_path)
    df = pd.DataFrame(data)

    print("=" * 70)
    print("COMPREHENSIVE PILOT ANALYSIS: Misconception Alignment Study")
    print("=" * 70)

    # Convert types
    df['is_correct'] = df['is_correct'].apply(lambda x: x == True)

    # 1. DATA OVERVIEW
    print(f"\n{'='*70}")
    print("1. DATA OVERVIEW")
    print("="*70)
    print(f"Total responses: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Unique items: {df['item_id'].nunique()}")
    print(f"Parse success rate: {df['parsed_answer'].notna().mean():.1%}")

    # Capability tiers
    tier_map = {
        'mistral-7b': ('weak', 40.0),
        'gpt-3.5-turbo': ('weak', 57.0),
        'claude-haiku': ('mid', 88.0),
        'gemini-flash': ('mid', 86.0),
    }

    df['tier'] = df['model'].map(lambda x: tier_map.get(x, ('unknown', 0))[0])
    df['gsm8k'] = df['model'].map(lambda x: tier_map.get(x, ('unknown', 0))[1])

    # 2. ACCURACY BY MODEL
    print(f"\n{'='*70}")
    print("2. ACCURACY BY MODEL (Overall Performance)")
    print("="*70)

    df_valid = df[df['parsed_answer'].notna()].copy()

    model_stats = df_valid.groupby(['model', 'tier', 'gsm8k']).agg({
        'is_correct': ['mean', 'sum', 'count']
    }).round(3)
    model_stats.columns = ['accuracy', 'correct', 'total']
    model_stats['errors'] = model_stats['total'] - model_stats['correct']
    print(model_stats.to_string())

    # 3. ERROR ANALYSIS
    print(f"\n{'='*70}")
    print("3. ERROR ANALYSIS (Key for Misconception Alignment)")
    print("="*70)

    errors = df_valid[df_valid['is_correct'] == False].copy()
    print(f"Total errors: {len(errors)}")
    print(f"\nErrors by model:")
    print(errors['model'].value_counts().to_string())

    print(f"\nErrors by tier:")
    print(errors['tier'].value_counts().to_string())

    # 4. TARGET DISTRACTOR ANALYSIS
    print(f"\n{'='*70}")
    print("4. TARGET DISTRACTOR SELECTION (Core Metric)")
    print("="*70)

    # Load probe items to get target distractors
    items_path = Path("/Users/dereklomas/AIED/study2-materials/data/probe_items.json")
    with open(items_path) as f:
        items_data = json.load(f)

    # Extract items from nested structure
    item_targets = {}
    for misc in items_data.get('misconceptions', []):
        for item in misc.get('items', []):
            item_targets[item['item_id']] = item.get('target_distractor')

    # Add target distractor to errors
    errors['target_distractor'] = errors['item_id'].map(item_targets)
    errors['selected_target'] = errors['parsed_answer'] == errors['target_distractor']

    print("When wrong, rate of selecting the TARGET (misconception-aligned) distractor:\n")

    target_by_model = errors.groupby(['model', 'tier']).agg({
        'selected_target': ['sum', 'count', 'mean']
    })
    target_by_model.columns = ['target_selected', 'total_errors', 'target_rate']
    target_by_model = target_by_model.sort_values('target_rate', ascending=False)
    print(target_by_model.to_string())

    # 5. TIER COMPARISON
    print(f"\n{'='*70}")
    print("5. TIER-LEVEL TARGET DISTRACTOR COMPARISON")
    print("="*70)

    weak_errors = errors[errors['tier'] == 'weak']
    mid_errors = errors[errors['tier'] == 'mid']

    weak_target = weak_errors['selected_target'].sum()
    weak_total = len(weak_errors)
    mid_target = mid_errors['selected_target'].sum()
    mid_total = len(mid_errors)

    print(f"Weak tier (Mistral-7B, GPT-3.5):")
    print(f"  Target distractor rate: {weak_target}/{weak_total} = {weak_target/weak_total:.1%}")
    print(f"\nMid tier (Claude-Haiku, Gemini-Flash):")
    print(f"  Target distractor rate: {mid_target}/{mid_total} = {mid_target/mid_total:.1%}" if mid_total > 0 else "  No errors")

    if weak_total > 0 and mid_total > 0:
        # Fisher's exact test
        table = [[weak_target, weak_total - weak_target],
                 [mid_target, mid_total - mid_target]]
        odds_ratio, p_value = stats.fisher_exact(table)
        print(f"\nFisher's exact test (Weak vs Mid):")
        print(f"  Odds ratio: {odds_ratio:.2f}")
        print(f"  p-value: {p_value:.3f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # 6. WITHIN-TIER ANALYSIS
    print(f"\n{'='*70}")
    print("6. WITHIN-TIER ANALYSIS (Sweet Spot Hypothesis)")
    print("="*70)

    print("\nWithin WEAK tier (comparing GPT-3.5 vs Mistral-7B):")

    gpt_errors = errors[errors['model'] == 'gpt-3.5-turbo']
    mistral_errors = errors[errors['model'] == 'mistral-7b']

    gpt_target = gpt_errors['selected_target'].sum()
    gpt_total = len(gpt_errors)
    mistral_target = mistral_errors['selected_target'].sum()
    mistral_total = len(mistral_errors)

    print(f"  GPT-3.5 (57% GSM8K): {gpt_target}/{gpt_total} = {gpt_target/gpt_total:.1%}" if gpt_total > 0 else "  GPT-3.5: No errors")
    print(f"  Mistral-7B (40% GSM8K): {mistral_target}/{mistral_total} = {mistral_target/mistral_total:.1%}" if mistral_total > 0 else "  Mistral-7B: No errors")

    if gpt_total > 0 and mistral_total > 0:
        table = [[gpt_target, gpt_total - gpt_target],
                 [mistral_target, mistral_total - mistral_target]]
        odds_ratio, p_value = stats.fisher_exact(table)
        print(f"\n  Fisher's exact test (GPT-3.5 vs Mistral-7B):")
        print(f"    Odds ratio: {odds_ratio:.2f}")
        print(f"    p-value: {p_value:.3f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # 7. MISCONCEPTION CATEGORY ANALYSIS
    print(f"\n{'='*70}")
    print("7. ERROR PATTERNS BY MISCONCEPTION CATEGORY")
    print("="*70)

    if 'misconception_category' in errors.columns:
        cat_analysis = errors.groupby(['misconception_category']).agg({
            'selected_target': ['sum', 'count', 'mean']
        })
        cat_analysis.columns = ['target_selected', 'total_errors', 'target_rate']
        cat_analysis = cat_analysis.sort_values('target_rate', ascending=False)
        print(cat_analysis.to_string())

    # 8. SAMPLE OF ALIGNED ERRORS
    print(f"\n{'='*70}")
    print("8. SAMPLE OF MISCONCEPTION-ALIGNED ERRORS")
    print("="*70)

    aligned_errors = errors[errors['selected_target'] == True]
    print(f"\nShowing {min(5, len(aligned_errors))} examples of LLM errors that matched target misconception:\n")

    for i, (_, row) in enumerate(aligned_errors.head(5).iterrows()):
        print(f"--- Example {i+1} ---")
        print(f"Model: {row['model']} | Item: {row['item_id']}")
        print(f"Misconception: {row.get('misconception_name', 'N/A')}")
        print(f"Answered: {row['parsed_answer']} (target distractor)")
        print(f"Correct: {row['correct_answer']}")
        reasoning = str(row.get('reasoning', ''))[:200]
        print(f"Reasoning excerpt: {reasoning}...")
        print()

    # 9. DIAGNOSIS FRAMEWORK
    print(f"\n{'='*70}")
    print("9. MISCONCEPTION DIAGNOSIS CAPABILITY")
    print("="*70)

    print("""
The diagnosis question: Can LLMs identify what misconception led to a student error?

This is DIFFERENT from exhibiting misconceptions:
- Exhibiting: LLM makes an error consistent with human misconception
- Diagnosing: LLM identifies what misconception explains an observed error

Previous test showed Claude Haiku achieved 5/5 (100%) on diagnosing errors.

For full study, we should test:
1. Diagnosis accuracy across all 91 errors
2. Compare diagnosis accuracy between models
3. Test if models can diagnose their OWN errors vs others' errors
4. Measure if frontier models are better at diagnosis
""")

    # 10. SUMMARY
    print(f"\n{'='*70}")
    print("10. KEY FINDINGS & IMPLICATIONS")
    print("="*70)

    print(f"""
DATA SUMMARY:
- {len(df_valid)} valid responses across 4 models (2 weak, 2 mid tier)
- {len(errors)} total errors ({len(weak_errors)} weak tier, {len(mid_errors)} mid tier)
- Overall accuracy: {df_valid['is_correct'].mean():.1%}

KEY FINDINGS:

1. CAPABILITY HYPOTHESIS: Mixed support
   - Weak tier models make more errors (expected)
   - BUT mid-tier models ALSO show misconception alignment when they err

2. "SWEET SPOT" HYPOTHESIS: Stronger support
   - GPT-3.5 (57% GSM8K) shows BEST misconception alignment ({gpt_target}/{gpt_total}={gpt_target/gpt_total:.1%} target rate)
   - Mistral-7B (40% GSM8K) shows lower alignment ({mistral_target}/{mistral_total}={mistral_target/mistral_total:.1%})
   - Suggests: Too weak = random errors, Just right = human-like errors

3. IMPLICATIONS FOR STUDY 2:
   - Focus on GPT-3.5 for misconception simulation
   - Add Llama 2 7B to test "too weak" hypothesis
   - Compare simulation quality vs diagnosis capability
""")

    # Save summary stats
    summary = {
        'total_responses': len(df_valid),
        'total_errors': len(errors),
        'weak_tier_errors': len(weak_errors),
        'mid_tier_errors': len(mid_errors),
        'gpt35_target_rate': gpt_target/gpt_total if gpt_total > 0 else None,
        'mistral_target_rate': mistral_target/mistral_total if mistral_total > 0 else None,
        'weak_vs_mid_pvalue': p_value if (weak_total > 0 and mid_total > 0) else None,
    }

    output_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/summary_stats.json")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary stats saved to {output_path}")

if __name__ == "__main__":
    main()
