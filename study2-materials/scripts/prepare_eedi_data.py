#!/usr/bin/env python3
"""
Prepare Eedi Data for Study 2: Reasoning Authenticity Gap

This script creates the processed datasets from raw source files:
1. Aggregates 15.8M student responses by QuestionId
2. Joins with Kaggle misconception labels
3. Filters to target misconceptions with high student selection rates

Source Data Required:
- data/eedi/train.csv (Kaggle 2024 - misconception labels)
- data/eedi/data/train_data/train_task_1_2.csv (NeurIPS 2020 - student responses)

Output Files:
- data/eedi/eedi_with_student_data.csv (1,869 items with response distributions)
- data/eedi/curated_eedi_items.csv (58 items filtered for target misconceptions)

Usage:
    python scripts/prepare_eedi_data.py
    python scripts/prepare_eedi_data.py --skip-join  # If eedi_with_student_data.csv exists
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "eedi"
KAGGLE_FILE = DATA_DIR / "train.csv"
NEURIPS_FILE = DATA_DIR / "data" / "train_data" / "train_task_1_2.csv"
JOINED_OUTPUT = DATA_DIR / "eedi_with_student_data.csv"
CURATED_OUTPUT = DATA_DIR / "curated_eedi_items.csv"

# Target misconceptions for Study 2
# Original design: 4 misconceptions with good Eedi coverage
TARGET_MISCONCEPTIONS = {
    "ORDER_OPS": {
        "type": "procedural",
        "ids": [1507],
        "description": "Carries out operations from left to right regardless of priority order",
    },
    "INVERSE_OPS": {
        "type": "procedural",
        "ids": [1214],
        "description": "When solving an equation, uses the same operation rather than the inverse",
    },
    "NEG_MULTIPLY": {
        "type": "conceptual",
        "ids": [1597],
        "description": "Believes multiplying two negatives gives a negative answer",
    },
    "FRAC_ADD": {
        "type": "conceptual",
        "ids": [217],
        "description": "When adding fractions, adds the numerators and denominators",
    },
}

# Curation thresholds
MIN_SELECTION_PCT = 0.0  # No minimum - include all items for misconception
MIN_RESPONSES = 30  # Minimum number of student responses (matches original)


def step1_aggregate_responses() -> pd.DataFrame:
    """Aggregate 15.8M student responses by QuestionId.

    IMPORTANT: resp_A/B/C/D and pct_A/B/C/D use POSITIONAL ordering from the
    NeurIPS dataset (AnswerValue 1→A, 2→B, 3→C, 4→D). This is a different
    ordering from the Kaggle dataset's AnswerA/B/C/D and CorrectAnswer fields.
    The neurips_correct_pos column stores the NeurIPS correct answer position
    (A/B/C/D positional) so downstream code can correctly identify which
    pct column corresponds to the correct answer.
    """
    print(f"Reading NeurIPS responses from {NEURIPS_FILE}...")

    if not NEURIPS_FILE.exists():
        print(f"ERROR: NeurIPS data not found at {NEURIPS_FILE}")
        print("Download from: https://eedi.com/projects/neurips-education-challenge")
        sys.exit(1)

    # Read in chunks to handle large file
    chunks = []
    correct_answer_chunks = []
    for chunk in pd.read_csv(NEURIPS_FILE, chunksize=1_000_000):
        # Group by QuestionId and AnswerValue (1-4 maps to A-D)
        agg = chunk.groupby(['QuestionId', 'AnswerValue']).size().reset_index(name='count')
        chunks.append(agg)
        # Also extract the NeurIPS CorrectAnswer per question (1-4)
        ca = chunk[['QuestionId', 'CorrectAnswer']].drop_duplicates()
        correct_answer_chunks.append(ca)

    print(f"  Read {len(chunks)} chunks, aggregating...")

    # Combine all chunks
    all_responses = pd.concat(chunks, ignore_index=True)

    # NeurIPS correct answer mapping (1-4 → A-D positional)
    num_to_pos = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    neurips_correct = pd.concat(correct_answer_chunks).drop_duplicates('QuestionId')
    neurips_correct['neurips_correct_pos'] = neurips_correct['CorrectAnswer'].map(num_to_pos)
    neurips_correct = neurips_correct[['QuestionId', 'neurips_correct_pos']]

    # Final aggregation
    final_agg = all_responses.groupby(['QuestionId', 'AnswerValue'])['count'].sum().reset_index()

    # Pivot to wide format
    pivoted = final_agg.pivot(index='QuestionId', columns='AnswerValue', values='count').fillna(0)
    pivoted.columns = ['resp_A', 'resp_B', 'resp_C', 'resp_D']
    pivoted = pivoted.reset_index()

    # Merge NeurIPS correct answer
    pivoted = pivoted.merge(neurips_correct, on='QuestionId', how='left')

    # Convert to int
    for col in ['resp_A', 'resp_B', 'resp_C', 'resp_D']:
        pivoted[col] = pivoted[col].astype(int)

    print(f"  Aggregated {len(pivoted)} questions with responses")
    total_responses = pivoted[['resp_A', 'resp_B', 'resp_C', 'resp_D']].sum().sum()
    print(f"  Total student responses: {total_responses:,}")

    return pivoted


def step2_join_with_kaggle(response_agg: pd.DataFrame) -> pd.DataFrame:
    """Join aggregated responses with Kaggle misconception labels."""
    print(f"\nReading Kaggle data from {KAGGLE_FILE}...")

    if not KAGGLE_FILE.exists():
        print(f"ERROR: Kaggle data not found at {KAGGLE_FILE}")
        print("Download from: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics")
        sys.exit(1)

    kaggle = pd.read_csv(KAGGLE_FILE)
    print(f"  {len(kaggle)} questions with misconception labels")

    # Left join to keep all Kaggle questions
    joined = kaggle.merge(response_agg, on='QuestionId', how='left')

    # Check for missing joins
    missing = joined['resp_A'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} questions have no response data")
        # Fill with zeros
        for col in ['resp_A', 'resp_B', 'resp_C', 'resp_D']:
            joined[col] = joined[col].fillna(0).astype(int)

    # Calculate totals and percentages
    joined['total_responses'] = joined[['resp_A', 'resp_B', 'resp_C', 'resp_D']].sum(axis=1)

    for letter in ['A', 'B', 'C', 'D']:
        joined[f'pct_{letter}'] = np.where(
            joined['total_responses'] > 0,
            (joined[f'resp_{letter}'] / joined['total_responses']) * 100,
            0
        )

    print(f"  Joined dataset: {len(joined)} questions")
    print(f"  Response stats: min={joined['total_responses'].min()}, "
          f"median={joined['total_responses'].median():.0f}, "
          f"max={joined['total_responses'].max()}")

    return joined


def step3_curate_items(joined: pd.DataFrame) -> pd.DataFrame:
    """Filter to target misconceptions with high student selection rates."""
    print(f"\nCurating items for target misconceptions...")

    curated_items = []

    for target_key, info in TARGET_MISCONCEPTIONS.items():
        misc_ids = info['ids']
        misc_type = info['type']
        misc_desc = info['description']

        count = 0
        for _, row in joined.iterrows():
            # Check each distractor for target misconception
            for letter in ['A', 'B', 'C', 'D']:
                misc_col = f'Misconception{letter}Id'
                misc_id = row.get(misc_col)

                if pd.notna(misc_id) and int(misc_id) in misc_ids:
                    selection_pct = row[f'pct_{letter}']
                    total_resp = row['total_responses']

                    # Apply thresholds
                    if selection_pct >= MIN_SELECTION_PCT and total_resp >= MIN_RESPONSES:
                        # correct_answer uses NeurIPS positional ordering
                        # (matches pct_A/B/C/D columns)
                        neurips_correct = row.get('neurips_correct_pos', '')
                        # target_distractor_kaggle is the Kaggle letter for the
                        # misconception distractor. We can't map to NeurIPS
                        # position without answer text matching.
                        curated_items.append({
                            'target_key': target_key,
                            'target_type': misc_type,
                            'QuestionId': row['QuestionId'],
                            'question_text': row['QuestionText'],
                            'correct_answer': neurips_correct,
                            'correct_answer_kaggle': row['CorrectAnswer'],
                            'target_distractor_kaggle': letter,
                            'target_answer': row[f'Answer{letter}Text'],
                            'misconception_id': int(misc_id),
                            'misconception_name': misc_desc,
                            'student_selection_pct_kaggle': round(selection_pct, 2),
                            'total_responses': int(total_resp),
                            'pct_A': round(row['pct_A'], 2),
                            'pct_B': round(row['pct_B'], 2),
                            'pct_C': round(row['pct_C'], 2),
                            'pct_D': round(row['pct_D'], 2),
                        })
                        count += 1
                        break  # Only one target per question

        print(f"  {target_key}: {count} items")

    curated_df = pd.DataFrame(curated_items)
    print(f"\nTotal curated items: {len(curated_df)}")

    return curated_df


def main():
    parser = argparse.ArgumentParser(description="Prepare Eedi data for Study 2")
    parser.add_argument('--skip-join', action='store_true',
                        help='Skip join step if eedi_with_student_data.csv exists')
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE EEDI DATA FOR STUDY 2")
    print("=" * 60)

    # Step 1 & 2: Aggregate and join (or load existing)
    if args.skip_join and JOINED_OUTPUT.exists():
        print(f"\nLoading existing joined data from {JOINED_OUTPUT}...")
        joined = pd.read_csv(JOINED_OUTPUT)
        print(f"  Loaded {len(joined)} questions")
    else:
        response_agg = step1_aggregate_responses()
        joined = step2_join_with_kaggle(response_agg)

        # Save joined data
        print(f"\nSaving joined data to {JOINED_OUTPUT}...")
        joined.to_csv(JOINED_OUTPUT, index=False)
        print(f"  Saved {len(joined)} rows")

    # Step 3: Curate items
    curated = step3_curate_items(joined)

    # Save curated data
    print(f"\nSaving curated items to {CURATED_OUTPUT}...")
    curated.to_csv(CURATED_OUTPUT, index=False)
    print(f"  Saved {len(curated)} items")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Joined dataset: {JOINED_OUTPUT}")
    print(f"  - {len(joined)} questions with misconception labels + student response %s")
    print(f"\nCurated items: {CURATED_OUTPUT}")
    print(f"  - {len(curated)} items across {curated['target_key'].nunique()} misconception types")
    print(f"  - Selection threshold: >{MIN_SELECTION_PCT}% of students")
    print(f"  - Response threshold: >={MIN_RESPONSES} total responses")

    print("\nItems per misconception:")
    for key, count in curated.groupby('target_key').size().items():
        avg_pct = curated[curated['target_key'] == key]['student_selection_pct_kaggle'].mean()
        print(f"  {key}: {count} items (avg {avg_pct:.1f}% selection)")


if __name__ == "__main__":
    main()
