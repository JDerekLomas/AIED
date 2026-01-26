#!/usr/bin/env python3
"""Quick exploration of Eedi dataset."""

import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "eedi"

# Load data
train = pd.read_csv(data_dir / "train.csv")
misconceptions = pd.read_csv(data_dir / "misconception_mapping.csv")

print("=== TRAIN DATA ===")
print(f"Items: {len(train)}")
print(f"Columns: {list(train.columns)}")
print(f"\nSubjects:\n{train['SubjectName'].value_counts()}")

print("\n=== SAMPLE ITEM ===")
item = train.iloc[0]
print(f"Question: {item['QuestionText'][:300]}")
print(f"A: {item['AnswerAText']}")
print(f"B: {item['AnswerBText']}")
print(f"C: {item['AnswerCText']}")
print(f"D: {item['AnswerDText']}")
print(f"Correct: {item['CorrectAnswer']}")

print("\n=== MISCONCEPTIONS ===")
print(f"Total unique: {len(misconceptions)}")
print(f"\nSample misconceptions:")
for _, row in misconceptions.head(5).iterrows():
    print(f"  {row['MisconceptionId']}: {row['MisconceptionName'][:70]}")

print("\n=== COVERAGE ===")
for col in ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']:
    non_null = train[col].notna().sum()
    print(f"{col}: {non_null} labeled ({non_null/len(train):.1%})")

# Items with all distractors labeled
has_all = (
    train['MisconceptionAId'].notna() &
    train['MisconceptionBId'].notna() &
    train['MisconceptionCId'].notna() &
    train['MisconceptionDId'].notna()
)
# But one should be correct (null)
# Actually, correct answer should have null misconception
print(f"\nItems where correct answer has null misconception (expected): checking...")

# Create misconception lookup
misc_lookup = dict(zip(misconceptions['MisconceptionId'], misconceptions['MisconceptionName']))

# Show a complete example
print("\n=== COMPLETE EXAMPLE ===")
for idx, item in train.iterrows():
    if idx > 5:
        break
    print(f"\nQ{item['QuestionId']}: {item['QuestionText'][:100]}...")
    print(f"  Correct: {item['CorrectAnswer']}")
    for opt in ['A', 'B', 'C', 'D']:
        misc_id = item[f'Misconception{opt}Id']
        misc_name = misc_lookup.get(misc_id, 'N/A (correct answer)')
        if pd.isna(misc_id):
            misc_name = "(correct answer)"
        else:
            misc_name = misc_lookup.get(int(misc_id), "Unknown")[:50]
        print(f"  {opt}: {item[f'Answer{opt}Text'][:30]:30} -> {misc_name}")
