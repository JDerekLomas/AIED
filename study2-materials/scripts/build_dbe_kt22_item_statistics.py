#!/usr/bin/env python3
"""Build item_statistics.json for DBE-KT22 dataset (no LLM calls, pure data processing)."""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "dbe-kt22"

# Load all tables
questions = pd.read_csv(DATA_DIR / "Questions.csv")
choices = pd.read_csv(DATA_DIR / "Question_Choices.csv")
transactions = pd.read_csv(DATA_DIR / "Transaction.csv")
kcs = pd.read_csv(DATA_DIR / "KCs.csv")
qkc = pd.read_csv(DATA_DIR / "Question_KC_Relationships.csv")

# Empirical p_correct from transactions
empirical = transactions.groupby("question_id").agg(
    n_responses=("answer_state", "count"),
    n_correct=("answer_state", "sum"),
).reset_index()
empirical["p_correct"] = empirical["n_correct"] / empirical["n_responses"]

questions = questions.merge(empirical, left_on="id", right_on="question_id", how="left")

# KC mapping: question_id -> list of KC names
kc_map = {}
kc_names = dict(zip(kcs["id"], kcs["name"]))
for _, row in qkc.iterrows():
    qid = row["question_id"]
    kc_name = kc_names.get(row["knowledgecomponent_id"], "Unknown")
    kc_map.setdefault(qid, []).append(kc_name)

# Options per question
choice_map = {}
for qid, group in choices.groupby("question_id"):
    group = group.sort_values("id")  # Deterministic label assignment regardless of CSV order
    opts = []
    for i, (_, row) in enumerate(group.iterrows()):
        label = chr(65 + i)
        opts.append({
            "label": label,
            "text": str(row["choice_text"]),
            "is_correct": bool(row["is_correct"]),
        })
    choice_map[qid] = opts

# Filter: text-only usable items (same criteria as run_dbe_kt22_validation.py)
has_img = questions["question_rich_text"].str.contains("<img", na=False)
usable = questions[
    (questions["question_text"].str.len() > 20) &
    (questions["n_responses"] >= 50) &
    (~has_img)
].copy()

print(f"Usable items: {len(usable)} / {len(questions)} total")
print(f"p_correct range: {usable['p_correct'].min():.3f} - {usable['p_correct'].max():.3f}")

# Build output
items = []
for _, row in usable.iterrows():
    qid = int(row["id"])
    items.append({
        "question_id": qid,
        "question_text": row["question_text"],
        "options": choice_map.get(qid, []),
        "n_responses": int(row["n_responses"]),
        "p_correct": round(float(row["p_correct"]), 4),
        "difficulty_author": int(row["difficulty"]) if pd.notna(row["difficulty"]) else None,
        "knowledge_components": kc_map.get(qid, []),
    })

out_path = DATA_DIR / "item_statistics.json"
with open(out_path, "w") as f:
    json.dump(items, f, indent=2)

print(f"Wrote {len(items)} items to {out_path}")
