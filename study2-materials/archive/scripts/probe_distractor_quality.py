#!/usr/bin/env python3
"""
Probe distractor quality: ask Gemini 3 Flash to rate each distractor individually,
then derive difficulty and discrimination predictions from the ratings.
"""

import json, os, re, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

# ── paths ──
BASE = "/Users/dereklomas/AIED/study2-materials"
DATA = f"{BASE}/data/dbe-kt22"
CACHE = f"{BASE}/pilot/dbe_kt22_validation/probe_distractor_quality.json"
ITEM_STATS = f"{DATA}/item_statistics.json"

# ── load data ──
questions_df = pd.read_csv(f"{DATA}/Questions.csv")
choices_df = pd.read_csv(f"{DATA}/Question_Choices.csv")
transactions_df = pd.read_csv(f"{DATA}/Transaction.csv")
item_stats = json.load(open(ITEM_STATS))

# Build lookup: question_id -> item stat
stat_by_qid = {s["question_id"]: s for s in item_stats}

# ── compute point-biserial discrimination from Transaction.csv ──
print("Computing point-biserial discrimination from Transaction.csv...", flush=True)

# Get correct choice ids
correct_choices = set(choices_df[choices_df["is_correct"] == True]["id"].values)

# Score each transaction
transactions_df["score"] = transactions_df["answer_choice_id"].isin(correct_choices).astype(int)

# First attempt per student per question
txn_sorted = transactions_df.sort_values("start_time")
first_attempts = txn_sorted.drop_duplicates(subset=["student_id", "question_id"], keep="first")

# Pivot to student x item
pivot = first_attempts.pivot(index="student_id", columns="question_id", values="score")

# Total score per student
student_totals = pivot.sum(axis=1)

# Point-biserial: correlation between item score and (total - item)
pb_disc = {}
for qid in pivot.columns:
    col = pivot[qid]
    valid = col.dropna()
    if len(valid) < 50:
        continue
    item_scores = valid.values
    totals_minus_item = student_totals[valid.index] - item_scores
    if np.std(item_scores) == 0 or np.std(totals_minus_item) == 0:
        continue
    r, _ = stats.pearsonr(item_scores, totals_minus_item)
    pb_disc[qid] = r

print(f"  Computed point-biserial for {len(pb_disc)} items", flush=True)

# ── filter to usable items ──
usable_items = []
for s in item_stats:
    qid = s["question_id"]
    if qid in pb_disc:
        s["pb_discrimination"] = pb_disc[qid]
        usable_items.append(s)

print(f"  Usable items with point-biserial: {len(usable_items)}", flush=True)

# ── build prompt per item ──
def build_prompt(item):
    options_text = ""
    labels = []
    for opt in item["options"]:
        options_text += f"{opt['label']}: {opt['text']}\n"
        labels.append(opt["label"])
    correct = [o["label"] for o in item["options"] if o["is_correct"]][0]

    extra_line = ""
    if "E" in labels:
        extra_line = "E: PLAUSIBILITY=X MISCONCEPTION=description\n"

    prompt = f"""You are an experienced psychometrician evaluating test item quality.

For this multiple choice question, evaluate EACH answer option.

Question: {item['question_text']}

{options_text.strip()}

Correct Answer: {correct}

For each option, provide:
1. PLAUSIBILITY (0-10): How attractive is this option to a student who doesn't fully understand the material? (0 = obviously wrong, 10 = very tempting)
2. MISCONCEPTION: What specific error or misconception would lead a student to choose this? Write "correct answer" for the right option, or "none - implausible" if the distractor is obviously wrong.

Then estimate:
3. DIFFICULTY: What proportion of students would answer correctly? (0.00 to 1.00)
4. DISCRIMINATION: How well does this item separate students who understand from those who don't? (0.0 = not at all, 1.0 = moderately, 2.0+ = very well)

Format your response EXACTLY like this:
A: PLAUSIBILITY=X MISCONCEPTION=description
B: PLAUSIBILITY=X MISCONCEPTION=description
C: PLAUSIBILITY=X MISCONCEPTION=description
D: PLAUSIBILITY=X MISCONCEPTION=description
{extra_line}DIFFICULTY: 0.XX
DISCRIMINATION: X.XX"""
    return prompt

# ── call Gemini ──
def call_gemini(prompt):
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text

# ── load/update cache ──
if os.path.exists(CACHE):
    cache = json.load(open(CACHE))
else:
    cache = {}

print(f"\nCache has {len(cache)} items, need {len(usable_items)} total", flush=True)

# Run missing items
missing = [it for it in usable_items if str(it["question_id"]) not in cache]
if missing:
    print(f"Calling Gemini for {len(missing)} missing items...", flush=True)
    for i, item in enumerate(missing):
        qid = str(item["question_id"])
        prompt = build_prompt(item)
        try:
            raw = call_gemini(prompt)
            cache[qid] = {"raw": raw}
            print(f"  [{i+1}/{len(missing)}] qid={qid} OK", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(missing)}] qid={qid} ERROR: {e}", flush=True)
            cache[qid] = {"raw": "", "error": str(e)}
        if (i + 1) % 20 == 0:
            json.dump(cache, open(CACHE, "w"), indent=2)
            print(f"  Saved checkpoint at {i+1}", flush=True)
        time.sleep(0.15)
    json.dump(cache, open(CACHE, "w"), indent=2)
    print("Cache saved.", flush=True)

# ── parse responses ──
def parse_response(raw, item):
    result = {"plausibilities": {}, "misconceptions": {}, "difficulty": None, "discrimination": None}

    for opt in item["options"]:
        label = opt["label"]
        m = re.search(rf"{label}:\s*PLAUSIBILITY=(\d+)", raw)
        if m:
            result["plausibilities"][label] = int(m.group(1))
        m2 = re.search(rf"{label}:\s*PLAUSIBILITY=\d+\s*MISCONCEPTION=(.*?)(?:\n|$)", raw)
        if m2:
            result["misconceptions"][label] = m2.group(1).strip()

    m = re.search(r"DIFFICULTY:\s*([\d.]+)", raw)
    if m:
        result["difficulty"] = float(m.group(1))

    m = re.search(r"DISCRIMINATION:\s*([\d.]+)", raw)
    if m:
        result["discrimination"] = float(m.group(1))

    return result

# ── build analysis dataframe ──
rows = []
for item in usable_items:
    qid = str(item["question_id"])
    if qid not in cache or not cache[qid].get("raw"):
        continue

    parsed = parse_response(cache[qid]["raw"], item)
    if parsed["difficulty"] is None or parsed["discrimination"] is None:
        continue

    correct_label = [o["label"] for o in item["options"] if o["is_correct"]][0]
    wrong_plaus = [v for k, v in parsed["plausibilities"].items() if k != correct_label]
    if not wrong_plaus:
        continue

    rows.append({
        "question_id": item["question_id"],
        "question_text": item["question_text"][:80],
        "p_correct": item["p_correct"],
        "pb_discrimination": item["pb_discrimination"],
        "pred_difficulty": parsed["difficulty"],
        "pred_discrimination": parsed["discrimination"],
        "max_plausibility": max(wrong_plaus),
        "mean_plausibility": np.mean(wrong_plaus),
        "n_plausible": sum(1 for p in wrong_plaus if p >= 5),
        "plausibility_range": max(wrong_plaus) - min(wrong_plaus),
        "plausibilities": parsed["plausibilities"],
        "misconceptions": parsed["misconceptions"],
        "n_options": len(item["options"]),
    })

df = pd.DataFrame(rows)
print(f"\n{'='*70}")
print(f"ANALYSIS: {len(df)} items with complete data")
print(f"{'='*70}\n")

# ── 1 & 2: Spearman correlations ──
metrics = ["pred_discrimination", "max_plausibility", "mean_plausibility",
           "n_plausible", "plausibility_range", "pred_difficulty"]

print("SPEARMAN CORRELATIONS")
print("-" * 70)
header = f"{'Metric':<25} {'vs pb_disc':>15} {'vs p_correct':>15}"
print(header)
print("-" * 70)

for m in metrics:
    rho_disc, p_disc = stats.spearmanr(df[m], df["pb_discrimination"])
    rho_diff, p_diff = stats.spearmanr(df[m], df["p_correct"])
    sig_d = "***" if p_disc < 0.001 else "**" if p_disc < 0.01 else "*" if p_disc < 0.05 else ""
    sig_p = "***" if p_diff < 0.001 else "**" if p_diff < 0.01 else "*" if p_diff < 0.05 else ""
    print(f"{m:<25} {rho_disc:>8.3f}{sig_d:<4} p={p_disc:.4f}   {rho_diff:>8.3f}{sig_p:<4} p={p_diff:.4f}")

print()

# ── 3: Multiple regression ──
print("MULTIPLE REGRESSION: Distractor features -> pb_discrimination")
print("-" * 70)

features = ["max_plausibility", "mean_plausibility", "n_plausible", "plausibility_range"]
X = df[features].values
y = df["pb_discrimination"].values

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2 = reg.score(X, y)
rho_reg, p_reg = stats.spearmanr(y_pred, y)

print(f"R2 = {r2:.4f}")
print(f"Spearman rho of regression predictions vs actual: {rho_reg:.3f} (p={p_reg:.4f})")
print(f"\nCoefficients:")
for feat, coef in zip(features, reg.coef_):
    print(f"  {feat:<25} {coef:>8.4f}")
print(f"  {'intercept':<25} {reg.intercept_:>8.4f}")
print()

# Also try predicting difficulty
print("MULTIPLE REGRESSION: Distractor features -> p_correct")
print("-" * 70)
y2 = df["p_correct"].values
reg2 = LinearRegression().fit(X, y2)
r2_2 = reg2.score(X, y2)
rho_reg2, p_reg2 = stats.spearmanr(reg2.predict(X), y2)
print(f"R2 = {r2_2:.4f}")
print(f"Spearman rho of regression predictions vs actual: {rho_reg2:.3f} (p={p_reg2:.4f})")
print(f"\nCoefficients:")
for feat, coef in zip(features, reg2.coef_):
    print(f"  {feat:<25} {coef:>8.4f}")
print(f"  {'intercept':<25} {reg2.intercept_:>8.4f}")
print()

# ── 4: Summary table ──
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Metric':<25} {'rho(disc)':>10} {'rho(diff)':>10}")
print("-" * 50)
for m in metrics:
    rho_disc, _ = stats.spearmanr(df[m], df["pb_discrimination"])
    rho_diff, _ = stats.spearmanr(df[m], df["p_correct"])
    print(f"{m:<25} {rho_disc:>10.3f} {rho_diff:>10.3f}")
print(f"{'regression(4feat)':<25} {rho_reg:>10.3f} {rho_reg2:>10.3f}")
print()

# ── 5: High plausibility examples ──
print("=" * 70)
print("HIGH PLAUSIBILITY DISTRACTORS (max_plausibility >= 8)")
print("=" * 70)
high = df[df["max_plausibility"] >= 8].sort_values("max_plausibility", ascending=False).head(5)
for _, row in high.iterrows():
    print(f"\nQ{row['question_id']}: {row['question_text']}...")
    print(f"  Plausibilities: {row['plausibilities']}")
    print(f"  Misconceptions: {row['misconceptions']}")
    print(f"  pb_discrimination={row['pb_discrimination']:.3f}  p_correct={row['p_correct']:.3f}")
    print(f"  pred_disc={row['pred_discrimination']:.2f}  pred_diff={row['pred_difficulty']:.2f}")

# ── 6: Low plausibility examples ──
print(f"\n{'='*70}")
print("LOW PLAUSIBILITY DISTRACTORS (max_plausibility <= 3)")
print("=" * 70)
low = df[df["max_plausibility"] <= 3].sort_values("max_plausibility").head(5)
if len(low) == 0:
    low = df.sort_values("max_plausibility").head(5)
    print("(No items with max_plausibility <= 3, showing 5 lowest)")
for _, row in low.iterrows():
    print(f"\nQ{row['question_id']}: {row['question_text']}...")
    print(f"  Plausibilities: {row['plausibilities']}")
    print(f"  Misconceptions: {row['misconceptions']}")
    print(f"  pb_discrimination={row['pb_discrimination']:.3f}  p_correct={row['p_correct']:.3f}")
    print(f"  pred_disc={row['pred_discrimination']:.2f}  pred_diff={row['pred_difficulty']:.2f}")

# ── descriptive stats ──
print(f"\n{'='*70}")
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
for col in ["p_correct", "pb_discrimination", "pred_difficulty", "pred_discrimination",
            "max_plausibility", "mean_plausibility", "n_plausible"]:
    print(f"{col:<25} mean={df[col].mean():.3f}  sd={df[col].std():.3f}  "
          f"min={df[col].min():.3f}  max={df[col].max():.3f}")
