#!/usr/bin/env python3
"""Test 3 decomposed prompts for predicting item discrimination on DBE-KT22."""

import json, os, re, time
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/dereklomas/AIED/study2-materials"
DATA = f"{BASE}/data/dbe-kt22"
CACHE_DIR = f"{BASE}/pilot/dbe_kt22_validation"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Load data ──
print("Loading data...", flush=True)
questions_df = pd.read_csv(f"{DATA}/Questions.csv")
choices_df = pd.read_csv(f"{DATA}/Question_Choices.csv")
trans_df = pd.read_csv(f"{DATA}/Transaction.csv")
item_stats = json.load(open(f"{DATA}/item_statistics.json"))
item_stats_map = {s["question_id"]: s for s in item_stats}

# ── Compute discrimination (point-biserial) from Transaction.csv ──
print("Computing discrimination from transactions...", flush=True)
trans_df["correct"] = trans_df["answer_state"].astype(str).str.lower() == "true"
student_scores = trans_df.groupby("student_id")["correct"].mean().rename("total_score")
trans_with_score = trans_df.merge(student_scores, on="student_id")

def compute_discrimination(group):
    if group["correct"].nunique() < 2:
        return np.nan
    corr, _ = stats.pointbiserialr(group["correct"].astype(int), group["total_score"])
    return corr

q_groups = trans_with_score.groupby("question_id")
n_responses = q_groups.size().rename("n_responses")
disc = q_groups.apply(compute_discrimination).rename("discrimination")
p_correct_computed = q_groups["correct"].mean().rename("p_correct")

item_data = pd.DataFrame({"n_responses": n_responses, "discrimination": disc, "p_correct": p_correct_computed})
item_data.index.name = "question_id"
item_data = item_data.reset_index()

# Override p_correct from item_statistics.json where available
for _, row in item_data.iterrows():
    qid = row["question_id"]
    if qid in item_stats_map and "p_correct" in item_stats_map[qid]:
        item_data.loc[item_data["question_id"] == qid, "p_correct"] = item_stats_map[qid]["p_correct"]

# ── Filter to usable items ──
print("Filtering items...", flush=True)
merged = item_data.merge(questions_df[["id", "question_text", "question_rich_text"]], left_on="question_id", right_on="id", how="inner")
mask = (
    (merged["question_text"].str.len() > 20) &
    (~merged["question_rich_text"].fillna("").str.contains("<img", case=False)) &
    (merged["n_responses"] >= 50) &
    (merged["discrimination"].notna())
)
filtered = merged[mask].copy()
print(f"Usable items: {len(filtered)}", flush=True)

# ── Build item prompts ──
choices_by_q = {}
for qid, group in choices_df.groupby("question_id"):
    opts = []
    correct = None
    for _, row in group.iterrows():
        label = chr(65 + len(opts))
        opts.append(f"{label}. {row['choice_text']}")
        if str(row["is_correct"]).lower() == "true":
            correct = f"{label}. {row['choice_text']}"
    choices_by_q[qid] = (opts, correct)

items = []
for _, row in filtered.iterrows():
    qid = int(row["question_id"])
    if qid not in choices_by_q:
        continue
    opts, correct = choices_by_q[qid]
    if correct is None:
        continue
    items.append({
        "qid": qid,
        "question_text": row["question_text"],
        "options": "\n".join(opts),
        "correct": correct,
        "discrimination": row["discrimination"],
        "p_correct": row["p_correct"],
    })

print(f"Items with choices: {len(items)}", flush=True)

# ── Prompts ──
PROMPTS = {
    "distractor_plausibility": """You are an experienced psychometrician reviewing test items.

For this multiple choice question, count how many of the WRONG answers would specifically attract a student who holds a particular misconception or makes a specific error. A plausible distractor targets a real mistake; an implausible one is obviously wrong.

Question: {question_text}

Options:
{options}

Correct Answer: {correct}

How many of the wrong answers are plausible distractors that target specific misconceptions? Reply with ONLY a number (0, 1, 2, 3, etc.).""",

    "guessability": """You are an experienced psychometrician reviewing test items.

For this multiple choice question, estimate how easily a student who doesn't know the material could get it right through guessing or elimination.

Question: {question_text}

Options:
{options}

Correct Answer: {correct}

On a scale from 0 to 10, how guessable is this item? (0 = impossible to guess correctly without knowledge, 10 = very easy to guess or eliminate wrong answers)

Reply with ONLY a number from 0 to 10.""",

    "misconception_specificity": """You are an experienced psychometrician reviewing test items.

For this multiple choice question, consider the most popular wrong answer. Does it target ONE specific misconception, or could students choose it for many different unrelated reasons?

Question: {question_text}

Options:
{options}

Correct Answer: {correct}

Rate the specificity of the primary distractor on a scale from 0 to 10:
0 = students would pick it for many unrelated reasons (random attractor)
10 = students pick it ONLY because of one specific, identifiable misconception

Reply with ONLY a number from 0 to 10.""",
}


def parse_number(text):
    m = re.search(r"(\d+(?:\.\d+)?)", text.strip())
    if m:
        return float(m.group(1))
    return None


# ── API setup ──
from google import genai

client = genai.Client()
MODEL = "gemini-3-flash-preview"


def call_llm(prompt_text):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_text,
        config={
            "temperature": 0,
            "max_output_tokens": 50,
            "thinking_config": {"thinking_budget": 0},
        },
    )
    return response.text


# ── Run each prompt ──
all_results = {}

for prompt_name, prompt_template in PROMPTS.items():
    cache_path = f"{CACHE_DIR}/probe_disc_{prompt_name}.json"

    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    else:
        cache = {}

    print(f"\n=== Prompt: {prompt_name} ===", flush=True)
    n_cached = sum(1 for item in items if str(item["qid"]) in cache)
    print(f"Cached: {n_cached}/{len(items)}", flush=True)

    for i, item in enumerate(items):
        key = str(item["qid"])
        if key in cache:
            continue

        prompt_text = prompt_template.format(
            question_text=item["question_text"],
            options=item["options"],
            correct=item["correct"],
        )

        try:
            resp = call_llm(prompt_text)
            cache[key] = {"response": resp, "parsed": parse_number(resp)}
        except Exception as e:
            print(f"  Error qid={item['qid']}: {e}", flush=True)
            cache[key] = {"response": str(e), "parsed": None}

        if (i + 1) % 20 == 0:
            json.dump(cache, open(cache_path, "w"), indent=2)
            print(f"  {i+1}/{len(items)} done", flush=True)

        time.sleep(0.1)

    json.dump(cache, open(cache_path, "w"), indent=2)
    print(f"  Complete: {len(cache)} cached", flush=True)
    all_results[prompt_name] = cache

# ── Analysis ──
print("\n" + "=" * 60, flush=True)
print("ANALYSIS", flush=True)
print("=" * 60, flush=True)

prompt_predictions = {}

for prompt_name in PROMPTS:
    cache = all_results[prompt_name]
    preds, discs, pcorrs, all_vals, qids = [], [], [], [], []
    n_parsed = n_total = 0

    for item in items:
        key = str(item["qid"])
        n_total += 1
        if key in cache and cache[key]["parsed"] is not None:
            val = cache[key]["parsed"]
            n_parsed += 1
            all_vals.append(val)
            preds.append(val)
            discs.append(item["discrimination"])
            pcorrs.append(item["p_correct"])
            qids.append(item["qid"])

    preds = np.array(preds)
    discs = np.array(discs)
    pcorrs = np.array(pcorrs)
    all_vals = np.array(all_vals)

    prompt_predictions[prompt_name] = {"preds": preds, "discs": discs, "pcorrs": pcorrs, "qids": qids}

    rho_disc, p_disc = stats.spearmanr(preds, discs)
    rho_pcorr, p_pcorr = stats.spearmanr(preds, pcorrs)

    print(f"\n--- {prompt_name} ---", flush=True)
    print(f"  Parsed: {n_parsed}/{n_total}", flush=True)
    print(f"  Distribution: mean={all_vals.mean():.2f}, std={all_vals.std():.2f}, min={all_vals.min():.1f}, max={all_vals.max():.1f}", flush=True)
    print(f"  Spearman rho vs discrimination: {rho_disc:.3f} (p={p_disc:.4f})", flush=True)
    print(f"  Spearman rho vs p_correct:      {rho_pcorr:.3f} (p={p_pcorr:.4f})", flush=True)

# ── Inter-prompt correlations ──
print(f"\n--- Inter-prompt correlations ---", flush=True)
prompt_names = list(PROMPTS.keys())
for i in range(len(prompt_names)):
    for j in range(i + 1, len(prompt_names)):
        n1, n2 = prompt_names[i], prompt_names[j]
        qids1 = set(prompt_predictions[n1]["qids"])
        qids2 = set(prompt_predictions[n2]["qids"])
        common = sorted(qids1 & qids2)
        v1 = [all_results[n1][str(q)]["parsed"] for q in common]
        v2 = [all_results[n2][str(q)]["parsed"] for q in common]
        rho, p = stats.spearmanr(v1, v2)
        print(f"  {n1} vs {n2}: rho={rho:.3f} (p={p:.4f}, n={len(common)})", flush=True)

# ── Composite ──
print(f"\n--- Composite (z-scored average) ---", flush=True)
common_qids = set(prompt_predictions[prompt_names[0]]["qids"])
for n in prompt_names[1:]:
    common_qids &= set(prompt_predictions[n]["qids"])
common_qids = sorted(common_qids)
print(f"  Common items: {len(common_qids)}", flush=True)

raw = {}
for n in prompt_names:
    raw[n] = np.array([all_results[n][str(q)]["parsed"] for q in common_qids])

z_scores = {}
for n in prompt_names:
    arr = raw[n]
    z_scores[n] = (arr - arr.mean()) / (arr.std() + 1e-10)

composite = np.mean([z_scores[n] for n in prompt_names], axis=0)

item_map = {item["qid"]: item for item in items}
gt_disc = np.array([item_map[q]["discrimination"] for q in common_qids])
gt_pcorr = np.array([item_map[q]["p_correct"] for q in common_qids])

rho_disc, p_disc = stats.spearmanr(composite, gt_disc)
rho_pcorr, p_pcorr = stats.spearmanr(composite, gt_pcorr)
print(f"  Spearman rho vs discrimination: {rho_disc:.3f} (p={p_disc:.4f})", flush=True)
print(f"  Spearman rho vs p_correct:      {rho_pcorr:.3f} (p={p_pcorr:.4f})", flush=True)

rho_base, p_base = stats.spearmanr(gt_disc, gt_pcorr)
print(f"\n--- Baseline: discrimination vs p_correct: rho={rho_base:.3f} (p={p_base:.4f}) ---", flush=True)

print("\nDone.", flush=True)
