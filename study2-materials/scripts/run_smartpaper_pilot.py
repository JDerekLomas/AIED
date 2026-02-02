"""
Pilot: Structured difficulty estimation on SmartPaper open-ended items.
Adapts the Eedi structured estimation approach for open-response items.

Small test: 20 items (5 calibration + 15 test), 3 reps each = 45 calls.
Uses Gemini 2.5 Flash.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

NUM_REPS = 3
OUTPUT_DIR = Path("pilot/smartpaper_structured_pilot")

# Ratings adapted for open-ended items (no distractors)
RATING_TAGS = [
    "concept_complexity",
    "procedural_steps",
    "abstraction_level",
    "prerequisite_knowledge",
    "rubric_specificity",       # replaces distractor_plausibility
    "common_error_likelihood",  # replaces misconception_strength
    "question_clarity",
    "cognitive_load",
    "age_appropriateness",
    "transfer_distance",
    "language_demand",          # replaces visual_complexity (OCR handwriting)
    "response_precision",       # replaces solution_uniqueness
]

PREDICTION_TAGS = [
    "classical_difficulty",     # proportion scoring 0
    "mean_score_proportion",    # mean score / max score
    "item_total_correlation",   # discrimination
]


def load_items():
    items = json.loads(Path("data/smartpaper/item_statistics.json").read_text())
    # Filter to items with question_text
    items = [i for i in items if i.get('question_text')]
    return items


def select_calibration(items, n=5):
    """Pick 5 items spanning the difficulty range."""
    sorted_items = sorted(items, key=lambda x: x['classical_difficulty'])
    percentiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    indices = [int(p * (len(sorted_items) - 1)) for p in percentiles]
    return [sorted_items[i] for i in indices]


def format_item(item):
    lines = [
        f"Subject: {item['subject']} (Grade {item['grade']})",
        f"Skill: {item['skill']}",
        f"Question: {item['question_text']}",
        f"Scoring rubric: {item['rubric']}",
        f"Max score: {item['max_score']}",
        f"Format: {item['content_type']}",
    ]
    return "\n".join(lines)


def format_calibration_block(cal_items):
    blocks = []
    for item in cal_items:
        text = format_item(item)
        params = (
            f"  Classical difficulty (prop scoring 0): {1 - item['classical_difficulty']:.3f}\n"
            f"  Mean score proportion: {item['classical_difficulty']:.3f}\n"
            f"  Item-total correlation: {item['item_total_correlation']:.3f}\n"
            f"  N responses: {item['n_responses']}"
        )
        blocks.append(f"{text}\n\nActual parameters:\n{params}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(cal_items, test_item):
    cal_block = format_calibration_block(cal_items)
    item_text = format_item(test_item)

    return f"""You are an expert psychometrician analyzing open-ended assessment items for Indian government school students (Grades 6-8). These items require written/short answers that are scored against rubrics. Student answers are handwritten and OCR'd.

## Calibration Examples
Below are 5 items with known psychometric parameters. Use these to calibrate your judgments.

{cal_block}

---

## Target Item
Analyze this item and provide your ratings and predictions.

{item_text}

## Instructions
1. First, think through the item carefully in <chain_of_thought> tags.
2. Then provide 12 ratings (integer 1-10) in <ratings> tags.
3. Finally, provide 3 direct predictions in <predictions> tags.

Rating dimensions (1-10 scale):
- concept_complexity: Number/depth of concepts involved
- procedural_steps: Number of sequential steps to answer
- abstraction_level: Concrete recall (1) to abstract reasoning (10)
- prerequisite_knowledge: Amount of prior knowledge needed
- rubric_specificity: How narrow/specific the acceptable answer is (10 = very specific)
- common_error_likelihood: How likely students are to make common errors
- question_clarity: How unambiguous the question is (10 = very clear)
- cognitive_load: Working memory demands
- age_appropriateness: Match to grade-level curriculum (10 = perfect match)
- transfer_distance: Distance from standard textbook presentation
- language_demand: Reading comprehension / language production required
- response_precision: How precisely students must express the answer (10 = exact wording needed)

Prediction ranges:
- classical_difficulty: proportion of students scoring 0, range 0 to 1 (higher = harder)
- mean_score_proportion: mean score / max score, range 0 to 1 (higher = easier)
- item_total_correlation: point-biserial r, typically 0.2 to 0.8

Respond in this exact format:

<chain_of_thought>
Your reasoning here...
</chain_of_thought>

<ratings>
<concept_complexity>N</concept_complexity>
<procedural_steps>N</procedural_steps>
<abstraction_level>N</abstraction_level>
<prerequisite_knowledge>N</prerequisite_knowledge>
<rubric_specificity>N</rubric_specificity>
<common_error_likelihood>N</common_error_likelihood>
<question_clarity>N</question_clarity>
<cognitive_load>N</cognitive_load>
<age_appropriateness>N</age_appropriateness>
<transfer_distance>N</transfer_distance>
<language_demand>N</language_demand>
<response_precision>N</response_precision>
</ratings>

<predictions>
<classical_difficulty>X.XX</classical_difficulty>
<mean_score_proportion>X.XX</mean_score_proportion>
<item_total_correlation>X.XX</item_total_correlation>
</predictions>"""


def parse_response(text):
    result = {}
    all_tags = RATING_TAGS + PREDICTION_TAGS
    for tag in all_tags:
        match = re.search(rf"<{tag}>\s*([-\d.]+)\s*</{tag}>", text)
        result[tag] = float(match.group(1)) if match else None
    result["_ok"] = sum(1 for v in result.values() if v is not None)
    return result


def run():
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    model_id = "gemini-2.5-flash"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_DIR / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    items = load_items()
    print(f"Loaded {len(items)} items with statistics")

    cal_items = select_calibration(items)
    cal_keys = {(c['assessment'], c['question_number']) for c in cal_items}
    print(f"Calibration items: {[(c['assessment'], c['question_number']) for c in cal_items]}")

    # Test on 15 items sampled across subjects and difficulty
    test_items = [i for i in items if (i['assessment'], i['question_number']) not in cal_keys]
    np.random.seed(42)
    test_items = [test_items[i] for i in np.random.choice(len(test_items), size=15, replace=False)]
    print(f"Test items: {len(test_items)}")

    results = []
    total = len(test_items) * NUM_REPS
    n = 0

    for item in test_items:
        item_key = f"{item['assessment']}_q{item['question_number']}"
        for rep in range(NUM_REPS):
            n += 1
            raw_path = raw_dir / f"{item_key}_rep{rep}.txt"

            if raw_path.exists():
                text = raw_path.read_text()
                parsed = parse_response(text)
                parsed["item_key"] = item_key
                parsed["rep"] = rep
                results.append(parsed)
                print(f"  [{n}/{total}] {item_key} rep={rep} (cached)")
                continue

            print(f"  [{n}/{total}] {item_key} rep={rep}", end="", flush=True)
            prompt = build_prompt(cal_items, item)

            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=4096,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                text = response.text
                raw_path.write_text(text)
                parsed = parse_response(text)
                parsed["item_key"] = item_key
                parsed["rep"] = rep
                results.append(parsed)
                print(f" -> {parsed['_ok']}/15 tags")
            except Exception as e:
                print(f" ERROR: {e}")
                results.append({"item_key": item_key, "rep": rep, "_error": str(e)})

            time.sleep(0.3)

    # Aggregate and correlate with ground truth
    print("\n=== RESULTS ===")
    from scipy import stats

    pred_diffs = []
    actual_diffs = []
    pred_itc = []
    actual_itc = []

    for item in test_items:
        item_key = f"{item['assessment']}_q{item['question_number']}"
        reps = [r for r in results if r.get("item_key") == item_key and r.get("_ok", 0) > 0]
        if not reps:
            continue

        # Mean predictions across reps
        mean_pred_diff = np.nanmean([r.get("mean_score_proportion") for r in reps if r.get("mean_score_proportion") is not None])
        mean_pred_itc = np.nanmean([r.get("item_total_correlation") for r in reps if r.get("item_total_correlation") is not None])

        if not np.isnan(mean_pred_diff):
            pred_diffs.append(mean_pred_diff)
            actual_diffs.append(item["classical_difficulty"])
        if not np.isnan(mean_pred_itc):
            pred_itc.append(mean_pred_itc)
            actual_itc.append(item["item_total_correlation"])

        # Print per-item
        ratings_mean = {tag: np.nanmean([r.get(tag) for r in reps if r.get(tag) is not None])
                        for tag in RATING_TAGS}
        print(f"{item_key}: actual_p={item['classical_difficulty']:.3f} pred_p={mean_pred_diff:.3f} | "
              f"actual_r={item['item_total_correlation']:.3f} pred_r={mean_pred_itc:.3f}")

    if len(pred_diffs) >= 5:
        r_diff, p_diff = stats.pearsonr(pred_diffs, actual_diffs)
        rho_diff, _ = stats.spearmanr(pred_diffs, actual_diffs)
        print(f"\nDifficulty: Pearson r={r_diff:.3f} (p={p_diff:.3f}), Spearman rho={rho_diff:.3f}")

    if len(pred_itc) >= 5:
        r_itc, p_itc = stats.pearsonr(pred_itc, actual_itc)
        rho_itc, _ = stats.spearmanr(pred_itc, actual_itc)
        print(f"Item-total r: Pearson r={r_itc:.3f} (p={p_itc:.3f}), Spearman rho={rho_itc:.3f}")

    # Also check rating-based prediction via simple correlation
    print("\n=== RATING CORRELATIONS WITH DIFFICULTY ===")
    for tag in RATING_TAGS:
        tag_vals = []
        diff_vals = []
        for item in test_items:
            item_key = f"{item['assessment']}_q{item['question_number']}"
            reps = [r for r in results if r.get("item_key") == item_key and r.get(tag) is not None]
            if reps:
                tag_vals.append(np.nanmean([r[tag] for r in reps]))
                diff_vals.append(item["classical_difficulty"])
        if len(tag_vals) >= 5:
            r, p = stats.pearsonr(tag_vals, diff_vals)
            print(f"  {tag:30s}: r={r:+.3f} (p={p:.3f})")

    # Save
    import pandas as pd
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "raw_results.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
