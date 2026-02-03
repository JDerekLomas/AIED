#!/usr/bin/env python3
"""
Structured IRT difficulty estimation experiment.

Calls Opus with a structured CoT prompt for each item, extracting
XML-tagged feature ratings and direct IRT predictions.

Design:
- 105 IRT items → 5 calibration (excluded) → 100 items
- 5 replications per item = 500 API calls
- 12 ratings (1-10) + 4 direct predictions per call
- Output: aggregated features.csv (100 rows × 28 columns)
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
NUM_REPLICATIONS = 5
RANDOM_SEED = 42

PROVIDER_CONFIGS = {
    "gemini-3-flash": {
        "provider": "gemini",
        "model_id": "gemini-3-flash-preview",
        "output_dir": "pilot/structured_estimation_gemini3flash",
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
        "output_dir": "pilot/structured_estimation_gemini25flash",
    },
    "opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "output_dir": "pilot/structured_estimation",
    },
}

RATING_TAGS = [
    "concept_complexity",
    "procedural_steps",
    "abstraction_level",
    "prerequisite_knowledge",
    "distractor_plausibility",
    "misconception_strength",
    "question_clarity",
    "cognitive_load",
    "age_appropriateness",
    "transfer_distance",
    "visual_complexity",
    "solution_uniqueness",
]

PREDICTION_TAGS = [
    "classical_difficulty",
    "irt_difficulty",
    "irt_discrimination",
    "guessing_probability",
]


def load_data():
    """Load and merge item data with IRT parameters."""
    irt = json.loads(Path("results/irt_proper_statistics.json").read_text())
    curated = pd.read_csv("data/eedi/curated_eedi_items.csv")
    eedi = pd.read_csv("data/eedi/eedi_with_student_data.csv")

    answer_cols = ["QuestionId", "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
    answers = eedi[answer_cols].drop_duplicates(subset=["QuestionId"])
    df = curated.merge(answers, on="QuestionId", how="left").drop_duplicates(subset=["QuestionId"])

    # Attach IRT params
    irt_items = irt["items"]
    irt_rows = []
    for qid_str, params in irt_items.items():
        row = {"QuestionId": int(qid_str)}
        row.update(params)
        irt_rows.append(row)
    irt_df = pd.DataFrame(irt_rows)
    df = df.merge(irt_df, on="QuestionId", how="inner")
    return df


def select_calibration_items(df, n=5):
    """Select 5 calibration items at 10th/30th/50th/70th/90th percentile of b_2pl."""
    sorted_df = df.sort_values("b_2pl").reset_index(drop=True)
    percentiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    indices = [int(p * (len(sorted_df) - 1)) for p in percentiles]
    return sorted_df.iloc[indices]


def format_item_text(row):
    """Format a single item for the prompt."""
    lines = [
        f"Question: {row['question_text']}",
        f"A) {row['AnswerAText']}",
        f"B) {row['AnswerBText']}",
        f"C) {row['AnswerCText']}",
        f"D) {row['AnswerDText']}",
        f"Correct answer: {row['correct_answer_kaggle']}",
        f"Misconception tested: {row['misconception_name']}",
    ]
    return "\n".join(lines)


def format_calibration_block(cal_items):
    """Format calibration items with revealed IRT parameters."""
    blocks = []
    for _, row in cal_items.iterrows():
        item_text = format_item_text(row)
        params = (
            f"  Classical difficulty (proportion incorrect): {row['difficulty_classical']:.3f}\n"
            f"  IRT difficulty (b, 2PL): {row['b_2pl']:.3f}\n"
            f"  IRT discrimination (a, 2PL): {row['a_2pl']:.3f}\n"
            f"  Guessing parameter (c, 3PL): {row['c_3pl_guess']:.3f}"
        )
        blocks.append(f"{item_text}\n\nActual parameters:\n{params}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(cal_items, test_item):
    """Build the full prompt for a single test item."""
    cal_block = format_calibration_block(cal_items)
    item_text = format_item_text(test_item)

    return f"""You are an expert psychometrician analyzing math assessment items for UK secondary school students (ages 11-16). These are 4-choice multiple-choice questions from diagnostic math assessments.

## Calibration Examples
Below are 5 items with their known psychometric parameters. Use these to calibrate your judgments.

{cal_block}

---

## Target Item
Analyze this item and provide your ratings and predictions.

{item_text}

## Instructions
1. First, think through the item carefully in <chain_of_thought> tags.
2. Then provide 12 ratings (integer 1-10) in <ratings> tags, each in its own XML tag.
3. Finally, provide 4 direct predictions in <predictions> tags.

Rating dimensions (1-10 scale):
- concept_complexity: Number/depth of math concepts involved
- procedural_steps: Number of sequential steps to solve
- abstraction_level: Concrete (1) to abstract (10)
- prerequisite_knowledge: Amount of prior math knowledge needed
- distractor_plausibility: How attractive the wrong answers are
- misconception_strength: How common the tested misconception is
- question_clarity: How unambiguous the question is (10 = very clear)
- cognitive_load: Working memory demands
- age_appropriateness: Match to 11-16 curriculum (10 = perfect match)
- transfer_distance: Distance from standard textbook presentation
- visual_complexity: Notation/diagram complexity
- solution_uniqueness: Whether partial knowledge helps (10 = must fully understand)

Prediction ranges:
- classical_difficulty: proportion of students answering INCORRECTLY, 0 to 1 (higher = harder)
- irt_difficulty: b parameter, typically -1 to 1 (most items), extreme range -5 to 2
- irt_discrimination: a parameter, roughly 0.3 to 3
- guessing_probability: c parameter, 0 to 0.5

Respond in this exact format:

<chain_of_thought>
Your reasoning here...
</chain_of_thought>

<ratings>
<concept_complexity>N</concept_complexity>
<procedural_steps>N</procedural_steps>
<abstraction_level>N</abstraction_level>
<prerequisite_knowledge>N</prerequisite_knowledge>
<distractor_plausibility>N</distractor_plausibility>
<misconception_strength>N</misconception_strength>
<question_clarity>N</question_clarity>
<cognitive_load>N</cognitive_load>
<age_appropriateness>N</age_appropriateness>
<transfer_distance>N</transfer_distance>
<visual_complexity>N</visual_complexity>
<solution_uniqueness>N</solution_uniqueness>
</ratings>

<predictions>
<classical_difficulty>X.XX</classical_difficulty>
<irt_difficulty>X.XX</irt_difficulty>
<irt_discrimination>X.XX</irt_discrimination>
<guessing_probability>X.XX</guessing_probability>
</predictions>"""


def parse_response(text):
    """Extract ratings and predictions from XML-tagged response."""
    result = {}
    all_tags = RATING_TAGS + PREDICTION_TAGS

    for tag in all_tags:
        match = re.search(rf"<{tag}>\s*([-\d.]+)\s*</{tag}>", text)
        if match:
            result[tag] = float(match.group(1))
        else:
            result[tag] = None

    # Count successful extractions
    n_found = sum(1 for v in result.values() if v is not None)
    result["_extraction_success"] = n_found
    result["_total_tags"] = len(all_tags)
    return result


def make_api_call(client, provider, model_id, prompt):
    """Call LLM API, return text response."""
    if provider == "anthropic":
        from anthropic import Anthropic as _A
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    elif provider == "gemini":
        from google.genai import types
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_experiment(model_key="gemini-3-flash"):
    """Run the full structured estimation experiment."""
    config = PROVIDER_CONFIGS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]
    OUTPUT_DIR = Path(config["output_dir"])

    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
    elif provider == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_DIR / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} items with IRT parameters")

    # Select calibration items
    cal_items = select_calibration_items(df)
    cal_qids = set(cal_items["QuestionId"].values)
    print(f"Calibration items (excluded from test): {sorted(cal_qids)}")

    # Test items = all minus calibration
    test_df = df[~df["QuestionId"].isin(cal_qids)].reset_index(drop=True)
    print(f"Test items: {len(test_df)}")

    # Train/test split (80/20 stratified by target_type)
    from sklearn.model_selection import train_test_split

    train_df, holdout_df = train_test_split(
        test_df, test_size=0.2, random_state=RANDOM_SEED, stratify=test_df["target_type"]
    )
    print(f"Train: {len(train_df)}, Holdout: {len(holdout_df)}")

    # Save split info
    split_info = {
        "calibration_qids": [int(q) for q in cal_qids],
        "train_qids": train_df["QuestionId"].tolist(),
        "holdout_qids": holdout_df["QuestionId"].tolist(),
    }
    (OUTPUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2))

    # Run all test items (train + holdout) through the LLM
    all_results = []
    total_calls = len(test_df) * NUM_REPLICATIONS
    call_num = 0

    for _, item_row in test_df.iterrows():
        qid = item_row["QuestionId"]
        for rep in range(NUM_REPLICATIONS):
            call_num += 1
            print(f"  [{call_num}/{total_calls}] QID={qid} rep={rep+1}")

            prompt = build_prompt(cal_items, item_row)

            # Resume support: skip if raw response already exists
            raw_path = raw_dir / f"qid{qid}_rep{rep}.txt"
            if raw_path.exists():
                text = raw_path.read_text()
                parsed = parse_response(text)
                parsed["QuestionId"] = qid
                parsed["replication"] = rep
                all_results.append(parsed)
                print(f"    (cached)")
                continue

            try:
                text = make_api_call(client, provider, model_id, prompt)

                raw_path.write_text(text)

                # Parse
                parsed = parse_response(text)
                parsed["QuestionId"] = qid
                parsed["replication"] = rep
                all_results.append(parsed)

                if parsed["_extraction_success"] < parsed["_total_tags"]:
                    print(f"    WARNING: only extracted {parsed['_extraction_success']}/{parsed['_total_tags']} tags")

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"QuestionId": qid, "replication": rep, "_error": str(e)})

            # Rate limiting
            time.sleep(0.5)

    # Save raw results
    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)

    # Aggregate: mean and std across replications
    feature_cols = RATING_TAGS + PREDICTION_TAGS
    agg_rows = []

    for qid in test_df["QuestionId"].values:
        qid_data = raw_df[(raw_df["QuestionId"] == qid) & raw_df["_extraction_success"].notna()]
        if len(qid_data) == 0:
            continue

        row = {"QuestionId": qid}
        for col in feature_cols:
            vals = qid_data[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
        agg_rows.append(row)

    features_df = pd.DataFrame(agg_rows)
    features_df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    # Report extraction success
    success_rate = raw_df["_extraction_success"].mean() / raw_df["_total_tags"].mean() * 100
    print(f"\nExtraction success rate: {success_rate:.1f}%")
    print(f"Saved {len(features_df)} aggregated feature rows to {OUTPUT_DIR / 'features.csv'}")
    print(f"Saved {len(raw_df)} raw results to {OUTPUT_DIR / 'raw_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3-flash", choices=list(PROVIDER_CONFIGS.keys()))
    args = parser.parse_args()
    print(f"Using model: {args.model} ({PROVIDER_CONFIGS[args.model]['model_id']})")
    run_experiment(args.model)
