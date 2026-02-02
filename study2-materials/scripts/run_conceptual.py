#!/usr/bin/env python3
"""Run experiment on conceptual misconceptions only (1597, 217)."""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from anthropic import Anthropic

MODELS = {
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o", "tier": "frontier", "rpm": 500},
    "claude-sonnet-4": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514", "tier": "frontier", "rpm": 50},
}

MISCONCEPTION_PROMPTS = {
    1597: {
        "S2_knowledge": """KNOW WELL: Multiplying positive numbers, what negative numbers mean
STILL LEARNING: Rules for multiplying with negative numbers
HEARD OF BUT FUZZY: Something about "two negatives make a positive" but that seems weird""",
        "S3_belief": """You believe negative numbers represent "negativity."
When you multiply two negative things together, you get something MORE negative, not positive.
Two wrongs don't make a right. Two negatives should make something even more negative.""",
        "S4_procedure": """Your procedure for multiplying with negatives:
STEP 1: Multiply the absolute values
STEP 2: Count the negative signs
STEP 3: If ANY negative signs, the answer is negative

Example: (-3) × (-4) → 3×4=12 → Two negatives → Answer: -12""",
    },
    217: {
        "S2_knowledge": """KNOW WELL: What fractions represent, adding whole numbers
STILL LEARNING: Adding fractions with different denominators
NEVER LEARNED: Why you need common denominators""",
        "S3_belief": """You believe fractions are just two numbers stacked. When you add fractions,
you add the top parts together and the bottom parts together.
1/2 + 1/3 = (1+1)/(2+3) = 2/5. This makes sense: combine the tops, combine the bottoms.""",
        "S4_procedure": """Your procedure for adding fractions:
STEP 1: Add the numerators together
STEP 2: Add the denominators together
STEP 3: Write as: sum of tops / sum of bottoms

Example: 1/2 + 2/3 → (1+2)/(2+3) → Answer: 3/5""",
    },
}


def build_prompt(spec_level, item):
    misc_id = item["misconception_id"]
    misc_prompts = MISCONCEPTION_PROMPTS[misc_id]
    question = item["question_text"]
    options = f"A) {item['answer_a']}\nB) {item['answer_b']}\nC) {item['answer_c']}\nD) {item['answer_d']}"

    if spec_level == "S1":
        return f"""You are a student who sometimes struggles with math. Show your thinking, then give your answer.

Question: {question}
Options:
{options}

Show your work, then state your answer as **Answer: X**"""
    elif spec_level == "S2":
        return f"""You are a student with the following math knowledge:

{misc_prompts['S2_knowledge']}

Show your thinking, then give your answer.

Question: {question}
Options:
{options}

Show your work, then state your answer as **Answer: X**"""
    elif spec_level == "S3":
        return f"""You are a student with a specific way of thinking about math.

{misc_prompts['S3_belief']}

Show your thinking, then give your answer.

Question: {question}
Options:
{options}

Show your work, then state your answer as **Answer: X**"""
    elif spec_level == "S4":
        return f"""You are a student who uses a specific procedure.

{misc_prompts['S4_procedure']}

Apply your procedure step by step.

Question: {question}
Options:
{options}

Show your work, then state your answer as **Answer: X**"""


def parse_answer(response):
    match = re.search(r'Answer:\s*\*?\*?([A-D])\)?\*?\*?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'\*\*([A-D])\)', response)
    if match:
        return match.group(1).upper()
    match = re.search(r'the answer is\s*\*?\*?([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-D])\)?\.?\s*$', response.strip())
    if match:
        return match.group(1).upper()
    return None


def call_model(model_name, prompt):
    config = MODELS[model_name]
    if config["provider"] == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    elif config["provider"] == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=config["model_id"],
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


def main():
    with open('data/conceptual_items.json') as f:
        items = json.load(f)

    output_dir = Path('results/full_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / 'responses_conceptual.jsonl'

    models = list(MODELS.keys())
    spec_levels = ["S1", "S2", "S3", "S4"]

    total = len(models) * len(items) * len(spec_levels)

    print("=" * 60)
    print("CONCEPTUAL MISCONCEPTIONS (1597, 217)")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Items: {len(items)} (1597: 26, 217: 14)")
    print(f"Total calls: {total}")
    print()

    count = 0

    for model in models:
        config = MODELS[model]
        delay = 60.0 / config["rpm"]

        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        for item in items:
            for spec in spec_levels:
                count += 1
                prompt = build_prompt(spec, item)

                try:
                    raw = call_model(model, prompt)
                    parsed = parse_answer(raw)
                    is_correct = parsed == item["correct_answer"]
                    hit_target = parsed == item["target_distractor"]

                    if is_correct:
                        status = "✓"
                    elif hit_target:
                        status = "⊛"
                    elif parsed is None:
                        status = "✗"
                    else:
                        status = "○"

                    result = {
                        "model": model,
                        "model_tier": config["tier"],
                        "misconception_id": item["misconception_id"],
                        "misconception_name": item["misconception_name"],
                        "misconception_type": item["misconception_type"],
                        "spec_level": spec,
                        "question_id": item["question_id"],
                        "correct_answer": item["correct_answer"],
                        "target_distractor": item["target_distractor"],
                        "human_target_rate": item["human_target_rate"],
                        "parsed_answer": parsed,
                        "is_correct": is_correct,
                        "hit_target": hit_target,
                        "rep": 1,
                        "raw_response": raw,
                        "timestamp": datetime.now().isoformat(),
                    }

                    with open(responses_file, 'a') as f:
                        f.write(json.dumps(result) + '\n')

                    print(f"  {status} M{item['misconception_id']} Q{item['question_id']} {spec}: {parsed} [{count}/{total}]")

                except Exception as e:
                    print(f"  ERROR: {e}")

                time.sleep(delay)

    print(f"\nResults saved to: {responses_file}")


if __name__ == "__main__":
    main()
