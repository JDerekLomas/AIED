#!/usr/bin/env python3
"""
Run pilot experiment with random sample of Eedi items.
"""

import json
import os
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from anthropic import Anthropic

# Models for pilot
MODELS = {
    "gpt-3.5-turbo": {"provider": "openai", "model_id": "gpt-3.5-turbo", "tier": "mid", "rpm": 500},
    "claude-3-haiku": {"provider": "anthropic", "model_id": "claude-3-haiku-20240307", "tier": "mid", "rpm": 50},
}

# Misconception-specific prompts
MISCONCEPTION_PROMPTS = {
    1507: {
        "S2_knowledge": """KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't sure how it works""",
        "S3_belief": """You think about math expressions like reading a sentence. Just as you read English left-to-right,
word by word, you believe math should be solved left-to-right, operation by operation.
When you see 3 + 4 × 2, you naturally do 3+4 first (getting 7), then ×2 (getting 14).""",
        "S4_procedure": """Your procedure for expressions:
STEP 1: Find the leftmost operation
STEP 2: Do that operation with the numbers on either side
STEP 3: Replace with the result
STEP 4: Repeat until one number remains

Example: 5 + 3 × 2 → (5+3)=8 → 8×2=16 → Answer: 16""",
    },
    1214: {
        "S2_knowledge": """KNOW WELL: Basic arithmetic, what equations look like
STILL LEARNING: Solving equations by isolating the variable
HEARD OF BUT FUZZY: "Inverse operations" - you know you're supposed to "do the opposite" but get confused""",
        "S3_belief": """You believe that to solve equations, you perform the same operation on both sides to keep them balanced.
If you see +3 on one side, you think "add 3 to both sides" to balance it out.
Doing the opposite seems like it would unbalance the equation.""",
        "S4_procedure": """Your procedure for solving equations:
STEP 1: Identify the operation next to the variable (like +3 or ×2)
STEP 2: Do the SAME operation to both sides
STEP 3: Simplify

Example: x + 5 = 12 → Add 5 to both sides → x + 10 = 17""",
    },
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


def build_prompt(spec_level: str, item: dict) -> str:
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


def parse_answer(response: str) -> Optional[str]:
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


def call_model(model: str, prompt: str) -> str:
    config = MODELS[model]
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
    # Load pilot items
    with open('data/pilot_items.json') as f:
        items = json.load(f)

    output_dir = Path('pilot/eedi_pilot')
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / 'responses.jsonl'

    models = list(MODELS.keys())
    spec_levels = ["S1", "S2", "S3", "S4"]
    reps = 2

    total = len(models) * len(items) * len(spec_levels) * reps

    print("=" * 60)
    print("PILOT EXPERIMENT WITH EEDI ITEMS")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Items: {len(items)}")
    print(f"Specs: {spec_levels}")
    print(f"Reps: {reps}")
    print(f"Total calls: {total}")
    print()

    count = 0
    results = []

    for model in models:
        config = MODELS[model]
        delay = 60.0 / config["rpm"]

        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        for item in items:
            for spec in spec_levels:
                for rep in range(1, reps + 1):
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
                            "misconception_type": item["misconception_type"],
                            "spec_level": spec,
                            "question_id": item["question_id"],
                            "correct_answer": item["correct_answer"],
                            "target_distractor": item["target_distractor"],
                            "human_target_rate": item["human_target_rate"],
                            "parsed_answer": parsed,
                            "is_correct": is_correct,
                            "hit_target": hit_target,
                            "rep": rep,
                            "raw_response": raw,
                        }
                        results.append(result)

                        with open(responses_file, 'a') as f:
                            f.write(json.dumps(result) + '\n')

                        print(f"  {status} Q{item['question_id']} {spec} r{rep}: {parsed} [{count}/{total}]")

                    except Exception as e:
                        print(f"  ERROR Q{item['question_id']} {spec} r{rep}: {e}")

                    time.sleep(delay)

    # Summary
    print("\n" + "=" * 60)
    print("PILOT RESULTS SUMMARY")
    print("=" * 60)

    print("\nBy Specification Level:")
    for spec in spec_levels:
        spec_results = [r for r in results if r["spec_level"] == spec]
        errors = [r for r in spec_results if not r["is_correct"]]
        targets = [r for r in errors if r["hit_target"]]

        error_rate = len(errors) / len(spec_results) * 100 if spec_results else 0
        target_rate = len(targets) / len(errors) * 100 if errors else 0
        avg_human = sum(r["human_target_rate"] for r in spec_results) / len(spec_results) * 100 if spec_results else 0

        print(f"  {spec}: {error_rate:.1f}% errors, {target_rate:.1f}% target rate (human: {avg_human:.1f}%)")

    print("\nBy Model:")
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        errors = [r for r in model_results if not r["is_correct"]]
        targets = [r for r in errors if r["hit_target"]]

        error_rate = len(errors) / len(model_results) * 100 if model_results else 0
        target_rate = len(targets) / len(errors) * 100 if errors else 0

        print(f"  {model}: {error_rate:.1f}% errors, {target_rate:.1f}% target rate")

    print("\nBy Misconception Type:")
    for mtype in ["procedural", "conceptual"]:
        type_results = [r for r in results if r["misconception_type"] == mtype]
        errors = [r for r in type_results if not r["is_correct"]]
        targets = [r for r in errors if r["hit_target"]]

        error_rate = len(errors) / len(type_results) * 100 if type_results else 0
        target_rate = len(targets) / len(errors) * 100 if errors else 0

        print(f"  {mtype}: {error_rate:.1f}% errors, {target_rate:.1f}% target rate")

    print(f"\nResults saved to: {responses_file}")


if __name__ == "__main__":
    main()
