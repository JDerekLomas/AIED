#!/usr/bin/env python3
"""
Run the S1-S4 Specification Level Experiment with Real Eedi Items

Uses actual Eedi items with human response distributions for comparison.
"""

import json
import os
import sys
import time
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import re
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API CLIENTS
# =============================================================================

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not installed")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: Anthropic not installed")


# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    # Frontier tier
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "tier": "frontier",
        "rpm": 500,
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "tier": "frontier",
        "rpm": 50,
    },
    # Mid tier
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "tier": "mid",
        "rpm": 500,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "tier": "mid",
        "rpm": 50,
    },
}

# Misconception-specific prompt components
MISCONCEPTION_PROMPTS = {
    1507: {
        "short_name": "left_to_right_ops",
        "S2_knowledge": """KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't sure how it works""",
        "S3_belief": """You think about math expressions like reading a sentence. Just as you read English left-to-right,
word by word, you believe math should be solved left-to-right, operation by operation.

When you see 3 + 4 × 2, you naturally do 3+4 first (getting 7), then ×2 (getting 14).
Why would you skip around randomly? That doesn't make sense.""",
        "S4_procedure": """Your procedure for expressions:
STEP 1: Find the leftmost operation
STEP 2: Do that operation with the numbers on either side
STEP 3: Replace those numbers and operator with your result
STEP 4: Repeat until only one number remains

Example: 5 + 3 × 2
- Leftmost op is +, so do 5+3=8
- Now have 8 × 2
- Do 8×2=16
- Answer: 16""",
    },
    1214: {
        "short_name": "same_op_not_inverse",
        "S2_knowledge": """KNOW WELL: Basic arithmetic, what equations look like
STILL LEARNING: Solving equations by isolating the variable
HEARD OF BUT FUZZY: "Inverse operations" - you know you're supposed to "do the opposite" but get confused about what that means""",
        "S3_belief": """You believe that to solve equations, you perform the same operation on both sides to keep them balanced.
If you see +3 on one side, you think "add 3 to both sides" to balance it out.
If you see ×2, you think "multiply both sides by 2."

Doing the opposite seems like it would unbalance the equation.""",
        "S4_procedure": """Your procedure for solving equations:
STEP 1: Identify the operation next to the variable (like +3 or ×2)
STEP 2: Do the SAME operation to both sides
STEP 3: Simplify both sides
STEP 4: Repeat until x is alone

Example: x + 5 = 12
- Operation next to x is +5
- Add 5 to both sides: x + 5 + 5 = 12 + 5
- Simplify: x + 10 = 17""",
    },
    1597: {
        "short_name": "neg_times_neg_is_neg",
        "S2_knowledge": """KNOW WELL: Multiplying positive numbers, what negative numbers mean
STILL LEARNING: Rules for multiplying with negative numbers
HEARD OF BUT FUZZY: Something about "two negatives make a positive" but that seems weird""",
        "S3_belief": """You believe negative numbers represent "negativity" or "badness."
When you multiply two negative things together, you get something MORE negative, not positive.
It's like: two wrongs don't make a right. Two negatives should make something even more negative.

The rule "negative times negative equals positive" seems arbitrary and counterintuitive to you.""",
        "S4_procedure": """Your procedure for multiplying with negatives:
STEP 1: Multiply the absolute values
STEP 2: Count the negative signs
STEP 3: If ANY negative signs, the answer is negative
STEP 4: More negatives = more negative

Example: (-3) × (-4)
- Absolute values: 3 × 4 = 12
- Two negative signs → definitely negative
- Answer: -12""",
    },
    217: {
        "short_name": "add_fractions_add_parts",
        "S2_knowledge": """KNOW WELL: What fractions represent, adding whole numbers
STILL LEARNING: Adding fractions with different denominators
NEVER LEARNED: Why you need common denominators""",
        "S3_belief": """You believe fractions are just two numbers stacked. When you add fractions,
you add the top parts together and the bottom parts together, like adding any two-part things.

If you're adding 1/2 + 1/3, you add 1+1 on top and 2+3 on bottom to get 2/5.
This makes sense: you're combining the numerators and combining the denominators.""",
        "S4_procedure": """Your procedure for adding fractions:
STEP 1: Add the numerators together
STEP 2: Add the denominators together
STEP 3: Write as a fraction: sum of tops / sum of bottoms
STEP 4: Simplify if possible

Example: 1/2 + 2/3
- Add numerators: 1 + 2 = 3
- Add denominators: 2 + 3 = 5
- Answer: 3/5""",
    },
}


def build_prompt(spec_level: str, item: dict) -> str:
    """Build a prompt for the given specification level and item."""
    misc_id = item["misconception_id"]
    misc_prompts = MISCONCEPTION_PROMPTS[misc_id]

    question = item["question_text"]
    options = f"A) {item['answer_a']}\nB) {item['answer_b']}\nC) {item['answer_c']}\nD) {item['answer_d']}"

    if spec_level == "S1":
        return f"""You are a student who sometimes struggles with math. You're taking a test and trying your best. Show your thinking, then give your answer.

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
        return f"""You are a student who uses a specific procedure for this type of problem.

{misc_prompts['S4_procedure']}

Apply your procedure step by step to solve this problem.

Question: {question}
Options:
{options}

Show your work, then state your answer as **Answer: X**"""

    else:
        raise ValueError(f"Unknown spec level: {spec_level}")


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_answer(response: str) -> Optional[str]:
    """Extract answer letter from response."""
    # Try **Answer: X)** or **Answer: X** pattern
    match = re.search(r'Answer:\s*\*?\*?([A-D])\)?\*?\*?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try **X) answer** pattern
    match = re.search(r'\*\*([A-D])\)', response)
    if match:
        return match.group(1).upper()

    # Try "The answer is X" pattern
    match = re.search(r'the answer is\s*\*?\*?([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try final letter pattern
    match = re.search(r'\b([A-D])\)?\.?\s*$', response.strip())
    if match:
        return match.group(1).upper()

    # Try any standalone letter in parentheses
    matches = re.findall(r'\(([A-D])\)|^([A-D])\)', response, re.MULTILINE)
    if matches:
        last = matches[-1]
        return (last[0] or last[1]).upper()

    return None


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str) -> str:
    """Call OpenAI API."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def call_anthropic(model_id: str, prompt: str) -> str:
    """Call Anthropic API."""
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_model(model: str, prompt: str) -> str:
    """Call the appropriate API for the model."""
    config = MODELS[model]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "openai":
        return call_openai(model_id, prompt)
    elif provider == "anthropic":
        return call_anthropic(model_id, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Response:
    response_id: str
    timestamp: str
    model: str
    model_tier: str
    misconception_id: int
    misconception_name: str
    misconception_type: str
    spec_level: str
    question_id: int
    construct_name: str
    question_text: str
    options: str
    correct_answer: str
    target_distractor: str
    human_correct_rate: float
    human_target_rate: float
    prompt: str
    raw_response: str
    parsed_answer: Optional[str]
    is_correct: bool
    hit_target: bool
    rep: int


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    output_dir: str,
    models: list,
    misconceptions: list,
    reps: int = 3,
    sample_per_misc: int = 0,  # 0 = all items
):
    """Run the full experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load items
    with open("data/experiment_items.json") as f:
        all_items = json.load(f)

    # Filter by misconception if specified
    if misconceptions:
        all_items = [i for i in all_items if i["misconception_id"] in misconceptions]

    # Sample if specified
    if sample_per_misc > 0:
        sampled = []
        for misc_id in set(i["misconception_id"] for i in all_items):
            misc_items = [i for i in all_items if i["misconception_id"] == misc_id]
            sampled.extend(misc_items[:sample_per_misc])
        all_items = sampled

    spec_levels = ["S1", "S2", "S3", "S4"]

    # Calculate total calls
    total_calls = len(models) * len(spec_levels) * len(all_items) * reps

    # Load existing responses for checkpointing
    responses_file = output_path / "responses.jsonl"
    completed = set()
    if responses_file.exists():
        with open(responses_file) as f:
            for line in f:
                r = json.loads(line)
                key = (r["model"], r["spec_level"], r["question_id"], r["rep"])
                completed.add(key)

    remaining = total_calls - len(completed)

    print("=" * 60)
    print("S1-S4 EXPERIMENT WITH EEDI ITEMS")
    print("=" * 60)
    print(f"Output: {output_path.absolute()}")
    print(f"Models: {models}")
    print(f"Specs: S1-S4")
    print(f"Items: {len(all_items)}")
    print(f"Reps: {reps}")
    print(f"\nTotal API calls planned: {total_calls}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {remaining}")
    print()

    call_count = len(completed)

    for model in models:
        if model not in MODELS:
            print(f"Skipping unknown model: {model}")
            continue

        model_config = MODELS[model]
        rpm = model_config["rpm"]
        delay = 60.0 / rpm

        print("=" * 60)
        print(f"MODEL: {model} (tier: {model_config['tier']}, rpm: {rpm})")
        print("=" * 60)

        for item in all_items:
            misc_id = item["misconception_id"]

            if misc_id not in MISCONCEPTION_PROMPTS:
                print(f"  Skipping item with unknown misconception: {misc_id}")
                continue

            for spec in spec_levels:
                for rep in range(1, reps + 1):
                    key = (model, spec, item["question_id"], rep)
                    if key in completed:
                        continue

                    call_count += 1

                    # Build prompt
                    prompt = build_prompt(spec, item)
                    options = f"A) {item['answer_a']}\nB) {item['answer_b']}\nC) {item['answer_c']}\nD) {item['answer_d']}"

                    # Call API
                    try:
                        raw_response = call_model(model, prompt)
                        parsed = parse_answer(raw_response)
                        is_correct = parsed == item["correct_answer"]
                        hit_target = parsed == item["target_distractor"]

                        # Status indicator
                        if is_correct:
                            status = "✓"
                        elif hit_target:
                            status = "⊛"  # Target distractor
                        elif parsed is None:
                            status = "✗"  # Parse failure
                        else:
                            status = "○"  # Other wrong answer

                        # Create response record
                        response = Response(
                            response_id=str(uuid.uuid4()),
                            timestamp=datetime.now().isoformat(),
                            model=model,
                            model_tier=model_config["tier"],
                            misconception_id=misc_id,
                            misconception_name=item["misconception_name"],
                            misconception_type=item["misconception_type"],
                            spec_level=spec,
                            question_id=item["question_id"],
                            construct_name=item["construct_name"],
                            question_text=item["question_text"],
                            options=options,
                            correct_answer=item["correct_answer"],
                            target_distractor=item["target_distractor"],
                            human_correct_rate=item["human_correct_rate"],
                            human_target_rate=item["human_target_rate"],
                            prompt=prompt,
                            raw_response=raw_response,
                            parsed_answer=parsed,
                            is_correct=is_correct,
                            hit_target=hit_target,
                            rep=rep,
                        )

                        # Save
                        with open(responses_file, "a") as f:
                            f.write(json.dumps(asdict(response)) + "\n")

                        print(f"  {status} Q{item['question_id']} {spec} r{rep}: {parsed} [{call_count}/{total_calls}]")

                    except Exception as e:
                        print(f"  ERROR Q{item['question_id']} {spec} r{rep}: {e}")

                    time.sleep(delay)

    print()
    print("=" * 60)
    print(f"COMPLETE: {call_count} responses collected")
    print(f"Output: {responses_file}")
    print("=" * 60)

    # Summary statistics
    analyze_results(responses_file)


def analyze_results(responses_file: Path):
    """Print summary statistics."""
    responses = []
    with open(responses_file) as f:
        for line in f:
            responses.append(json.loads(line))

    if not responses:
        return

    print(f"\nTotal responses: {len(responses)}")

    # By spec level
    print("\nBy Specification Level:")
    print("-" * 60)
    for spec in ["S1", "S2", "S3", "S4"]:
        spec_resp = [r for r in responses if r["spec_level"] == spec]
        if not spec_resp:
            continue

        errors = [r for r in spec_resp if not r["is_correct"]]
        target_hits = [r for r in errors if r["hit_target"]]

        error_rate = len(errors) / len(spec_resp) * 100
        target_rate = len(target_hits) / len(errors) * 100 if errors else 0

        # Average human target rate for comparison
        avg_human_target = sum(r["human_target_rate"] for r in spec_resp) / len(spec_resp) * 100

        print(f"  {spec}: {len(errors)}/{len(spec_resp)} errors ({error_rate:.1f}%)")
        print(f"      LLM target rate: {target_rate:.1f}%")
        print(f"      Human target rate: {avg_human_target:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run S1-S4 experiment with Eedi items")
    parser.add_argument("--output", "-o", default="pilot/eedi_experiment", help="Output directory")
    parser.add_argument("--models", "-m", nargs="+", default=list(MODELS.keys()), help="Models to test")
    parser.add_argument("--misconceptions", nargs="+", type=int, default=[1507, 1214, 1597, 217], help="Misconception IDs")
    parser.add_argument("--reps", "-r", type=int, default=3, help="Repetitions per condition")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Items per misconception (0=all)")

    args = parser.parse_args()

    run_experiment(
        output_dir=args.output,
        models=args.models,
        misconceptions=args.misconceptions,
        reps=args.reps,
        sample_per_misc=args.sample,
    )


if __name__ == "__main__":
    main()
