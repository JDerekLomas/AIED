#!/usr/bin/env python3
"""
Run the S1-S4 Specification Level Experiment

This script runs the full factorial experiment:
- 4 specification levels (S1, S2, S3, S4)
- 6 models across 3 capability tiers
- 4 misconceptions (2 procedural, 2 conceptual)
- 7 items per misconception (5 near-transfer, 2 discriminant)
- 3 repetitions

Total: 2,016 API calls
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

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: Groq not installed")

try:
    import together
    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False
    print("Warning: Together not installed")


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
    # Weak tier
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "tier": "weak",
        "rpm": 30,
    },
    "mistral-7b": {
        "provider": "together",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "tier": "weak",
        "rpm": 60,
    },
}

MISCONCEPTIONS = {
    "PROC_ORDER_OPS": {
        "type": "procedural",
        "description": "Left-to-right instead of PEMDAS",
    },
    "PROC_SUBTRACT_REVERSE": {
        "type": "procedural",
        "description": "Smaller-from-larger in each column",
    },
    "CONC_FRAC_DENOM": {
        "type": "conceptual",
        "description": "Larger denominator = larger fraction",
    },
    "CONC_MULT_INCREASE": {
        "type": "conceptual",
        "description": "Multiplication always increases",
    },
}


# =============================================================================
# PROMPTS (S1-S4 for each misconception)
# =============================================================================

PROMPTS = {
    "PROC_ORDER_OPS": {
        "S1": """You are a 6th grade student who sometimes struggles with math. You're taking a test and trying your best. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following math knowledge:

KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't sure exactly how it works

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student who thinks about math expressions like reading a sentence. Just as you read English left-to-right, word by word, you believe math should be solved left-to-right, operation by operation.

When teachers mention "order of operations," you think this means "the order you see them" - first operation you encounter, then the next, and so on.

This makes sense to you. Why would you skip around randomly?

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S4": """You solve math expressions using this exact procedure:

STEP 1: Find the leftmost operation (like + or ×)
STEP 2: Do that operation with the numbers on either side
STEP 3: Replace those numbers and operator with your result
STEP 4: Repeat from Step 1 until only one number remains

Example of your process: 5 + 3 × 2
- Leftmost operation is +
- Do 5 + 3 = 8
- Now have: 8 × 2
- Do 8 × 2 = 16
- Answer: 16

Apply your procedure step by step, showing your work.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**"""
    },

    "PROC_SUBTRACT_REVERSE": {
        "S1": """You are a 4th grade student who learned subtraction last year. You follow the steps you were taught. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following subtraction knowledge:

KNOW WELL: Subtracting single digits, place value, lining up numbers
PARTIALLY LEARNED: Column subtraction - you learned a version but missed some days when "borrowing" was taught
NEVER FULLY UNDERSTOOD: When and how to "regroup" or "borrow"

You do your best with what you know.

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student who believes subtraction is about "finding the difference" between two digits in each column.

You think: "The difference between 3 and 8 is 5. It doesn't matter which is on top - the difference is the same either way."

You've noticed that 8-3 = 5 and |3-8| = 5, so this confirms your understanding. Subtraction finds how far apart two numbers are.

You apply this column by column, finding the difference in each place.

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S4": """Your procedure for multi-digit subtraction:

STEP 1: Write the numbers vertically, aligned by place value
STEP 2: Start with the rightmost column
STEP 3: In each column:
   - Look at the two digits
   - Subtract the smaller from the larger
   - Write that result below the line
STEP 4: Move left to the next column and repeat Step 3
STEP 5: The bottom number is your answer

Example of your process: 52 - 38
  52
- 38
----
- Ones column: 8 and 2. Larger minus smaller: 8-2 = 6. Write 6.
- Tens column: 5 and 3. Larger minus smaller: 5-3 = 2. Write 2.
- Answer: 26

Apply your procedure column by column.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**"""
    },

    "CONC_FRAC_DENOM": {
        "S1": """You are a 5th grade student who finds fractions confusing. You're trying your best on this test. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following fraction knowledge:

KNOW WELL: What numerators and denominators are, basic fraction notation
STILL LEARNING: How to compare fractions with different denominators
NEVER LEARNED: Why smaller denominators can mean larger fractions

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student who thinks about fractions using a "pizza slices" mental model, but with a specific belief:

You think: "The denominator tells you how many slices. More slices = more pizza. So 1/6 means 6 slices, which is more than 1/4 which is only 4 slices."

You've seen pizza cut into 8 slices, and that seemed like more pizza than when it was cut into 4 slices. This confirms your understanding.

When comparing fractions, you look at denominators - bigger denominator means more, so that fraction is larger.

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S4": """You compare fractions using this procedure:

STEP 1: Look at the denominators (bottom numbers) of each fraction
STEP 2: The larger denominator means more pieces
STEP 3: More pieces = more stuff = bigger fraction
STEP 4: Choose the fraction with the larger denominator as the bigger one

Special case - same denominators: Compare numerators directly

Example of your process: Which is larger, 1/3 or 1/8?
- Denominators: 3 and 8
- 8 > 3, so 8 means more pieces
- Therefore 1/8 > 1/3
- Answer: 1/8 is larger

Apply your procedure, showing your work.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**"""
    },

    "CONC_MULT_INCREASE": {
        "S1": """You are a 6th grade student who is good at basic multiplication but still learning about decimals. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following knowledge:

KNOW WELL: Multiplying whole numbers, times tables
STILL LEARNING: What happens when you multiply by decimals or fractions
CORE BELIEF FROM EXPERIENCE: Every multiplication you've practiced (2×3, 5×4, etc.) made numbers bigger

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student with a strong intuition about multiplication:

"Multiplication makes things bigger. That's what 'times' means - you're making multiple copies, so you end up with more than you started with."

You learned this from years of practice: 3 × 4 = 12 (bigger than both), 7 × 8 = 56 (much bigger). This always works.

When you see decimals like 0.5, you think of them as "small numbers." Multiplying by a small number should still make things bigger, just not by as much.

The idea that multiplication could make something smaller contradicts everything you've experienced.

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S4": """Your understanding of multiplication:

RULE 1: Multiplication means "making copies" or "groups of"
RULE 2: More copies = more stuff = bigger result
RULE 3: The result of A × B is always larger than both A and B
RULE 4: If a multiplication result seems smaller, check your arithmetic

When comparing before and after multiplication:
- If asked "is the result bigger or smaller?" → always answer BIGGER
- If the calculated answer seems smaller, you assume you made an error

Example of your thinking: 8 × 0.5
- You calculate: 8 × 0.5 = 4
- But wait, 4 is smaller than 8
- Multiplication makes things bigger, so result should be bigger than 8
- The answer must be 8 (same) or larger

Apply your understanding.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**"""
    }
}


# =============================================================================
# TEST ITEMS
# =============================================================================

ITEMS = {
    "PROC_ORDER_OPS": {
        "near_transfer": [
            {"id": "OPS_01", "question": "Calculate: 5 + 3 × 2", "options": "A) 16  B) 11  C) 13  D) 10", "correct": "B", "target": "A"},
            {"id": "OPS_02", "question": "Calculate: 2 + 6 ÷ 2", "options": "A) 4  B) 5  C) 8  D) 6", "correct": "B", "target": "A"},
            {"id": "OPS_03", "question": "Calculate: 3 × 4 + 2 × 5", "options": "A) 22  B) 70  C) 50  D) 26", "correct": "A", "target": "B"},
            {"id": "OPS_04", "question": "Calculate: 10 - 2 × 3", "options": "A) 24  B) 4  C) 16  D) 6", "correct": "B", "target": "A"},
            {"id": "OPS_05", "question": "Calculate: 8 + 4 ÷ 2 - 1", "options": "A) 5  B) 9  C) 6  D) 10", "correct": "B", "target": "A"},
        ],
        "discriminant": [
            {"id": "OPS_D1", "question": "What is 52 - 38?", "options": "A) 26  B) 14  C) 24  D) 16", "correct": "B", "target": "A", "tests": "PROC_SUBTRACT_REVERSE"},
            {"id": "OPS_D2", "question": "Which is larger: 1/4 or 1/6?", "options": "A) 1/4  B) 1/6  C) Equal  D) Cannot tell", "correct": "A", "target": "B", "tests": "CONC_FRAC_DENOM"},
        ]
    },
    "PROC_SUBTRACT_REVERSE": {
        "near_transfer": [
            {"id": "SUB_01", "question": "What is 52 - 38?", "options": "A) 26  B) 14  C) 24  D) 16", "correct": "B", "target": "A"},
            {"id": "SUB_02", "question": "What is 64 - 38?", "options": "A) 34  B) 26  C) 36  D) 24", "correct": "B", "target": "A"},
            {"id": "SUB_03", "question": "What is 71 - 45?", "options": "A) 34  B) 26  C) 36  D) 24", "correct": "B", "target": "A"},
            {"id": "SUB_04", "question": "What is 83 - 47?", "options": "A) 44  B) 36  C) 46  D) 34", "correct": "B", "target": "A"},
            {"id": "SUB_05", "question": "What is 92 - 58?", "options": "A) 46  B) 34  C) 44  D) 36", "correct": "B", "target": "A"},
        ],
        "discriminant": [
            {"id": "SUB_D1", "question": "Calculate: 5 + 3 × 2", "options": "A) 16  B) 11  C) 13  D) 10", "correct": "B", "target": "A", "tests": "PROC_ORDER_OPS"},
            {"id": "SUB_D2", "question": "What is 8 × 0.5?", "options": "A) 8  B) 4  C) 16  D) 0.5", "correct": "B", "target": "A", "tests": "CONC_MULT_INCREASE"},
        ]
    },
    "CONC_FRAC_DENOM": {
        "near_transfer": [
            {"id": "FRAC_01", "question": "Which is larger: 1/4 or 1/6?", "options": "A) 1/4  B) 1/6  C) They are equal  D) Cannot determine", "correct": "A", "target": "B"},
            {"id": "FRAC_02", "question": "Which is larger: 1/3 or 1/5?", "options": "A) 1/3  B) 1/5  C) They are equal  D) Cannot determine", "correct": "A", "target": "B"},
            {"id": "FRAC_03", "question": "Which is larger: 2/3 or 2/7?", "options": "A) 2/3  B) 2/7  C) They are equal  D) Cannot determine", "correct": "A", "target": "B"},
            {"id": "FRAC_04", "question": "Order from smallest to largest: 1/2, 1/4, 1/8", "options": "A) 1/8, 1/4, 1/2  B) 1/2, 1/4, 1/8  C) 1/4, 1/2, 1/8  D) 1/4, 1/8, 1/2", "correct": "A", "target": "B"},
            {"id": "FRAC_05", "question": "Which is larger: 3/4 or 3/8?", "options": "A) 3/4  B) 3/8  C) They are equal  D) Cannot determine", "correct": "A", "target": "B"},
        ],
        "discriminant": [
            {"id": "FRAC_D1", "question": "Calculate: 10 - 2 × 3", "options": "A) 24  B) 4  C) 16  D) 6", "correct": "B", "target": "A", "tests": "PROC_ORDER_OPS"},
            {"id": "FRAC_D2", "question": "What is 83 - 47?", "options": "A) 44  B) 36  C) 46  D) 34", "correct": "B", "target": "A", "tests": "PROC_SUBTRACT_REVERSE"},
        ]
    },
    "CONC_MULT_INCREASE": {
        "near_transfer": [
            {"id": "MULT_01", "question": "What is 8 × 0.5?", "options": "A) 8  B) 4  C) 16  D) 0.5", "correct": "B", "target": "A"},
            {"id": "MULT_02", "question": "If you multiply 10 by 0.25, the result is:", "options": "A) Larger than 10  B) Smaller than 10  C) Equal to 10  D) Cannot determine", "correct": "B", "target": "A"},
            {"id": "MULT_03", "question": "What is 12 × 1/2?", "options": "A) 24  B) 6  C) 12  D) 14", "correct": "B", "target": "A"},
            {"id": "MULT_04", "question": "Multiplying a number by 0.1 makes it:", "options": "A) 10 times larger  B) 10 times smaller  C) Stay the same  D) Slightly larger", "correct": "B", "target": "A"},
            {"id": "MULT_05", "question": "What is 20 × 0.5?", "options": "A) 10  B) 40  C) 20  D) 100", "correct": "A", "target": "C"},
        ],
        "discriminant": [
            {"id": "MULT_D1", "question": "Which is larger: 1/3 or 1/5?", "options": "A) 1/3  B) 1/5  C) They are equal  D) Cannot determine", "correct": "A", "target": "B", "tests": "CONC_FRAC_DENOM"},
            {"id": "MULT_D2", "question": "What is 71 - 45?", "options": "A) 34  B) 26  C) 36  D) 24", "correct": "B", "target": "A", "tests": "PROC_SUBTRACT_REVERSE"},
        ]
    }
}


# =============================================================================
# API CALLERS
# =============================================================================

def call_openai(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_groq(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content


def call_together(model_id: str, prompt: str, temperature: float = 0.7) -> str:
    client = together.Together()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content


def call_model(model_name: str, prompt: str) -> str:
    """Route to appropriate API."""
    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "openai":
        return call_openai(model_id, prompt)
    elif provider == "anthropic":
        return call_anthropic(model_id, prompt)
    elif provider == "groq":
        return call_groq(model_id, prompt)
    elif provider == "together":
        return call_together(model_id, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_answer(response: str) -> Optional[str]:
    """Extract answer letter from response."""
    # Try **Answer: X)** or **Answer: X** pattern (with optional bold markers around letter)
    match = re.search(r'Answer:\s*\*?\*?([A-D])\)?\*?\*?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try **X) answer** pattern (letter at start of bold)
    match = re.search(r'\*\*([A-D])\)', response)
    if match:
        return match.group(1).upper()

    # Try "The answer is X" pattern
    match = re.search(r'the answer is\s*\*?\*?([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try final letter pattern (end of response)
    match = re.search(r'\b([A-D])\)?\.?\s*$', response.strip())
    if match:
        return match.group(1).upper()

    # Try any standalone letter in parentheses like (A) or A)
    matches = re.findall(r'\(([A-D])\)|^([A-D])\)', response, re.MULTILINE)
    if matches:
        # Return the last match
        last = matches[-1]
        return (last[0] or last[1]).upper()

    return None


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

@dataclass
class Response:
    response_id: str
    timestamp: str
    model: str
    model_tier: str
    misconception_id: str
    misconception_type: str
    spec_level: str
    item_id: str
    item_type: str
    question: str
    options: str
    correct_answer: str
    target_distractor: str
    prompt: str
    raw_response: str
    parsed_answer: Optional[str]
    is_correct: bool
    hit_target: bool
    rep: int


def run_experiment(
    output_dir: Path,
    models: list = None,
    spec_levels: list = None,
    misconceptions: list = None,
    reps: int = 3,
    resume: bool = True
):
    """Run the full experiment."""

    # Defaults
    if models is None:
        models = list(MODELS.keys())
    if spec_levels is None:
        spec_levels = ["S1", "S2", "S3", "S4"]
    if misconceptions is None:
        misconceptions = list(MISCONCEPTIONS.keys())

    # Setup output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "responses.jsonl"
    checkpoint_file = output_dir / "checkpoint.json"

    # Load checkpoint if resuming
    completed = set()
    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            completed = set(checkpoint.get("completed", []))
        print(f"Resuming from checkpoint: {len(completed)} calls completed")

    # Calculate total
    total_calls = 0
    for model in models:
        for misc in misconceptions:
            for spec in spec_levels:
                items = ITEMS[misc]["near_transfer"] + ITEMS[misc]["discriminant"]
                total_calls += len(items) * reps

    print(f"Total API calls planned: {total_calls}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {total_calls - len(completed)}")
    print()

    # Run experiment
    call_count = 0

    for model in models:
        model_config = MODELS[model]
        rpm = model_config["rpm"]
        delay = 60.0 / rpm  # seconds between calls

        print(f"\n{'='*60}")
        print(f"MODEL: {model} (tier: {model_config['tier']}, rpm: {rpm})")
        print('='*60)

        for misc_id in misconceptions:
            misc_config = MISCONCEPTIONS[misc_id]
            print(f"\n  Misconception: {misc_id}")

            for spec in spec_levels:
                print(f"    Spec Level: {spec}")

                prompt_template = PROMPTS[misc_id][spec]
                items = ITEMS[misc_id]["near_transfer"] + ITEMS[misc_id]["discriminant"]

                for item in items:
                    item_type = "near_transfer" if item["id"].find("_D") == -1 else "discriminant"

                    for rep in range(1, reps + 1):
                        # Check if already done
                        call_id = f"{model}|{misc_id}|{spec}|{item['id']}|{rep}"
                        if call_id in completed:
                            continue

                        # Format prompt
                        prompt = prompt_template.format(
                            question=item["question"],
                            options=item["options"]
                        )

                        # Call API
                        try:
                            raw_response = call_model(model, prompt)
                            parsed = parse_answer(raw_response)
                            is_correct = parsed == item["correct"]
                            hit_target = parsed == item["target"]

                            # Create response record
                            response = Response(
                                response_id=str(uuid.uuid4()),
                                timestamp=datetime.now().isoformat(),
                                model=model,
                                model_tier=model_config["tier"],
                                misconception_id=misc_id,
                                misconception_type=misc_config["type"],
                                spec_level=spec,
                                item_id=item["id"],
                                item_type=item_type,
                                question=item["question"],
                                options=item["options"],
                                correct_answer=item["correct"],
                                target_distractor=item["target"],
                                prompt=prompt,
                                raw_response=raw_response,
                                parsed_answer=parsed,
                                is_correct=is_correct,
                                hit_target=hit_target,
                                rep=rep
                            )

                            # Save response
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(asdict(response)) + '\n')

                            # Update checkpoint
                            completed.add(call_id)
                            with open(checkpoint_file, 'w') as f:
                                json.dump({"completed": list(completed)}, f)

                            # Progress
                            call_count += 1
                            status = "✓" if is_correct else ("⊛" if hit_target else "✗")
                            print(f"      {status} {item['id']} rep{rep}: {parsed} [{call_count}/{total_calls}]")

                            # Rate limiting
                            time.sleep(delay)

                        except Exception as e:
                            print(f"      ERROR {item['id']} rep{rep}: {e}")
                            time.sleep(delay * 2)  # Extra delay on error

    print(f"\n{'='*60}")
    print(f"COMPLETE: {call_count} responses collected")
    print(f"Output: {output_file}")
    print('='*60)

    return output_file


# =============================================================================
# QUICK ANALYSIS
# =============================================================================

def quick_analysis(responses_file: Path):
    """Quick summary of results."""

    responses = []
    with open(responses_file) as f:
        for line in f:
            responses.append(json.loads(line))

    print(f"\nTotal responses: {len(responses)}")

    # By spec level
    from collections import defaultdict
    by_spec = defaultdict(lambda: {"correct": 0, "target": 0, "total": 0})

    for r in responses:
        if r["item_type"] == "near_transfer":
            spec = r["spec_level"]
            by_spec[spec]["total"] += 1
            if r["is_correct"]:
                by_spec[spec]["correct"] += 1
            if r["hit_target"]:
                by_spec[spec]["target"] += 1

    print("\nBy Specification Level (near-transfer items only):")
    print("-" * 50)
    for spec in ["S1", "S2", "S3", "S4"]:
        d = by_spec[spec]
        if d["total"] > 0:
            errors = d["total"] - d["correct"]
            error_rate = errors / d["total"] * 100
            target_rate = d["target"] / errors * 100 if errors > 0 else 0
            print(f"  {spec}: {errors}/{d['total']} errors ({error_rate:.1f}%), {target_rate:.1f}% target rate")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run S1-S4 Specification Experiment")
    parser.add_argument("--output", type=str, default="pilot/spec_experiment",
                       help="Output directory")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to test (default: all)")
    parser.add_argument("--specs", nargs="+", default=None,
                       help="Spec levels to test (default: S1-S4)")
    parser.add_argument("--misconceptions", nargs="+", default=None,
                       help="Misconceptions to test (default: all 4)")
    parser.add_argument("--reps", type=int, default=3,
                       help="Repetitions per condition")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh (don't resume from checkpoint)")
    parser.add_argument("--analyze-only", type=str, default=None,
                       help="Just analyze existing results file")

    args = parser.parse_args()

    if args.analyze_only:
        quick_analysis(Path(args.analyze_only))
        return

    output_dir = Path("/Users/dereklomas/AIED/study2-materials") / args.output

    print("="*60)
    print("S1-S4 SPECIFICATION LEVEL EXPERIMENT")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Models: {args.models or 'all'}")
    print(f"Specs: {args.specs or 'S1-S4'}")
    print(f"Misconceptions: {args.misconceptions or 'all 4'}")
    print(f"Reps: {args.reps}")
    print()

    output_file = run_experiment(
        output_dir=output_dir,
        models=args.models,
        spec_levels=args.specs,
        misconceptions=args.misconceptions,
        reps=args.reps,
        resume=not args.no_resume
    )

    # Quick analysis
    quick_analysis(output_file)


if __name__ == "__main__":
    main()
