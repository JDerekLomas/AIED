#!/usr/bin/env python3
"""
Test S1-S4 Specification Levels for Misconception Simulation

Compares four theoretically-grounded prompt specification levels:
- S1: Persona only
- S2: Knowledge state
- S3: Mental model
- S4: Production rules (KLI-informed)

Tests on 4 misconceptions (2 procedural, 2 conceptual) with:
- Training items (used in prompt examples, if any)
- Near transfer items (same misconception, different surface)
- Discriminant items (different misconception - should NOT trigger)
"""

import json
import os
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import re
from dotenv import load_dotenv

load_dotenv()

# Try imports for different API clients
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# =============================================================================
# SPECIFICATION LEVEL PROMPTS
# =============================================================================

PROMPTS = {
    "PROC_ORDER_OPS": {
        "S1": """You are a 6th grade student who sometimes struggles with math. You're taking
a test and trying your best. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following math knowledge:

KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't
sure exactly how it works

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student who thinks about math expressions like reading a sentence.
Just as you read English left-to-right, word by word, you believe math should
be solved left-to-right, operation by operation.

When teachers mention "order of operations," you think this means "the order
you see them" - first operation you encounter, then the next, and so on.

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

    "CONC_FRAC_DENOM": {
        "S1": """You are a 5th grade student who finds fractions confusing. You're trying
your best on this test. Show your thinking, then give your answer.

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

        "S3": """You are a student who thinks about fractions using a "pizza slices" mental
model, but with a specific belief:

You think: "The denominator tells you how many slices. More slices = more
pizza. So 1/6 means 6 slices, which is more than 1/4 which is only 4 slices."

You've seen pizza cut into 8 slices, and that seemed like more pizza than
when it was cut into 4 slices. This confirms your understanding.

When comparing fractions, you look at denominators - bigger denominator
means more, so that fraction is larger.

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

    "PROC_SUBTRACT_REVERSE": {
        "S1": """You are a 4th grade student who learned subtraction last year. You follow
the steps you were taught. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following subtraction knowledge:

KNOW WELL: Subtracting single digits, place value, lining up numbers
PARTIALLY LEARNED: Column subtraction - you learned a version but missed
some days when "borrowing" was taught
NEVER FULLY UNDERSTOOD: When and how to "regroup" or "borrow"

You do your best with what you know.

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student who believes subtraction is about "finding the difference"
between two digits in each column.

You think: "The difference between 3 and 8 is 5. It doesn't matter which is
on top - the difference is the same either way."

You've noticed that 8-3 = 5 and |3-8| = 5, so this confirms your understanding.
Subtraction finds how far apart two numbers are.

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

    "CONC_MULT_INCREASE": {
        "S1": """You are a 6th grade student who is good at basic multiplication but still
learning about decimals. Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S2": """You are a student with the following knowledge:

KNOW WELL: Multiplying whole numbers, times tables
STILL LEARNING: What happens when you multiply by decimals or fractions
CORE BELIEF FROM EXPERIENCE: Every multiplication you've practiced (2×3,
5×4, etc.) made numbers bigger

Show your thinking, then give your answer.

Question: {question}
Options: {options}

Show your work, then state your answer as **Answer: X**""",

        "S3": """You are a student with a strong intuition about multiplication:

"Multiplication makes things bigger. That's what 'times' means - you're
making multiple copies, so you end up with more than you started with."

You learned this from years of practice: 3 × 4 = 12 (bigger than both),
7 × 8 = 56 (much bigger). This always works.

When you see decimals like 0.5, you think of them as "small numbers."
Multiplying by a small number should still make things bigger, just not
by as much.

The idea that multiplication could make something smaller contradicts
everything you've experienced.

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

TEST_ITEMS = {
    "PROC_ORDER_OPS": {
        "near_transfer": [
            {
                "id": "OPS_01",
                "question": "Calculate: 5 + 3 × 2",
                "options": "A) 16  B) 11  C) 13  D) 10",
                "correct": "B",
                "target_distractor": "A",  # Left-to-right gives 16
                "target_answer_value": 16
            },
            {
                "id": "OPS_02",
                "question": "Calculate: 10 - 4 + 2",
                "options": "A) 4  B) 8  C) 6  D) 12",
                "correct": "B",
                "target_distractor": "B",  # L-to-R actually correct here
                "target_answer_value": 8
            },
            {
                "id": "OPS_03",
                "question": "Calculate: 2 + 6 ÷ 2",
                "options": "A) 4  B) 5  C) 8  D) 6",
                "correct": "B",
                "target_distractor": "A",  # L-to-R: (2+6)÷2 = 4
                "target_answer_value": 4
            },
            {
                "id": "OPS_04",
                "question": "Calculate: 8 ÷ 2 × 2",
                "options": "A) 2  B) 8  C) 4  D) 16",
                "correct": "B",
                "target_distractor": "B",  # L-to-R correct here
                "target_answer_value": 8
            },
            {
                "id": "OPS_05",
                "question": "Calculate: 3 × 4 + 2 × 5",
                "options": "A) 22  B) 70  C) 50  D) 26",
                "correct": "A",
                "target_distractor": "B",  # L-to-R: 3×4=12, 12+2=14, 14×5=70
                "target_answer_value": 70
            },
        ],
        "discriminant": [
            {
                "id": "OPS_DISC_01",
                "question": "What is 64 - 38?",
                "options": "A) 34  B) 26  C) 36  D) 24",
                "correct": "B",
                "misconception": "PROC_SUBTRACT_REVERSE",
                "note": "Should NOT trigger order of ops misconception"
            }
        ]
    },

    "CONC_FRAC_DENOM": {
        "near_transfer": [
            {
                "id": "FRAC_01",
                "question": "Which is larger: 1/4 or 1/6?",
                "options": "A) 1/4  B) 1/6  C) They are equal  D) Cannot determine",
                "correct": "A",
                "target_distractor": "B",  # Larger denom = larger fraction
                "target_answer_value": "1/6"
            },
            {
                "id": "FRAC_02",
                "question": "Which is larger: 1/3 or 1/5?",
                "options": "A) 1/3  B) 1/5  C) They are equal  D) Cannot determine",
                "correct": "A",
                "target_distractor": "B",
                "target_answer_value": "1/5"
            },
            {
                "id": "FRAC_03",
                "question": "Which is larger: 2/3 or 2/7?",
                "options": "A) 2/3  B) 2/7  C) They are equal  D) Cannot determine",
                "correct": "A",
                "target_distractor": "B",
                "target_answer_value": "2/7"
            },
            {
                "id": "FRAC_04",
                "question": "Order from smallest to largest: 1/2, 1/4, 1/8",
                "options": "A) 1/8, 1/4, 1/2  B) 1/2, 1/4, 1/8  C) 1/4, 1/2, 1/8  D) 1/4, 1/8, 1/2",
                "correct": "A",
                "target_distractor": "B",  # Reverses order
                "target_answer_value": "1/2, 1/4, 1/8"
            },
            {
                "id": "FRAC_05",
                "question": "Which is larger: 3/4 or 3/8?",
                "options": "A) 3/4  B) 3/8  C) They are equal  D) Cannot determine",
                "correct": "A",
                "target_distractor": "B",
                "target_answer_value": "3/8"
            },
        ],
        "discriminant": [
            {
                "id": "FRAC_DISC_01",
                "question": "Calculate: 5 + 3 × 2",
                "options": "A) 16  B) 11  C) 13  D) 10",
                "correct": "B",
                "misconception": "PROC_ORDER_OPS",
                "note": "Should NOT trigger fraction misconception"
            }
        ]
    },

    "PROC_SUBTRACT_REVERSE": {
        "near_transfer": [
            {
                "id": "SUB_01",
                "question": "What is 52 - 38?",
                "options": "A) 26  B) 14  C) 24  D) 16",
                "correct": "B",
                "target_distractor": "A",  # 8-2=6, 5-3=2 → 26
                "target_answer_value": 26
            },
            {
                "id": "SUB_02",
                "question": "What is 64 - 38?",
                "options": "A) 34  B) 26  C) 36  D) 24",
                "correct": "B",
                "target_distractor": "A",  # 8-4=4, 6-3=3 → 34
                "target_answer_value": 34
            },
            {
                "id": "SUB_03",
                "question": "What is 71 - 45?",
                "options": "A) 34  B) 26  C) 36  D) 24",
                "correct": "B",
                "target_distractor": "A",  # 5-1=4, 7-4=3 → 34
                "target_answer_value": 34
            },
            {
                "id": "SUB_04",
                "question": "What is 83 - 47?",
                "options": "A) 44  B) 36  C) 46  D) 34",
                "correct": "B",
                "target_distractor": "A",  # 7-3=4, 8-4=4 → 44
                "target_answer_value": 44
            },
            {
                "id": "SUB_05",
                "question": "What is 92 - 58?",
                "options": "A) 46  B) 34  C) 44  D) 36",
                "correct": "B",
                "target_distractor": "A",  # 8-2=6, 9-5=4 → 46
                "target_answer_value": 46
            },
        ],
        "discriminant": [
            {
                "id": "SUB_DISC_01",
                "question": "Which is larger: 1/4 or 1/6?",
                "options": "A) 1/4  B) 1/6  C) They are equal  D) Cannot determine",
                "correct": "A",
                "misconception": "CONC_FRAC_DENOM",
                "note": "Should NOT trigger subtraction misconception"
            }
        ]
    },

    "CONC_MULT_INCREASE": {
        "near_transfer": [
            {
                "id": "MULT_01",
                "question": "What is 8 × 0.5?",
                "options": "A) 8  B) 4  C) 16  D) 0.5",
                "correct": "B",
                "target_distractor": "A",  # Thinks result must be ≥ 8
                "target_answer_value": 8
            },
            {
                "id": "MULT_02",
                "question": "If you multiply 10 by 0.25, the result is:",
                "options": "A) Larger than 10  B) Smaller than 10  C) Equal to 10  D) Cannot determine",
                "correct": "B",
                "target_distractor": "A",
                "target_answer_value": "Larger"
            },
            {
                "id": "MULT_03",
                "question": "What is 12 × 1/2?",
                "options": "A) 24  B) 6  C) 12  D) 14",
                "correct": "B",
                "target_distractor": "A",  # 12 × 2 thinking
                "target_answer_value": 24
            },
            {
                "id": "MULT_04",
                "question": "Multiplying a number by 0.1 makes it:",
                "options": "A) 10 times larger  B) 10 times smaller  C) Stay the same  D) Slightly larger",
                "correct": "B",
                "target_distractor": "A",
                "target_answer_value": "larger"
            },
            {
                "id": "MULT_05",
                "question": "What is 20 × 0.5?",
                "options": "A) 10  B) 40  C) 20  D) 100",
                "correct": "A",
                "target_distractor": "C",  # Result should be ≥ 20
                "target_answer_value": 20
            },
        ],
        "discriminant": [
            {
                "id": "MULT_DISC_01",
                "question": "What is 71 - 45?",
                "options": "A) 34  B) 26  C) 36  D) 24",
                "correct": "B",
                "misconception": "PROC_SUBTRACT_REVERSE",
                "note": "Should NOT trigger multiplication misconception"
            }
        ]
    }
}


# =============================================================================
# MODEL CLIENTS
# =============================================================================

@dataclass
class Response:
    misconception_id: str
    item_id: str
    spec_level: str
    model: str
    question: str
    correct_answer: str
    target_distractor: str
    raw_response: str
    parsed_answer: Optional[str]
    is_correct: bool
    hit_target: bool
    item_type: str  # "near_transfer" or "discriminant"
    repetition: int = 1  # Which repetition this is


def parse_answer(response_text: str) -> Optional[str]:
    """Extract answer letter from response."""
    # Look for **Answer: X** pattern
    match = re.search(r'\*\*Answer:\s*([A-D])\*\*', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end
    match = re.search(r'\b([A-D])\)?\.?\s*$', response_text)
    if match:
        return match.group(1).upper()

    return None


def call_openai(model: str, prompt: str) -> str:
    """Call OpenAI API."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


def call_anthropic(model: str, prompt: str) -> str:
    """Call Anthropic API."""
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_groq(model: str, prompt: str) -> str:
    """Call Groq API."""
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


# Model registry with capability tiers
MODELS = {
    # Frontier tier (~95% GSM8K)
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "gpt-4o": ("openai", "gpt-4o"),
    # Mid tier (~75% GSM8K)
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    # Weak tier (~50% GSM8K)
    "llama-3.1-8b": ("groq", "llama-3.1-8b-instant"),
    "mixtral-8x7b": ("groq", "mixtral-8x7b-32768"),
}

MODEL_TIERS = {
    "frontier": ["claude-sonnet-4", "gpt-4o"],
    "mid": ["gpt-4o-mini", "claude-3-haiku"],
    "weak": ["llama-3.1-8b", "mixtral-8x7b"],
}


def call_model(model_name: str, prompt: str) -> str:
    """Call the appropriate API for a model."""
    provider, model_id = MODELS[model_name]

    if provider == "openai" and HAS_OPENAI:
        return call_openai(model_id, prompt)
    elif provider == "anthropic" and HAS_ANTHROPIC:
        return call_anthropic(model_id, prompt)
    elif provider == "groq" and HAS_GROQ:
        return call_groq(model_id, prompt)
    else:
        raise ValueError(f"Provider {provider} not available")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_test(
    misconception_id: str,
    spec_level: str,
    model_name: str,
    items: list,
    item_type: str
) -> list[Response]:
    """Run tests for one misconception × spec level × model combination."""

    results = []
    prompt_template = PROMPTS[misconception_id][spec_level]

    for item in items:
        # Format prompt
        prompt = prompt_template.format(
            question=item["question"],
            options=item["options"]
        )

        try:
            raw_response = call_model(model_name, prompt)
            parsed = parse_answer(raw_response)

            is_correct = parsed == item["correct"]
            hit_target = parsed == item.get("target_distractor", "")

            results.append(Response(
                misconception_id=misconception_id,
                item_id=item["id"],
                spec_level=spec_level,
                model=model_name,
                question=item["question"],
                correct_answer=item["correct"],
                target_distractor=item.get("target_distractor", "N/A"),
                raw_response=raw_response,
                parsed_answer=parsed,
                is_correct=is_correct,
                hit_target=hit_target,
                item_type=item_type
            ))

            # Progress indicator
            status = "✓" if is_correct else ("⊛" if hit_target else "✗")
            print(f"  {status} {item['id']}: {parsed} (correct: {item['correct']})")

        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}")

    return results


def run_full_experiment(
    models: list[str] = None,
    misconceptions: list[str] = None,
    spec_levels: list[str] = None,
    repetitions: int = 1,
    model_tier: str = None
):
    """Run the full experiment.

    Args:
        models: Specific models to test (overrides model_tier)
        misconceptions: Which misconceptions to test (default: all 4)
        spec_levels: Which specification levels (default: S1-S4)
        repetitions: Number of times to repeat each condition
        model_tier: "frontier", "mid", "weak", or "all" (ignored if models specified)
    """

    # Resolve models
    if models is None:
        if model_tier == "all":
            models = list(MODELS.keys())
        elif model_tier in MODEL_TIERS:
            models = MODEL_TIERS[model_tier]
        else:
            models = ["gpt-4o-mini", "llama-3.1-8b"]  # Default to mid/weak

    if misconceptions is None:
        misconceptions = list(PROMPTS.keys())
    if spec_levels is None:
        spec_levels = ["S1", "S2", "S3", "S4"]

    all_results = []
    total_calls = len(misconceptions) * len(models) * len(spec_levels) * 6 * repetitions  # 5 near + 1 disc
    call_count = 0

    for rep in range(repetitions):
        rep_label = f" (rep {rep+1}/{repetitions})" if repetitions > 1 else ""

        for misconception in misconceptions:
            print(f"\n{'='*60}")
            print(f"MISCONCEPTION: {misconception}{rep_label}")
            print('='*60)

            items = TEST_ITEMS[misconception]

            for model in models:
                print(f"\n  Model: {model}")

                for spec in spec_levels:
                    print(f"\n    Spec Level: {spec}")
                    print(f"    {'-'*40}")

                    # Run near transfer items
                    print("    Near Transfer:")
                    results = run_test(
                        misconception, spec, model,
                        items["near_transfer"], "near_transfer"
                    )
                    # Add repetition number to results
                    for r in results:
                        r.repetition = rep + 1
                    all_results.extend(results)
                    call_count += len(items["near_transfer"])

                    # Run discriminant items
                    print("    Discriminant:")
                    results = run_test(
                        misconception, spec, model,
                        items["discriminant"], "discriminant"
                    )
                    for r in results:
                        r.repetition = rep + 1
                    all_results.extend(results)
                    call_count += len(items["discriminant"])

                    print(f"    Progress: {call_count}/{total_calls} ({100*call_count/total_calls:.1f}%)")

    return all_results


def analyze_results(results: list[Response]):
    """Analyze and summarize results."""

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Group by spec level
    from collections import defaultdict

    by_spec = defaultdict(lambda: {"correct": 0, "target": 0, "total": 0})
    by_model_spec = defaultdict(lambda: {"correct": 0, "target": 0, "total": 0})
    by_misconception_spec = defaultdict(lambda: {"correct": 0, "target": 0, "total": 0})

    for r in results:
        if r.item_type == "near_transfer":  # Only analyze target items
            key = r.spec_level
            by_spec[key]["total"] += 1
            if r.is_correct:
                by_spec[key]["correct"] += 1
            if r.hit_target:
                by_spec[key]["target"] += 1

            key2 = (r.model, r.spec_level)
            by_model_spec[key2]["total"] += 1
            if r.is_correct:
                by_model_spec[key2]["correct"] += 1
            if r.hit_target:
                by_model_spec[key2]["target"] += 1

            key3 = (r.misconception_id, r.spec_level)
            by_misconception_spec[key3]["total"] += 1
            if r.is_correct:
                by_misconception_spec[key3]["correct"] += 1
            if r.hit_target:
                by_misconception_spec[key3]["target"] += 1

    # Print by spec level
    print("\n1. BY SPECIFICATION LEVEL (Near Transfer Items)")
    print("-" * 50)
    print(f"{'Spec':<8} {'Total':<8} {'Errors':<8} {'Target Rate':<12} {'Error Rate'}")
    for spec in ["S1", "S2", "S3", "S4"]:
        data = by_spec[spec]
        if data["total"] > 0:
            errors = data["total"] - data["correct"]
            error_rate = errors / data["total"] * 100
            target_rate = data["target"] / errors * 100 if errors > 0 else 0
            print(f"{spec:<8} {data['total']:<8} {errors:<8} {target_rate:>6.1f}%      {error_rate:>6.1f}%")

    # Print by model × spec
    print("\n2. BY MODEL × SPECIFICATION LEVEL")
    print("-" * 60)
    models = sorted(set(r.model for r in results))
    for model in models:
        print(f"\n  {model}:")
        for spec in ["S1", "S2", "S3", "S4"]:
            data = by_model_spec[(model, spec)]
            if data["total"] > 0:
                errors = data["total"] - data["correct"]
                error_rate = errors / data["total"] * 100
                target_rate = data["target"] / errors * 100 if errors > 0 else 0
                print(f"    {spec}: {errors}/{data['total']} errors, {target_rate:.1f}% target rate")

    # Print by misconception × spec
    print("\n3. BY MISCONCEPTION × SPECIFICATION LEVEL")
    print("-" * 60)
    for misc in ["PROC_ORDER_OPS", "CONC_FRAC_DENOM", "PROC_SUBTRACT_REVERSE", "CONC_MULT_INCREASE"]:
        print(f"\n  {misc}:")
        for spec in ["S1", "S2", "S3", "S4"]:
            data = by_misconception_spec[(misc, spec)]
            if data["total"] > 0:
                errors = data["total"] - data["correct"]
                target_rate = data["target"] / errors * 100 if errors > 0 else 0
                print(f"    {spec}: {target_rate:.1f}% target rate ({data['target']}/{errors} errors)")

    # Discriminant analysis
    print("\n4. DISCRIMINANT VALIDITY")
    print("-" * 50)
    disc_results = [r for r in results if r.item_type == "discriminant"]
    correct_disc = sum(1 for r in disc_results if r.is_correct)
    print(f"Accuracy on discriminant items: {correct_disc}/{len(disc_results)} ({correct_disc/len(disc_results)*100:.1f}%)")
    print("(Higher = better discriminant validity - not triggering wrong misconception)")

    return by_spec, by_model_spec, by_misconception_spec


def save_results(results: list[Response], output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(exist_ok=True)

    # Save as JSONL
    jsonl_path = output_dir / "spec_level_results.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')
    print(f"\nResults saved to: {jsonl_path}")

    # Save summary
    summary_path = output_dir / "spec_level_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SPECIFICATION LEVEL EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")

        # Add summary stats
        by_spec = defaultdict(lambda: {"correct": 0, "target": 0, "total": 0})
        for r in results:
            if r.item_type == "near_transfer":
                by_spec[r.spec_level]["total"] += 1
                if r.is_correct:
                    by_spec[r.spec_level]["correct"] += 1
                if r.hit_target:
                    by_spec[r.spec_level]["target"] += 1

        f.write("By Specification Level:\n")
        for spec in ["S1", "S2", "S3", "S4"]:
            data = by_spec[spec]
            if data["total"] > 0:
                errors = data["total"] - data["correct"]
                target_rate = data["target"] / errors * 100 if errors > 0 else 0
                f.write(f"  {spec}: {target_rate:.1f}% target rate ({data['target']}/{errors} errors)\n")

    print(f"Summary saved to: {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test S1-S4 specification levels")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to test (overrides --tier)")
    parser.add_argument("--tier", type=str, default=None,
                       choices=["frontier", "mid", "weak", "all"],
                       help="Model tier to test: frontier, mid, weak, or all")
    parser.add_argument("--misconceptions", nargs="+", default=None,
                       help="Misconceptions to test (default: all 4)")
    parser.add_argument("--specs", nargs="+", default=["S1", "S2", "S3", "S4"],
                       help="Specification levels to test")
    parser.add_argument("--reps", type=int, default=1,
                       help="Number of repetitions per condition")
    parser.add_argument("--output", type=str,
                       default="/Users/dereklomas/AIED/study2-materials/results",
                       help="Output directory")

    args = parser.parse_args()

    # Resolve models for display
    if args.models:
        display_models = args.models
    elif args.tier:
        display_models = MODEL_TIERS.get(args.tier, []) if args.tier != "all" else list(MODELS.keys())
    else:
        display_models = ["gpt-4o-mini", "llama-3.1-8b"]

    print("="*70)
    print("SPECIFICATION LEVEL EXPERIMENT")
    print("Testing S1 (Persona) through S4 (Production Rules)")
    print("="*70)
    print(f"\nModels: {display_models}")
    print(f"Misconceptions: {args.misconceptions or 'all 4'}")
    print(f"Spec Levels: {args.specs}")
    print(f"Repetitions: {args.reps}")

    # Calculate expected API calls
    n_models = len(display_models)
    n_misc = len(args.misconceptions) if args.misconceptions else 4
    n_specs = len(args.specs)
    n_items = 6  # 5 near + 1 disc per misconception
    total = n_models * n_misc * n_specs * n_items * args.reps
    print(f"Expected API calls: {total}")

    # Run experiment
    results = run_full_experiment(
        models=args.models,
        misconceptions=args.misconceptions,
        spec_levels=args.specs,
        repetitions=args.reps,
        model_tier=args.tier
    )

    # Analyze
    analyze_results(results)

    # Save
    save_results(results, Path(args.output))

    print("\nDone!")


if __name__ == "__main__":
    main()
