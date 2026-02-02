#!/usr/bin/env python3
"""
Study 2: Data Collection with Eedi Items + Student Response Data

Uses curated Eedi items where:
- Each distractor has expert-labeled misconceptions
- We have actual student response distributions (N=15M+ responses)
- We can compare LLM selections to real student behavior

Key metrics:
- Target distractor rate: Does LLM select same distractor as students?
- Distribution alignment: Does LLM distribution match student distribution?
- Reasoning authenticity: Does LLM CoT match the target misconception?
"""

import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

# API clients
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
# MODEL CONFIGURATION
# =============================================================================

MODELS = {
    # Frontier tier
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "gpt-4o": ("openai", "gpt-4o"),
    # Mid tier
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    # Weak tier
    "llama-3.1-8b": ("groq", "llama-3.1-8b-instant"),
    "mixtral-8x7b": ("groq", "mixtral-8x7b-32768"),
}

MODEL_TIERS = {
    "frontier": ["claude-sonnet-4", "gpt-4o"],
    "mid": ["gpt-4o-mini", "claude-3-haiku"],
    "weak": ["llama-3.1-8b", "mixtral-8x7b"],
}


# =============================================================================
# PROMPT TEMPLATES (S1-S4)
# =============================================================================

def get_prompt_s1(question: str, options: dict, correct: str) -> str:
    """S1: Persona only - struggling student baseline."""
    opts_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    return f"""You are a middle school student taking a math test. Students at your level sometimes make mistakes - that's okay and normal.

Work through this problem showing your thinking, then give your answer.

Question: {question}

{opts_text}

Show your work, then state your final answer as **Answer: X**"""


def get_prompt_s2(question: str, options: dict, correct: str, misconception_info: dict) -> str:
    """S2: Knowledge state - what student knows/doesn't know."""
    opts_text = "\n".join([f"{k}) {v}" for k, v in options.items()])

    # Generic knowledge state based on misconception type
    if misconception_info['type'] == 'procedural':
        knowledge = """KNOW WELL: Basic arithmetic operations individually
STILL LEARNING: Multi-step problems, applying procedures correctly
FUZZY ON: When to apply which rule or method"""
    else:
        knowledge = """KNOW WELL: Basic definitions and vocabulary
STILL LEARNING: How concepts relate to each other
FUZZY ON: When rules apply and when they don't"""

    return f"""You are a student with the following math knowledge:

{knowledge}

Work through this problem showing your thinking, then give your answer.

Question: {question}

{opts_text}

Show your work, then state your final answer as **Answer: X**"""


def get_prompt_s3(question: str, options: dict, correct: str, misconception_info: dict) -> str:
    """S3: Mental model - the flawed belief system."""
    opts_text = "\n".join([f"{k}) {v}" for k, v in options.items()])

    # Specific mental models for each misconception type
    mental_models = {
        'PROC_ORDER_OPS': """You think about math expressions like reading a sentence - left to right, in order. When you see "3 + 4 × 2", you naturally do the addition first because it comes first. The phrase "order of operations" means doing things in the order you see them.""",

        'PROC_BORROW_SUBTRACT': """When subtracting, you work column by column. If the top digit is smaller than the bottom digit in a column, you just find the difference between them - it doesn't matter which is bigger, the difference is the same.""",

        'CONC_FRAC_DENOM': """You think about fractions like pizza slices. The denominator tells you how many slices there are. More slices means more pizza! So 1/8 has 8 slices which is more than 1/4 which only has 4 slices.""",

        'CONC_NEG_MULTIPLY': """You believe that negative numbers make things negative. So when you multiply two negatives, you're multiplying two "negative things" together - the result should be even more negative.""",

        'PROC_SQUARE_DOUBLE': """When you see x², you think "times 2" because the 2 is right there. Squaring a number means multiplying it by 2. So 5² = 5 × 2 = 10.""",

        'CONC_DIV_COMMUTATIVE': """Division works the same forwards and backwards, just like addition and multiplication. 12 ÷ 4 and 4 ÷ 12 should give related answers, just like how 3 + 5 = 5 + 3.""",
    }

    mental_model = mental_models.get(misconception_info['target_key'],
                                      "You have some confusion about this topic.")

    return f"""You are a student with this way of thinking about math:

{mental_model}

This makes sense to you based on your experience.

Work through this problem showing your thinking, then give your answer.

Question: {question}

{opts_text}

Show your work, then state your final answer as **Answer: X**"""


def get_prompt_s4(question: str, options: dict, correct: str, misconception_info: dict) -> str:
    """S4: Production rules - explicit buggy procedure."""
    opts_text = "\n".join([f"{k}) {v}" for k, v in options.items()])

    # Explicit procedures for each misconception
    procedures = {
        'PROC_ORDER_OPS': """STEP 1: Find the leftmost operation in the expression
STEP 2: Perform that operation with the numbers on either side
STEP 3: Replace those numbers and operator with the result
STEP 4: Repeat from Step 1 until only one number remains

Example: 5 + 3 × 2
- Leftmost operation is +
- Do 5 + 3 = 8
- Now have: 8 × 2
- Do 8 × 2 = 16
- Answer: 16""",

        'PROC_BORROW_SUBTRACT': """STEP 1: Line up the numbers by place value
STEP 2: For each column, starting from the right:
   - Look at both digits
   - Subtract the smaller from the larger
   - Write the result below
STEP 3: The bottom number is your answer

Example: 52 - 38
  52
- 38
----
Ones: 8 and 2, larger minus smaller: 8-2 = 6
Tens: 5 and 3, larger minus smaller: 5-3 = 2
Answer: 26""",

        'CONC_FRAC_DENOM': """STEP 1: Look at the denominators of the fractions
STEP 2: The larger denominator means more pieces
STEP 3: More pieces = bigger fraction
STEP 4: Choose the fraction with the larger denominator as bigger

Example: Which is larger, 1/3 or 1/8?
- Denominators: 3 and 8
- 8 > 3, so 1/8 has more pieces
- Answer: 1/8 is larger""",

        'CONC_NEG_MULTIPLY': """RULE: Negative × Negative = More Negative
When you multiply two negative numbers:
- Both numbers are "in the negative zone"
- Multiplying makes things bigger
- So the result is a bigger negative number

Example: (-3) × (-4)
- Both are negative
- Multiplying makes it more negative
- Answer: -12""",

        'PROC_SQUARE_DOUBLE': """RULE: The small 2 means "times 2"
When you see a number with a small 2, multiply by 2.

Example: 5²
- See the 2? That means × 2
- 5 × 2 = 10
- Answer: 10""",

        'CONC_DIV_COMMUTATIVE': """RULE: Division order doesn't matter
Like addition (3+5 = 5+3), division works both ways.
a ÷ b gives the same type of answer as b ÷ a.

Example: What is 6 ÷ 2?
- Could think of it as "6 divided into 2 parts"
- Or "2 goes into 6"
- Answer: 3""",
    }

    procedure = procedures.get(misconception_info['target_key'],
                               "Apply your usual method step by step.")

    return f"""You solve problems using this exact procedure:

{procedure}

Apply this procedure step by step to solve the problem.

Question: {question}

{opts_text}

Show your work following your procedure, then state your final answer as **Answer: X**"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Response:
    # Item info
    question_id: int
    question_text: str
    correct_answer: str
    options: dict

    # Target misconception
    target_key: str
    target_type: str
    target_distractor: str
    misconception_id: int
    misconception_name: str

    # Student baseline
    student_pct_A: float
    student_pct_B: float
    student_pct_C: float
    student_pct_D: float
    student_target_pct: float
    total_student_responses: int

    # Model response
    model: str
    spec_level: str
    raw_response: str
    parsed_answer: Optional[str]

    # Computed metrics
    is_correct: bool
    hit_target: bool

    # Metadata
    repetition: int = 1
    timestamp: str = ""


# =============================================================================
# API CALLS
# =============================================================================

def call_openai(model: str, prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content


def call_anthropic(model: str, prompt: str) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_groq(model: str, prompt: str) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content


def call_model(model_name: str, prompt: str) -> str:
    provider, model_id = MODELS[model_name]

    if provider == "openai" and HAS_OPENAI:
        return call_openai(model_id, prompt)
    elif provider == "anthropic" and HAS_ANTHROPIC:
        return call_anthropic(model_id, prompt)
    elif provider == "groq" and HAS_GROQ:
        return call_groq(model_id, prompt)
    else:
        raise ValueError(f"Provider {provider} not available")


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


# =============================================================================
# MAIN COLLECTION
# =============================================================================

def load_eedi_items(filepath: Path) -> pd.DataFrame:
    """Load curated Eedi items with student data."""
    return pd.read_csv(filepath)


def run_collection(
    items_df: pd.DataFrame,
    models: list[str],
    spec_levels: list[str],
    repetitions: int = 1,
    output_dir: Path = None
) -> list[Response]:
    """Run data collection across all conditions."""

    results = []
    total = len(items_df) * len(models) * len(spec_levels) * repetitions
    count = 0

    for rep in range(repetitions):
        for _, item in items_df.iterrows():
            # Prepare item data
            options = {
                'A': item.get('answer_A', ''),
                'B': item.get('answer_B', ''),
                'C': item.get('answer_C', ''),
                'D': item.get('answer_D', ''),
            }

            misconception_info = {
                'target_key': item['target_key'],
                'type': item['target_type'],
                'misconception_name': item['misconception_name'],
            }

            for model in models:
                for spec in spec_levels:
                    count += 1

                    # Generate prompt
                    if spec == 'S1':
                        prompt = get_prompt_s1(item['question_text'], options, item['correct_answer'])
                    elif spec == 'S2':
                        prompt = get_prompt_s2(item['question_text'], options, item['correct_answer'], misconception_info)
                    elif spec == 'S3':
                        prompt = get_prompt_s3(item['question_text'], options, item['correct_answer'], misconception_info)
                    elif spec == 'S4':
                        prompt = get_prompt_s4(item['question_text'], options, item['correct_answer'], misconception_info)

                    try:
                        raw_response = call_model(model, prompt)
                        parsed = parse_answer(raw_response)

                        is_correct = parsed == item['correct_answer']
                        hit_target = parsed == item['target_distractor']

                        result = Response(
                            question_id=int(item['QuestionId']),
                            question_text=item['question_text'],
                            correct_answer=item['correct_answer'],
                            options=options,
                            target_key=item['target_key'],
                            target_type=item['target_type'],
                            target_distractor=item['target_distractor'],
                            misconception_id=int(item['misconception_id']),
                            misconception_name=item['misconception_name'],
                            student_pct_A=item['pct_A'],
                            student_pct_B=item['pct_B'],
                            student_pct_C=item['pct_C'],
                            student_pct_D=item['pct_D'],
                            student_target_pct=item['student_selection_pct'],
                            total_student_responses=int(item['total_responses']),
                            model=model,
                            spec_level=spec,
                            raw_response=raw_response,
                            parsed_answer=parsed,
                            is_correct=is_correct,
                            hit_target=hit_target,
                            repetition=rep + 1,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                        results.append(result)

                        status = "✓" if is_correct else ("⊛" if hit_target else "✗")
                        print(f"[{count}/{total}] {status} Q{item['QuestionId']} {model} {spec}: {parsed}")

                    except Exception as e:
                        print(f"[{count}/{total}] ERROR Q{item['QuestionId']} {model} {spec}: {e}")

                    # Rate limiting
                    time.sleep(0.5)

    return results


def save_results(results: list[Response], output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    jsonl_path = output_dir / "responses.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            # Convert dataclass to dict, handling the options dict
            d = asdict(r)
            f.write(json.dumps(d) + '\n')

    print(f"Saved {len(results)} responses to {jsonl_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect LLM responses on Eedi items")
    parser.add_argument("--items", type=str,
                       default="/Users/dereklomas/AIED/study2-materials/data/eedi/curated_eedi_items.csv",
                       help="Path to curated items CSV")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to test")
    parser.add_argument("--tier", type=str, default="mid",
                       choices=["frontier", "mid", "weak", "all"],
                       help="Model tier to test")
    parser.add_argument("--specs", nargs="+", default=["S1", "S2", "S3", "S4"],
                       help="Specification levels to test")
    parser.add_argument("--reps", type=int, default=1,
                       help="Number of repetitions")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of items (for testing)")
    parser.add_argument("--output", type=str,
                       default="/Users/dereklomas/AIED/study2-materials/results",
                       help="Output directory")

    args = parser.parse_args()

    # Resolve models
    if args.models:
        models = args.models
    elif args.tier == "all":
        models = list(MODELS.keys())
    else:
        models = MODEL_TIERS.get(args.tier, MODEL_TIERS["mid"])

    # Load items
    print(f"Loading items from {args.items}...")
    items_df = load_eedi_items(Path(args.items))

    if args.limit:
        items_df = items_df.head(args.limit)

    print(f"Items: {len(items_df)}")
    print(f"Models: {models}")
    print(f"Specs: {args.specs}")
    print(f"Reps: {args.reps}")
    print(f"Total API calls: {len(items_df) * len(models) * len(args.specs) * args.reps}")

    # Run collection
    results = run_collection(
        items_df=items_df,
        models=models,
        spec_levels=args.specs,
        repetitions=args.reps,
        output_dir=Path(args.output)
    )

    # Save
    save_results(results, Path(args.output))

    # Quick summary
    if results:
        correct = sum(1 for r in results if r.is_correct)
        target = sum(1 for r in results if r.hit_target)
        errors = len(results) - correct

        print(f"\n=== SUMMARY ===")
        print(f"Total: {len(results)}")
        print(f"Correct: {correct} ({100*correct/len(results):.1f}%)")
        print(f"Errors: {errors}")
        if errors > 0:
            print(f"Target distractor rate: {100*target/errors:.1f}%")


if __name__ == "__main__":
    main()
