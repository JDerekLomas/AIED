#!/usr/bin/env python3
"""
Misconception Diagnosis Test

Tests whether LLMs can identify what misconception led a student to select
a particular wrong answer. This is a different capability than exhibiting
misconceptions - it tests metacognitive diagnosis ability.
"""

import json
import os
import sys
from pathlib import Path
from anthropic import Anthropic

def load_errors(path: Path) -> list:
    """Load incorrect responses from JSONL."""
    errors = []
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get('is_correct') == False and record.get('parsed_answer'):
                    errors.append(record)
    return errors

def create_diagnosis_prompt(error: dict) -> str:
    """Create a prompt asking the model to diagnose the misconception."""
    # Extract the original question from the prompt
    prompt_text = error.get('prompt', '')

    # Find the question portion
    if 'Question:' in prompt_text:
        question_start = prompt_text.find('Question:')
        question_end = prompt_text.find('Reasoning:') if 'Reasoning:' in prompt_text else prompt_text.find('Your thinking:')
        if question_end == -1:
            question_end = len(prompt_text)
        question = prompt_text[question_start:question_end].strip()
    else:
        question = "Could not extract question"

    student_answer = error.get('parsed_answer', '?')
    correct_answer = error.get('correct_answer', '?')
    student_reasoning = error.get('reasoning', 'No reasoning provided')

    # Get the possible misconceptions for this item
    misconception_name = error.get('misconception_name', 'Unknown')

    diagnosis_prompt = f"""You are an expert math teacher analyzing a student's error.

{question}

The student selected answer: {student_answer}
The correct answer was: {correct_answer}

Here is the student's reasoning:
{student_reasoning}

Based on the student's wrong answer and reasoning, identify the most likely misconception.
Choose from these common misconceptions:

1. "Carries out operations from left to right regardless of priority order" - Student ignores order of operations (PEMDAS)
2. "Confuses percentages with decimals" - Student treats 50% as 50 instead of 0.5
3. "Reverses operations when solving equations" - Student subtracts when should add, etc.
4. "Uses the wrong fraction operation" - Student multiplies when should divide fractions, etc.
5. "Misreads negative signs" - Student treats negatives incorrectly
6. "Confuses area and perimeter" - Student calculates wrong measurement type
7. "Believes digit values don't change with position" - Place value confusion
8. "Applies additive rather than multiplicative reasoning" - Ratio/proportion errors

Which misconception best explains this student's error? Respond with ONLY the number (1-8) of the most likely misconception."""

    return diagnosis_prompt

def test_diagnosis(client, errors: list, n_tests: int = 20) -> dict:
    """Run diagnosis tests on a sample of errors."""
    import random

    # Sample errors if we have more than n_tests
    if len(errors) > n_tests:
        test_errors = random.sample(errors, n_tests)
    else:
        test_errors = errors

    results = []

    # Map misconception IDs to diagnosis categories
    misconception_map = {
        'PROC_ORDER_OPS': 1,  # Order of operations
        'NUM_PERCENT_DEC': 2,  # Percentages/decimals
        'ALG_REVERSE_OPS': 3,  # Reversing operations
        'FRAC_WRONG_OP': 4,  # Wrong fraction operation
        'NUM_NEG_SIGN': 5,  # Negative signs
        'GEOM_AREA_PERIM': 6,  # Area vs perimeter
        'NUM_PLACE_VALUE': 7,  # Place value
        'RATIO_ADDITIVE': 8,  # Additive vs multiplicative
    }

    for i, error in enumerate(test_errors):
        prompt = create_diagnosis_prompt(error)

        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )

            diagnosis_text = response.content[0].text.strip()

            # Parse the diagnosis number
            try:
                diagnosed = int(diagnosis_text.split()[0].replace('.', '').replace(')', ''))
            except:
                diagnosed = None

            # Get the actual misconception
            actual_misc_id = error.get('misconception_id', '').split('_')[0] + '_' + '_'.join(error.get('misconception_id', '').split('_')[1:3])

            # Simple prefix matching for category
            actual_category = None
            for key, value in misconception_map.items():
                if error.get('misconception_id', '').startswith(key.replace('_', '_')):
                    actual_category = value
                    break

            # For PROC_ORDER_OPS items specifically
            if 'ORDER_OPS' in error.get('misconception_id', ''):
                actual_category = 1

            is_correct = diagnosed == actual_category if (diagnosed and actual_category) else None

            results.append({
                'item_id': error.get('item_id'),
                'model_that_erred': error.get('model'),
                'student_answer': error.get('parsed_answer'),
                'correct_answer': error.get('correct_answer'),
                'actual_misconception': error.get('misconception_name'),
                'actual_category': actual_category,
                'diagnosed_category': diagnosed,
                'diagnosis_correct': is_correct,
                'raw_diagnosis': diagnosis_text
            })

            print(f"  [{i+1}/{len(test_errors)}] {error.get('item_id')}: diagnosed={diagnosed}, actual={actual_category}, correct={is_correct}")

        except Exception as e:
            print(f"  Error on {error.get('item_id')}: {e}")
            results.append({
                'item_id': error.get('item_id'),
                'error': str(e)
            })

    return results

def main():
    # Load errors from single-item data
    data_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/all_single_item.jsonl")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print("Loading errors from single-item data...")
    errors = load_errors(data_path)
    print(f"Found {len(errors)} incorrect responses")

    # Group by model
    by_model = {}
    for e in errors:
        model = e.get('model', 'unknown')
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(e)

    print("\nErrors by model:")
    for model, errs in sorted(by_model.items()):
        print(f"  {model}: {len(errs)} errors")

    # Initialize Anthropic client
    client = Anthropic()

    print("\n" + "="*60)
    print("MISCONCEPTION DIAGNOSIS TEST")
    print("="*60)
    print("Testing Claude Haiku's ability to diagnose misconceptions")
    print("from student errors...\n")

    # Run diagnosis test
    results = test_diagnosis(client, errors, n_tests=30)

    # Analyze results
    valid_results = [r for r in results if r.get('diagnosis_correct') is not None]
    correct = sum(1 for r in valid_results if r.get('diagnosis_correct'))

    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)
    print(f"Total tests: {len(results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Correct diagnoses: {correct}")
    print(f"Accuracy: {correct/len(valid_results)*100:.1f}%" if valid_results else "N/A")

    # By original model that made the error
    print("\nDiagnosis accuracy by error source model:")
    for model in by_model.keys():
        model_results = [r for r in valid_results if r.get('model_that_erred') == model]
        if model_results:
            model_correct = sum(1 for r in model_results if r.get('diagnosis_correct'))
            print(f"  {model}: {model_correct}/{len(model_results)} ({model_correct/len(model_results)*100:.1f}%)")

    # Save results
    output_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/diagnosis_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
