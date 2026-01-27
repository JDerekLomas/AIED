#!/usr/bin/env python3
"""Code S3/S4 errors for reasoning authenticity - does the reasoning match the misconception?"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from collections import defaultdict

load_dotenv()

MISCONCEPTION_CRITERIA = {
    1507: {
        "name": "Order of operations (left-to-right)",
        "authentic_indicators": [
            "Explicitly does operations left-to-right",
            "Does addition/subtraction before multiplication/division",
            "Ignores PEMDAS/BODMAS/BIDMAS priority",
            "Processes expression sequentially as encountered"
        ],
        "NOT_authentic": [
            "Follows correct order but makes arithmetic error",
            "Mentions PEMDAS but applies it wrong",
            "Random computational mistake"
        ]
    },
    1597: {
        "name": "Negative × Negative = Negative",
        "authentic_indicators": [
            "Explicitly states two negatives give negative",
            "Applies 'any negative makes result negative' rule",
            "Counts negative signs and concludes negative",
            "Expresses belief that negatives 'stay negative'"
        ],
        "NOT_authentic": [
            "Correct sign rule but arithmetic error",
            "Forgets a negative sign accidentally",
            "Random computational mistake"
        ]
    },
    217: {
        "name": "Fraction addition (add numerators and denominators)",
        "authentic_indicators": [
            "Adds numerators together",
            "Adds denominators together",
            "Shows (a+c)/(b+d) pattern",
            "Treats fractions as 'two numbers stacked'"
        ],
        "NOT_authentic": [
            "Finds common denominator but makes arithmetic error",
            "Correct method but wrong calculation",
            "Random fraction mistake"
        ]
    }
}

CODING_PROMPT = """You are coding student reasoning for a research study on mathematical misconceptions.

TARGET MISCONCEPTION: {misconception_name}
QUESTION: {question_text}
CORRECT ANSWER: {correct_answer}
STUDENT'S ANSWER: {student_answer} (Target distractor: {target_distractor})
STUDENT'S REASONING:
{reasoning}

AUTHENTIC MISCONCEPTION INDICATORS:
{authentic_indicators}

NOT AUTHENTIC (these don't count):
{not_authentic}

TASK: Does the student's reasoning AUTHENTICALLY demonstrate the target misconception?

Classify as:
- FULL_MATCH: Reasoning explicitly shows the misconception pattern
- PARTIAL_MATCH: Some evidence of misconception thinking, but not fully explicit
- DIFFERENT_ERROR: Made an error but reasoning doesn't match target misconception
- CORRECT_METHOD_WRONG_CALC: Used correct method but made arithmetic mistake
- UNCLEAR: Can't determine from reasoning

Respond in JSON format:
{{"classification": "...", "evidence": "brief quote or description", "confidence": "high/medium/low"}}
"""


def load_responses():
    """Load all S3/S4 error responses."""
    responses = []
    for f in ['results/full_experiment/responses_20260126_173930.jsonl',
              'results/full_experiment/responses_conceptual.jsonl']:
        try:
            with open(f) as file:
                for line in file:
                    r = json.loads(line)
                    # Filter to S3/S4 errors only
                    if r['spec_level'] in ['S3', 'S4'] and not r['is_correct']:
                        responses.append(r)
        except FileNotFoundError:
            pass
    return responses


def load_items():
    """Load item data for question text."""
    with open('data/experiment_items.json') as f:
        items = json.load(f)
    return {item['question_id']: item for item in items}


def code_response(client, response, item):
    """Use Claude to code a single response."""
    misc_id = response['misconception_id']
    criteria = MISCONCEPTION_CRITERIA.get(misc_id, {})

    prompt = CODING_PROMPT.format(
        misconception_name=criteria.get('name', response['misconception_name']),
        question_text=item.get('question_text', f"Question {response['question_id']}"),
        correct_answer=response['correct_answer'],
        student_answer=response['parsed_answer'],
        target_distractor=response['target_distractor'],
        reasoning=response['raw_response'],
        authentic_indicators="\n".join(f"- {i}" for i in criteria.get('authentic_indicators', [])),
        not_authentic="\n".join(f"- {i}" for i in criteria.get('NOT_authentic', []))
    )

    result = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = result.content[0].text
    # Parse JSON from response
    try:
        # Find JSON in response
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return {"classification": "PARSE_ERROR", "evidence": text[:200], "confidence": "low"}


def main():
    client = Anthropic()

    responses = load_responses()
    items = load_items()

    print(f"Coding {len(responses)} S3/S4 errors...")

    output_dir = Path('results/misconception_coding')
    output_dir.mkdir(parents=True, exist_ok=True)

    coded = []
    stats = defaultdict(int)

    for i, r in enumerate(responses):
        item = items.get(r['question_id'], {})

        coding = code_response(client, r, item)

        coded_response = {
            **r,
            'question_text': item.get('question_text', ''),
            'coding': coding
        }
        coded.append(coded_response)

        stats[coding['classification']] += 1

        symbol = {
            'FULL_MATCH': '✓✓',
            'PARTIAL_MATCH': '✓',
            'DIFFERENT_ERROR': '✗',
            'CORRECT_METHOD_WRONG_CALC': '○',
            'UNCLEAR': '?',
            'PARSE_ERROR': '!'
        }.get(coding['classification'], '?')

        print(f"[{i+1}/{len(responses)}] {r['spec_level']} M{r['misconception_id']} Q{r['question_id']}: {symbol} {coding['classification']}")

        # Save incrementally
        if (i + 1) % 10 == 0:
            with open(output_dir / 'coded_s3s4_errors.json', 'w') as f:
                json.dump(coded, f, indent=2)

    # Final save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_dir / f'coded_s3s4_errors_{timestamp}.json', 'w') as f:
        json.dump(coded, f, indent=2)
    with open(output_dir / 'coded_s3s4_errors.json', 'w') as f:
        json.dump(coded, f, indent=2)

    # Summary report
    print("\n" + "="*60)
    print("CODING SUMMARY")
    print("="*60)

    total = len(coded)
    full_match = stats['FULL_MATCH']
    partial_match = stats['PARTIAL_MATCH']
    authentic = full_match + partial_match

    print(f"Total S3/S4 errors: {total}")
    print(f"Target distractor hits: {sum(1 for r in coded if r['hit_target'])} ({100*sum(1 for r in coded if r['hit_target'])/total:.1f}%)")
    print()
    print("Reasoning Authenticity:")
    print(f"  FULL_MATCH:    {full_match} ({100*full_match/total:.1f}%)")
    print(f"  PARTIAL_MATCH: {partial_match} ({100*partial_match/total:.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  AUTHENTIC:     {authentic} ({100*authentic/total:.1f}%)")
    print()
    print(f"  DIFFERENT_ERROR: {stats['DIFFERENT_ERROR']} ({100*stats['DIFFERENT_ERROR']/total:.1f}%)")
    print(f"  CORRECT_METHOD:  {stats['CORRECT_METHOD_WRONG_CALC']} ({100*stats['CORRECT_METHOD_WRONG_CALC']/total:.1f}%)")
    print(f"  UNCLEAR:         {stats['UNCLEAR']} ({100*stats['UNCLEAR']/total:.1f}%)")

    # Reasoning Authenticity Gap
    target_rate = 100 * sum(1 for r in coded if r['hit_target']) / total
    alignment_rate = 100 * authentic / total
    gap = target_rate - alignment_rate

    print()
    print("="*60)
    print("REASONING AUTHENTICITY GAP")
    print("="*60)
    print(f"Target Distractor Rate: {target_rate:.1f}%")
    print(f"Reasoning Alignment:    {alignment_rate:.1f}%")
    print(f"GAP:                    {gap:.1f} percentage points")

    # By spec level
    print()
    print("By Specification Level:")
    for spec in ['S3', 'S4']:
        spec_responses = [r for r in coded if r['spec_level'] == spec]
        if spec_responses:
            spec_authentic = sum(1 for r in spec_responses
                               if r['coding']['classification'] in ['FULL_MATCH', 'PARTIAL_MATCH'])
            spec_target = sum(1 for r in spec_responses if r['hit_target'])
            print(f"  {spec}: Target={100*spec_target/len(spec_responses):.1f}%, Authentic={100*spec_authentic/len(spec_responses):.1f}%")

    # Save report
    report = {
        "total_errors": total,
        "target_rate": target_rate,
        "alignment_rate": alignment_rate,
        "gap": gap,
        "classifications": dict(stats),
        "timestamp": timestamp
    }
    with open(output_dir / 'coding_summary.json', 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
