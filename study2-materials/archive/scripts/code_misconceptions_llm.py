#!/usr/bin/env python3
"""
LLM-Assisted Misconception Coding for Study 2

Uses a small, fast model (Claude Haiku) to code whether chain-of-thought
reasoning reflects the target misconception - the key metric for
measuring the Reasoning Authenticity Gap.

Coding scheme:
- FULL_MATCH: Reasoning explicitly demonstrates the target misconception
- PARTIAL_MATCH: Reasoning shows elements of the misconception but incomplete
- DIFFERENT_ERROR: Wrong answer with clearly different reasoning
- UNCLEAR: Cannot determine reasoning from response
"""

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Misconception definitions for the coding prompt
MISCONCEPTION_DEFINITIONS = {
    "PROC_ORDER_OPS": {
        "name": "Left-to-right order of operations",
        "description": "Student performs operations left-to-right instead of following PEMDAS/BODMAS",
        "indicators": [
            "Performs addition/subtraction before multiplication/division",
            "Processes expression sequentially from left to right",
            "Ignores operator precedence rules",
            "Mentions reading math 'like a sentence'"
        ],
        "example_error": "5 + 3 × 2 = 16 (does 5+3=8, then 8×2=16)"
    },
    "CONC_FRAC_DENOM": {
        "name": "Larger denominator means larger fraction",
        "description": "Student believes fractions with larger denominators are larger",
        "indicators": [
            "Compares denominators to determine which fraction is larger",
            "States that more pieces means more/bigger",
            "Confuses number of slices with size of slices",
            "Says 1/8 > 1/4 because 8 > 4"
        ],
        "example_error": "1/6 > 1/4 because 6 is bigger than 4"
    },
    "PROC_SUBTRACT_REVERSE": {
        "name": "Smaller-from-larger subtraction",
        "description": "Student subtracts smaller digit from larger in each column regardless of position",
        "indicators": [
            "In each column, subtracts the smaller digit from the larger",
            "Ignores which number is being subtracted from which",
            "Avoids borrowing/regrouping by reversing subtraction",
            "Treats subtraction as finding absolute difference per column"
        ],
        "example_error": "52 - 38 = 26 (ones: 8-2=6, tens: 5-3=2)"
    },
    "CONC_MULT_INCREASE": {
        "name": "Multiplication always increases",
        "description": "Student believes multiplication always makes numbers larger",
        "indicators": [
            "States that multiplying always increases the result",
            "Rejects or questions results smaller than inputs",
            "Confusion when multiplying by decimals < 1 or fractions",
            "References 'times means more' or 'copies' metaphor inappropriately"
        ],
        "example_error": "8 × 0.5 must be ≥ 8 because multiplication makes bigger"
    }
}


CODING_PROMPT = """You are an expert at analyzing student mathematical reasoning. Your task is to determine whether a student's reasoning reflects a specific misconception.

## Target Misconception: {misconception_name}

**Description**: {misconception_description}

**Key indicators that reasoning matches this misconception**:
{indicators}

**Example of this error**: {example_error}

## Student Response to Analyze

**Question**: {question}
**Correct Answer**: {correct_answer}
**Student's Answer**: {student_answer}
**Target Distractor**: {target_distractor}

**Student's Reasoning**:
{reasoning}

## Your Task

Analyze whether the student's REASONING (not just their answer) reflects the target misconception.

Respond with EXACTLY ONE of these codes and a brief justification:

1. **FULL_MATCH** - The reasoning explicitly demonstrates the target misconception. Multiple indicators are present.

2. **PARTIAL_MATCH** - The reasoning shows some elements of the target misconception but is incomplete or mixed with other approaches.

3. **DIFFERENT_ERROR** - The student got the wrong answer but their reasoning shows a DIFFERENT error pattern, not the target misconception.

4. **UNCLEAR** - Cannot determine the reasoning pattern from the response (too brief, contradictory, or no work shown).

Format your response as:
CODE: [FULL_MATCH/PARTIAL_MATCH/DIFFERENT_ERROR/UNCLEAR]
EVIDENCE: [1-2 sentence justification citing specific reasoning from the response]
"""


@dataclass
class CodingResult:
    item_id: str
    model: str
    spec_level: str
    misconception_id: str
    question: str
    correct_answer: str
    student_answer: str
    target_distractor: str
    hit_target: bool
    reasoning: str
    code: str
    evidence: str
    coder_model: str
    coding_time: float


def load_responses(filepath: Path) -> list[dict]:
    """Load responses from JSONL file."""
    responses = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def extract_errors(responses: list[dict]) -> list[dict]:
    """Extract only incorrect responses for coding."""
    return [r for r in responses if not r.get('is_correct', True)]


def code_reasoning_llm(error: dict, client: Anthropic, coder_model: str = "claude-3-haiku-20240307") -> CodingResult:
    """Use LLM to code whether reasoning matches target misconception."""

    misconception_id = error.get('misconception_id', '')
    definition = MISCONCEPTION_DEFINITIONS.get(misconception_id, {})

    if not definition:
        return CodingResult(
            item_id=error.get('item_id', ''),
            model=error.get('model', ''),
            spec_level=error.get('spec_level', ''),
            misconception_id=misconception_id,
            question=error.get('question', ''),
            correct_answer=error.get('correct_answer', ''),
            student_answer=error.get('parsed_answer', ''),
            target_distractor=error.get('target_distractor', ''),
            hit_target=error.get('hit_target', False),
            reasoning=error.get('raw_response', '')[:500],
            code='UNCLEAR',
            evidence='Unknown misconception type',
            coder_model=coder_model,
            coding_time=0
        )

    # Format the coding prompt
    prompt = CODING_PROMPT.format(
        misconception_name=definition['name'],
        misconception_description=definition['description'],
        indicators='\n'.join(f"- {ind}" for ind in definition['indicators']),
        example_error=definition['example_error'],
        question=error.get('question', ''),
        correct_answer=error.get('correct_answer', ''),
        student_answer=error.get('parsed_answer', ''),
        target_distractor=error.get('target_distractor', ''),
        reasoning=error.get('raw_response', '')
    )

    start_time = time.time()

    try:
        response = client.messages.create(
            model=coder_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text
        coding_time = time.time() - start_time

        # Parse the response
        code = 'UNCLEAR'
        evidence = ''

        for line in response_text.split('\n'):
            if line.startswith('CODE:'):
                code_text = line.replace('CODE:', '').strip()
                if 'FULL_MATCH' in code_text:
                    code = 'FULL_MATCH'
                elif 'PARTIAL_MATCH' in code_text:
                    code = 'PARTIAL_MATCH'
                elif 'DIFFERENT_ERROR' in code_text:
                    code = 'DIFFERENT_ERROR'
                else:
                    code = 'UNCLEAR'
            elif line.startswith('EVIDENCE:'):
                evidence = line.replace('EVIDENCE:', '').strip()

        return CodingResult(
            item_id=error.get('item_id', ''),
            model=error.get('model', ''),
            spec_level=error.get('spec_level', ''),
            misconception_id=misconception_id,
            question=error.get('question', ''),
            correct_answer=error.get('correct_answer', ''),
            student_answer=error.get('parsed_answer', ''),
            target_distractor=error.get('target_distractor', ''),
            hit_target=error.get('hit_target', False),
            reasoning=error.get('raw_response', '')[:500],
            code=code,
            evidence=evidence,
            coder_model=coder_model,
            coding_time=coding_time
        )

    except Exception as e:
        return CodingResult(
            item_id=error.get('item_id', ''),
            model=error.get('model', ''),
            spec_level=error.get('spec_level', ''),
            misconception_id=misconception_id,
            question=error.get('question', ''),
            correct_answer=error.get('correct_answer', ''),
            student_answer=error.get('parsed_answer', ''),
            target_distractor=error.get('target_distractor', ''),
            hit_target=error.get('hit_target', False),
            reasoning=error.get('raw_response', '')[:500],
            code='UNCLEAR',
            evidence=f'Coding error: {str(e)}',
            coder_model=coder_model,
            coding_time=0
        )


def compute_reasoning_authenticity_gap(coded_results: list[CodingResult]) -> dict:
    """Compute the Reasoning Authenticity Gap and related metrics."""

    total = len(coded_results)
    if total == 0:
        return {}

    # Overall metrics
    target_hits = sum(1 for r in coded_results if r.hit_target)
    full_matches = sum(1 for r in coded_results if r.code == 'FULL_MATCH')
    partial_matches = sum(1 for r in coded_results if r.code == 'PARTIAL_MATCH')
    different_errors = sum(1 for r in coded_results if r.code == 'DIFFERENT_ERROR')
    unclear = sum(1 for r in coded_results if r.code == 'UNCLEAR')

    target_rate = target_hits / total
    # Misconception alignment = FULL_MATCH only (strict) or FULL + PARTIAL (lenient)
    alignment_strict = full_matches / total
    alignment_lenient = (full_matches + partial_matches) / total

    gap_strict = target_rate - alignment_strict
    gap_lenient = target_rate - alignment_lenient

    # By specification level
    by_spec = defaultdict(lambda: {'total': 0, 'target': 0, 'full': 0, 'partial': 0})
    for r in coded_results:
        by_spec[r.spec_level]['total'] += 1
        if r.hit_target:
            by_spec[r.spec_level]['target'] += 1
        if r.code == 'FULL_MATCH':
            by_spec[r.spec_level]['full'] += 1
        elif r.code == 'PARTIAL_MATCH':
            by_spec[r.spec_level]['partial'] += 1

    # By model
    by_model = defaultdict(lambda: {'total': 0, 'target': 0, 'full': 0, 'partial': 0})
    for r in coded_results:
        by_model[r.model]['total'] += 1
        if r.hit_target:
            by_model[r.model]['target'] += 1
        if r.code == 'FULL_MATCH':
            by_model[r.model]['full'] += 1
        elif r.code == 'PARTIAL_MATCH':
            by_model[r.model]['partial'] += 1

    # By misconception type
    by_misc = defaultdict(lambda: {'total': 0, 'target': 0, 'full': 0, 'partial': 0})
    for r in coded_results:
        by_misc[r.misconception_id]['total'] += 1
        if r.hit_target:
            by_misc[r.misconception_id]['target'] += 1
        if r.code == 'FULL_MATCH':
            by_misc[r.misconception_id]['full'] += 1
        elif r.code == 'PARTIAL_MATCH':
            by_misc[r.misconception_id]['partial'] += 1

    return {
        'overall': {
            'total_errors': total,
            'target_distractor_rate': target_rate,
            'alignment_strict': alignment_strict,
            'alignment_lenient': alignment_lenient,
            'gap_strict': gap_strict,
            'gap_lenient': gap_lenient,
            'code_distribution': {
                'FULL_MATCH': full_matches,
                'PARTIAL_MATCH': partial_matches,
                'DIFFERENT_ERROR': different_errors,
                'UNCLEAR': unclear
            }
        },
        'by_spec_level': dict(by_spec),
        'by_model': dict(by_model),
        'by_misconception': dict(by_misc)
    }


def generate_report(coded_results: list[CodingResult], metrics: dict) -> str:
    """Generate a comprehensive coding report."""

    lines = []
    lines.append("=" * 80)
    lines.append("REASONING AUTHENTICITY GAP ANALYSIS")
    lines.append("LLM-Assisted Misconception Coding Results")
    lines.append("=" * 80)
    lines.append("")

    overall = metrics['overall']

    lines.append("## 1. KEY FINDING: REASONING AUTHENTICITY GAP")
    lines.append("-" * 60)
    lines.append(f"Total errors coded: {overall['total_errors']}")
    lines.append(f"Target distractor rate: {100*overall['target_distractor_rate']:.1f}%")
    lines.append(f"Misconception alignment (strict): {100*overall['alignment_strict']:.1f}%")
    lines.append(f"Misconception alignment (lenient): {100*overall['alignment_lenient']:.1f}%")
    lines.append("")
    lines.append(f">>> REASONING AUTHENTICITY GAP (strict): {100*overall['gap_strict']:.1f} percentage points <<<")
    lines.append(f">>> REASONING AUTHENTICITY GAP (lenient): {100*overall['gap_lenient']:.1f} percentage points <<<")
    lines.append("")

    lines.append("## 2. CODE DISTRIBUTION")
    lines.append("-" * 60)
    for code, count in overall['code_distribution'].items():
        pct = 100 * count / overall['total_errors']
        lines.append(f"  {code}: {count} ({pct:.1f}%)")
    lines.append("")

    lines.append("## 3. BY SPECIFICATION LEVEL")
    lines.append("-" * 60)
    lines.append(f"{'Spec':<8} {'Errors':<8} {'Target%':<10} {'Align%':<10} {'Gap':<10}")
    for spec in ['S1', 'S2', 'S3', 'S4']:
        data = metrics['by_spec_level'].get(spec, {})
        total = data.get('total', 0)
        if total > 0:
            target_rate = 100 * data.get('target', 0) / total
            align_rate = 100 * (data.get('full', 0) + data.get('partial', 0)) / total
            gap = target_rate - align_rate
            lines.append(f"{spec:<8} {total:<8} {target_rate:<10.1f} {align_rate:<10.1f} {gap:<10.1f}")
    lines.append("")

    lines.append("## 4. BY MODEL")
    lines.append("-" * 60)
    for model, data in sorted(metrics['by_model'].items()):
        total = data.get('total', 0)
        if total > 0:
            target_rate = 100 * data.get('target', 0) / total
            align_rate = 100 * (data.get('full', 0) + data.get('partial', 0)) / total
            gap = target_rate - align_rate
            lines.append(f"{model}:")
            lines.append(f"  Errors: {total}, Target: {target_rate:.1f}%, Align: {align_rate:.1f}%, Gap: {gap:.1f}pp")
    lines.append("")

    lines.append("## 5. BY MISCONCEPTION TYPE")
    lines.append("-" * 60)
    for misc, data in sorted(metrics['by_misconception'].items()):
        total = data.get('total', 0)
        if total > 0:
            target_rate = 100 * data.get('target', 0) / total
            align_rate = 100 * (data.get('full', 0) + data.get('partial', 0)) / total
            gap = target_rate - align_rate
            misc_type = "PROC" if misc.startswith("PROC") else "CONC"
            lines.append(f"{misc} ({misc_type}):")
            lines.append(f"  Errors: {total}, Target: {target_rate:.1f}%, Align: {align_rate:.1f}%, Gap: {gap:.1f}pp")
    lines.append("")

    # Sample coded responses
    lines.append("## 6. SAMPLE CODED RESPONSES")
    lines.append("-" * 60)

    for code_type in ['FULL_MATCH', 'PARTIAL_MATCH', 'DIFFERENT_ERROR']:
        examples = [r for r in coded_results if r.code == code_type][:2]
        if examples:
            lines.append(f"\n### {code_type} Examples:")
            for ex in examples:
                lines.append(f"\nItem: {ex.item_id} | Model: {ex.model} | Spec: {ex.spec_level}")
                lines.append(f"Question: {ex.question}")
                lines.append(f"Answer: {ex.student_answer} (correct: {ex.correct_answer}, target: {ex.target_distractor})")
                lines.append(f"Target hit: {ex.hit_target}")
                lines.append(f"Reasoning excerpt: {ex.reasoning[:200]}...")
                lines.append(f">>> Code: {ex.code}")
                lines.append(f">>> Evidence: {ex.evidence}")

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-assisted misconception coding")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file with responses")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: same as input)")
    parser.add_argument("--coder-model", type=str, default="claude-3-haiku-20240307",
                       help="Model to use for coding")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of errors to code (for testing)")

    args = parser.parse_args()

    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed")
        return

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "coding_results"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading responses from {input_path}...")
    responses = load_responses(input_path)
    print(f"Total responses: {len(responses)}")

    print("Extracting errors...")
    errors = extract_errors(responses)
    print(f"Total errors to code: {len(errors)}")

    if args.limit:
        errors = errors[:args.limit]
        print(f"Limited to {len(errors)} errors")

    print(f"\nCoding with {args.coder_model}...")
    client = Anthropic()

    coded_results = []
    for i, error in enumerate(errors):
        result = code_reasoning_llm(error, client, args.coder_model)
        coded_results.append(result)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Coded {i+1}/{len(errors)} ({100*(i+1)/len(errors):.1f}%)")

    print("\nComputing metrics...")
    metrics = compute_reasoning_authenticity_gap(coded_results)

    print("\nGenerating report...")
    report = generate_report(coded_results, metrics)

    # Save outputs
    report_path = output_dir / "coding_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    coded_path = output_dir / "coded_errors.jsonl"
    with open(coded_path, 'w') as f:
        for r in coded_results:
            f.write(json.dumps(asdict(r)) + '\n')
    print(f"Coded data saved to: {coded_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Target distractor rate: {100*metrics['overall']['target_distractor_rate']:.1f}%")
    print(f"Misconception alignment: {100*metrics['overall']['alignment_lenient']:.1f}%")
    print(f"REASONING AUTHENTICITY GAP: {100*metrics['overall']['gap_lenient']:.1f} percentage points")


if __name__ == "__main__":
    main()
