#!/usr/bin/env python3
"""
Misconception Coding Script for Study 2

This script extracts all error responses and codes whether the reasoning
reflects the target misconception - not just whether the model selected
the target distractor.

Coding scheme:
- FULL_MATCH: Reasoning explicitly shows the target misconception
- PARTIAL_MATCH: Reasoning shows related but not exact misconception
- DIFFERENT_ERROR: Wrong answer but different reasoning than target
- UNCLEAR: Cannot determine reasoning from response
"""

import json
from pathlib import Path
from collections import defaultdict
import re

# Misconception patterns for automated pre-coding
MISCONCEPTION_PATTERNS = {
    "CONC_AREA_PERIMETER": {
        "indicators": [
            r"perimeter.*area|area.*perimeter",
            r"add.*sides|sides.*add",
            r"multiply.*length.*width.*perimeter",
            r"2.*\+.*2|length.*\+.*width.*for area",
        ],
        "description": "Confuses area formula (L×W) with perimeter formula (2L+2W)"
    },
    "CONC_FRAC_DENOM": {
        "indicators": [
            r"bigger.*denominator.*bigger|larger.*denominator.*larger",
            r"6.*bigger.*4|6.*larger.*4|6.*more.*4",
            r"more.*pieces.*means.*more",
        ],
        "description": "Believes larger denominator = larger fraction"
    },
    "CONC_MULT_INCREASE": {
        "indicators": [
            r"multiply.*bigger|multiply.*increase|multiply.*more",
            r"times.*always.*bigger",
            r"0\.5.*makes.*bigger|half.*makes.*more",
        ],
        "description": "Believes multiplying always increases the number"
    },
    "CONC_PERCENT_BASE": {
        "indicators": [
            r"percent.*of.*wrong|base.*confused",
            r"20.*percent.*of.*25|25.*percent.*of.*20",
            r"switched.*numbers",
        ],
        "description": "Confuses which number is the base in percentage problems"
    },
    "PROC_ORDER_OPS": {
        "indicators": [
            r"left.*to.*right|reading.*order",
            r"add.*first.*then.*multiply|addition.*before.*multiplication",
            r"5.*\+.*3.*=.*8.*then|first.*add",
        ],
        "description": "Performs operations left-to-right instead of PEMDAS"
    },
    "PROC_NEGATIVE_SUBTRACT": {
        "indicators": [
            r"minus.*minus.*minus|negative.*negative.*negative",
            r"subtract.*negative.*subtract",
            r"double.*negative.*positive",
            r"-.*-.*=.*-",
        ],
        "description": "Mishandles subtraction of negative numbers"
    },
    "PROC_SUBTRACT_REVERSE": {
        "indicators": [
            r"smaller.*from.*larger|small.*minus.*big",
            r"reverse.*subtract|backward.*subtract",
            r"3.*-.*5.*instead|wrong.*order",
        ],
        "description": "Subtracts in wrong direction (smaller from larger in each column)"
    },
    "PROC_DECIMAL_PLACE": {
        "indicators": [
            r"ignore.*decimal|decimal.*ignore",
            r"25.*bigger.*5|0?\.?25.*>.*0?\.?5",
            r"more.*digits.*bigger",
        ],
        "description": "Ignores decimal point position when comparing"
    },
    "PROC_ALGEBRA_COMBINE": {
        "indicators": [
            r"combine.*unlike|add.*x.*and.*number",
            r"3x.*\+.*5.*=.*8x|coefficient.*add.*constant",
            r"x.*\+.*\d+.*=.*\d+x",
        ],
        "description": "Incorrectly combines unlike terms in algebra"
    },
    "INTERP_GRAPH_AXES": {
        "indicators": [
            r"x.*axis.*y|y.*axis.*x|axes.*confused|axes.*switched",
            r"horizontal.*vertical.*confused",
            r"read.*wrong.*axis",
        ],
        "description": "Confuses x and y axes when reading graphs"
    },
}


def load_responses(filepath):
    """Load all responses from JSONL file."""
    responses = []
    with open(filepath, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    return responses


def extract_errors(responses):
    """Extract only incorrect responses."""
    return [r for r in responses if not r.get('is_correct', True)]


def check_target_distractor(error):
    """Check if model selected the target distractor."""
    return error.get('parsed_answer') == error.get('target_distractor')


def auto_code_reasoning(error):
    """
    Attempt automated pre-coding of reasoning.
    Returns (code, confidence, evidence)
    """
    reasoning = error.get('reasoning', '').lower()
    misconception_id = error.get('misconception_id', '')

    if not reasoning or len(reasoning) < 10:
        return 'UNCLEAR', 'low', 'No reasoning provided'

    patterns = MISCONCEPTION_PATTERNS.get(misconception_id, {})
    indicators = patterns.get('indicators', [])

    matches = []
    for pattern in indicators:
        if re.search(pattern, reasoning, re.IGNORECASE):
            matches.append(pattern)

    if len(matches) >= 2:
        return 'FULL_MATCH', 'high', f"Multiple indicators: {matches[:2]}"
    elif len(matches) == 1:
        return 'PARTIAL_MATCH', 'medium', f"Single indicator: {matches[0]}"
    else:
        # Check if there's ANY reasoning that led to wrong answer
        if len(reasoning) > 50:
            return 'DIFFERENT_ERROR', 'medium', 'Has reasoning but no misconception indicators'
        else:
            return 'UNCLEAR', 'low', 'Minimal reasoning'


def analyze_errors(errors):
    """Analyze all errors and generate coding."""
    results = []

    for error in errors:
        target_hit = check_target_distractor(error)
        auto_code, confidence, evidence = auto_code_reasoning(error)

        results.append({
            'item_id': error.get('item_id'),
            'model': error.get('model'),
            'condition': error.get('condition'),
            'misconception_id': error.get('misconception_id'),
            'misconception_name': error.get('misconception_name'),
            'misconception_category': error.get('misconception_category'),
            'parsed_answer': error.get('parsed_answer'),
            'correct_answer': error.get('correct_answer'),
            'target_distractor': error.get('target_distractor'),
            'target_hit': target_hit,
            'reasoning': error.get('reasoning', '')[:500],  # Truncate for readability
            'auto_code': auto_code,
            'code_confidence': confidence,
            'code_evidence': evidence,
        })

    return results


def generate_report(coded_errors):
    """Generate comprehensive coding report."""

    # Overall statistics
    total = len(coded_errors)
    target_hits = sum(1 for e in coded_errors if e['target_hit'])

    # Auto-coding breakdown
    code_counts = defaultdict(int)
    for e in coded_errors:
        code_counts[e['auto_code']] += 1

    # By model
    model_stats = defaultdict(lambda: {'total': 0, 'target_hits': 0, 'full_match': 0, 'partial': 0})
    for e in coded_errors:
        model = e['model']
        model_stats[model]['total'] += 1
        if e['target_hit']:
            model_stats[model]['target_hits'] += 1
        if e['auto_code'] == 'FULL_MATCH':
            model_stats[model]['full_match'] += 1
        elif e['auto_code'] == 'PARTIAL_MATCH':
            model_stats[model]['partial'] += 1

    # By condition
    cond_stats = defaultdict(lambda: {'total': 0, 'target_hits': 0, 'full_match': 0})
    for e in coded_errors:
        cond = e['condition']
        cond_stats[cond]['total'] += 1
        if e['target_hit']:
            cond_stats[cond]['target_hits'] += 1
        if e['auto_code'] == 'FULL_MATCH':
            cond_stats[cond]['full_match'] += 1

    # By misconception
    misc_stats = defaultdict(lambda: {'total': 0, 'target_hits': 0, 'full_match': 0})
    for e in coded_errors:
        misc = e['misconception_id']
        misc_stats[misc]['total'] += 1
        if e['target_hit']:
            misc_stats[misc]['target_hits'] += 1
        if e['auto_code'] == 'FULL_MATCH':
            misc_stats[misc]['full_match'] += 1

    report = []
    report.append("=" * 80)
    report.append("MISCONCEPTION CODING REPORT")
    report.append("Study 2: LLM Synthetic Student Misconception Alignment")
    report.append("=" * 80)
    report.append("")

    report.append("## 1. OVERVIEW")
    report.append(f"Total errors coded: {total}")
    report.append(f"Target distractor hits: {target_hits} ({100*target_hits/total:.1f}%)")
    report.append("")
    report.append("Auto-coding breakdown:")
    for code, count in sorted(code_counts.items()):
        report.append(f"  {code}: {count} ({100*count/total:.1f}%)")
    report.append("")

    # Key metric: Misconception alignment rate
    full_matches = code_counts.get('FULL_MATCH', 0)
    partial_matches = code_counts.get('PARTIAL_MATCH', 0)
    misconception_aligned = full_matches + partial_matches

    report.append("## 2. KEY FINDING: MISCONCEPTION ALIGNMENT")
    report.append(f"Errors with reasoning matching target misconception: {misconception_aligned}/{total} ({100*misconception_aligned/total:.1f}%)")
    report.append(f"  - Full match: {full_matches} ({100*full_matches/total:.1f}%)")
    report.append(f"  - Partial match: {partial_matches} ({100*partial_matches/total:.1f}%)")
    report.append("")
    report.append(f"Comparison to target distractor rate: {100*target_hits/total:.1f}%")
    report.append(f"Gap (distractor - misconception): {100*(target_hits - misconception_aligned)/total:.1f} percentage points")
    report.append("")

    report.append("## 3. BY MODEL")
    report.append("-" * 60)
    for model, stats in sorted(model_stats.items(), key=lambda x: -x[1]['full_match']/max(x[1]['total'],1)):
        total_m = stats['total']
        target_rate = 100 * stats['target_hits'] / total_m if total_m > 0 else 0
        full_rate = 100 * stats['full_match'] / total_m if total_m > 0 else 0
        partial_rate = 100 * stats['partial'] / total_m if total_m > 0 else 0
        misc_rate = full_rate + partial_rate
        report.append(f"{model}:")
        report.append(f"  Errors: {total_m}")
        report.append(f"  Target distractor rate: {target_rate:.1f}%")
        report.append(f"  Misconception alignment: {misc_rate:.1f}% (full: {full_rate:.1f}%, partial: {partial_rate:.1f}%)")
        report.append("")

    report.append("## 4. BY CONDITION")
    report.append("-" * 60)
    for cond, stats in sorted(cond_stats.items()):
        total_c = stats['total']
        target_rate = 100 * stats['target_hits'] / total_c if total_c > 0 else 0
        full_rate = 100 * stats['full_match'] / total_c if total_c > 0 else 0
        report.append(f"{cond}: {total_c} errors, target={target_rate:.1f}%, full_match={full_rate:.1f}%")
    report.append("")

    report.append("## 5. BY MISCONCEPTION TYPE")
    report.append("-" * 60)
    for misc, stats in sorted(misc_stats.items(), key=lambda x: -x[1]['full_match']/max(x[1]['total'],1)):
        total_m = stats['total']
        target_rate = 100 * stats['target_hits'] / total_m if total_m > 0 else 0
        full_rate = 100 * stats['full_match'] / total_m if total_m > 0 else 0
        report.append(f"{misc}: {total_m} errors, target={target_rate:.1f}%, full_match={full_rate:.1f}%")
    report.append("")

    report.append("## 6. SAMPLE CODED ERRORS (for manual verification)")
    report.append("-" * 60)

    # Show examples of each code type
    for code_type in ['FULL_MATCH', 'PARTIAL_MATCH', 'DIFFERENT_ERROR', 'UNCLEAR']:
        examples = [e for e in coded_errors if e['auto_code'] == code_type][:2]
        if examples:
            report.append(f"\n### {code_type} Examples:")
            for ex in examples:
                report.append(f"\nItem: {ex['item_id']}")
                report.append(f"Model: {ex['model']} | Condition: {ex['condition']}")
                report.append(f"Target misconception: {ex['misconception_name']}")
                report.append(f"Answer: {ex['parsed_answer']} (correct: {ex['correct_answer']}, target: {ex['target_distractor']})")
                report.append(f"Target hit: {ex['target_hit']}")
                report.append(f"Reasoning: {ex['reasoning'][:300]}...")
                report.append(f"Evidence: {ex['code_evidence']}")
                report.append("")

    return "\n".join(report), coded_errors


def main():
    # Paths
    data_dir = Path("/Users/dereklomas/AIED/study2-materials/pilot/full_collection")
    responses_file = data_dir / "batch_responses.jsonl"
    output_dir = data_dir / "misconception_coding"
    output_dir.mkdir(exist_ok=True)

    print("Loading responses...")
    responses = load_responses(responses_file)
    print(f"Total responses: {len(responses)}")

    print("Extracting errors...")
    errors = extract_errors(responses)
    print(f"Total errors: {len(errors)}")

    print("Coding misconceptions...")
    coded_errors = analyze_errors(errors)

    print("Generating report...")
    report_text, coded_data = generate_report(coded_errors)

    # Save report
    report_file = output_dir / "coding_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"Report saved to: {report_file}")

    # Save coded data as JSON for further analysis
    coded_file = output_dir / "coded_errors.json"
    with open(coded_file, 'w') as f:
        json.dump(coded_data, f, indent=2)
    print(f"Coded data saved to: {coded_file}")

    # Print report to console
    print("\n" + report_text)

    return coded_data


if __name__ == "__main__":
    main()
