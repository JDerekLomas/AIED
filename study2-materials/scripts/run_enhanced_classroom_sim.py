#!/usr/bin/env python3
"""
Enhanced Classroom Simulation
Extends Kröger et al. replication with two innovations:
1. Backstory generation: Rich student personas per ability level
2. Cognitive simulation: Flawed reasoning chains based on persona

Uses Groq (llama-3.3-70b-versatile) for both phases.
Compares against actual student error rates via neurips_correct_pos.
"""

import json
import os
import sys
import time
import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats

# =============================================================================
# CONFIG
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MODEL_ID = DEFAULT_MODEL  # overridden by --model flag
RPM = 6000
DELAY = 60.0 / RPM

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
]

ABILITY_LEVELS = {
    "below_basic": {"proportion": 0.25},
    "basic": {"proportion": 0.35},
    "proficient": {"proportion": 0.25},
    "advanced": {"proportion": 0.15},
}

PERSONAS_PER_LEVEL = 5

# =============================================================================
# BACKSTORY GENERATION PROMPTS
# =============================================================================

BACKSTORY_PROMPT = """Generate {n} distinct student personas for a UK secondary school math class. These students are all at the "{level}" ability level.

Ability level guidelines:
- below_basic: Struggles significantly, large gaps in foundational skills, frequently lost
- basic: Can do routine problems, but falters with multi-step or unfamiliar formats
- proficient: Solid understanding, occasional careless errors, handles moderate complexity
- advanced: Strong grasp of concepts, rarely wrong, can reason through novel problems

For each persona, provide a JSON array of objects with these fields:
- "name": A first name
- "math_strengths": What math topics/skills they're relatively OK at (1-2 sentences)
- "math_weaknesses": What they specifically struggle with (1-2 sentences)
- "typical_errors": The kinds of mistakes they characteristically make (1-2 sentences)
- "emotional_profile": Their emotional relationship to math — confidence, anxiety, engagement (1 sentence)
- "personal_context": Brief background that affects their learning (1 sentence)

Make the personas diverse and realistic for UK Year 9-10 students. Return ONLY a JSON array, no other text."""

# =============================================================================
# COGNITIVE SIMULATION PROMPT
# =============================================================================

COGNITIVE_SIM_PROMPT = """You are {name}, a UK secondary school student taking a maths test.

Your profile:
- Strengths: {math_strengths}
- Weaknesses: {math_weaknesses}
- Typical errors: {typical_errors}
- How you feel about maths: {emotional_profile}
- Background: {personal_context}

You must answer this multiple choice question. Think through it AS THIS STUDENT WOULD — showing your actual thought process, including any confusion, second-guessing, or mistakes that would be natural for someone with your profile. If this student would make an error, make that error naturally in the reasoning. Do NOT be artificially correct if this student would struggle.

Question: {question}

{options}

Show your thinking (2-4 sentences, staying in character), then state your final answer as "Answer: X" where X is A, B, C, or D."""

# =============================================================================
# BASELINE PROMPT (for comparison — same as original replication)
# =============================================================================

BASELINE_DESCRIPTIONS = {
    "below_basic": """You are a student who struggles significantly with math.
You often feel confused by math problems and frequently make errors.
You have difficulty with multi-step problems and often forget procedures.
You tend to guess when you're unsure, which is often.""",
    "basic": """You are a student with basic math skills.
You can handle straightforward problems but struggle with anything complex.
You sometimes mix up procedures or apply the wrong method.
You can usually get simple calculations right but make errors on harder ones.""",
    "proficient": """You are a proficient math student.
You understand most concepts and can apply procedures correctly.
You occasionally make careless errors but generally perform well.
You can handle moderately complex problems without much difficulty.""",
    "advanced": """You are an advanced math student who excels at the subject.
You rarely make errors and can tackle complex problems confidently.
You understand the underlying concepts, not just the procedures.
You might occasionally make a careless mistake but almost always get problems right.""",
}

BASELINE_PROMPT = """You are taking a math test. {ability_description}

Answer the following question. Think briefly about how you would approach it given your skill level, then give your final answer.

Question: {question}

{options}

First show your thinking (1-2 sentences), then state your answer as "Answer: X" where X is A, B, C, or D."""

# =============================================================================
# API
# =============================================================================

from groq import Groq

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def call_groq(prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> str:
    client = get_client()
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                # Extract wait time if available
                import re as _re
                wait_match = _re.search(r'(\d+)m(\d+)', err_str)
                if wait_match:
                    wait_secs = int(wait_match.group(1)) * 60 + int(wait_match.group(2))
                else:
                    wait_secs = 30 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait_secs}s...", end="", flush=True)
                time.sleep(wait_secs + 2)
            else:
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise


def parse_answer(response: str) -> str:
    # Strip qwen-style <think>...</think> tags, search both inside and outside
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    full_text = response
    if think_match:
        # Check outside think block first, fall back to inside
        outside = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if re.search(r'ANSWER:\s*([A-D])', outside.upper()):
            full_text = outside
        else:
            full_text = think_match.group(1) + " " + outside

    response_upper = full_text.upper()
    match = re.search(r'ANSWER:\s*([A-D])', response_upper)
    if match:
        return match.group(1)
    match = re.search(r'THE ANSWER IS\s*([A-D])', response_upper)
    if match:
        return match.group(1)
    match = re.search(r'\b([A-D])\)?\.?\s*$', response.strip().upper())
    if match:
        return match.group(1)
    matches = re.findall(r'\b([A-D])\b', response_upper)
    if matches:
        return matches[-1]
    return None


# =============================================================================
# PHASE 1: BACKSTORY GENERATION
# =============================================================================

def generate_backstories(output_dir: Path) -> dict:
    """Generate diverse student personas for each ability level."""
    backstory_file = output_dir / "backstories.json"

    # Resume from existing if available
    if backstory_file.exists():
        print("Loading existing backstories...")
        with open(backstory_file) as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("PHASE 1: GENERATING STUDENT BACKSTORIES")
    print("=" * 60)

    backstories = {}

    for level in ABILITY_LEVELS:
        print(f"\nGenerating {PERSONAS_PER_LEVEL} personas for {level}...")
        prompt = BACKSTORY_PROMPT.format(n=PERSONAS_PER_LEVEL, level=level)

        for attempt in range(3):
            try:
                raw = call_groq(prompt, temperature=0.9, max_tokens=2000)
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if json_match:
                    personas = json.loads(json_match.group())
                    backstories[level] = personas[:PERSONAS_PER_LEVEL]
                    print(f"  Generated {len(backstories[level])} personas")
                    for p in backstories[level]:
                        print(f"    - {p['name']}: {p['emotional_profile'][:60]}...")
                    break
                else:
                    print(f"  Attempt {attempt+1}: No JSON found, retrying...")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Attempt {attempt+1}: Parse error ({e}), retrying...")
            time.sleep(1)
        else:
            print(f"  FAILED for {level}, using fallback")
            backstories[level] = [
                {
                    "name": f"Student_{level}_{i}",
                    "math_strengths": "Basic arithmetic",
                    "math_weaknesses": "Most topics",
                    "typical_errors": "Procedural and conceptual errors",
                    "emotional_profile": "Neutral about math",
                    "personal_context": "Average UK student",
                }
                for i in range(PERSONAS_PER_LEVEL)
            ]

        time.sleep(DELAY)

    with open(backstory_file, 'w') as f:
        json.dump(backstories, f, indent=2)
    print(f"\nBackstories saved to {backstory_file}")

    return backstories


# =============================================================================
# PHASE 2: COGNITIVE SIMULATION
# =============================================================================

def run_enhanced_simulation(items: list, backstories: dict, output_dir: Path) -> list:
    """Run cognitive simulation with persona-based prompts."""
    intermediate_file = output_dir / "enhanced_intermediate.json"

    # Resume support
    done_ids = set()
    results = []
    if intermediate_file.exists():
        with open(intermediate_file) as f:
            results = json.load(f)
        done_ids = {r['item_id'] for r in results}
        print(f"Resuming: {len(done_ids)} items already done")

    print(f"\n{'='*60}")
    print("PHASE 2: ENHANCED COGNITIVE SIMULATION")
    print(f"Items: {len(items)}, Personas: {sum(len(v) for v in backstories.values())}")
    print('='*60)

    for i, item in enumerate(items):
        if item['id'] in done_ids:
            continue

        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        all_responses = []
        level_responses = {level: [] for level in ABILITY_LEVELS}
        reasoning_traces = []

        for level_name in ABILITY_LEVELS:
            personas = backstories.get(level_name, [])
            print(f"  {level_name} ({len(personas)} personas): ", end="", flush=True)

            for persona in personas:
                prompt = COGNITIVE_SIM_PROMPT.format(
                    name=persona['name'],
                    math_strengths=persona['math_strengths'],
                    math_weaknesses=persona['math_weaknesses'],
                    typical_errors=persona['typical_errors'],
                    emotional_profile=persona['emotional_profile'],
                    personal_context=persona['personal_context'],
                    question=item['question'],
                    options=item['options'],
                )

                try:
                    raw = call_groq(prompt, temperature=0.7, max_tokens=300)
                    answer = parse_answer(raw)

                    if answer:
                        all_responses.append({
                            "level": level_name,
                            "persona": persona['name'],
                            "answer": answer,
                            "weight": ABILITY_LEVELS[level_name]['proportion'],
                        })
                        level_responses[level_name].append(answer)
                        reasoning_traces.append({
                            "level": level_name,
                            "persona": persona['name'],
                            "answer": answer,
                            "reasoning": raw,
                        })
                        print(answer, end="", flush=True)
                    else:
                        print("?", end="", flush=True)

                    time.sleep(DELAY)

                except Exception as e:
                    print("E", end="", flush=True)
                    time.sleep(DELAY * 3)

            print()

        # Weighted aggregation
        weighted_counts = defaultdict(float)
        for resp in all_responses:
            weighted_counts[resp['answer']] += resp['weight']
        total_weight = sum(weighted_counts.values())
        if total_weight > 0:
            sim_dist = {opt: weighted_counts.get(opt, 0) / total_weight for opt in ['A', 'B', 'C', 'D']}
        else:
            sim_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        # Unweighted
        unweighted_counts = defaultdict(int)
        for resp in all_responses:
            unweighted_counts[resp['answer']] += 1
        total_uw = sum(unweighted_counts.values())
        if total_uw > 0:
            uw_dist = {opt: unweighted_counts.get(opt, 0) / total_uw for opt in ['A', 'B', 'C', 'D']}
        else:
            uw_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        correct = item['correct']
        result = {
            "item_id": item['id'],
            "correct_answer": correct,
            "student_difficulty": item['student_difficulty'],
            "enhanced_difficulty_weighted": 1 - sim_dist.get(correct, 0),
            "enhanced_difficulty_unweighted": 1 - uw_dist.get(correct, 0),
            "student_dist": item['student_dist'],
            "enhanced_dist_weighted": sim_dist,
            "enhanced_dist_unweighted": uw_dist,
            "level_responses": {k: list(v) for k, v in level_responses.items()},
            "reasoning_traces": reasoning_traces,
            "n_responses": len(all_responses),
        }
        results.append(result)

        print(f"  Student difficulty: {item['student_difficulty']:.2f}")
        print(f"  Enhanced difficulty (weighted): {result['enhanced_difficulty_weighted']:.2f}")

        # Save intermediate
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


# =============================================================================
# PHASE 2b: BASELINE (for direct comparison)
# =============================================================================

def run_baseline_simulation(items: list, output_dir: Path) -> list:
    """Run the original generic-prompt simulation for comparison."""
    intermediate_file = output_dir / "baseline_intermediate.json"

    done_ids = set()
    results = []
    if intermediate_file.exists():
        with open(intermediate_file) as f:
            results = json.load(f)
        done_ids = {r['item_id'] for r in results}
        print(f"Resuming baseline: {len(done_ids)} items already done")

    print(f"\n{'='*60}")
    print("BASELINE SIMULATION (generic prompts, same model)")
    print('='*60)

    for i, item in enumerate(items):
        if item['id'] in done_ids:
            continue

        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        all_responses = []
        level_responses = {level: [] for level in ABILITY_LEVELS}

        for level_name, level_config in ABILITY_LEVELS.items():
            print(f"  {level_name}: ", end="", flush=True)

            for _ in range(PERSONAS_PER_LEVEL):
                prompt = BASELINE_PROMPT.format(
                    ability_description=BASELINE_DESCRIPTIONS[level_name],
                    question=item['question'],
                    options=item['options'],
                )

                try:
                    raw = call_groq(prompt, temperature=0.7, max_tokens=200)
                    answer = parse_answer(raw)

                    if answer:
                        all_responses.append({
                            "level": level_name,
                            "answer": answer,
                            "weight": level_config['proportion'],
                        })
                        level_responses[level_name].append(answer)
                        print(answer, end="", flush=True)
                    else:
                        print("?", end="", flush=True)

                    time.sleep(DELAY)

                except Exception as e:
                    print("E", end="", flush=True)
                    time.sleep(DELAY * 3)

            print()

        # Weighted
        weighted_counts = defaultdict(float)
        for resp in all_responses:
            weighted_counts[resp['answer']] += resp['weight']
        total_weight = sum(weighted_counts.values())
        if total_weight > 0:
            sim_dist = {opt: weighted_counts.get(opt, 0) / total_weight for opt in ['A', 'B', 'C', 'D']}
        else:
            sim_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        # Unweighted
        uw_counts = defaultdict(int)
        for resp in all_responses:
            uw_counts[resp['answer']] += 1
        total_uw = sum(uw_counts.values())
        if total_uw > 0:
            uw_dist = {opt: uw_counts.get(opt, 0) / total_uw for opt in ['A', 'B', 'C', 'D']}
        else:
            uw_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        correct = item['correct']
        result = {
            "item_id": item['id'],
            "correct_answer": correct,
            "student_difficulty": item['student_difficulty'],
            "baseline_difficulty_weighted": 1 - sim_dist.get(correct, 0),
            "baseline_difficulty_unweighted": 1 - uw_dist.get(correct, 0),
            "student_dist": item['student_dist'],
            "baseline_dist_weighted": sim_dist,
            "baseline_dist_unweighted": uw_dist,
            "level_responses": {k: list(v) for k, v in level_responses.items()},
            "n_responses": len(all_responses),
        }
        results.append(result)

        print(f"  Student difficulty: {item['student_difficulty']:.2f}")
        print(f"  Baseline difficulty (weighted): {result['baseline_difficulty_weighted']:.2f}")

        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eedi_items(csv_path: str, limit: int = None) -> list:
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        if 'AnswerAText' not in row or pd.isna(row.get('neurips_correct_pos')):
            continue

        options = f"A) {row['AnswerAText']}\nB) {row['AnswerBText']}\nC) {row['AnswerCText']}\nD) {row['AnswerDText']}"
        correct = row['CorrectAnswer']  # Kaggle letter for LLM prompt
        neurips_correct = row['neurips_correct_pos']
        correct_pct = row[f'pct_{neurips_correct}'] / 100
        difficulty = 1 - correct_pct

        items.append({
            "id": int(row['QuestionId']),
            "question": row['QuestionText'],
            "options": options,
            "correct": correct,
            "student_dist": {
                "A": row['pct_A'] / 100,
                "B": row['pct_B'] / 100,
                "C": row['pct_C'] / 100,
                "D": row['pct_D'] / 100,
            },
            "student_difficulty": difficulty,
            "total_responses": int(row['total_responses']),
        })
        if limit and len(items) >= limit:
            break
    return items


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(enhanced: list, baseline: list, output_dir: Path) -> dict:
    """Compare enhanced vs baseline simulation."""
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: Enhanced vs Baseline")
    print('='*60)

    # Match items by ID
    baseline_map = {r['item_id']: r for r in baseline}
    matched = [(e, baseline_map[e['item_id']]) for e in enhanced if e['item_id'] in baseline_map]

    if not matched:
        print("No matched items!")
        return {}

    student_diff = [e['student_difficulty'] for e, _ in matched]
    enh_diff = [e['enhanced_difficulty_weighted'] for e, _ in matched]
    base_diff = [b['baseline_difficulty_weighted'] for _, b in matched]

    # Pearson
    r_enh, p_enh = stats.pearsonr(enh_diff, student_diff)
    r_base, p_base = stats.pearsonr(base_diff, student_diff)

    # Spearman
    rho_enh, rho_p_enh = stats.spearmanr(enh_diff, student_diff)
    rho_base, rho_p_base = stats.spearmanr(base_diff, student_diff)

    # RMSE / MAE
    rmse_enh = np.sqrt(np.mean((np.array(enh_diff) - np.array(student_diff))**2))
    rmse_base = np.sqrt(np.mean((np.array(base_diff) - np.array(student_diff))**2))
    mae_enh = np.mean(np.abs(np.array(enh_diff) - np.array(student_diff)))
    mae_base = np.mean(np.abs(np.array(base_diff) - np.array(student_diff)))

    print(f"\nMatched items: {len(matched)}")
    print(f"\n{'Metric':<25} {'Enhanced':>12} {'Baseline':>12} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Pearson r':<25} {r_enh:>12.3f} {r_base:>12.3f} {r_enh - r_base:>+10.3f}")
    print(f"{'Spearman rho':<25} {rho_enh:>12.3f} {rho_base:>12.3f} {rho_enh - rho_base:>+10.3f}")
    print(f"{'RMSE':<25} {rmse_enh:>12.3f} {rmse_base:>12.3f} {rmse_enh - rmse_base:>+10.3f}")
    print(f"{'MAE':<25} {mae_enh:>12.3f} {mae_base:>12.3f} {mae_enh - mae_base:>+10.3f}")
    print(f"\nBenchmark (Kröger et al.): r=0.75-0.82")

    # Distribution correlation
    enh_probs, base_probs, student_probs = [], [], []
    for e, b in matched:
        for opt in ['A', 'B', 'C', 'D']:
            enh_probs.append(e['enhanced_dist_weighted'].get(opt, 0))
            base_probs.append(b['baseline_dist_weighted'].get(opt, 0))
            student_probs.append(e['student_dist'].get(opt, 0))

    dist_r_enh, _ = stats.pearsonr(enh_probs, student_probs)
    dist_r_base, _ = stats.pearsonr(base_probs, student_probs)
    print(f"\nDistribution correlation (all options):")
    print(f"  Enhanced: r={dist_r_enh:.3f}")
    print(f"  Baseline: r={dist_r_base:.3f}")

    # Per-level accuracy analysis (enhanced only)
    print(f"\nPer-level accuracy (Enhanced):")
    for level in ABILITY_LEVELS:
        correct_count = 0
        total_count = 0
        for e, _ in matched:
            for ans in e['level_responses'].get(level, []):
                total_count += 1
                if ans == e['correct_answer']:
                    correct_count += 1
        if total_count > 0:
            print(f"  {level}: {correct_count}/{total_count} = {correct_count/total_count:.1%}")

    analysis = {
        "n_items": len(matched),
        "enhanced": {
            "pearson_r": float(r_enh), "pearson_p": float(p_enh),
            "spearman_rho": float(rho_enh), "spearman_p": float(rho_p_enh),
            "rmse": float(rmse_enh), "mae": float(mae_enh),
            "distribution_r": float(dist_r_enh),
        },
        "baseline": {
            "pearson_r": float(r_base), "pearson_p": float(p_base),
            "spearman_rho": float(rho_base), "spearman_p": float(rho_p_base),
            "rmse": float(rmse_base), "mae": float(mae_base),
            "distribution_r": float(dist_r_base),
        },
        "delta": {
            "pearson_r": float(r_enh - r_base),
            "spearman_rho": float(rho_enh - rho_base),
            "rmse": float(rmse_enh - rmse_base),
            "mae": float(mae_enh - mae_base),
            "distribution_r": float(dist_r_enh - dist_r_base),
        },
    }

    with open(output_dir / "comparative_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {output_dir / 'comparative_analysis.json'}")

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Classroom Simulation")
    script_dir = Path(__file__).parent.parent
    parser.add_argument("--data", type=str,
                       default=str(script_dir / "data/eedi/eedi_with_student_data.csv"))
    parser.add_argument("--output", type=str,
                       default=str(script_dir / "pilot/replications/enhanced_classroom_sim"))
    parser.add_argument("--items", type=int, default=None, help="Limit items (default: all)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline, only run enhanced")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    parser.add_argument("--students", type=int, default=5,
                       help="Personas per ability level (default: 5)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       choices=AVAILABLE_MODELS, help=f"Groq model (default: {DEFAULT_MODEL})")
    parser.add_argument("--fresh", action="store_true",
                       help="Delete intermediate files and start fresh")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    global PERSONAS_PER_LEVEL, MODEL_ID
    PERSONAS_PER_LEVEL = args.students
    MODEL_ID = args.model

    if args.fresh:
        for f in ['backstories.json', 'enhanced_intermediate.json', 'baseline_intermediate.json']:
            p = output_dir / f
            if p.exists():
                p.unlink()
                print(f"Removed {p}")

    if args.analyze_only:
        enh_file = output_dir / "enhanced_intermediate.json"
        base_file = output_dir / "baseline_intermediate.json"
        if enh_file.exists() and base_file.exists():
            with open(enh_file) as f:
                enhanced = json.load(f)
            with open(base_file) as f:
                baseline = json.load(f)
            analyze_results(enhanced, baseline, output_dir)
        else:
            print("Missing result files for analysis")
        return

    print("=" * 60)
    print("ENHANCED CLASSROOM SIMULATION")
    print("=" * 60)
    print(f"Model: {MODEL_ID} (Groq)")
    print(f"Personas per level: {PERSONAS_PER_LEVEL}")
    print(f"Total simulated per item: {PERSONAS_PER_LEVEL * len(ABILITY_LEVELS)}")
    print(f"Item limit: {args.items or 'all'}")

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")
    if not items:
        print("No items loaded!")
        return

    # Phase 1: Generate backstories
    backstories = generate_backstories(output_dir)

    # Phase 2a: Enhanced simulation
    enhanced_results = run_enhanced_simulation(items, backstories, output_dir)

    # Save final enhanced results
    with open(output_dir / "enhanced_results.json", 'w') as f:
        json.dump(enhanced_results, f, indent=2)

    if not args.skip_baseline:
        # Phase 2b: Baseline simulation
        baseline_results = run_baseline_simulation(items, output_dir)

        with open(output_dir / "baseline_results.json", 'w') as f:
            json.dump(baseline_results, f, indent=2)

        # Phase 3: Comparative analysis
        analyze_results(enhanced_results, baseline_results, output_dir)
    else:
        # Analyze enhanced only
        student_diff = [r['student_difficulty'] for r in enhanced_results]
        enh_diff = [r['enhanced_difficulty_weighted'] for r in enhanced_results]
        r, p = stats.pearsonr(enh_diff, student_diff)
        rho, _ = stats.spearmanr(enh_diff, student_diff)
        print(f"\nEnhanced only: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={rho:.3f}")


if __name__ == "__main__":
    main()
