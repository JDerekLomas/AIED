#!/usr/bin/env python3
"""
Student Simulation Experiment for EEDI Items

Uses teacher-prediction prompting with realistic UK student profiles.
Multi-model ensemble: weaker models for struggling students, stronger for confident.

Key design choices (from literature):
- Teacher-prediction framing, not direct role-play
- One question per API call
- Bidirectional confusion examples in profile
- Same prompt structure for all models (no cheating)
"""

import json
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
import random
from dataclasses import dataclass
from typing import Optional
import re

# API clients
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq

# Configuration
OUTPUT_DIR = Path("pilot/student_simulation")
FINAL_ITEMS_PATH = Path("results/final_items.json")
STUDENT_PROFILES_PATH = Path("data/uk_student_profiles.json")

# Model mapping by proficiency
MODEL_CONFIG = {
    "struggling": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "developing": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "secure": {"provider": "openai", "model": "gpt-4o-mini"},
    "confident": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
}

@dataclass
class SimulationResult:
    student_id: str
    student_name: str
    proficiency: str
    question_id: int
    misconception_id: int
    selected_answer: str
    correct_answer: str
    target_distractor: str
    is_correct: bool
    selected_target: bool
    reasoning: str
    model: str
    provider: str
    timestamp: str
    latency_ms: int


def load_items():
    """Load the 36 final items, excluding empirical statistics."""
    with open(FINAL_ITEMS_PATH) as f:
        data = json.load(f)

    items = []
    for item in data["items"]:
        # Keep only information a student would see + misconception name for analysis
        clean_item = {
            "question_id": item["question_id"],
            "question_text": item["question_text"],
            "option_A": item["option_A"],
            "option_B": item["option_B"],
            "option_C": item["option_C"],
            "option_D": item["option_D"],
            "correct_answer": item["correct_answer"],
            "target_distractor": item["target_distractor"],
            "misconception_id": item["misconception_id"],
            "misconception_name": item["misconception_name"],
            # DO NOT include: pct_A, pct_B, pct_C, pct_D, total_responses, human_target_rate
        }
        items.append(clean_item)

    return items, data["misconceptions"]


def load_students():
    """Load UK student profiles."""
    with open(STUDENT_PROFILES_PATH) as f:
        data = json.load(f)
    return data["students"]


def build_student_prompt(student: dict, item: dict) -> str:
    """
    Build teacher-prediction prompt.

    Key elements:
    - Teacher predicting student behavior (not direct role-play)
    - Student profile with bidirectional examples
    - Single question format
    - Request for letter answer + brief reasoning
    """
    # Build confusion examples string
    confusion_str = ""
    for pair in student.get("confused_pairs", []):
        if pair.get("misconception_id"):
            wrong = pair["example_wrong"]
            correct = pair["example_partial_correct"]
            confusion_str += f"""
  - Misconception: {pair['description']}
    When they get it WRONG: Q: "{wrong['question']}" → They answered: {wrong['student_answer']} (Correct: {wrong['correct_answer']})
    When they get it RIGHT: Q: "{correct['question']}" → They answered: {correct['student_answer']} (Correct: {correct['correct_answer']})"""

    prompt = f"""You are a UK secondary school maths teacher predicting how a specific student will answer a diagnostic question.

STUDENT PROFILE:
Name: {student['name']}
Year Group: {student['year_group']} (age {student['age']})
School: {student['school_type']}
Target Grade: {student['target_grade']} ({student['tier']} tier)
Background: {student['background']}

Topics they have mastered:
{chr(10).join('- ' + m for m in student.get('mastered', []))}

Known areas of confusion (with examples of when they get things wrong AND right):
{confusion_str if confusion_str else '- Generally accurate, occasional careless errors'}

Topics not yet assessed: {', '.join(student.get('unassessed_topics', []))}

QUESTION:
{item['question_text']}

A) {item['option_A']}
B) {item['option_B']}
C) {item['option_C']}
D) {item['option_D']}

Based on your knowledge of {student['name']}'s mathematical understanding, which answer would they MOST LIKELY select?

Respond with:
1. The letter (A, B, C, or D) they would choose
2. A brief explanation of their likely reasoning (1-2 sentences)

Format your response as:
ANSWER: [letter]
REASONING: [explanation]"""

    return prompt


def parse_response(response_text: str) -> tuple[str, str]:
    """Extract answer letter and reasoning from model response."""
    answer = ""
    reasoning = ""

    # Look for ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*([A-Da-d])', response_text, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).upper()
    else:
        # Fallback: look for standalone letter
        letter_match = re.search(r'\b([A-Da-d])\)', response_text)
        if letter_match:
            answer = letter_match.group(1).upper()

    # Look for REASONING: pattern
    reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Use everything after the answer as reasoning
        reasoning = response_text.strip()

    return answer, reasoning


class ModelClient:
    """Unified interface for different API providers."""

    def __init__(self):
        self.anthropic = Anthropic()
        self.openai = OpenAI()
        self.groq = Groq()

    def call(self, provider: str, model: str, prompt: str) -> tuple[str, int]:
        """Call model and return (response_text, latency_ms)."""
        start = time.time()

        if provider == "anthropic":
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text

        elif provider == "openai":
            response = self.openai.chat.completions.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content

        elif provider == "groq":
            response = self.groq.chat.completions.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {provider}")

        latency = int((time.time() - start) * 1000)
        return text, latency


def run_simulation(
    items: list,
    students: list,
    output_path: Path,
    n_items_per_student: Optional[int] = None,
    seed: int = 42
):
    """
    Run the full simulation.

    Each student answers each question once.
    Model is selected based on student proficiency.
    """
    random.seed(seed)
    client = ModelClient()

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Optionally limit items per student for testing
    if n_items_per_student:
        test_items = random.sample(items, min(n_items_per_student, len(items)))
    else:
        test_items = items

    total_calls = len(students) * len(test_items)
    call_num = 0

    print(f"Running simulation: {len(students)} students × {len(test_items)} items = {total_calls} calls")
    print(f"Output: {output_path}")
    print()

    for student in students:
        config = MODEL_CONFIG[student["proficiency"]]
        provider = config["provider"]
        model = config["model"]

        print(f"Student: {student['name']} ({student['proficiency']}) → {model}")

        for item in test_items:
            call_num += 1

            prompt = build_student_prompt(student, item)

            try:
                response_text, latency = client.call(provider, model, prompt)
                answer, reasoning = parse_response(response_text)

                result = SimulationResult(
                    student_id=student["id"],
                    student_name=student["name"],
                    proficiency=student["proficiency"],
                    question_id=item["question_id"],
                    misconception_id=item["misconception_id"],
                    selected_answer=answer,
                    correct_answer=item["correct_answer"],
                    target_distractor=item["target_distractor"],
                    is_correct=(answer == item["correct_answer"]),
                    selected_target=(answer == item["target_distractor"]),
                    reasoning=reasoning[:500],  # Truncate long reasoning
                    model=model,
                    provider=provider,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency
                )
                results.append(result)

                status = "✓" if result.is_correct else ("T" if result.selected_target else "✗")
                print(f"  [{call_num}/{total_calls}] Q{item['question_id']}: {answer} ({status}) {latency}ms")

            except Exception as e:
                print(f"  [{call_num}/{total_calls}] Q{item['question_id']}: ERROR - {e}")
                continue

            # Rate limiting
            if provider == "groq":
                time.sleep(0.1)  # Groq has aggressive rate limits
            else:
                time.sleep(0.05)

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"results_{timestamp}.jsonl"

    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "n_students": len(students),
        "n_items": len(test_items),
        "n_calls": len(results),
        "models_used": {p: MODEL_CONFIG[p] for p in set(s["proficiency"] for s in students)},
        "accuracy": sum(r.is_correct for r in results) / len(results) if results else 0,
        "target_rate": sum(r.selected_target for r in results) / len(results) if results else 0,
    }

    summary_file = output_path / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Results saved to {results_file}")
    print(f"Summary: {len(results)} responses, {summary['accuracy']:.1%} correct, {summary['target_rate']:.1%} target misconception")

    return results, summary


def main():
    """Main entry point."""
    print("Loading data...")
    items, misconceptions = load_items()
    students = load_students()

    print(f"Loaded {len(items)} items across {len(misconceptions)} misconceptions")
    print(f"Loaded {len(students)} students")
    print()

    # Run full simulation
    results, summary = run_simulation(
        items=items,
        students=students,
        output_path=OUTPUT_DIR,
        n_items_per_student=None,  # Set to small number for testing
    )

    # Print breakdown by proficiency
    print("\nBreakdown by proficiency:")
    for prof in ["struggling", "developing", "secure", "confident"]:
        prof_results = [r for r in results if r.proficiency == prof]
        if prof_results:
            acc = sum(r.is_correct for r in prof_results) / len(prof_results)
            target = sum(r.selected_target for r in prof_results) / len(prof_results)
            print(f"  {prof}: {len(prof_results)} responses, {acc:.1%} correct, {target:.1%} target")


if __name__ == "__main__":
    main()
