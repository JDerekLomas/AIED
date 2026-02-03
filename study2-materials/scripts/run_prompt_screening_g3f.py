#!/usr/bin/env python3
"""
Prompt Framing Experiment — Gemini 3 Flash, Full 140 Items
==========================================================
7 framings × 2 temps × 3 reps × 140 items = 5,880 calls.
No base-rate leakage. Clean comparison on same model.

Framing 0: teacher (current best, ρ=0.550 on 140 — the baseline to beat)
Framing 1: error_analysis (System 1 error-focused, best non-teacher existing)
Framing 2: devil_advocate (bias correction framing)
Framing 3: prerequisite_chain (KC Theory — count failure points)
Framing 4: error_affordance (BUGGY — count plausible error paths)
Framing 5: cognitive_load (Sweller CLT — element interactivity)
Framing 6: familiarity_gradient (distance from textbook drills)

Dropped from prior version:
- contrastive: ρ=0.077 on 140 items despite ρ=0.694 on 20 probes
- imagine_classroom: similar to teacher but weaker
- comparative_difficulty: difficulty rating → % is an extra indirection
- population_aware: subsumes into all prompts via population paragraph

Design decisions:
- No "29% average pass rate" — that's ground truth leakage
- Population context kept (government school, EWS, Hindi-medium) — that's public knowledge
- Output format: "0.XX" not "XX%" — matches Phase 2, better parse rate
- 3 reps not 5 — Gemini is very consistent (σ=0.005 across reps)
- Full 140 items — 20-probe screening doesn't transfer (contrastive proved this)

Usage:
  python3 scripts/run_prompt_screening_g3f.py
  python3 scripts/run_prompt_screening_g3f.py --framing teacher
  python3 scripts/run_prompt_screening_g3f.py --analyze
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

OUTPUT_DIR = Path("pilot/prompt_framing_experiment")
MODEL = "gemini-3-flash-preview"  # overridden by --model
N_REPS = 3
TEMPERATURES = [1.0, 2.0]

POPULATION = ("Your students are mostly from economically weaker sections. "
              "Many come from Hindi-medium backgrounds and have limited English proficiency.")


def load_items():
    path = Path("data/smartpaper/item_statistics.json")
    return json.loads(path.read_text())


def item_key(item):
    return f"{item['assessment']}_q{item['question_number']}"


def call_llm(prompt, temperature=2.0, max_tokens=512):
    if MODEL.startswith("gemini"):
        return _call_gemini(prompt, temperature, max_tokens)
    else:
        return _call_groq(prompt, temperature, max_tokens)


def _call_gemini(prompt, temperature, max_tokens=512):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            return response.text
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2 ** attempt)
    return ""


def _call_groq(prompt, temperature, max_tokens=512):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=min(temperature, 2.0),
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2 ** attempt)
    return ""


def parse_proportion(text):
    """Parse a proportion (0-1) from LLM output. Tries multiple formats."""
    if not text:
        return None
    text = text.strip()

    # Direct decimal (e.g., "0.45" or "Your estimate: 0.45")
    m = re.search(r'(?:estimate|answer|proportion|correct)\s*:?\s*(0\.\d+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # "ESTIMATED PROPORTION CORRECT: XX%" format (theory prompts)
    m = re.search(r'PROPORTION CORRECT\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0

    # "CORRECT: XX%" format
    m = re.search(r'CORRECT\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0

    # Bare decimal on its own line (e.g., just "0.35")
    m = re.search(r'^(0\.\d+)$', text, re.MULTILINE)
    if m:
        return float(m.group(1))

    # Last percentage in text
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        return float(matches[-1]) / 100.0

    # Last bare decimal
    matches = re.findall(r'\b(0\.\d+)\b', text)
    if matches:
        return float(matches[-1])

    return None


# ============================================================
# PROMPTS — no base-rate leakage
# ============================================================

def prompt_teacher(item):
    """Clean teacher baseline — matches Phase 2 model survey format."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in India.

For this open-ended question, estimate what proportion of students would score full marks.

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

Think about:
- What specific errors or misunderstandings would cause students to lose marks?
- How clearly does the question communicate what's expected?
- What prerequisite knowledge is needed?
- How likely are students at this grade level to have that knowledge?

Respond with ONLY a number between 0 and 1 representing the proportion of students who would get full marks.
For example: 0.45

Your estimate:"""


def prompt_error_analysis(item):
    """Error-focused framing — strongest non-teacher existing prompt."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

For this question, think about what ERRORS would lead students to get this wrong:
- What misconceptions about {item['subject']} would cause incorrect answers?
- What language misunderstandings could occur?
- Would students attempt but fail, or not attempt at all?
- What would the most common wrong answer look like?

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

After analyzing likely errors, estimate what proportion would get full marks.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


def prompt_devil_advocate(item):
    """Bias correction — challenges expert overconfidence."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

IMPORTANT: Teachers and experts consistently OVERESTIMATE how well students will do because they forget:
- How limited many students' English actually is
- That "basic" concepts aren't basic for struggling learners
- That exam anxiety causes blank responses
- That many students have never practiced this exact question type

Challenge your first instinct. If you think "most students could do this," ask yourself: could a student who barely reads English? Could a student who was absent for this topic?

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

After challenging your assumptions, estimate what proportion would get full marks.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


def prompt_prerequisite_chain(item):
    """KC Theory — count prerequisite failure points."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

For this question, identify the prerequisite knowledge and skills a student needs. Count how many independent things must ALL go right for a correct answer. Each prerequisite is a potential failure point.

Examples of prerequisites: reading comprehension, specific vocabulary, a math operation, a concept definition, multi-step reasoning, writing ability.

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

List the prerequisites, then estimate what proportion would get full marks.

PREREQUISITES: [list them]
COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_error_affordance(item):
    """BUGGY — count plausible error paths."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

Count the number of distinct, plausible ways a student could get this wrong:
- Procedural slips (computation errors, skipped steps)
- Conceptual confusions (wrong rule applied)
- Language barriers (misunderstanding the question)
- Partial knowledge (knows some but not all required steps)
- Non-attempts (leaves blank, writes something irrelevant)

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

List the plausible errors, then estimate what proportion would get full marks.

PLAUSIBLE ERRORS: [list them]
ERROR COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_cognitive_load(item):
    """Sweller's CLT — element interactivity."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

Rate the element interactivity of this question: how many pieces of information must a student hold in working memory simultaneously to solve it?

Elements include: numbers to manipulate, rules to apply, vocabulary to recall, sentence structures to parse, steps in a procedure, constraints to satisfy.

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

List the elements, then estimate what proportion would get full marks.

ELEMENTS: [list them]
ELEMENT COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_familiarity_gradient(item):
    """Conceptual change — distance from textbook drills."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

How typical is this question compared to standard textbook exercises for Grade {item['grade']} {item['subject']} in Indian government schools?

Consider: does it match textbook drills exactly, or require transfer, novel phrasing, or non-obvious connections?

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

Rate typicality, then estimate what proportion would get full marks.

TYPICALITY: [very typical / slight variation / novel / requires insight]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_verbalized_sampling(item):
    """Verbalized Sampling — ask for multiple estimates with confidence weights."""
    return f"""You are an experienced teacher in {item['subject']} for Grade {item['grade']} students in Indian government schools.

{POPULATION}

For this question, generate 5 INDEPENDENT estimates of what proportion of students would get full marks. For each estimate, adopt a slightly different perspective:

1. As an optimistic teacher who believes in student potential
2. As a pessimistic teacher who has seen many students struggle
3. As a data analyst looking at similar items' pass rates
4. As a curriculum designer who knows what was taught
5. As a student's parent who sees homework struggles

Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

For each perspective, give a proportion (0 to 1) and a confidence weight (how reliable you think this perspective is, 0 to 1):

ESTIMATE 1: [proportion] (weight: [w])
ESTIMATE 2: [proportion] (weight: [w])
ESTIMATE 3: [proportion] (weight: [w])
ESTIMATE 4: [proportion] (weight: [w])
ESTIMATE 5: [proportion] (weight: [w])

WEIGHTED AVERAGE: [single number 0 to 1]

Respond with ONLY the weighted average on the last line.
For example: 0.45

Your estimate:"""


def prompt_teacher_decomposed(item):
    """Teacher judgment decomposed by proficiency level — from RSM v2.
    Asks teacher to estimate per-level pass rates, then we weight them."""
    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

{POPULATION} The school has mixed ability levels, with about 35% struggling significantly, 35% at basic level, 20% competent, and 10% advanced.

Question: {item['question_text']}
Scoring rubric: {item['rubric']}
Max score: {item['max_score']}

Based on your experience, what percentage of students AT EACH LEVEL would score full marks on this item?

Reply in this EXACT format (numbers only, no explanation):
struggling: XX%
basic: XX%
competent: XX%
advanced: XX%"""


def prompt_classroom_sim(item):
    """Classroom simulation — LLM role-plays 20 students at 4 proficiency levels
    answering the question, then estimates overall pass rate from the simulation."""
    return f"""Imagine a classroom of 20 Class {item['grade']} students in an Indian government school taking a paper exam. {POPULATION}

The class has:
- 7 struggling students (barely read English, often leave blank or copy irrelevant text)
- 7 basic students (read Hindi well, weak English, handle straightforward recall)
- 4 competent students (understand most concepts, some grammar errors)
- 2 advanced students (deep understanding, clear answers, rarely make errors)

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

Mentally simulate each student attempting this question. How many of the 20 would score full marks?

Respond with ONLY a number between 0 and 1 representing the proportion who would get full marks.
For example: 0.45

Your estimate:"""


def parse_decomposed(text):
    """Parse proficiency-decomposed output and return weighted proportion."""
    if not text:
        return None
    weights = {"struggling": 0.35, "basic": 0.35, "competent": 0.20, "advanced": 0.10}
    level_scores = []
    for level, weight in weights.items():
        pattern = rf'{level}\s*:\s*(\d+(?:\.\d+)?)\s*%'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            level_scores.append((float(match.group(1)) / 100.0, weight))
    if level_scores:
        return sum(p * w for p, w in level_scores) / sum(w for _, w in level_scores)
    # Fallback to standard parser
    return None


def prompt_contrastive(item):
    """Contrastive framing — compare to similar content. Clean (no base-rate leak)."""
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would score full marks on this question.

{POPULATION}

Think about what makes this SPECIFIC question easy or hard compared to similar {item['subject']} content at the Class {item['grade']} level:
- What prior knowledge is assumed?
- What language demands does it place on students with limited English?
- Is this recall, application, or reasoning?
- How does it compare in difficulty to typical textbook exercises?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

After your contrastive analysis, respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


def prompt_imagine_classroom(item):
    """Imagine classroom framing — trust classroom instinct. Clean (no base-rate leak)."""
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would score full marks on this question.

{POPULATION}

Picture the moment after students submit their exam papers in a government school classroom of 40 students:
- The back-benchers who barely opened their papers
- The students who tried hard but made errors
- The few who wrote confident, correct answers
- The ones who copied something irrelevant

Trust your classroom instinct. How many papers would have a correct answer for this question?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:"""


# ============================================================
# Synthetic Student Simulation (two-stage)
# ============================================================

PERSONA_GENERATION_PROMPT = f"""Generate 10 diverse student profiles for a Class {{grade}} government school in India. {POPULATION}

The class distribution should reflect a typical government school:
- 4 students: Below Basic (barely literate in English, struggle with basic concepts)
- 3 students: Basic (can read Hindi well, handle simple recall, weak English)
- 2 students: Competent (understand most concepts, some errors)
- 1 student: Advanced (strong understanding, rarely makes mistakes)

For each student, provide a short profile in this exact format:
STUDENT 1: [Name] | Level: [Below Basic/Basic/Competent/Advanced] | [2-3 specific traits: reading level, attention, home language, study habits, specific strengths/weaknesses]
STUDENT 2: ...
...
STUDENT 10: ...

Make each student distinct and realistic. Include a mix of genders and backgrounds."""


def generate_personas(grade, config_dir):
    """Generate or load cached student personas."""
    persona_path = config_dir / "personas.txt"
    if persona_path.exists():
        return persona_path.read_text()
    prompt = PERSONA_GENERATION_PROMPT.format(grade=grade)
    text = call_llm(prompt, temperature=1.0, max_tokens=2048)
    if text:
        persona_path.write_text(text)
    return text


def prompt_synthetic_student(item, persona_line):
    """Prompt for one student attempting one item."""
    return f"""You are role-playing as this student:
{persona_line}

This student is in Class {item['grade']} at an Indian government school and is taking a written exam.

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

Role-play this specific student attempting this question. Consider their reading ability, knowledge level, attention, and typical behaviors. Write their actual response as they would write it, then score it.

Format your answer as:
RESPONSE: [what this student would actually write]
SCORE: [0 to {item['max_score']}]"""


def parse_student_score(text, max_score):
    """Parse a student simulation response to get score."""
    if not text:
        return None
    m = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
    if m:
        score = float(m.group(1))
        return min(score / float(max_score), 1.0)  # normalize to 0-1
    return None


def run_synthetic_students(temp=2.0):
    """Two-stage synthetic student simulation.
    Stage 1: Generate 10 student personas (cached, one-time)
    Stage 2: Each student attempts each item, aggregate pass rates
    """
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_name = f"synthetic_students_t{temp}"
    config_dir = OUTPUT_DIR / config_name
    config_dir.mkdir(exist_ok=True)

    # Stage 1: Generate personas (use grade from first item, all same grade)
    grades = set(item['grade'] for item in items)
    all_personas = {}
    for grade in grades:
        grade_dir = config_dir / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)
        persona_text = generate_personas(grade, grade_dir)
        # Parse persona lines
        lines = [l.strip() for l in persona_text.split('\n') if re.match(r'\**STUDENT\s+\d', l.strip())]
        all_personas[grade] = lines
        print(f"  Grade {grade}: {len(lines)} personas generated", flush=True)

    # Stage 2: Each student attempts each item
    n_api = 0
    n_parse_fail = 0
    item_estimates = {}

    for i, item in enumerate(items):
        ik = item_key(item)
        grade = item['grade']
        personas = all_personas.get(grade, [])
        if not personas:
            continue

        scores = []
        for si, persona_line in enumerate(personas):
            raw_path = config_dir / f"grade_{grade}" / f"{ik}_s{si}.txt"

            if raw_path.exists():
                text = raw_path.read_text()
            else:
                prompt = prompt_synthetic_student(item, persona_line)
                text = call_llm(prompt, temperature=temp)
                if text:
                    raw_path.write_text(text)
                n_api += 1
                time.sleep(0.1)

            score = parse_student_score(text, item['max_score'])
            if score is not None:
                scores.append(score)
            else:
                n_parse_fail += 1

        if scores:
            item_estimates[ik] = {
                "mean": float(np.mean(scores)),
                "sd": float(np.std(scores)),
                "n": len(scores),
            }

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

    # Evaluate
    sim, actual = [], []
    for item in items:
        ik = item_key(item)
        if ik in item_estimates:
            sim.append(item_estimates[ik]["mean"])
            actual.append(item["classical_difficulty"])

    if len(sim) >= 10:
        rho, p_rho = stats.spearmanr(sim, actual)
        mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
        bias = float(np.mean(np.array(sim) - np.array(actual)))

        rng = np.random.default_rng(42)
        boot_rhos = []
        for _ in range(2000):
            idx = rng.integers(0, len(sim), len(sim))
            br, _ = stats.spearmanr(np.array(sim)[idx], np.array(actual)[idx])
            boot_rhos.append(br)
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        print(f"\n  SYNTHETIC STUDENTS (t={temp})")
        print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | n={len(sim)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

        # Save predictions
        pred_path = config_dir / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(item_estimates, f, indent=2)


def run_synthetic_batched(temp=2.0):
    """Batched synthetic students — all 10 students simulated in ONE call per item.
    140 API calls total (vs 1,400 for individual). Tests if batch degrades quality."""
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_name = f"synthetic_batched_t{temp}"
    config_dir = OUTPUT_DIR / config_name
    config_dir.mkdir(exist_ok=True)

    # Stage 1: Generate personas (reuse from individual if available)
    grades = set(item['grade'] for item in items)
    all_personas = {}
    for grade in grades:
        grade_dir = config_dir / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)
        # Try to reuse personas from individual run
        individual_dir = OUTPUT_DIR / f"synthetic_students_t{temp}" / f"grade_{grade}"
        if (individual_dir / "personas.txt").exists():
            import shutil
            if not (grade_dir / "personas.txt").exists():
                shutil.copy(individual_dir / "personas.txt", grade_dir / "personas.txt")
        persona_text = generate_personas(grade, grade_dir)
        lines = [l.strip() for l in persona_text.split('\n') if re.match(r'\**STUDENT\s+\d', l.strip())]
        all_personas[grade] = lines
        print(f"  Grade {grade}: {len(lines)} personas", flush=True)

    n_api = 0
    n_parse_fail = 0
    item_estimates = {}

    for i, item in enumerate(items):
        ik = item_key(item)
        grade = item['grade']
        personas = all_personas.get(grade, [])
        if not personas:
            continue

        raw_path = config_dir / f"grade_{grade}" / f"{ik}_batch.txt"

        if raw_path.exists():
            text = raw_path.read_text()
        else:
            persona_block = "\n".join(personas)
            prompt = f"""You are simulating how each of these students would perform on an exam question.

STUDENT PROFILES:
{persona_block}

This is a Class {item['grade']} exam at an Indian government school.

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

For EACH student, briefly describe what they would write and assign a score.

Format (one per student):
STUDENT 1: [brief response description] → SCORE: [0-{item['max_score']}]
STUDENT 2: [brief response description] → SCORE: [0-{item['max_score']}]
...
STUDENT {len(personas)}: [brief response description] → SCORE: [0-{item['max_score']}]"""
            text = call_llm(prompt, temperature=temp, max_tokens=2048)
            if text:
                raw_path.write_text(text)
            n_api += 1
            time.sleep(0.1)

        # Parse all scores from batch response
        scores = []
        if text:
            max_s = float(item['max_score'])
            for m in re.finditer(r'SCORE:\s*(\d+(?:\.\d+)?)', text):
                score = float(m.group(1))
                scores.append(min(score / max_s, 1.0))
            if not scores:
                n_parse_fail += 1

        if scores:
            item_estimates[ik] = {
                "mean": float(np.mean(scores)),
                "sd": float(np.std(scores)),
                "n": len(scores),
            }

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

    # Evaluate
    sim, actual = [], []
    for item in items:
        ik = item_key(item)
        if ik in item_estimates:
            sim.append(item_estimates[ik]["mean"])
            actual.append(item["classical_difficulty"])

    if len(sim) >= 10:
        rho, p_rho = stats.spearmanr(sim, actual)
        mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
        bias = float(np.mean(np.array(sim) - np.array(actual)))
        rng = np.random.default_rng(42)
        boot_rhos = [stats.spearmanr(np.array(sim)[(idx:=rng.integers(0,len(sim),len(sim)))], np.array(actual)[idx])[0] for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        print(f"\n  SYNTHETIC BATCHED (t={temp})")
        print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | n={len(sim)}, {n_api} calls", flush=True)

        pred_path = config_dir / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(item_estimates, f, indent=2)


def run_synthetic_misconception(temp=2.0):
    """Misconception-aware synthetic students.
    Stage 1: Generate personas with specific misconceptions (cached)
    Stage 2: For each item, identify likely misconceptions
    Stage 3: Simulate students with those misconceptions attempting the item
    All in one call per item (batched) but misconception-enriched."""
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_name = f"synthetic_misconception_t{temp}"
    config_dir = OUTPUT_DIR / config_name
    config_dir.mkdir(exist_ok=True)

    # Generate misconception-aware personas per grade
    grades = set(item['grade'] for item in items)
    all_personas = {}
    for grade in grades:
        grade_dir = config_dir / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)
        persona_path = grade_dir / "personas.txt"
        if persona_path.exists():
            persona_text = persona_path.read_text()
        else:
            prompt = f"""Generate 10 diverse student profiles for a Class {grade} government school in India. {POPULATION}

Distribution: 4 Below Basic, 3 Basic, 2 Competent, 1 Advanced.

For each student, include:
- Name, gender, home language
- Proficiency level and specific academic weaknesses
- COMMON MISCONCEPTIONS they hold (e.g., "thinks all -ed endings are past tense", "confuses area with perimeter", "believes multiplication always increases a number")
- Typical test-taking behaviors (rushes, leaves blanks, copies from board, etc.)

Format:
STUDENT 1: [Name] | Level: [level] | Misconceptions: [list 2-3 specific misconceptions] | Behaviors: [test behaviors]
...
STUDENT 10: ..."""
            persona_text = call_llm(prompt, temperature=1.0, max_tokens=2048)
            if persona_text:
                persona_path.write_text(persona_text)
        lines = [l.strip() for l in persona_text.split('\n') if re.match(r'\**STUDENT\s+\d', l.strip())]
        all_personas[grade] = lines
        print(f"  Grade {grade}: {len(lines)} misconception-aware personas", flush=True)

    n_api = 0
    n_parse_fail = 0
    item_estimates = {}

    for i, item in enumerate(items):
        ik = item_key(item)
        grade = item['grade']
        personas = all_personas.get(grade, [])
        if not personas:
            continue

        raw_path = config_dir / f"grade_{grade}" / f"{ik}_misconception.txt"

        if raw_path.exists():
            text = raw_path.read_text()
        else:
            persona_block = "\n".join(personas)
            prompt = f"""STUDENT PROFILES (with known misconceptions):
{persona_block}

EXAM QUESTION (Class {item['grade']}, Indian government school):
{item['question_text']}

Rubric: {item['rubric']}
Max score: {item['max_score']}

First, identify which misconceptions from the student profiles are RELEVANT to this specific question. Then simulate each student's attempt, considering:
- Would their specific misconceptions cause errors on THIS question?
- Would their reading level prevent them from understanding the question?
- Would their test-taking behavior (rushing, leaving blank) affect their score?

Format:
RELEVANT MISCONCEPTIONS: [which student misconceptions apply to this question]
STUDENT 1: [what happens when they attempt it] → SCORE: [0-{item['max_score']}]
...
STUDENT {len(personas)}: [what happens] → SCORE: [0-{item['max_score']}]"""
            text = call_llm(prompt, temperature=temp, max_tokens=2048)
            if text:
                raw_path.write_text(text)
            n_api += 1
            time.sleep(0.1)

        scores = []
        if text:
            max_s = float(item['max_score'])
            for m in re.finditer(r'SCORE:\s*(\d+(?:\.\d+)?)', text):
                score = float(m.group(1))
                scores.append(min(score / max_s, 1.0))
            if not scores:
                n_parse_fail += 1

        if scores:
            item_estimates[ik] = {
                "mean": float(np.mean(scores)),
                "sd": float(np.std(scores)),
                "n": len(scores),
            }

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

    sim, actual = [], []
    for item in items:
        ik = item_key(item)
        if ik in item_estimates:
            sim.append(item_estimates[ik]["mean"])
            actual.append(item["classical_difficulty"])

    if len(sim) >= 10:
        rho, p_rho = stats.spearmanr(sim, actual)
        mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
        bias = float(np.mean(np.array(sim) - np.array(actual)))
        rng = np.random.default_rng(42)
        boot_rhos = [stats.spearmanr(np.array(sim)[(idx:=rng.integers(0,len(sim),len(sim)))], np.array(actual)[idx])[0] for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        print(f"\n  SYNTHETIC MISCONCEPTION (t={temp})")
        print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | n={len(sim)}, {n_api} calls", flush=True)

        pred_path = config_dir / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(item_estimates, f, indent=2)


# ============================================================
# IRT-fitted Simulation (inspired by SMART / "Take Out Your Calculators")
# ============================================================
# Instead of averaging simulated scores, we build a binary response matrix
# (students × items) and fit a 1PL IRT model to extract difficulty parameters.
# This separates item difficulty from student ability mathematically.

def fit_1pl_irt(response_matrix):
    """Fit a 1PL (Rasch) model via joint maximum likelihood.
    response_matrix: np.array of shape (n_students, n_items), values 0/1 or NaN.
    Returns: item difficulties (higher = harder = lower p_correct)."""
    R = np.array(response_matrix, dtype=float)
    n_students, n_items = R.shape

    # Initialize abilities and difficulties
    theta = np.zeros(n_students)   # student abilities
    b = np.zeros(n_items)          # item difficulties

    for iteration in range(50):
        # E-step: update abilities given difficulties
        for j in range(n_students):
            mask = ~np.isnan(R[j])
            if mask.sum() == 0:
                continue
            scores = R[j, mask]
            diffs = b[mask]
            # Newton-Raphson step
            p = 1.0 / (1.0 + np.exp(-(theta[j] - diffs)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            grad = np.sum(scores - p)
            hess = -np.sum(p * (1 - p))
            if abs(hess) > 1e-10:
                theta[j] -= grad / hess

        # M-step: update difficulties given abilities
        for i in range(n_items):
            mask = ~np.isnan(R[:, i])
            if mask.sum() == 0:
                continue
            scores = R[mask, i]
            abilities = theta[mask]
            p = 1.0 / (1.0 + np.exp(-(abilities - b[i])))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            grad = np.sum(scores - p)
            hess = -np.sum(p * (1 - p))
            if abs(hess) > 1e-10:
                b[i] -= grad / hess

    # Convert difficulty to proportion correct (using mean ability = 0)
    p_correct = 1.0 / (1.0 + np.exp(b))
    return p_correct, b, theta


def run_irt_simulation(temp=2.0, n_students=50):
    """Simulate n_students at different proficiency levels, build response matrix,
    fit 1PL IRT to extract difficulty. Inspired by SMART (EMNLP 2025)."""
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_name = f"irt_sim_n{n_students}_t{temp}"
    config_dir = OUTPUT_DIR / config_name
    config_dir.mkdir(exist_ok=True)

    # Define student ability levels matching "Take Out Your Calculators" distribution
    # 25% Below Basic, 35% Basic, 25% Competent, 15% Advanced
    levels = (
        [("Below Basic", "You struggle with basic reading and writing. You have minimal understanding of grade-level content. You often leave answers blank or copy irrelevant text.")] * int(0.25 * n_students) +
        [("Basic", "You can handle simple recall questions in Hindi but struggle with English. You understand basic concepts but make frequent errors on multi-step problems.")] * int(0.35 * n_students) +
        [("Competent", "You understand most concepts and can apply them. You sometimes make careless errors but generally produce reasonable answers.")] * int(0.25 * n_students) +
        [("Advanced", "You have strong understanding of all grade-level content. You rarely make mistakes and can explain your reasoning clearly.")] * int(0.15 * n_students)
    )
    # Pad to exact n_students
    while len(levels) < n_students:
        levels.append(levels[0])

    print(f"  {n_students} students: {sum(1 for l,_ in levels if l=='Below Basic')} BB, "
          f"{sum(1 for l,_ in levels if l=='Basic')} B, "
          f"{sum(1 for l,_ in levels if l=='Competent')} C, "
          f"{sum(1 for l,_ in levels if l=='Advanced')} A", flush=True)

    # Build response matrix — one API call per (item, student-batch)
    # Batch students by level to reduce API calls: 4 levels × 140 items = 560 calls
    level_groups = {}
    for s_idx, (level_name, level_desc) in enumerate(levels):
        level_groups.setdefault(level_name, []).append((s_idx, level_desc))

    n_api = 0
    n_parse_fail = 0
    response_matrix = np.full((n_students, len(items)), np.nan)

    for i, item in enumerate(items):
        ik = item_key(item)
        max_s = float(item['max_score'])

        for level_name, student_list in level_groups.items():
            n_in_group = len(student_list)
            raw_path = config_dir / f"{ik}_{level_name.replace(' ', '_')}.txt"

            if raw_path.exists():
                text = raw_path.read_text()
            else:
                prompt = f"""You are simulating {n_in_group} students at the "{level_name}" proficiency level attempting an exam question.

Student profile: {student_list[0][1]}
{POPULATION}

Class {item['grade']} exam, Indian government school.

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

For each student, decide if they would get this right or wrong. Vary responses — not all students at the same level perform identically.

Format (one line per student):
STUDENT 1: [correct/incorrect] — [1-sentence reason]
STUDENT 2: [correct/incorrect] — [reason]
...
STUDENT {n_in_group}: [correct/incorrect] — [reason]"""
                text = call_llm(prompt, temperature=temp, max_tokens=2048)
                if text:
                    raw_path.write_text(text)
                n_api += 1
                time.sleep(0.1)

            # Parse correct/incorrect
            if text:
                correct_count = 0
                total_parsed = 0
                for m in re.finditer(r'STUDENT\s+\d+:\s*(correct|incorrect)', text, re.IGNORECASE):
                    s_local_idx = total_parsed
                    if s_local_idx < n_in_group:
                        global_idx = student_list[s_local_idx][0]
                        response_matrix[global_idx, i] = 1.0 if m.group(1).lower() == 'correct' else 0.0
                    total_parsed += 1
                if total_parsed == 0:
                    n_parse_fail += 1

        if (i + 1) % 20 == 0:
            valid = np.sum(~np.isnan(response_matrix[:, :i+1]))
            total = n_students * (i + 1)
            print(f"    {i+1}/{len(items)} items, {n_api} api calls, "
                  f"{n_parse_fail} parse fails, {valid}/{total} responses filled", flush=True)

    # Fit IRT
    print(f"\n  Fitting 1PL IRT on {n_students}×{len(items)} response matrix...", flush=True)
    valid_rate = np.sum(~np.isnan(response_matrix)) / response_matrix.size
    print(f"  Response matrix fill rate: {valid_rate:.1%}", flush=True)

    # Simple proportion as baseline
    prop_correct = np.nanmean(response_matrix, axis=0)

    # IRT-fitted difficulty
    p_irt, difficulties, abilities = fit_1pl_irt(response_matrix)

    # Evaluate both
    items_list = items
    actual = [item['classical_difficulty'] for item in items_list]

    item_estimates_prop = {}
    item_estimates_irt = {}
    for i, item in enumerate(items_list):
        ik = item_key(item)
        if not np.isnan(prop_correct[i]):
            item_estimates_prop[ik] = {"mean": float(prop_correct[i])}
            item_estimates_irt[ik] = {"mean": float(p_irt[i]), "difficulty_b": float(difficulties[i])}

    for label, estimates in [("RAW PROPORTION", item_estimates_prop), ("IRT-FITTED", item_estimates_irt)]:
        sim_vals, act_vals = [], []
        for item in items_list:
            ik = item_key(item)
            if ik in estimates:
                sim_vals.append(estimates[ik]["mean"])
                act_vals.append(item["classical_difficulty"])

        if len(sim_vals) >= 10:
            rho, _ = stats.spearmanr(sim_vals, act_vals)
            mae = float(np.mean(np.abs(np.array(sim_vals) - np.array(act_vals))))
            bias = float(np.mean(np.array(sim_vals) - np.array(act_vals)))
            rng = np.random.default_rng(42)
            boot_rhos = [stats.spearmanr(np.array(sim_vals)[(idx:=rng.integers(0,len(sim_vals),len(sim_vals)))], np.array(act_vals)[idx])[0] for _ in range(2000)]
            ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
            print(f"\n  {label} (n={n_students}, t={temp})")
            print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | n={len(sim_vals)}, {n_api} calls", flush=True)

    # Save both
    pred_path = config_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(item_estimates_irt, f, indent=2)
    pred_path_raw = config_dir / "predictions_raw.json"
    with open(pred_path_raw, "w") as f:
        json.dump(item_estimates_prop, f, indent=2)

    # Save response matrix
    np.save(config_dir / "response_matrix.npy", response_matrix)
    print(f"  Saved response matrix and predictions to {config_dir}", flush=True)


# ============================================================
# KC Mastery Vector Simulation
# ============================================================
# Stage 0: LLM generates KC taxonomy for each grade/subject
# Stage 1: Generate students as KC mastery vectors
# Stage 2: LLM identifies required KCs per item
# Stage 3: Mechanical scoring: P(correct) = product of P(mastered KC_i) for required KCs

def run_kc_mastery(temp=2.0, n_students=10):
    """KC mastery vector simulation.
    Students defined by which knowledge components they've mastered.
    Items scored mechanically based on KC requirements."""
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_name = f"kc_mastery_n{n_students}_t{temp}"
    config_dir = OUTPUT_DIR / config_name
    config_dir.mkdir(exist_ok=True)

    # Stage 0: Generate KC taxonomy per grade+subject
    grade_subjects = set((item['grade'], item['subject']) for item in items)
    kc_taxonomies = {}

    for grade, subject in grade_subjects:
        kc_path = config_dir / f"kc_taxonomy_{grade}_{subject.replace(' ', '_')}.json"
        if kc_path.exists():
            kc_taxonomies[(grade, subject)] = json.loads(kc_path.read_text())
        else:
            prompt = f"""List the 12-15 most important knowledge components (specific skills/concepts) that a Class {grade} student in India needs for {subject}.

Order from most foundational to most advanced. {POPULATION}

Format as a JSON list of objects:
[
  {{"id": "KC1", "name": "basic reading comprehension", "prerequisite_for": ["KC3", "KC5"]}},
  {{"id": "KC2", "name": "number recognition 1-100", "prerequisite_for": ["KC4"]}},
  ...
]

Return ONLY valid JSON, no other text."""
            text = call_llm(prompt, temperature=0.5, max_tokens=2048)
            try:
                # Extract JSON from response
                json_match = re.search(r'\[[\s\S]*\]', text)
                if json_match:
                    kcs = json.loads(json_match.group())
                    kc_taxonomies[(grade, subject)] = kcs
                    kc_path.write_text(json.dumps(kcs, indent=2))
                    print(f"  Grade {grade} {subject}: {len(kcs)} KCs generated", flush=True)
                else:
                    print(f"  WARNING: Failed to parse KC taxonomy for Grade {grade} {subject}", flush=True)
                    kc_taxonomies[(grade, subject)] = []
            except json.JSONDecodeError:
                print(f"  WARNING: JSON parse failed for Grade {grade} {subject}", flush=True)
                kc_taxonomies[(grade, subject)] = []

    # Stage 1: Generate student mastery vectors
    # Students are defined by WHICH KCs they've mastered, based on proficiency level
    student_profiles_path = config_dir / "student_profiles.json"
    if student_profiles_path.exists():
        all_student_profiles = json.loads(student_profiles_path.read_text())
    else:
        all_student_profiles = {}
        for (grade, subject), kcs in kc_taxonomies.items():
            if not kcs:
                continue
            kc_names = [kc['name'] for kc in kcs]
            kc_ids = [kc['id'] for kc in kcs]

            prompt = f"""Here are the knowledge components for Class {grade} {subject}, ordered foundational → advanced:
{json.dumps([{"id": kc["id"], "name": kc["name"]} for kc in kcs], indent=2)}

Generate {n_students} students with realistic mastery patterns for an Indian government school. {POPULATION}

Distribution: {int(0.25*n_students)} Below Basic, {int(0.35*n_students)} Basic, {int(0.25*n_students)} Competent, {int(0.15*n_students)} Advanced.

Rules:
- Below Basic: mastered 1-3 foundational KCs only
- Basic: mastered 4-6 KCs, mostly foundational + some middle
- Competent: mastered 7-10 KCs, gaps in advanced topics
- Advanced: mastered 11+ KCs, may have 1-2 gaps

Format as JSON:
[
  {{"student": 1, "level": "Below Basic", "mastered": ["KC1", "KC2"]}},
  {{"student": 2, "level": "Basic", "mastered": ["KC1", "KC2", "KC3", "KC5"]}},
  ...
]

Return ONLY valid JSON."""
            text = call_llm(prompt, temperature=1.0, max_tokens=2048)
            try:
                json_match = re.search(r'\[[\s\S]*\]', text)
                if json_match:
                    profiles = json.loads(json_match.group())
                    key = f"{grade}_{subject}"
                    all_student_profiles[key] = profiles
                    print(f"  Grade {grade} {subject}: {len(profiles)} student profiles", flush=True)
            except json.JSONDecodeError:
                print(f"  WARNING: Failed to parse student profiles for Grade {grade} {subject}", flush=True)

        student_profiles_path.write_text(json.dumps(all_student_profiles, indent=2))

    # Stage 2: Identify required KCs per item
    n_api = 0
    n_parse_fail = 0
    item_estimates = {}

    for i, item in enumerate(items):
        ik = item_key(item)
        grade, subject = item['grade'], item['subject']
        kcs = kc_taxonomies.get((grade, subject), [])
        key = f"{grade}_{subject}"
        profiles = all_student_profiles.get(key, [])
        if not kcs or not profiles:
            continue

        kc_req_path = config_dir / f"{ik}_kc_requirements.json"

        if kc_req_path.exists():
            required_kcs = json.loads(kc_req_path.read_text())
        else:
            kc_list = "\n".join(f"  {kc['id']}: {kc['name']}" for kc in kcs)
            prompt = f"""Given these knowledge components for Class {grade} {subject}:
{kc_list}

Which KCs are REQUIRED to correctly answer this question?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

List ONLY the KC IDs that are necessary. A student who has mastered ALL listed KCs should get full marks.
A student missing ANY listed KC will likely lose marks.

Format: ["KC1", "KC3", "KC7"]
Return ONLY the JSON list."""
            text = call_llm(prompt, temperature=0.5, max_tokens=256)
            n_api += 1
            time.sleep(0.1)
            try:
                json_match = re.search(r'\[[\s\S]*?\]', text)
                if json_match:
                    required_kcs = json.loads(json_match.group())
                    kc_req_path.write_text(json.dumps(required_kcs))
                else:
                    required_kcs = []
                    n_parse_fail += 1
            except json.JSONDecodeError:
                required_kcs = []
                n_parse_fail += 1

        # Stage 3: Mechanical scoring
        if required_kcs:
            scores = []
            for student in profiles:
                mastered = set(student.get('mastered', []))
                required = set(required_kcs)
                # Proportion of required KCs that student has mastered
                if required:
                    kc_coverage = len(mastered & required) / len(required)
                else:
                    kc_coverage = 1.0
                # Simple model: score = coverage (could use threshold instead)
                scores.append(kc_coverage)

            item_estimates[ik] = {
                "mean": float(np.mean(scores)),
                "sd": float(np.std(scores)),
                "n": len(scores),
                "required_kcs": required_kcs,
            }

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

    # Evaluate
    sim, actual = [], []
    for item in items:
        ik = item_key(item)
        if ik in item_estimates:
            sim.append(item_estimates[ik]["mean"])
            actual.append(item["classical_difficulty"])

    if len(sim) >= 10:
        rho, _ = stats.spearmanr(sim, actual)
        mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
        bias = float(np.mean(np.array(sim) - np.array(actual)))
        rng = np.random.default_rng(42)
        boot_rhos = [stats.spearmanr(np.array(sim)[(idx:=rng.integers(0,len(sim),len(sim)))], np.array(actual)[idx])[0] for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        print(f"\n  KC MASTERY VECTOR (n={n_students}, t={temp})")
        print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | n={len(sim)}, {n_api} calls", flush=True)

        pred_path = config_dir / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(item_estimates, f, indent=2)


# ============================================================
# Cognitive-modeling prompts (holistic estimation after enumeration)
# ============================================================

def prompt_buggy_rules(item):
    """Brown & Burton (1978) — systematic procedural errors as difficulty scaffolding."""
    return f"""You are an expert in mathematical cognition and systematic student errors (Brown & Burton, 1978). You are analyzing a test item from Indian government schools.

{POPULATION}

For the following test item, analyze the cognitive demands:

Grade: {item['grade']}
Subject: {item['subject']}
Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

Step 1: List the specific procedural steps a student must execute correctly.
Step 2: For each step, identify any known "buggy rules" — systematic procedural errors students commonly make (e.g., subtracting smaller from larger regardless of position, forgetting to carry, distributing only to first term).
Step 3: Consider the target student population (grade level, open-ended format requiring recall not recognition).
Step 4: Taking into account ALL of the above analysis holistically — the number of steps, the severity and commonality of bugs, the grade level, and that stronger students may avoid multiple bugs while weaker students may hit several — estimate what proportion of students in this population would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_misconception_holistic(item):
    """Misconception-based difficulty analysis with holistic estimation."""
    return f"""You are an expert in mathematics education and student misconceptions. You are analyzing a test item from Indian government schools.

{POPULATION}

For the following test item, analyze what makes it easy or difficult:

Grade: {item['grade']}
Subject: {item['subject']}
Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

Step 1: Identify the mathematical concepts and skills required.
Step 2: List the most common misconceptions students at this grade level hold about these concepts. For each, note whether it would lead to an incorrect answer on THIS specific item.
Step 3: Consider factors that make the item easier: Is the context familiar? Are the numbers simple? Is the required operation well-practiced at this grade level?
Step 4: Consider factors that make it harder: Open-ended format (no answer choices), multi-step reasoning, abstract notation, large numbers, etc.
Step 5: Weighing ALL factors holistically, estimate what proportion of students at this grade level would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


def prompt_cognitive_profile(item):
    """Multi-dimensional cognitive difficulty profile with holistic estimation."""
    return f"""You are an expert psychometrician analyzing test item difficulty for Indian government schools.

{POPULATION}

For the following test item, build a cognitive difficulty profile:

Grade: {item['grade']}
Subject: {item['subject']}
Question: {item['question_text']}
Rubric: {item['rubric']}
Maximum score: {item['max_score']}

Analyze these dimensions:
- Concept familiarity: How well-practiced is this topic at this grade level? (very familiar / familiar / less familiar)
- Computational complexity: How many arithmetic/algebraic steps? (1 / 2-3 / 4+)
- Common error traps: Are there well-known misconceptions or procedural bugs that apply? (none / minor / major)
- Format demand: Open-ended requires recall; is partial credit possible or must the answer be exact?
- Number friendliness: Are the numbers easy to work with mentally? (easy / moderate / hard)

After building this profile, estimate holistically what proportion of students at this grade level would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:"""


# ============================================================
# Registry
# ============================================================

FRAMINGS = {
    "teacher": prompt_teacher,
    "error_analysis": prompt_error_analysis,
    "devil_advocate": prompt_devil_advocate,
    "prerequisite_chain": prompt_prerequisite_chain,
    "error_affordance": prompt_error_affordance,
    "cognitive_load": prompt_cognitive_load,
    "familiarity_gradient": prompt_familiarity_gradient,
    "verbalized_sampling": prompt_verbalized_sampling,
    "teacher_decomposed": prompt_teacher_decomposed,
    "classroom_sim": prompt_classroom_sim,
    "contrastive": prompt_contrastive,
    "imagine_classroom": prompt_imagine_classroom,
    "buggy_rules": prompt_buggy_rules,
    "misconception_holistic": prompt_misconception_holistic,
    "cognitive_profile": prompt_cognitive_profile,
}

# Special parsers for non-standard output formats
SPECIAL_PARSERS = {
    "teacher_decomposed": parse_decomposed,
}


def run_experiment(only_framing=None):
    items = load_items()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    framings = {only_framing: FRAMINGS[only_framing]} if only_framing else FRAMINGS
    temps = TEMPERATURES

    all_results = {}
    # Load existing results
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    total_configs = len(framings) * len(temps)
    config_idx = 0
    total_api = 0

    for framing_name, prompt_fn in framings.items():
        for temp in temps:
            config_idx += 1
            config_name = f"{framing_name}_t{temp}"
            config_dir = OUTPUT_DIR / config_name
            config_dir.mkdir(exist_ok=True)

            existing = list(config_dir.glob("rep*/*.txt"))
            expected = len(items) * N_REPS
            print(f"[{config_idx}/{total_configs}] {config_name} — {len(existing)}/{expected} files", flush=True)

            item_estimates = {}
            n_api = 0
            n_parse_fail = 0

            for i, item in enumerate(items):
                ik = item_key(item)
                estimates = []

                for rep in range(N_REPS):
                    raw_path = config_dir / f"rep{rep}" / f"{ik}.txt"
                    raw_path.parent.mkdir(exist_ok=True)

                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = prompt_fn(item)
                        # Chain-of-thought prompts need more tokens
                        mtk = 1024 if framing_name in ("buggy_rules", "misconception_holistic", "cognitive_profile") else 512
                        text = call_llm(prompt, temperature=temp, max_tokens=mtk)
                        if text:
                            raw_path.write_text(text)
                        n_api += 1
                        time.sleep(0.1)

                    special = SPECIAL_PARSERS.get(framing_name)
                    p = special(text) if special else None
                    if p is None:
                        p = parse_proportion(text)
                    if p is not None:
                        estimates.append(p)
                    else:
                        n_parse_fail += 1

                if estimates:
                    item_estimates[ik] = {
                        "mean": float(np.mean(estimates)),
                        "sd": float(np.std(estimates)),
                        "n": len(estimates),
                    }

                # Progress every 20 items
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(items)} items, {n_api} api calls, {n_parse_fail} parse fails", flush=True)

            total_api += n_api

            # Evaluate
            sim, actual = [], []
            for item in items:
                ik = item_key(item)
                if ik in item_estimates:
                    sim.append(item_estimates[ik]["mean"])
                    actual.append(item["classical_difficulty"])

            if len(sim) >= 10:
                rho, p_rho = stats.spearmanr(sim, actual)
                mae = float(np.mean(np.abs(np.array(sim) - np.array(actual))))
                bias = float(np.mean(np.array(sim) - np.array(actual)))

                # Bootstrap CI
                rng = np.random.default_rng(42)
                boot_rhos = []
                for _ in range(2000):
                    idx = rng.integers(0, len(sim), len(sim))
                    br, _ = stats.spearmanr(np.array(sim)[idx], np.array(actual)[idx])
                    boot_rhos.append(br)
                ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

                all_results[config_name] = {
                    "framing": framing_name,
                    "temperature": temp,
                    "spearman_rho": round(float(rho), 3),
                    "spearman_p": float(f"{p_rho:.2e}"),
                    "ci_95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
                    "mae": round(mae, 3),
                    "bias": round(bias, 3),
                    "n_items": len(sim),
                    "n_api_calls": n_api,
                    "n_parse_failures": n_parse_fail,
                    "parse_rate": round(1 - n_parse_fail / (len(items) * N_REPS), 3),
                }
                print(f"  ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | MAE={mae:.3f} | bias={bias:+.3f} | parse={1 - n_parse_fail / (len(items) * N_REPS):.1%}", flush=True)

                # Save incrementally
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

                # Save per-item predictions
                pred_path = config_dir / "predictions.json"
                with open(pred_path, "w") as f:
                    json.dump(item_estimates, f, indent=2)

    print_summary(all_results)


def analyze_existing():
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print("No results.json found")
        return
    all_results = json.loads(results_path.read_text())
    print_summary(all_results)


def print_summary(all_results):
    print(f"\n{'=' * 80}")
    print(f"PROMPT FRAMING EXPERIMENT — {MODEL}, {N_REPS} reps, 140 items, no base-rate")
    print(f"{'=' * 80}\n")

    print(f"{'Config':<30} {'ρ':>7} {'95% CI':>16} {'MAE':>7} {'Bias':>7} {'Parse':>7}")
    print("-" * 80)
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["spearman_rho"]):
        ci = res.get("ci_95", [0, 0])
        pr = res.get("parse_rate", 0)
        print(f"{name:<30} {res['spearman_rho']:>7.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {res['mae']:>7.3f} {res['bias']:>+7.3f} {pr:>6.1%}")

    # Temperature comparison
    print(f"\n  Temperature effect (ρ):")
    print(f"  {'Framing':<25} {'t=1.0':>7} {'t=2.0':>7} {'Δ':>7}")
    for framing in FRAMINGS:
        k1 = f"{framing}_t1.0"
        k2 = f"{framing}_t2.0"
        r1 = all_results.get(k1, {}).get("spearman_rho")
        r2 = all_results.get(k2, {}).get("spearman_rho")
        if r1 is not None and r2 is not None:
            print(f"  {framing:<25} {r1:>7.3f} {r2:>7.3f} {r2-r1:>+7.3f}")

    # Best overall
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]["spearman_rho"])
        print(f"\n  Best: {best[0]} (ρ={best[1]['spearman_rho']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framing", choices=list(FRAMINGS.keys()),
                        help="Run only this framing")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results only")
    parser.add_argument("--synthetic", choices=["individual", "batched", "misconception", "irt", "kc_mastery"],
                        help="Run synthetic student simulation variant")
    parser.add_argument("--temp", type=float, default=2.0,
                        help="Temperature for synthetic students (default: 2.0)")
    parser.add_argument("--temps", type=float, nargs="+", default=None,
                        help="Override TEMPERATURES for experiment (e.g. --temps 0.5 1.0 2.0)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model (e.g. meta-llama/llama-4-maverick-17b-128e-instruct for Groq)")
    args = parser.parse_args()

    if args.model:
        _m = sys.modules[__name__]
        _m.MODEL = args.model
        model_short = args.model.split("/")[-1].split("-instruct")[0]
        _m.OUTPUT_DIR = Path(f"pilot/prompt_framing_experiment/{model_short}")

    if args.temps:
        _m = sys.modules[__name__]
        _m.TEMPERATURES = args.temps

    os.chdir(Path(__file__).parent.parent)

    if args.analyze:
        analyze_existing()
    elif args.synthetic == "individual":
        run_synthetic_students(temp=args.temp)
    elif args.synthetic == "batched":
        run_synthetic_batched(temp=args.temp)
    elif args.synthetic == "misconception":
        run_synthetic_misconception(temp=args.temp)
    elif args.synthetic == "irt":
        run_irt_simulation(temp=args.temp, n_students=50)
    elif args.synthetic == "kc_mastery":
        run_kc_mastery(temp=args.temp, n_students=10)
    else:
        run_experiment(only_framing=args.framing)
