#!/usr/bin/env python3
"""
SmartPaper RSM v2 — Fixed scoring, filtered items, calibrated personas.

Key changes from v1:
1. Exclude visual-dependent items (6 items requiring images)
2. LLM-as-judge scoring with calibration examples
3. Better population-calibrated student personas
4. Score caching to minimize API calls

Quick test first: 3 configs (one per prompt_style) on 20 probe items.
Then full Box-Behnken sweep if quick test shows signal.

Phase 4 (--phase eedi_prompts): Eedi cognitive framing experiment.
Tests whether cognitive prompt framings from Eedi (contrastive, error_analysis,
devil_advocate, imagine_classroom, comparative_difficulty) improve difficulty
estimation on SmartPaper items. The KEY use case is zero-calibration estimation:
predicting difficulty of novel/LLM-generated items where no real student data
exists (calibration="none"). The "errors" and "anchors" calibration levels are
controls to measure how much real data helps on top of the framing.
"""

import argparse
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

RANDOM_SEED = 42
OUTPUT_DIR = Path("pilot/smartpaper_rsm_v2")
STUDENTS_PER_ITEM = 20
PROBE_N = 20

VISUAL_KEYWORDS = ["given figure", "in this picture", "based on images",
                   "observe the picture", "following figures", "in the given"]

MODEL_IDS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash": "gemini-3-flash-preview",
}

# Calibrated for Indian govt school population
# Key insight: many students have very limited English, especially in Hindi-medium schools
PROFICIENCY_DISTRIBUTION = [
    ("struggling", 0.35),
    ("basic", 0.35),
    ("competent", 0.20),
    ("advanced", 0.10),
]

LEVEL_DESCRIPTIONS = {
    "struggling": (
        "You are a Class {grade} student in a Hindi-medium Indian government school. "
        "You struggle a lot with academics. Your English is extremely limited — you can "
        "barely read English and can only write a few words. Even in Hindi, you find "
        "most questions confusing. You often don't understand what is being asked. "
        "For math, you know basic addition and subtraction but struggle with anything "
        "more complex. You frequently leave answers blank or copy something irrelevant."
    ),
    "basic": (
        "You are a Class {grade} student in an Indian government school. "
        "You have basic academic ability. You can read Hindi well but your English "
        "is weak — you understand simple English words but write with many errors. "
        "You can recall facts from textbooks if the question is straightforward, "
        "but you struggle with questions needing explanation or reasoning. "
        "For math, you can do basic operations but often make errors with fractions, "
        "decimals, and word problems."
    ),
    "competent": (
        "You are a Class {grade} student in an Indian government school who does "
        "reasonably well. You understand most concepts and can write answers in "
        "English with some grammatical errors. You handle recall and simple "
        "application questions well but sometimes miss details in more complex "
        "questions. For math, you are comfortable with grade-level concepts."
    ),
    "advanced": (
        "You are a Class {grade} student in an Indian government school who excels. "
        "You understand concepts deeply and can write clear, correct answers. "
        "You handle both recall and application questions well. You rarely make "
        "errors and can reason through multi-step problems."
    ),
}

# --- Scoring judge prompt with calibration ---
JUDGE_PROMPT = """You are scoring a student's handwritten answer (OCR'd) from an Indian government school. These students often have limited English and may write short, imperfect answers.

Question: {question}
Rubric: {rubric}
Max score: {max_score}

Student's answer: {answer}

Score this answer. Be calibrated:
- Score 1 if the answer demonstrates the key knowledge/skill in the rubric, even if poorly written
- Score 0 if the answer is blank, irrelevant, or does not show the required understanding
- For math: check if the approach/answer is correct, ignore minor notation issues
- For English: focus on whether the rubric criteria are met, not grammar/spelling

Reply with ONLY "1" or "0":"""


def load_data():
    items = json.loads(Path("data/smartpaper/item_statistics.json").read_text())
    # Filter: must have question_text, exclude visual items
    clean = []
    for item in items:
        if not item.get("question_text"):
            continue
        q = item["question_text"].lower()
        if any(kw in q for kw in VISUAL_KEYWORDS):
            continue
        clean.append(item)
    return clean


def select_probe_items(items, n=PROBE_N):
    np.random.seed(RANDOM_SEED)
    sorted_items = sorted(items, key=lambda x: x["classical_difficulty"])
    per_q = n // 5
    selected = []
    quintile_size = len(sorted_items) // 5
    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else len(sorted_items)
        pool = sorted_items[start:end]
        idx = np.random.choice(len(pool), size=min(per_q, len(pool)), replace=False)
        selected.extend([pool[i] for i in idx])
    return selected


def item_key(item):
    return f"{item['assessment']}_q{item['question_number']}"


def make_api_call(client, model_key, prompt, temperature=0.7, max_tokens=1024):
    if model_key.startswith("deepseek"):
        return _deepseek_call(model_key, prompt, temperature, max_tokens)
    from google.genai import types
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_IDS[model_key],
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""


def _deepseek_call(model_key, prompt, temperature, max_tokens):
    import openai
    client = openai.OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat" if model_key == "deepseek-chat" else "deepseek-reasoner"
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    DeepSeek error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""


def llm_judge(client, item, answer, model_key="gemini-2.5-flash"):
    """Score with calibrated LLM judge."""
    if not answer or answer.strip().lower() in ["", "blank", "[blank]", "i don't know", "-", "..."]:
        return 0
    prompt = JUDGE_PROMPT.format(
        question=item["question_text"],
        rubric=item["rubric"],
        max_score=item["max_score"],
        answer=answer,
    )
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=MODEL_IDS[model_key],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return 1 if "1" in response.text.strip()[:3] else 0
    except Exception as e:
        print(f"    Judge error: {e}")
        return 0


# --- Prompt builders ---

def build_individual_prompt(item, level):
    desc = LEVEL_DESCRIPTIONS[level].format(grade=item["grade"])
    return f"""{desc}

Answer this question from your school exam. Write your answer as you would on paper.
If you don't know, you can leave it blank or write what you think.

Subject: {item['subject']} (Class {item['grade']})
Question: {item['question_text']}

YOUR ANSWER:"""


def build_classroom_prompt(item, n_students):
    roster = []
    for level, weight in PROFICIENCY_DISTRIBUTION:
        count = max(1, round(n_students * weight))
        roster.extend([level] * count)
    roster = roster[:n_students]
    np.random.shuffle(roster)

    roster_str = ", ".join(f"Student {i+1} ({level})" for i, level in enumerate(roster))

    return f"""Simulate {n_students} Class {item['grade']} students in an Indian government school writing answers on a paper exam. Many students are from Hindi-medium backgrounds with limited English. Some may leave answers blank.

Ability levels: {roster_str}

Subject: {item['subject']}
Question: {item['question_text']}

Write each student's answer as they would actually write it (with typical spelling/grammar errors for their level):
{chr(10).join(f'Student {i+1} ({roster[i]}):' for i in range(min(3, n_students)))}
{"...all " + str(n_students) + " students." if n_students > 3 else ""}""", roster


def build_teacher_prompt(item):
    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

Your students are mostly from economically weaker sections. Many come from Hindi-medium backgrounds and have limited English proficiency. The school has mixed ability levels, with about 35% struggling significantly, 35% at basic level, 20% competent, and 10% advanced.

Question: {item['question_text']}
Scoring rubric: {item['rubric']}
Max score: {item['max_score']}

Based on your experience, what percentage of your students would score full marks on this item?

Think about:
- Language barriers (many can't read English well)
- Whether this requires recall vs. reasoning
- How specific the required answer is
- Common mistakes your students make

Reply in this format:
struggling: XX% full marks
basic: XX% full marks
competent: XX% full marks
advanced: XX% full marks"""


# --- Quick test: one config per style ---

def run_quick_test():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_DIR / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    items = load_data()
    print(f"Loaded {len(items)} text-only items (excluded visual)")

    probe = select_probe_items(items)
    print(f"Selected {len(probe)} probe items")
    with open(OUTPUT_DIR / "probe_items.json", "w") as f:
        json.dump(probe, f, indent=2)

    # Score cache
    cache_path = OUTPUT_DIR / "score_cache.json"
    score_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    configs = [
        {"name": "teacher_prediction", "style": "teacher"},
        {"name": "classroom_batch_5", "style": "classroom", "spc": 5},
        {"name": "individual_roleplay", "style": "individual"},
    ]

    for config in configs:
        cname = config["name"]
        cdir = raw_dir / cname
        cdir.mkdir(exist_ok=True)

        print(f"\n=== {cname} ===")
        item_mean_scores = {}
        n_api = 0

        for item in probe:
            ik = item_key(item)
            scores = []

            if config["style"] == "teacher":
                raw_path = cdir / f"{ik}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    prompt = build_teacher_prompt(item)
                    text = make_api_call(client, "gemini-2.5-flash", prompt, temperature=0.7)
                    raw_path.write_text(text)
                    n_api += 1
                    time.sleep(0.1)

                # Parse predictions
                for level, weight in PROFICIENCY_DISTRIBUTION:
                    pattern = rf'{level}\s*:\s*(\d+)\s*%'
                    match = re.search(pattern, text, re.IGNORECASE)
                    p_correct = int(match.group(1)) / 100.0 if match else 0.3
                    n_students = max(1, round(STUDENTS_PER_ITEM * weight))
                    for _ in range(n_students):
                        scores.append(1 if np.random.random() < p_correct else 0)

            elif config["style"] == "classroom":
                spc = config["spc"]
                n_calls = max(1, STUDENTS_PER_ITEM // spc)
                for batch_idx in range(n_calls):
                    raw_path = cdir / f"{ik}_batch{batch_idx}.txt"
                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt, roster = build_classroom_prompt(item, spc)
                        text = make_api_call(client, "gemini-2.5-flash", prompt, temperature=0.9, max_tokens=2048)
                        raw_path.write_text(text)
                        n_api += 1
                        time.sleep(0.1)

                    # Parse and score each answer
                    for i in range(1, spc + 1):
                        pattern = rf'Student\s*{i}\s*\([^)]*\)\s*:\s*(.+?)(?=\n\*?\*?Student\s*\d|\Z)'
                        match = re.search(pattern, text, re.DOTALL)
                        ans = match.group(1).strip().strip('*').strip() if match else ""

                        cache_key = f"{ik}||{ans[:200]}"
                        if cache_key in score_cache:
                            scores.append(score_cache[cache_key])
                        else:
                            s = llm_judge(client, item, ans)
                            score_cache[cache_key] = s
                            scores.append(s)
                            n_api += 1
                            time.sleep(0.05)

            elif config["style"] == "individual":
                student_idx = 0
                for level, weight in PROFICIENCY_DISTRIBUTION:
                    n_this = max(1, round(STUDENTS_PER_ITEM * weight))
                    for si in range(n_this):
                        if student_idx >= STUDENTS_PER_ITEM:
                            break
                        raw_path = cdir / f"{ik}_s{student_idx}.txt"
                        if raw_path.exists():
                            text = raw_path.read_text()
                        else:
                            prompt = build_individual_prompt(item, level)
                            text = make_api_call(client, "gemini-2.5-flash", prompt, temperature=0.9, max_tokens=300)
                            raw_path.write_text(text)
                            n_api += 1
                            time.sleep(0.1)

                        match = re.search(r'(?:YOUR ANSWER|ANSWER)\s*:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                        ans = match.group(1).strip() if match else text.strip().split('\n')[-1]

                        cache_key = f"{ik}||{ans[:200]}"
                        if cache_key in score_cache:
                            scores.append(score_cache[cache_key])
                        else:
                            s = llm_judge(client, item, ans)
                            score_cache[cache_key] = s
                            scores.append(s)
                            n_api += 1
                            time.sleep(0.05)

                        student_idx += 1

            if scores:
                item_mean_scores[ik] = sum(scores) / len(scores)

        # Save cache periodically
        cache_path.write_text(json.dumps(score_cache))

        # Evaluate
        sim, actual = [], []
        for item in probe:
            ik = item_key(item)
            if ik in item_mean_scores:
                sim.append(item_mean_scores[ik])
                actual.append(item["classical_difficulty"])

        if len(sim) >= 5:
            rho, p_rho = stats.spearmanr(sim, actual)
            r, p_r = stats.pearsonr(sim, actual)
            print(f"  API calls: {n_api}")
            print(f"  Spearman rho: {rho:.3f} (p={p_rho:.4f})")
            print(f"  Pearson r: {r:.3f} (p={p_r:.4f})")
            print(f"  Items: {len(sim)}")

            # Per-item comparison
            print(f"\n  {'Item':<40} {'Actual':>7} {'Sim':>7} {'Gap':>7}")
            for item in sorted(probe, key=lambda x: x['classical_difficulty']):
                ik = item_key(item)
                if ik in item_mean_scores:
                    gap = item_mean_scores[ik] - item['classical_difficulty']
                    print(f"  {ik:<40} {item['classical_difficulty']:>7.3f} {item_mean_scores[ik]:>7.3f} {gap:>+7.3f}")

    cache_path.write_text(json.dumps(score_cache))
    print(f"\nScore cache: {len(score_cache)} entries")


## ---- Phase 1: Calibrated Teacher Prediction ----

CALIBRATION_DIR = OUTPUT_DIR / "calibration"
N_REPS = 5

# Target difficulty levels for anchor items (proportion correct)
ANCHOR_TARGETS = [0.05, 0.15, 0.30, 0.50, 0.70]


def select_anchor_items(items, targets=ANCHOR_TARGETS):
    """Select items closest to target difficulty levels as calibration anchors."""
    anchors = []
    used = set()
    for target in targets:
        best = None
        best_dist = float("inf")
        for item in items:
            ik = item_key(item)
            if ik in used:
                continue
            dist = abs(item["classical_difficulty"] - target)
            if dist < best_dist:
                best_dist = dist
                best = item
        if best:
            anchors.append(best)
            used.add(item_key(best))
    return anchors


def sample_real_responses(anchor_items, responses_df, n_per_item=5):
    """Sample real student responses for anchor items (mix of correct/incorrect)."""
    samples = {}
    for item in anchor_items:
        ik = item_key(item)
        mask = (
            (responses_df["AssessmentName"] == item["assessment"])
            & (responses_df["QuestionNumber"] == item["question_number"])
            & (responses_df["StudentAnswer"].notna())
            & (responses_df["StudentAnswer"].str.strip() != "")
        )
        item_df = responses_df[mask].copy()
        if len(item_df) == 0:
            samples[ik] = []
            continue

        # Sample mix: ~half correct, ~half incorrect
        correct = item_df[item_df["StudentScore"] >= 1]
        incorrect = item_df[item_df["StudentScore"] == 0]
        n_correct = min(len(correct), max(1, n_per_item // 2))
        n_incorrect = min(len(incorrect), n_per_item - n_correct)
        n_correct = min(len(correct), n_per_item - n_incorrect)

        sampled = pd.concat([
            correct.sample(n=n_correct, random_state=RANDOM_SEED) if n_correct > 0 else pd.DataFrame(),
            incorrect.sample(n=n_incorrect, random_state=RANDOM_SEED) if n_incorrect > 0 else pd.DataFrame(),
        ])
        samples[ik] = [
            {"answer": row["StudentAnswer"], "score": int(row["StudentScore"])}
            for _, row in sampled.iterrows()
        ]
    return samples


def get_population_stats(items):
    """Compute population-level statistics from all items."""
    diffs = [i["classical_difficulty"] for i in items]
    return {
        "n_items": len(diffs),
        "mean_difficulty": np.mean(diffs),
        "median_difficulty": np.median(diffs),
        "sd_difficulty": np.std(diffs),
        "p10": np.quantile(diffs, 0.10),
        "p25": np.quantile(diffs, 0.25),
        "p75": np.quantile(diffs, 0.75),
        "p90": np.quantile(diffs, 0.90),
        "pct_below_20": np.mean([d < 0.20 for d in diffs]),
        "pct_below_30": np.mean([d < 0.30 for d in diffs]),
    }


def get_error_patterns(responses_df, items):
    """Extract common error patterns per subject from real data."""
    patterns = {}
    for subject in responses_df["AssessmentName"].str.extract(r"— (.+)")[0].dropna().unique():
        subj_items = [i for i in items if i["subject"] == subject]
        if not subj_items:
            continue
        # Find hardest items (lowest difficulty) and look at wrong answers
        hard = sorted(subj_items, key=lambda x: x["classical_difficulty"])[:5]
        errors = []
        for item in hard:
            mask = (
                (responses_df["AssessmentName"] == item["assessment"])
                & (responses_df["QuestionNumber"] == item["question_number"])
                & (responses_df["StudentScore"] == 0)
                & (responses_df["StudentAnswer"].notna())
                & (responses_df["StudentAnswer"].str.strip() != "")
            )
            wrong = responses_df.loc[mask, "StudentAnswer"].head(10).tolist()
            if wrong:
                errors.append({
                    "question": item["question_text"][:100],
                    "difficulty": item["classical_difficulty"],
                    "wrong_answers": wrong[:5],
                })
        patterns[subject] = errors
    return patterns


# --- Calibration prompt builders ---

def build_teacher_baseline(item):
    """Strategy 1: Baseline teacher prompt (same as existing)."""
    return build_teacher_prompt(item)


def build_teacher_popstats(item, pop_stats):
    """Strategy 2: Teacher prompt + population statistics."""
    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

IMPORTANT CALIBRATION DATA from actual student performance on this exact exam:
- Average item difficulty (proportion correct): {pop_stats['mean_difficulty']:.2f}
- Median: {pop_stats['median_difficulty']:.2f}
- 10th-90th percentile range: {pop_stats['p10']:.2f} to {pop_stats['p90']:.2f}
- {pop_stats['pct_below_20']:.0%} of items have <20% pass rate
- {pop_stats['pct_below_30']:.0%} of items have <30% pass rate
- This population performs MUCH lower than you might expect. The mean is ~{pop_stats['mean_difficulty']:.0%}, not 50-60%.

Your students are mostly from economically weaker sections. Many come from Hindi-medium backgrounds and have limited English proficiency.

Question: {item['question_text']}
Scoring rubric: {item['rubric']}
Max score: {item['max_score']}

Based on the calibration data above and your experience, what percentage of your students would score full marks on this item?

IMPORTANT: Most items have very low pass rates. Anchor your estimate to the population mean of {pop_stats['mean_difficulty']:.0%}.

Reply in this format:
ESTIMATED PROPORTION CORRECT: XX%"""


def build_teacher_anchors(item, anchor_items, anchor_responses):
    """Strategy 3: Teacher prompt + anchor items with real student responses."""
    anchor_text = ""
    for ai in anchor_items:
        aik = item_key(ai)
        responses = anchor_responses.get(aik, [])
        resp_text = ""
        for r in responses:
            resp_text += f"  - \"{r['answer'][:120]}\" → Score: {r['score']}\n"
        anchor_text += f"""
ANCHOR ITEM (Actual pass rate: {ai['classical_difficulty']:.0%}):
  Q: {ai['question_text'][:150]}
  Rubric: {ai['rubric'][:100]}
  Real student responses:
{resp_text}"""

    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

Here are REAL items from this exam with their ACTUAL student pass rates and real student responses. Use these to calibrate your prediction.
{anchor_text}

Now predict the pass rate for this NEW item:
Question: {item['question_text']}
Scoring rubric: {item['rubric']}
Max score: {item['max_score']}

Compare this item's difficulty to the anchor items above. Is it easier or harder? Why?

Reply in this format:
REASONING: [1-2 sentences comparing to anchors]
ESTIMATED PROPORTION CORRECT: XX%"""


def build_teacher_errors(item, error_patterns):
    """Strategy 4: Teacher prompt + error patterns from real data."""
    subject = item["subject"]
    patterns = error_patterns.get(subject, [])
    error_text = ""
    if patterns:
        error_text = f"\nCOMMON ERROR PATTERNS from real {subject} responses on this exam:\n"
        for p in patterns[:3]:
            error_text += f"  Q: \"{p['question'][:80]}\" (pass rate: {p['difficulty']:.0%})\n"
            for wa in p["wrong_answers"][:3]:
                error_text += f"    Wrong answer: \"{wa[:100]}\"\n"

    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

Your students are mostly from economically weaker sections with limited English. The average pass rate across all items is about 29%.
{error_text}
Question: {item['question_text']}
Scoring rubric: {item['rubric']}
Max score: {item['max_score']}

Based on the error patterns above and your experience, what percentage of your students would score full marks?

Reply in this format:
REASONING: [1-2 sentences about likely errors]
ESTIMATED PROPORTION CORRECT: XX%"""


def build_teacher_hybrid(item, pop_stats, anchor_items, anchor_responses, error_patterns):
    """Strategy 5: Hybrid — pop stats + 3 anchors + error patterns."""
    # Use 3 closest anchors
    sorted_anchors = sorted(anchor_items,
                            key=lambda a: abs(a["classical_difficulty"] - 0.30))[:3]
    anchor_text = ""
    for ai in sorted_anchors:
        aik = item_key(ai)
        responses = anchor_responses.get(aik, [])
        resp_text = ""
        for r in responses[:3]:
            resp_text += f"    \"{r['answer'][:100]}\" → {r['score']}\n"
        anchor_text += f"  {ai['question_text'][:100]} → {ai['classical_difficulty']:.0%} pass rate\n{resp_text}"

    subject = item["subject"]
    patterns = error_patterns.get(subject, [])
    error_text = ""
    if patterns:
        for p in patterns[:2]:
            error_text += f"  - \"{p['question'][:60]}\" ({p['difficulty']:.0%}): "
            error_text += f"students wrote: \"{p['wrong_answers'][0][:60]}\"\n"

    return f"""You are an experienced teacher in an Indian government school, Class {item['grade']} {item['subject']}.

POPULATION DATA: Mean pass rate = {pop_stats['mean_difficulty']:.0%}, range {pop_stats['p10']:.0%}–{pop_stats['p90']:.0%}.

ANCHOR ITEMS with real pass rates and student responses:
{anchor_text}
COMMON ERRORS in {subject}:
{error_text}
NEW ITEM to predict:
Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

Compare to anchors. Is this easier or harder? Consider language demands, reasoning required, and common errors.

Reply in this format:
REASONING: [1-2 sentences]
ESTIMATED PROPORTION CORRECT: XX%"""


# --- Eedi-style cognitive framing prompt builders ---

def build_anchor_context(anchor_items, anchor_responses):
    """Build anchor context string for calibration."""
    text = ""
    for ai in anchor_items:
        aik = item_key(ai)
        responses = anchor_responses.get(aik, [])
        resp_text = ""
        for r in responses[:3]:
            resp_text += f"    \"{r['answer'][:100]}\" → Score: {r['score']}\n"
        text += f"  Item (pass rate {ai['classical_difficulty']:.0%}): {ai['question_text'][:120]}\n{resp_text}"
    return text


def build_error_context(item, error_patterns):
    """Build error pattern context string for calibration."""
    subject = item["subject"]
    patterns = error_patterns.get(subject, [])
    if not patterns:
        return ""
    text = f"COMMON ERRORS in {subject} on this exam:\n"
    for p in patterns[:3]:
        text += f"  Q: \"{p['question'][:80]}\" (pass rate: {p['difficulty']:.0%})\n"
        for wa in p["wrong_answers"][:3]:
            text += f"    Wrong answer: \"{wa[:80]}\"\n"
    return text


def _eedi_calibration_block(item, calibration, anchor_items=None, anchor_responses=None, error_patterns=None):
    """Return calibration text block based on calibration level."""
    if calibration == "errors" and error_patterns:
        return "\n" + build_error_context(item, error_patterns) + "\n"
    elif calibration == "anchors" and anchor_items and anchor_responses:
        return "\nCALIBRATION ANCHORS from this exact exam with real student pass rates:\n" + build_anchor_context(anchor_items, anchor_responses) + "\n"
    return ""


def build_eedi_contrastive(item, calibration=None, **cal_data):
    """Contrastive framing: compare what makes this specific question easy or hard."""
    cal_block = _eedi_calibration_block(item, calibration, **cal_data)
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would answer this question correctly.

These students are mostly from economically weaker sections. Many attend Hindi-medium schools with limited English proficiency. Average pass rate across items on this exam is about 29%.
{cal_block}
Think about what makes this SPECIFIC question easy or hard compared to similar {item['subject']} content at the Class {item['grade']} level:
- What prior knowledge is assumed?
- What language demands does it place on students with limited English?
- Is this recall, application, or reasoning?
- How does it compare in difficulty to typical textbook exercises?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

After your contrastive analysis, provide your estimate.

ESTIMATED PROPORTION CORRECT: XX%"""


def build_eedi_error_analysis(item, calibration=None, **cal_data):
    """Error analysis framing: focus on what errors lead to wrong answers."""
    cal_block = _eedi_calibration_block(item, calibration, **cal_data)
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would answer this question correctly.

These students are mostly from economically weaker sections with limited English. Average pass rate is about 29%.
{cal_block}
Based on experience marking papers, think about what ERRORS would lead students to get this wrong:
- What misconceptions about {item['subject']} would cause incorrect answers?
- What language misunderstandings could occur?
- Would students attempt but fail, or not attempt at all?
- What would the most common wrong answer look like?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

After analyzing likely errors, estimate the pass rate.

ESTIMATED PROPORTION CORRECT: XX%"""


def build_eedi_devil_advocate(item, calibration=None, **cal_data):
    """Devil's advocate framing: challenge assumptions about difficulty."""
    cal_block = _eedi_calibration_block(item, calibration, **cal_data)
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would answer this question correctly.

These students are mostly from economically weaker sections with limited English. Average pass rate is about 29%.
{cal_block}
IMPORTANT: Teachers and experts consistently MISJUDGE item difficulty. They overestimate how well students will do because they forget:
- How limited many students' English actually is
- That "basic" concepts aren't basic for struggling learners
- That exam anxiety causes blank responses
- That many students have never practiced this exact question type

Challenge your first instinct. If you think "most students could do this," ask yourself: could a student who barely reads English? Could a student who was absent for this topic?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

After challenging your assumptions, provide your calibrated estimate.

ESTIMATED PROPORTION CORRECT: XX%"""


def build_eedi_imagine_classroom(item, calibration=None, **cal_data):
    """Imagine classroom framing: trust classroom instinct."""
    cal_block = _eedi_calibration_block(item, calibration, **cal_data)
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would answer this question correctly.

These students are mostly from economically weaker sections with limited English. Average pass rate is about 29%.
{cal_block}
Picture the moment after students submit their exam papers in a government school classroom of 40 students:
- The back-benchers who barely opened their papers
- The students who tried hard but made errors
- The few who wrote confident, correct answers
- The ones who copied something irrelevant

Trust your classroom instinct. How many papers would have a correct answer for this question?

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

Based on your classroom visualization, estimate the pass rate.

ESTIMATED PROPORTION CORRECT: XX%"""


def build_eedi_comparative_difficulty(item, calibration=None, **cal_data):
    """Comparative difficulty framing: rate difficulty first, then estimate."""
    cal_block = _eedi_calibration_block(item, calibration, **cal_data)
    return f"""You are estimating what proportion of Class {item['grade']} students in Indian government schools would answer this question correctly.

These students are mostly from economically weaker sections with limited English. Average pass rate is about 29%.
{cal_block}
Step 1: Rate this question's difficulty on a scale of 1-10 for this population:
  1 = trivially easy (>80% correct)
  3 = easy (60-80%)
  5 = moderate (30-50%)
  7 = hard (15-30%)
  9 = very hard (5-15%)
  10 = nearly impossible (<5%)

Step 2: Convert your difficulty rating to a proportion correct estimate.

Question: {item['question_text']}
Rubric: {item['rubric']}
Max score: {item['max_score']}

DIFFICULTY RATING: X/10
ESTIMATED PROPORTION CORRECT: XX%"""


EEDI_FRAMINGS = {
    "contrastive": build_eedi_contrastive,
    "error_analysis": build_eedi_error_analysis,
    "devil_advocate": build_eedi_devil_advocate,
    "imagine_classroom": build_eedi_imagine_classroom,
    "comparative_difficulty": build_eedi_comparative_difficulty,
}

CALIBRATION_LEVELS = ["none", "errors", "anchors"]


def run_eedi_prompt_sweep():
    """Phase 4: Test Eedi cognitive framings on SmartPaper items.

    Motivation: Can we estimate difficulty of novel items (e.g. LLM-generated)
    without any real student data? The "none" calibration level represents this
    realistic scenario. "errors" and "anchors" levels are controls showing
    how much real data improves over pure cognitive framing.

    Grid: 5 framings × 3 calibration levels × 2 temps × 5 reps × 20 items = 3000 calls.
    """
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    eedi_dir = OUTPUT_DIR / "eedi_prompts"
    eedi_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    items = load_data()
    responses_df = pd.read_csv("data/smartpaper/export_item_responses.csv")
    probe = json.loads((OUTPUT_DIR / "probe_items.json").read_text())
    print(f"Loaded {len(probe)} probe items, {len(items)} total items")

    # Build calibration data
    probe_keys = {item_key(i) for i in probe}
    non_probe = [i for i in items if item_key(i) not in probe_keys]
    anchors = select_anchor_items(non_probe)
    anchor_responses = sample_real_responses(anchors, responses_df)
    error_patterns = get_error_patterns(responses_df, items)

    temperatures = [1.0, 2.0]
    n_reps = 5
    all_results = {}

    total_configs = len(EEDI_FRAMINGS) * len(CALIBRATION_LEVELS) * len(temperatures)
    config_idx = 0

    for framing_name, prompt_fn in EEDI_FRAMINGS.items():
        for cal_level in CALIBRATION_LEVELS:
            for temp in temperatures:
                config_idx += 1
                config_name = f"{framing_name}_{cal_level}_t{temp}"
                config_dir = eedi_dir / config_name
                config_dir.mkdir(exist_ok=True)

                print(f"\n[{config_idx}/{total_configs}] === {config_name} ===")
                item_estimates = {}
                n_api = 0
                n_parse_fail = 0

                for item in probe:
                    ik = item_key(item)
                    estimates = []

                    for rep in range(n_reps):
                        raw_path = config_dir / f"rep{rep}" / f"{ik}.txt"
                        raw_path.parent.mkdir(parents=True, exist_ok=True)

                        if raw_path.exists():
                            text = raw_path.read_text()
                        else:
                            cal_data = {}
                            if cal_level == "errors":
                                cal_data["error_patterns"] = error_patterns
                            elif cal_level == "anchors":
                                cal_data["anchor_items"] = anchors
                                cal_data["anchor_responses"] = anchor_responses
                            prompt = prompt_fn(item, calibration=cal_level, **cal_data)
                            text = make_api_call(client, "gemini-2.5-flash", prompt, temperature=temp)
                            if text:
                                raw_path.write_text(text)
                            n_api += 1
                            time.sleep(0.2)

                        p = parse_proportion(text) if text else None
                        if p is not None:
                            estimates.append(p)
                        else:
                            n_parse_fail += 1

                    if estimates:
                        item_estimates[ik] = {
                            "mean": np.mean(estimates),
                            "sd": np.std(estimates),
                            "n": len(estimates),
                        }

                # Evaluate
                sim, actual = [], []
                for item in probe:
                    ik = item_key(item)
                    if ik in item_estimates:
                        sim.append(item_estimates[ik]["mean"])
                        actual.append(item["classical_difficulty"])

                if len(sim) >= 5:
                    rho, p_rho = stats.spearmanr(sim, actual)
                    r, p_r = stats.pearsonr(sim, actual)
                    mae = np.mean(np.abs(np.array(sim) - np.array(actual)))
                    bias = np.mean(np.array(sim) - np.array(actual))

                    all_results[config_name] = {
                        "framing": framing_name,
                        "calibration": cal_level,
                        "temperature": temp,
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_rho),
                        "pearson_r": float(r),
                        "mae": float(mae),
                        "bias": float(bias),
                        "n_items": len(sim),
                        "n_api_calls": n_api,
                        "n_parse_failures": n_parse_fail,
                    }
                    print(f"  API: {n_api} | parse fails: {n_parse_fail} | ρ={rho:.3f} (p={p_rho:.4f}) | r={r:.3f} | MAE={mae:.3f} | bias={bias:+.3f}")

    # Ranked summary
    print("\n" + "=" * 90)
    print("EEDI PROMPT SWEEP RESULTS — Ranked by Spearman ρ")
    print("=" * 90)
    print(f"{'Config':<45} {'ρ':>7} {'p':>8} {'r':>7} {'MAE':>7} {'Bias':>7} {'Parse%':>7}")
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["spearman_rho"]):
        total_possible = len(probe) * n_reps
        parse_rate = 1 - res["n_parse_failures"] / total_possible if total_possible > 0 else 0
        print(f"{name:<45} {res['spearman_rho']:>7.3f} {res['spearman_p']:>8.4f} {res['pearson_r']:>7.3f} {res['mae']:>7.3f} {res['bias']:>+7.3f} {parse_rate:>6.0%}")

    # Framing × calibration breakdown
    print("\n  Spearman ρ by framing × calibration (t=1.0 / t=2.0):")
    print(f"  {'Framing':<25}", end="")
    for cal in CALIBRATION_LEVELS:
        print(f"  {cal:<20}", end="")
    print()
    for framing in EEDI_FRAMINGS:
        print(f"  {framing:<25}", end="")
        for cal in CALIBRATION_LEVELS:
            vals = []
            for t in temperatures:
                key = f"{framing}_{cal}_t{t}"
                if key in all_results:
                    vals.append(f"{all_results[key]['spearman_rho']:.3f}")
                else:
                    vals.append("—")
            print(f"  {'/'.join(vals):<20}", end="")
        print()

    # Compare to baseline
    print("\n  Reference: Gemini anchors t=2.0 ρ=0.875")

    json.dump(all_results, open(eedi_dir / "results.json", "w"), indent=2)
    print(f"\nResults saved to {eedi_dir / 'results.json'}")


def parse_proportion(text):
    """Extract proportion from LLM response. Handles both XX% and per-level formats."""
    # Try single proportion format first
    m = re.search(r'ESTIMATED PROPORTION CORRECT\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0

    # Try per-level format (baseline)
    level_scores = []
    for level, weight in PROFICIENCY_DISTRIBUTION:
        pattern = rf'{level}\s*:\s*(\d+)\s*%'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            level_scores.append((float(match.group(1)) / 100.0, weight))
    if level_scores:
        return sum(p * w for p, w in level_scores) / sum(w for _, w in level_scores)

    # Fallback: find any percentage
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def run_calibration_experiment():
    """Phase 1: Test 5 calibration strategies on 20 probe items."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    items = load_data()
    print(f"Loaded {len(items)} text-only items")

    responses_df = pd.read_csv("data/smartpaper/export_item_responses.csv")
    print(f"Loaded {len(responses_df)} student responses")

    # Load probe items (same as quick test)
    probe_path = OUTPUT_DIR / "probe_items.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
    else:
        probe = select_probe_items(items)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(probe, open(probe_path, "w"), indent=2)
    print(f"Using {len(probe)} probe items")

    # Step 1: Build calibration anchors
    probe_keys = {item_key(i) for i in probe}
    non_probe = [i for i in items if item_key(i) not in probe_keys]
    anchors = select_anchor_items(non_probe)
    print(f"Selected {len(anchors)} anchor items:")
    for a in anchors:
        print(f"  {item_key(a)}: difficulty={a['classical_difficulty']:.3f}")

    anchor_responses = sample_real_responses(anchors, responses_df)
    for aik, resps in anchor_responses.items():
        print(f"  {aik}: {len(resps)} sampled responses")

    pop_stats = get_population_stats(items)
    print(f"Population stats: mean={pop_stats['mean_difficulty']:.3f}, median={pop_stats['median_difficulty']:.3f}")

    error_patterns = get_error_patterns(responses_df, items)
    for subj, pats in error_patterns.items():
        print(f"  {subj}: {len(pats)} error patterns")

    # Save calibration data
    json.dump({
        "anchors": [{"key": item_key(a), "difficulty": a["classical_difficulty"],
                      "question": a["question_text"][:200]} for a in anchors],
        "anchor_responses": {k: v for k, v in anchor_responses.items()},
        "population_stats": {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in pop_stats.items()},
    }, open(CALIBRATION_DIR / "calibration_data.json", "w"), indent=2)

    # Step 2: Run 5 strategies
    strategies = {
        "baseline": lambda item: build_teacher_baseline(item),
        "popstats": lambda item: build_teacher_popstats(item, pop_stats),
        "anchors": lambda item: build_teacher_anchors(item, anchors, anchor_responses),
        "errors": lambda item: build_teacher_errors(item, error_patterns),
        "hybrid": lambda item: build_teacher_hybrid(item, pop_stats, anchors, anchor_responses, error_patterns),
    }

    all_results = {}

    for strat_name, prompt_fn in strategies.items():
        strat_dir = CALIBRATION_DIR / strat_name
        strat_dir.mkdir(exist_ok=True)

        print(f"\n=== Strategy: {strat_name} ===")
        item_estimates = {}  # ik -> list of estimates across reps
        n_api = 0

        for item in probe:
            ik = item_key(item)
            estimates = []

            for rep in range(N_REPS):
                raw_path = strat_dir / f"{ik}_rep{rep}.txt"
                if raw_path.exists():
                    text = raw_path.read_text()
                else:
                    prompt = prompt_fn(item)
                    text = make_api_call(client, "gemini-2.5-flash", prompt, temperature=1.0)
                    raw_path.write_text(text)
                    n_api += 1
                    time.sleep(0.2)

                p = parse_proportion(text)
                if p is not None:
                    estimates.append(p)

            if estimates:
                item_estimates[ik] = {
                    "mean": np.mean(estimates),
                    "sd": np.std(estimates),
                    "estimates": estimates,
                }

        print(f"  API calls: {n_api}")
        print(f"  Items with estimates: {len(item_estimates)}/{len(probe)}")

        # Evaluate
        sim, actual = [], []
        for item in probe:
            ik = item_key(item)
            if ik in item_estimates:
                sim.append(item_estimates[ik]["mean"])
                actual.append(item["classical_difficulty"])

        if len(sim) >= 5:
            rho, p_rho = stats.spearmanr(sim, actual)
            r, p_r = stats.pearsonr(sim, actual)
            mae = np.mean(np.abs(np.array(sim) - np.array(actual)))
            bias = np.mean(np.array(sim) - np.array(actual))

            all_results[strat_name] = {
                "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "pearson_r": float(r),
                "pearson_p": float(p_r),
                "mae": float(mae),
                "bias": float(bias),
                "n_items": len(sim),
                "n_api_calls": n_api,
            }

            print(f"  Spearman rho: {rho:.3f} (p={p_rho:.4f})")
            print(f"  Pearson r:    {r:.3f} (p={p_r:.4f})")
            print(f"  MAE:          {mae:.3f}")
            print(f"  Bias:         {bias:+.3f}")

            # Per-item comparison
            print(f"\n  {'Item':<40} {'Actual':>7} {'Est':>7} {'SD':>6} {'Gap':>7}")
            for item in sorted(probe, key=lambda x: x['classical_difficulty']):
                ik = item_key(item)
                if ik in item_estimates:
                    est = item_estimates[ik]
                    gap = est["mean"] - item["classical_difficulty"]
                    print(f"  {ik:<40} {item['classical_difficulty']:>7.3f} {est['mean']:>7.3f} {est['sd']:>6.3f} {gap:>+7.3f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<15} {'rho':>7} {'p':>8} {'r':>7} {'MAE':>7} {'Bias':>7}")
    for strat_name, res in sorted(all_results.items(), key=lambda x: -x[1]["spearman_rho"]):
        print(f"{strat_name:<15} {res['spearman_rho']:>7.3f} {res['spearman_p']:>8.4f} {res['pearson_r']:>7.3f} {res['mae']:>7.3f} {res['bias']:>+7.3f}")

    json.dump(all_results, open(CALIBRATION_DIR / "results.json", "w"), indent=2)
    print(f"\nResults saved to {CALIBRATION_DIR / 'results.json'}")


def run_temp_sweep(model_key="gemini-2.5-flash"):
    """Temperature sweep on best strategies — mirrors Eedi temp analysis."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    sweep_dir = OUTPUT_DIR / "temp_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    items = load_data()
    responses_df = pd.read_csv("data/smartpaper/export_item_responses.csv")
    probe = json.loads((OUTPUT_DIR / "probe_items.json").read_text())

    probe_keys = {item_key(i) for i in probe}
    non_probe = [i for i in items if item_key(i) not in probe_keys]
    anchors = select_anchor_items(non_probe)
    anchor_responses = sample_real_responses(anchors, responses_df)
    pop_stats = get_population_stats(items)
    error_patterns = get_error_patterns(responses_df, items)

    temperatures = [0.5, 1.0, 1.5, 2.0]
    strategies = {
        "baseline": lambda item: build_teacher_baseline(item),
        "errors": lambda item: build_teacher_errors(item, error_patterns),
        "anchors": lambda item: build_teacher_anchors(item, anchors, anchor_responses),
    }

    all_results = {}
    model_tag = model_key.replace("-", "_").replace(".", "")

    for strat_name, prompt_fn in strategies.items():
        for temp in temperatures:
            config_name = f"{strat_name}_t{temp}" if model_key == "gemini-2.5-flash" else f"{model_tag}_{strat_name}_t{temp}"
            config_dir = sweep_dir / config_name
            config_dir.mkdir(exist_ok=True)

            print(f"\n=== {config_name} ===")
            item_estimates = {}
            n_api = 0

            for item in probe:
                ik = item_key(item)
                estimates = []
                for rep in range(N_REPS):
                    raw_path = config_dir / f"{ik}_rep{rep}.txt"
                    if raw_path.exists():
                        text = raw_path.read_text()
                    else:
                        prompt = prompt_fn(item)
                        text = make_api_call(client, model_key, prompt, temperature=temp)
                        if text:
                            raw_path.write_text(text)
                        n_api += 1
                        time.sleep(0.3)
                    p = parse_proportion(text) if text else None
                    if p is not None:
                        estimates.append(p)

                if estimates:
                    item_estimates[ik] = np.mean(estimates)

            sim, actual = [], []
            for item in probe:
                ik = item_key(item)
                if ik in item_estimates:
                    sim.append(item_estimates[ik])
                    actual.append(item["classical_difficulty"])

            if len(sim) >= 5:
                rho, p_rho = stats.spearmanr(sim, actual)
                r, p_r = stats.pearsonr(sim, actual)
                mae = np.mean(np.abs(np.array(sim) - np.array(actual)))
                bias = np.mean(np.array(sim) - np.array(actual))
                all_results[config_name] = {
                    "strategy": strat_name, "temperature": temp, "model": model_key,
                    "spearman_rho": float(rho), "spearman_p": float(p_rho),
                    "pearson_r": float(r), "mae": float(mae), "bias": float(bias),
                    "n_api_calls": n_api,
                }
                print(f"  API: {n_api} | ρ={rho:.3f} (p={p_rho:.4f}) | r={r:.3f} | MAE={mae:.3f} | bias={bias:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print(f"TEMPERATURE SWEEP RESULTS — {model_key}")
    print("=" * 70)
    print(f"{'Config':<35} {'ρ':>7} {'p':>8} {'r':>7} {'MAE':>7} {'Bias':>7}")
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["spearman_rho"]):
        print(f"{name:<35} {res['spearman_rho']:>7.3f} {res['spearman_p']:>8.4f} {res['pearson_r']:>7.3f} {res['mae']:>7.3f} {res['bias']:>+7.3f}")

    # Temperature gradient table
    print("\n  Temperature gradient (ρ):")
    print(f"  {'Strategy':<15}", end="")
    for t in temperatures:
        print(f"  t={t:<4}", end="")
    print()
    for strat in strategies:
        print(f"  {strat:<15}", end="")
        for t in temperatures:
            key = f"{strat}_t{t}" if model_key == "gemini-2.5-flash" else f"{model_tag}_{strat}_t{t}"
            if key in all_results:
                print(f"  {all_results[key]['spearman_rho']:.3f}", end="")
            else:
                print(f"  {'—':>5}", end="")
        print()

    # Merge with existing results if present
    results_path = sweep_dir / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing.update(all_results)
    json.dump(existing, open(results_path, "w"), indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="quick", choices=["quick", "sweep", "analyze", "calibrate", "temp_sweep", "temp_sweep_deepseek", "eedi_prompts"])
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    np.random.seed(RANDOM_SEED)

    if args.phase == "quick":
        run_quick_test()
    elif args.phase == "calibrate":
        run_calibration_experiment()
    elif args.phase == "temp_sweep":
        run_temp_sweep()
    elif args.phase == "temp_sweep_deepseek":
        run_temp_sweep(model_key="deepseek-chat")
    elif args.phase == "eedi_prompts":
        run_eedi_prompt_sweep()
