# Handoff: Eedi Cognitive Framings → Generic Prompt Templates

**Date:** 2026-02-02
**Status:** Phase 1 v1 invalidated (base-rate leakage). Clean v2 running via `run_prompt_screening_g3f.py`.

## What happened

1. Implemented 5 Eedi-style cognitive framings (contrastive, error_analysis, devil_advocate, imagine_classroom, comparative_difficulty) × 3 calibration levels (none, errors, anchors) × 2 temps in `run_smartpaper_rsm_v2.py --phase eedi_prompts`.

2. Ran partial sweep on Gemini 2.5 Flash (17/30 configs completed before crashes + restarts). Results showed contrastive_errors_t1.0 ρ=0.811, contrastive_none_t2.0 ρ=0.694.

3. **Invalidated all v1 results** — every prompt contained "Average pass rate across items on this exam is about 29%", which is base-rate leakage from ground truth. The "none" calibration level wasn't truly zero-calibration.

4. Killed the 2.5 Flash run (wrong model anyway — should be Gemini 3 Flash Preview).

5. Clean v2 experiment launched via `scripts/run_prompt_screening_g3f.py` with:
   - No base-rate in any prompt
   - 7 framings (teacher, error_analysis, devil_advocate, prerequisite_chain, error_affordance, cognitive_load, familiarity_gradient)
   - Full 140 items (not 20 probes)
   - Gemini 3 Flash Preview
   - Zero-calibration only

## Key insight: zero-calibration is the use case

The practical motivation is estimating difficulty of **novel/LLM-generated items** where no real student data exists. Calibration data (anchors, error patterns, base rates) won't be available. The only input you'd have is:
- A population description (who's taking the test)
- The item itself (question, rubric, max score)

This means prompts must work with NO leakage from ground truth distributions.

## Generic prompt templates (7 framings)

All use these slots:
- `{subject}`, `{grade}` — item metadata
- `{population}` — one sentence describing the test-taker population
- `{question_text}`, `{rubric}`, `{max_score}` — item content

### 1. Teacher (baseline)
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

For this open-ended question, estimate what proportion of students would score full marks.

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Think about:
- What specific errors or misunderstandings would cause students to lose marks?
- How clearly does the question communicate what's expected?
- What prerequisite knowledge is needed?
- How likely are students at this grade level to have that knowledge?

Respond with ONLY a number between 0 and 1 representing the proportion of students who would get full marks.
For example: 0.45

Your estimate:
```

### 2. Error Analysis
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

For this question, think about what ERRORS would lead students to get this wrong:
- What misconceptions about {subject} would cause incorrect answers?
- What language misunderstandings could occur?
- Would students attempt but fail, or not attempt at all?
- What would the most common wrong answer look like?

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

After analyzing likely errors, estimate what proportion would get full marks.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:
```

### 3. Devil's Advocate
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

IMPORTANT: Teachers and experts consistently OVERESTIMATE how well students will do because they forget:
- How limited some students' foundational skills actually are
- That "basic" concepts aren't basic for struggling learners
- That exam anxiety causes blank responses
- That many students have never practiced this exact question type

Challenge your first instinct. If you think "most students could do this," ask yourself: could the weakest students in this population?

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

After challenging your assumptions, estimate what proportion would get full marks.

Respond with ONLY a number between 0 and 1.
For example: 0.45

Your estimate:
```

### 4. Prerequisite Chain (KC Theory)
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

For this question, identify the prerequisite knowledge and skills a student needs. Count how many independent things must ALL go right for a correct answer. Each prerequisite is a potential failure point.

Examples of prerequisites: reading comprehension, specific vocabulary, a math operation, a concept definition, multi-step reasoning, writing ability.

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

List the prerequisites, then estimate what proportion would get full marks.

PREREQUISITES: [list them]
COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

### 5. Error Affordance (BUGGY)
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

Count the number of distinct, plausible ways a student could get this wrong:
- Procedural slips (computation errors, skipped steps)
- Conceptual confusions (wrong rule applied)
- Language barriers (misunderstanding the question)
- Partial knowledge (knows some but not all required steps)
- Non-attempts (leaves blank, writes something irrelevant)

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

List the plausible errors, then estimate what proportion would get full marks.

PLAUSIBLE ERRORS: [list them]
ERROR COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

### 6. Cognitive Load (Sweller CLT)
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

Rate the element interactivity of this question: how many pieces of information must a student hold in working memory simultaneously to solve it?

Elements include: numbers to manipulate, rules to apply, vocabulary to recall, sentence structures to parse, steps in a procedure, constraints to satisfy.

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

List the elements, then estimate what proportion would get full marks.

ELEMENTS: [list them]
ELEMENT COUNT: [N]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

### 7. Familiarity Gradient (Conceptual Change)
```
You are an experienced teacher in {subject} for Grade {grade} students.

{population}

How typical is this question compared to standard textbook exercises for Grade {grade} {subject}?

Consider: does it match textbook drills exactly, or require transfer, novel phrasing, or non-obvious connections?

Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Rate typicality, then estimate what proportion would get full marks.

TYPICALITY: [very typical / slight variation / novel / requires insight]

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

## Files
- `scripts/run_smartpaper_rsm_v2.py` — Phase 4 eedi_prompts (v1, invalidated)
- `scripts/run_prompt_screening_g3f.py` — v2 clean experiment (running)
- `pilot/smartpaper_rsm_v2/eedi_prompts/` — v1 results (invalidated, base-rate leakage)
- `pilot/prompt_framing_experiment/` — v2 results (in progress)
- `.claude/handoffs/2026-02-02-paper-bigpicture.md` — master doc (updated with v2 design)

## V1 results (INVALIDATED — for reference only)

These had "29% average pass rate" in every prompt. Archived, not usable.

| Config | ρ | Note |
|---|---|---|
| contrastive_errors_t1.0 | 0.811 | inflated by base-rate |
| devil_advocate_errors_t2.0 | 0.774 | inflated |
| contrastive_none_t2.0 | 0.694 | inflated |
| devil_advocate_anchors_t1.0 | 0.692 | inflated |
