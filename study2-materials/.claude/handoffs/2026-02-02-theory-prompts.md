# Handoff: Theory-Driven Prompt Optimizations (2026-02-02)

## Current State

### Paper Direction
Megaexperiment optimization/methodology paper. Eedi dropped. Three datasets remain:

| Dataset | Domain | n | Best ρ | Method |
|---------|--------|---|--------|--------|
| DBE-KT22 | Univ DB Systems MCQ | 168 | 0.440 | Contrastive, Gemini 3 Flash, 3-rep |
| Indian State | K-12 Math/Eng/Hindi MCQ | 210 | 0.410 | Direct, Gemini 3 Flash, 3-rep |
| SmartPaper | K-12 Math open-ended | 140 | 0.547 | Teacher prediction, Gemini 3 Flash, 3-rep |

### Key Findings from Hypothesis Registry (18 hypotheses tested)
- **System 1 > System 2**: Fast pattern matching beats deliberation (H5, H11)
- **Error-focused framing works**: 3-4x improvement over generic prompts (H12)
- **Deliberation hurts**: Thinking/CoT/reasoning models degrade performance (H11)
- **Misconception hints hurt**: Explicit KC specification narrows reasoning (H7)
- **Multi-rep averaging helps**: Consistent boost, especially high-variance models (H9)
- **Model size doesn't predict success**: GPT-4o < Scout (H10)
- **Calibration anchors are dataset-dependent**: Help on SmartPaper, hurt on Eedi (H16)

## Theory-Driven Prompt Optimization Proposals

### Theoretical Grounding

The hypothesis registry tested prompts inspired by four learning science traditions:
1. **KC Theory** (Koedinger et al.) — Knowledge components as units of learning
2. **BUGGY** (Brown & Burton) — Systematic procedural errors
3. **Conceptual Change** (Vosniadou, Chi) — Naive theories that resist instruction
4. **ESS** (Epistemic State Specification) — Levels of knowing/unknowing

All four produced null or negative results when used as *cognitive simulation scaffolding* (H5). But the "System 1" finding suggests a different use: these theories should inform **what the model attends to**, not **how the model reasons**.

### Proposed Theory-Driven Prompts

Each prompt operationalizes a learning science insight as an *attention directive* rather than a *reasoning scaffold*.

#### 1. Prerequisite Chain Prompt (KC Theory)
**Theory**: Item difficulty is driven by the number and mastery-likelihood of prerequisite knowledge components. Items requiring more prerequisites are harder because each is a potential failure point.

```
For this question, identify the prerequisite knowledge/skills a student needs.
Count how many independent things must go right for a correct answer.
More prerequisites = harder item.

Question: {question}
Options: {options}

What percentage of students would get all prerequisites right and answer correctly?
Reply with just a number between 0 and 100.
```

**Why it might work**: Doesn't ask the model to simulate cognition — asks it to count failure points. Leverages the model's curriculum knowledge (what topics are prerequisites) without asking it to model student learning.

**Datasets**: All three. Especially strong for SmartPaper (multi-step math) and DBE-KT22 (SQL requiring multiple concepts).

#### 2. Error Affordance Prompt (BUGGY + Distractor Analysis)
**Theory**: Difficulty depends on how many plausible errors the item *affords*. Items with many "reasonable wrong paths" are harder than items with one clear correct path plus obviously wrong alternatives.

```
For this question, count the number of distinct, plausible errors a student could make.
Consider: procedural slips, conceptual confusions, partial knowledge, and misreading.
Items with MORE plausible error paths are HARDER.

Question: {question}
Options: {options}

How many distinct plausible errors exist? Then estimate what percentage would answer correctly.
Reply in format: ERRORS: N, CORRECT: XX%
```

**Why it might work**: Operationalizes Brown & Burton's insight (errors are systematic, not random) as a counting task. The model identifies error paths using its absorbed PCK, then uses the count as a difficulty heuristic. This is "System 1" compatible — it's pattern-matching on error affordances, not simulating student reasoning.

**Datasets**: Especially MCQs (DBE-KT22, Indian State) where distractors define the error space.

#### 3. Cognitive Load Prompt (Intrinsic Load Theory)
**Theory**: Sweller's cognitive load theory — difficulty is driven by element interactivity (how many elements must be processed simultaneously). High element interactivity = high intrinsic load = harder.

```
Rate the element interactivity of this question: how many pieces of information must a student hold in working memory simultaneously to solve it?

1-2 elements: Very easy (most students succeed)
3-4 elements: Moderate
5+ elements: Hard (many students fail)

Question: {question}
Options: {options}

Element count: [N]. Estimated percentage correct: [XX%].
Reply with just the percentage.
```

**Why it might work**: Cognitive load is a well-validated predictor of difficulty in educational psychology. The model counts interacting elements — a text-legible property — rather than simulating working memory limitations.

**Datasets**: SmartPaper (multi-step math problems have clear element counts), DBE-KT22 (SQL queries have measurable element interactivity).

#### 4. Familiarity Gradient Prompt (Conceptual Change / Frequency)
**Theory**: Items testing familiar, frequently-practiced content are easier than items testing novel applications or transfer. Difficulty correlates with distance from "textbook examples."

```
How typical is this question compared to standard textbook exercises for this topic?

- Very typical (standard drill): ~80%+ correct
- Slight variation: ~60-70% correct
- Novel application or transfer: ~40-50% correct
- Requires insight or non-obvious connection: ~20-30% correct

Question: {question}
Options: {options}

Rate typicality, then estimate percentage correct.
Reply with just a number between 0 and 100.
```

**Why it might work**: Leverages the model's training distribution directly. The model has seen many textbook exercises and can judge how "typical" a question is. This converts the model's frequency-based knowledge into a difficulty estimate without cognitive simulation.

**Datasets**: All three, but especially Indian State (government school assessments likely vary in typicality).

#### 5. Population-Aware Prompt (Developmental Psychology)
**Theory**: Difficulty depends on the population, not just the item. Grade 4 students (age 9-10) have different cognitive constraints than Grade 7 or university students. The Indian State data showed Grade 4 is a "dead zone" (ρ=0.045) — the model treats all Grade 4 items as equally moderate.

```
You are estimating difficulty for {population_description}.

Key developmental considerations for this population:
- Working memory capacity: {wm_note}
- Reading fluency: {reading_note}
- Typical instruction coverage: {coverage_note}

Question: {question}
Options: {options}

Given these population-specific constraints, what percentage would answer correctly?
Reply with just a number between 0 and 100.
```

Population notes would be filled in per dataset:
- Indian State G3: "Age 8-9, limited working memory, reading may not be fluent, basic arithmetic only"
- Indian State G7: "Age 12-13, formal operations emerging, multi-step reasoning possible"
- DBE-KT22: "University undergraduates, first database course, wide ability range"

**Why it might work**: The Grade 4 dead zone suggests the model lacks population-specific calibration. Explicit developmental context might help it distinguish "easy for Grade 4" from "hard for Grade 4" rather than defaulting to a generic moderate estimate.

**Datasets**: Indian State (where grade-level effects are dramatic), DBE-KT22.

### Experimental Design

**Phase 1 screening** on SmartPaper 20-item probes (consistent with existing DOE):
- 5 theory prompts × 2 temperatures (1.0, 2.0) × 3 reps = 30 configs
- Compare against existing best: teacher prediction baseline (ρ=0.875 on probes)
- Gemini 3 Flash (our anchor model)

**If any beat baseline**: expand to full 140 items, then cross-validate on DBE-KT22 and Indian State.

**If none beat baseline**: the theory contribution is strengthened — even theory-informed prompts can't improve on "System 1" pattern matching.

### Key Predictions

Based on the hypothesis registry findings:

1. **Prerequisite Chain** and **Error Affordance** are most likely to help — they operationalize learning science as *counting tasks* rather than *simulation tasks*, compatible with the System 1 finding.

2. **Cognitive Load** is a moderate bet — element interactivity is a well-validated construct but may be too abstract for the model to operationalize consistently.

3. **Familiarity Gradient** is interesting because it explicitly asks the model to use its training distribution rather than fighting against it.

4. **Population-Aware** is specifically targeted at the Indian State Grade 4 dead zone — if it doesn't fix that, the dead zone is likely un-fixable by prompting.

5. **None of these should use deliberation/thinking** — all prompts are designed for fast output (System 1 compatible).

## Files
- Hypothesis registry: `.claude/handoffs/2026-02-02-hypothesis-registry.md`
- Big picture: `.claude/handoffs/2026-02-02-paper-bigpicture.md`
- Literature review: `literature-review-synthetic-students.md`
- SmartPaper screening script: `scripts/run_smartpaper_screening.py` (if exists)
- DBE-KT22 validation: `scripts/run_dbe_kt22_validation.py`
- Indian State: `scripts/run_indian_state_assessment.py`

## Next Steps
1. Implement the 5 theory-driven prompts
2. Run Phase 1 screening on SmartPaper probes
3. Analyze which (if any) beat the teacher prediction baseline
4. Cross-validate winners on DBE-KT22 and Indian State
5. Update hypothesis registry with H19-H23 results
