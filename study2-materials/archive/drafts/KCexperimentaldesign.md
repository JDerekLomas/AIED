# Study 2: KC-Grounded Experimental Design
## Can Theoretically-Grounded Prompts Produce Authentic Misconception Simulation?

---

## Research Question

**Can structured prompt specifications close the Reasoning Authenticity Gap?**

From pilot data:
- Target distractor rate: 50.0%
- Misconception alignment rate: 8.7%
- **Gap: 41.3 percentage points**

LLMs select the "right wrong answer" but their reasoning doesn't match the target misconception. Can better prompts fix this?

---

## Design: 3 × 4 × 2 Factorial

| Factor | Levels | N |
|--------|--------|---|
| Model Capability | Frontier, Mid, Weak | 3 |
| Specification Level | S1, S2, S3, S4 | 4 |
| Misconception Type | Procedural, Conceptual | 2 |

**Cells**: 24
**Responses per cell**: 2 models × 5 items × 3 reps = 30
**Total**: ~720 core responses + discriminant items

---

## Factor 1: Specification Level

| Level | Name | What's Specified | Example (Order of Ops) |
|-------|------|------------------|------------------------|
| S1 | Persona | Role only | "You're a struggling 6th grader" |
| S2 | Knowledge State | What's known/unknown | "You know single operations; fuzzy on order of operations" |
| S3 | Mental Model | The belief/theory held | "You believe math is solved left-to-right like reading" |
| S4 | Production Rules | Step-by-step procedure | "STEP 1: Find leftmost operation. STEP 2: Apply it. STEP 3: Repeat." |

### Theoretical Grounding

| Level | Based On |
|-------|----------|
| S1 | Baseline (no theory) |
| S2 | Knowledge Component Theory (Koedinger) |
| S3 | Misconception-as-Theory (Chi et al.) |
| S4 | Buggy Procedures (Brown & Burton) |

### Why This Avoids "Prompt Hacking"

Bad prompt (instruction following):
> "You think 1/6 > 1/4. Answer B."

Good prompt (cognitive state):
> "You believe more slices = more pizza. The denominator tells you how many slices."

The error should **emerge** from the specified state, not be directly instructed.

---

## Factor 2: Model Capability

| Tier | Models | GSM8K | Expected Error Rate |
|------|--------|-------|---------------------|
| Frontier | GPT-4o, Claude 3.5 Sonnet | ~95% | ~0% (ceiling) |
| Mid | GPT-3.5-turbo, Claude Haiku | ~75% | ~15-25% |
| Weak | Llama-3.1-8B, Mistral-7B | ~50% | ~40-50% |

**Note**: Frontier models may resist error induction even with S4 prompts. This is a finding, not a bug.

---

## Factor 3: Misconception Type

| Type | Example | Cognitive Mechanism |
|------|---------|---------------------|
| Procedural | Left-to-right calculation | Algorithm misapplication |
| Conceptual | "Bigger denominator = bigger fraction" | Flawed mental model |

### Test Misconceptions (4 total for pilot)

**Procedural:**
- PROC_ORDER_OPS: Left-to-right instead of PEMDAS
- PROC_SUBTRACT_REVERSE: Smaller-from-larger in each column

**Conceptual:**
- CONC_FRAC_DENOM: Larger denominator = larger fraction
- CONC_MULT_INCREASE: Multiplication always increases

---

## Dependent Variables

| Metric | Formula | Measures |
|--------|---------|----------|
| Error Rate | 1 - Accuracy | Base rate |
| Target Distractor Rate | P(target \| error) | Surface alignment |
| Misconception Alignment | P(reasoning matches \| error) | Authentic simulation |
| **Reasoning Authenticity Gap** | Target Rate - Alignment | **Primary DV** |

---

## Hypotheses

### H1: Specification Level Main Effect
Misconception alignment increases with specification level:
```
S4 > S3 > S2 > S1
```

### H2: Type × Specification Interaction
- Procedural misconceptions: S4 (production rules) most effective
- Conceptual misconceptions: S3 (mental model) most effective

### H3: Capability × Specification Interaction
- At S1 (persona only): Weak models > Frontier (capability paradox)
- At S4 (full specification): Gap narrows or reverses

### H4: Gap Reduction
The Reasoning Authenticity Gap decreases with specification level:
```
Gap_S1 > Gap_S2 > Gap_S3 > Gap_S4
```

---

## Validity Tests

### 1. Generalization (Near Transfer)
- Prompt constructed using Example Set A
- Test on Item Set B (same misconception, different surface)
- **Pass**: Errors generalize beyond prompt examples

### 2. Discriminant Validity
- Include items targeting different misconceptions
- **Pass**: Model does NOT produce off-target errors

### 3. Consistency
- Multiple items per misconception
- **Pass**: Same error pattern across items

---

## On the Stability Assumption

The cognitive science literature (Brown & Burton, Chi et al.) often treats misconceptions as **stable cognitive structures**. But this may be an idealization.

**Empirical question**: Are student misconceptions actually stable, or do they fluctuate?

**For our purposes**: We don't need to assume stability in students. We're testing whether LLMs can **simulate** a stable misconception when prompted. If they can't maintain consistency across items, that's a finding about LLM simulation, not a claim about human cognition.

**What we measure**:
- **Within-session consistency**: Does the model produce the same error pattern across 5 items in one prompt?
- **Cross-item generalization**: Does the induced "misconception" transfer to new items?

If LLMs show low consistency even with S4 prompts, this suggests they don't form stable error representations—regardless of whether humans do.

---

## Analysis Plan

### Primary Analysis
**3-way ANOVA** on Misconception Alignment Rate
- IVs: Specification Level (4) × Misconception Type (2) × Model Tier (3)
- Key tests: Main effect of Specification, Type × Specification interaction

### Planned Contrasts
1. S4 vs S1 (overall specification effect)
2. S4 vs S3 within Procedural (algorithms vs beliefs for procedures)
3. S3 vs S4 within Conceptual (beliefs vs algorithms for concepts)
4. Frontier-S4 vs Weak-S1 (can specification overcome capability paradox?)

### Secondary Analyses
1. Reasoning Authenticity Gap by cell
2. Consistency analysis (within-misconception variance)
3. Generalization rate (near transfer accuracy)
4. Discriminant validity (off-target error rate)

---

## Possible Outcomes

### Outcome A: Specification Works
S4 >> S1 across the board. Gap shrinks significantly.
- **Implication**: Detailed cognitive specification enables authentic simulation
- **Practical**: Use structured KC profiles for synthetic students

### Outcome B: Type-Specific Effects
S4 works for procedural, S3 works for conceptual.
- **Implication**: Match specification type to misconception type
- **Practical**: Different prompting strategies for different error types

### Outcome C: Nothing Works
Gap persists across all specification levels.
- **Implication**: LLMs fundamentally can't represent stable misconceptions
- **Practical**: Distractor alignment metrics are misleading; need different approaches

### Outcome D: Only Weak Models Work
Weak models show alignment at S3/S4; frontier models resist.
- **Implication**: Capability paradox is fundamental, not overcome by prompting
- **Practical**: Use weak models for misconception simulation

---

## Timeline

| Task | Date |
|------|------|
| Finalize S1-S4 prompts (4 misconceptions) | Jan 27 |
| Run pilot (2 models × 4 specs × 4 misconceptions) | Jan 28 |
| Analyze pilot, refine | Jan 28 |
| Full data collection (6 models) | Jan 29-30 |
| Misconception coding | Jan 31 |
| Analysis | Feb 1 |
| Write up | Feb 2 |
| **Submit** | **Feb 2** |

---

## Files

| File | Purpose |
|------|---------|
| `KCexperimentaldesign.md` | This document |
| `scripts/test_specification_levels.py` | Test runner |
| `experimental-design-v2.md` | Previous design (superseded) |

---

## Summary

We test whether **theoretically-grounded prompt specifications** (S1-S4) can close the **Reasoning Authenticity Gap** (41.3 points in pilot).

The design is grounded in cognitive science (KC theory, misconception-as-theory, buggy procedures) but does not assume misconception stability—we measure it.

Primary outcome: Does the Gap shrink with better specification?
