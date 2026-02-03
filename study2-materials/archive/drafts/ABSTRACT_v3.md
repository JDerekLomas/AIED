# The Selectivity Problem: LLMs Apply Misconceptions Uniformly, Not Like Students

## Abstract (200 words)

Large language models are proposed as synthetic students, but effective simulation requires both (1) producing misconception-consistent errors and (2) predicting which items trigger them. We tested both capabilities across six experiments using EEDI diagnostic items with 27,000+ empirical student responses.

**Experiment 1 (S1-S4 Specification):** "Mental model" prompts (S3) successfully induced target misconceptions (88.6% alignment), with conceptual errors (68%) outperforming procedural (55%). However, item-level correlation with human difficulty was near-zero (r=-0.07).

**Experiment 2 (Reasoning Authenticity):** While LLMs selected target distractors 50% of the time, reasoning matched the misconception only 8.7%—a 41-point "authenticity gap." Chain-of-thought improved answer selection but not reasoning alignment.

**Experiment 3 (Student Simulation):** Proficiency-matched models with UK student profiles differentiated ability levels internally (35% vs 94% accuracy) but showed no correlation with human performance (r=-0.19).

**Experiments 4-6 (Difficulty Estimation):** Direct estimation, classroom simulation, and error alignment approaches all yielded correlations between r=-0.27 and r=0.14.

We term this the **selectivity problem**: LLMs apply misconceptions uniformly across items rather than selectively like students. This limits their use for item calibration while preserving utility for controlled misconception generation.

**Keywords:** synthetic students, selectivity problem, misconception simulation, item difficulty

---

## Complete Results Summary

### Experiment 1: S1-S4 Specification Levels (n=322)

| Spec Level | Error Rate | Target Rate | Human Correlation |
|------------|------------|-------------|-------------------|
| S1 (Persona) | 4.9% | 0.0% | r = 0.00 |
| S2 (Knowledge) | 5.0% | 0.0% | r = 0.00 |
| **S3 (Mental Model)** | **43.8%** | **88.6%** | r = -0.07 |
| S4 (Production Rules) | 35.0% | 78.6% | r = -0.09 |

**By Misconception Type (S3):**
| Type | Target Rate |
|------|-------------|
| Conceptual | 68.2% |
| Procedural | 54.8% |

### Experiment 2: Reasoning Authenticity (n=208 errors)

| Metric | Value |
|--------|-------|
| Target distractor selection | 50.0% |
| Reasoning matches misconception | 8.7% |
| **Authenticity Gap** | **41.3 pp** |

**Reasoning Classification:**
- DIFFERENT_ERROR: 46.2%
- UNCLEAR: 45.2%
- PARTIAL_MATCH: 8.2%
- FULL_MATCH: 0.5%

### Experiment 3: Student Simulation (n=432)

| Proficiency | Model | Accuracy | Target Rate |
|-------------|-------|----------|-------------|
| Struggling | Llama-3.1-8B | 35.2% | 27.8% |
| Developing | GPT-3.5-turbo | 30.6% | 36.1% |
| Secure | GPT-4o-mini | 47.2% | 23.6% |
| Confident | Claude Sonnet 4 | 93.5% | 1.9% |

**Human Correlation:** r = -0.19 (correct rate), r = -0.09 (target rate)

### Experiment 4: Direct Difficulty Estimation (n=100)

| Prompting Style | Pearson r |
|-----------------|-----------|
| Basic | 0.14 |
| Expert | -0.03 |
| IRT-informed | -0.27 |
| Comparative | 0.12 |

### Experiment 5: Classroom Simulation (n=30)

| Metric | Value |
|--------|-------|
| Pearson r | -0.03 |
| Spearman ρ | 0.03 |

### Experiment 6: Opus Difficulty Estimation (n=250)

| Metric | Value |
|--------|-------|
| Pearson r | 0.19 |
| MAE | 39 pp |
| Within-item consistency | 2.5 pp std |

---

## The Selectivity Problem

**Definition:** LLMs can be induced to exhibit misconceptions but apply them uniformly across items, unlike humans who are selectively triggered by specific problem features.

**Evidence:**
1. S3 prompting induces 88.6% target alignment—but r ≈ 0 with human patterns
2. Models differentiate proficiency levels internally—but don't match human item difficulty
3. Case study: Q1430 (2/7+1/7) shows 91.7% LLM accuracy vs 5.8% human accuracy
4. Model reasoning describes correct procedure while labeling it a "misconception"

**Root cause:** LLMs judge difficulty by mathematical complexity. Humans apply misconceptions based on surface features that trigger automatic (wrong) procedures.

---

## Implications

| Use Case | Viable? | Evidence |
|----------|---------|----------|
| Generating misconception examples | ✓ Yes | 88.6% target rate with S3 |
| Testing tutor error detection | ✓ Yes | Controllable error induction |
| Item difficulty calibration | ✗ No | r ≈ 0 across all methods |
| Replacing student pilots | ✗ No | No item-level correlation |
| Adaptive system calibration | ✗ No | Selectivity problem |

---

## Future Directions

1. **Fine-tuning on student data:** Train on actual response distributions
2. **Explicit difficulty priors:** Include IRT parameters in prompts
3. **Hybrid approaches:** LLM generation + misconception library validation
4. **Neuro-symbolic systems:** Rule-based misconception triggering
5. **Feature-based triggering:** Identify surface features that activate misconceptions
