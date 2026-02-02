# Pilot Results: S1-S4 Specification Level Experiment

## Setup

| Parameter | Value |
|-----------|-------|
| Items | 12 (3 per misconception, random sample) |
| Models (Mid-tier) | gpt-3.5-turbo, claude-3-haiku |
| Models (Frontier) | gpt-4o, claude-sonnet-4 |
| Specification Levels | S1, S2, S3, S4 |
| Repetitions | 2 |
| Total API calls | 384 (192 mid-tier + 192 frontier) |
| Date | 2026-01-26 |

---

## Key Finding: S3 (Mental Model) Produces Highest Alignment Across All Tiers

### Mid-tier Models (gpt-3.5-turbo, claude-3-haiku)

| Spec Level | Error Rate | Target Rate | vs Human (38.2%) |
|------------|------------|-------------|------------------|
| S1 (Persona) | 18.8% | 0.0% | -38.2 pp |
| S2 (Knowledge State) | 37.5% | 16.7% | -21.5 pp |
| **S3 (Mental Model)** | **52.1%** | **72.0%** | **+33.8 pp** |
| S4 (Production Rules) | 58.3% | 46.4% | +8.2 pp |

### Frontier Models (gpt-4o, claude-sonnet-4)

| Spec Level | Error Rate | Target Rate | vs Human (38.2%) |
|------------|------------|-------------|------------------|
| S1 (Persona) | 4.2% | 0.0% | -38.2 pp |
| S2 (Knowledge State) | 8.3% | 0.0% | -38.2 pp |
| **S3 (Mental Model)** | **52.1%** | **68.0%** | **+29.8 pp** |
| S4 (Production Rules) | 45.8% | 63.6% | +25.4 pp |

**Key insight:** Frontier models are "smarter" (lower error rates at S1/S2), but S3/S4 prompts successfully induce misconceptions with similar target alignment. The S3 > S4 pattern holds across model tiers.

---

## Model-by-Model Comparison

| Model | Tier | Error Rate | Target Rate |
|-------|------|------------|-------------|
| gpt-3.5-turbo | mid | 42.7% | 31.7% |
| claude-3-haiku | mid | 40.6% | 53.8% |
| gpt-4o | frontier | 28.1% | 40.7% |
| **claude-sonnet-4** | **frontier** | **27.1%** | **76.9%** |

**Key finding:** Claude Sonnet 4 shows the highest target alignment (76.9%) when it makes errors. The model gets 100% correct at S1/S2, but when induced to err via S3/S4, it reliably hits the target distractor.

---

## Claude Sonnet 4: Binary Behavior Pattern

| Spec Level | Error Rate | Target Rate |
|------------|------------|-------------|
| S1 | **0.0%** | N/A |
| S2 | **0.0%** | N/A |
| S3 | 54.2% | **76.9%** |
| S4 | 54.2% | **76.9%** |

Claude Sonnet 4 exhibits a binary pattern: it either answers correctly (S1/S2) or consistently hits the target misconception (S3/S4). This suggests frontier Claude models may be ideal for controlled synthetic student simulation.

---

## Target Rate by Specification × Misconception Type (Mid-tier)

| Spec | Procedural | Conceptual |
|------|------------|------------|
| S1 | 0.0% (6 err) | 0.0% (3 err) |
| S2 | 11.1% (9 err) | 22.2% (9 err) |
| S3 | **71.4%** (7 err) | **72.2%** (18 err) |
| S4 | 15.4% (13 err) | **73.3%** (15 err) |

**Key finding:** S3 works well for both procedural and conceptual misconceptions, while S4 works primarily for conceptual.

---

## Authentic Reasoning Examples

### Example 1: Fraction Addition Misconception (217)

**Item:** Q1390 - Adding fractions
**Prompt type:** S3 (Mental Model)
**Target distractor:** C (= 2/9)

**LLM Response:**
> "Now, we can add these two fractions using the method mentioned earlier:
> 1/3 + 1/6 = (1+1)/(3+6) = **2/9**"

✓ Reasoning explicitly shows adding numerators AND denominators

### Example 2: Negative Multiplication Misconception (1597)

**Item:** Q139 - Expanding brackets with negatives
**Prompt type:** S3 (Mental Model)
**Target distractor:** B (= -12)

**LLM Response:**
> "when multiplying the two terms indicated by the arrows (-4 and -3), Alison should get **-12**"

✓ Reasoning shows incorrect belief that neg × neg = neg

---

## Implications for Full Experiment

1. **S3 is the most effective specification level** - contrary to initial hypothesis that S4 (production rules) would be best for procedural errors

2. **Pattern holds across model tiers** - both mid-tier and frontier models show S3 > S4 for misconception alignment

3. **Frontier Claude shows ideal behavior** - 0% errors at baseline, high target alignment when induced

4. **Human baseline is achievable** - LLM target rates (68-77%) exceed human target rate (38%)

5. **Reasoning appears authentic** - LLMs explain their thinking in ways consistent with the misconception

---

## Revised Hypotheses

| Original | Revised |
|----------|---------|
| H2: S4 for procedural, S3 for conceptual | S3 works for both; S4 only for conceptual |
| H4: Gap shrinks with specification | Gap can flip (LLM > human at S3/S4) |
| (new) H5: Frontier models resist prompting | Frontier models are MORE controllable, not less |

---

## Extended Results (n=322)

Ran partial full experiment with frontier models on all 113 items.

### Combined Results by Spec Level

| Spec | Error Rate | Target Rate | n |
|------|------------|-------------|---|
| S1 | 4.9% | 0.0% | 82 |
| S2 | 5.0% | 0.0% | 80 |
| **S3** | **43.8%** | **88.6%** | 80 |
| S4 | 35.0% | 78.6% | 80 |

### Misconception-Specific Findings

| Misconception | Type | S3 Error Rate | S3 Target Rate |
|---------------|------|---------------|----------------|
| 1507 (left-to-right) | procedural | **54.7%** | **88.6%** |
| 1214 (same operation) | procedural | 0.0% | N/A |

**Key finding:** Not all misconceptions are equally inducible. 1507 responds well to S3 prompting (55% error, 89% target), while 1214 is resistant (0% errors even with S3/S4).

---

## Conclusions

1. **S3 (Mental Model) is the optimal specification level** - 88.6% target alignment when errors occur
2. **Frontier models show controlled behavior** - Low error rates at S1/S2, high target alignment at S3/S4
3. **Misconception resistance varies** - Some misconceptions (1214) may require different prompting strategies
4. **Human baseline exceeded** - LLM target rates (88.6%) far exceed human baseline (38.2%)

---

*Extended experiment: 2026-01-26*
