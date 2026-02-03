# Final Dataset: 36-Item Diagnostic Mathematics Assessment

## Overview

This document describes the final dataset used in our AIED 2026 study investigating whether large language models can simulate students with specific mathematical misconceptions.

**Dataset size:** 36 item-misconception pairs (35 unique questions; Q1390 tests two misconceptions)

The items were selected from the Eedi NeurIPS 2020 Education Challenge dataset using psychometric criteria to ensure high-quality diagnostic items where misconceptions are genuinely observable in human student data.

---

## Selection Process

### Source Pool

We began with 113 diagnostic items from the Eedi platform covering four common mathematical misconceptions. These items were originally selected because they had misconception-tagged distractors with measurable human selection rates.

### Psychometric Filtering

We applied five sequential filters based on Item Response Theory (IRT) and Bayesian Knowledge Tracing (BKT) analyses:

| Step | Criterion | Threshold | Items Remaining | Rationale |
|------|-----------|-----------|-----------------|-----------|
| 0 | Source pool | - | 113 | Original curated items |
| 1 | Exclude misconception 1214 | P(L₀) < 0.95 | 72 | BKT showed 97% initial mastery—students rarely exhibit this misconception |
| 2 | Discrimination | r > 0.50 | 68 | Items must effectively distinguish high/low ability students |
| 3 | Response count | n > 100 | 51 | Ensures stable human baseline estimates (SE < 5%) |
| 4 | Human target rate | > 10% | 38 | Misconception must be observably exhibited by humans |
| 5 | Difficulty | 20-80% incorrect | 36 | Avoids floor/ceiling effects |

### Excluded Misconception: 1214

Misconception 1214 ("When solving an equation, uses the same operation rather than the inverse") was excluded entirely based on BKT analysis showing P(L₀) = 0.967. This means 97% of students already understood inverse operations before encountering these items—the "misconception" is rarely exhibited even when students make errors.

This exclusion is further supported by our LLM experiment results: both GPT-4o and Claude Sonnet 4 showed 0% errors on misconception 1214 items across all prompting conditions, suggesting this misconception cannot be reliably induced.

---

## Final Dataset Composition

### Items by Misconception

| ID | Misconception | Type | Items | Human Target Rate |
|----|---------------|------|-------|-------------------|
| 1507 | Carries out operations from left to right regardless of priority order | Procedural | 15 | M=27.3%, SD=15.8% |
| 1597 | Believes multiplying two negatives gives a negative answer | Conceptual | 13 | M=30.5%, SD=21.3% |
| 217 | When adding fractions, adds the numerators and denominators | Conceptual | 8 | M=46.4%, SD=29.9% |
| **Total** | | | **36** | **M=32.7%, SD=22.7%** |

### Items by Type

| Type | Items | Description |
|------|-------|-------------|
| Procedural | 15 | Rule-following errors (incorrect procedure applied consistently) |
| Conceptual | 21 | Belief-based errors (incorrect mental model of mathematical concepts) |

### Note on Question 1390

Question 1390 (`1/3 + 1/2 × 1/3 = ?`) appears twice in the dataset because it diagnoses two different misconceptions depending on which wrong answer is selected:

- **Target B** (misconception 1507): Student ignores order of operations, computing left-to-right
- **Target C** (misconception 217): Student adds fractions incorrectly

Both mappings were tested in the LLM experiment, and both successfully induced the respective misconceptions in S3/S4 conditions.

---

## Psychometric Properties

### Item Response Theory (IRT) Parameters

| Parameter | Mean | SD | Min | Max | Interpretation |
|-----------|------|-----|-----|-----|----------------|
| Difficulty (Classical) | 0.41 | 0.12 | 0.21 | 0.67 | Proportion incorrect; moderate difficulty |
| Discrimination | 0.72 | 0.14 | 0.51 | 0.95 | Point-biserial correlation; all items discriminate well |
| Guessing Estimate | 0.26 | 0.07 | 0.14 | 0.42 | Near chance level for 4-choice items |

**Interpretation:**
- All items have discrimination > 0.50, indicating they effectively separate students who understand from those who don't
- Mean difficulty of 0.41 (41% incorrect) provides good variance for detecting misconception-aligned errors
- Guessing estimates near 0.25 suggest minimal floor effects

### Bayesian Knowledge Tracing (BKT) Parameters

| Misconception | P(L₀) | P(T) | P(G) | P(S) | Sequences |
|---------------|-------|------|------|------|-----------|
| 1507: Order of operations | 68.0% | 10.0% | 33.9% | 18.9% | 2,216 |
| 1597: Negative multiplication | 90.9% | 10.0% | 7.9% | 34.3% | 3,240 |
| 217: Fraction addition | 78.3% | 10.0% | 1.0% | 42.2% | 280 |

**Parameter Definitions:**
- **P(L₀)**: Initial probability of mastery before encountering items
- **P(T)**: Probability of learning (transition from unknown to known)
- **P(G)**: Probability of guessing correctly when skill not mastered
- **P(S)**: Probability of slipping (error despite mastery)

**Key Observations:**
- Misconception 1507 (order of operations) has the lowest initial mastery (68%) and highest guess rate (34%), suggesting this is the most challenging concept and students often succeed by chance
- Misconception 217 (fraction addition) has the highest slip rate (42%), indicating even students who understand fractions frequently make this error
- All misconceptions show elevated slip rates (19-42%), meaning a substantial portion of human errors may be careless rather than misconception-driven

---

## Human Response Data

### Student Population

The human baseline data comes from naturalistic Eedi platform usage:

- **Students**: ~48,000 who answered items in this subset
- **Total responses**: 73,340 across the 36 items
- **Collection period**: September 2018 - May 2020
- **Demographics**: UK students, primarily ages 10-17

### Response Distribution

| Statistic | Value |
|-----------|-------|
| Mean responses per item | 845 |
| Minimum responses | 103 |
| Maximum responses | 4,825 |
| Items with n > 500 | 17 (47%) |
| Items with n > 1,000 | 8 (22%) |

All items have sufficient responses (n > 100) for stable baseline estimates with standard errors below 5%.

---

## Data Files

### Primary Files

| File | Format | Rows | Description |
|------|--------|------|-------------|
| `results/final_items.csv` | CSV | 36 | Item-misconception pairs with metadata |
| `results/final_items.json` | JSON | - | Items with nested structure, BKT, and experiment summary |
| `results/final_responses.csv` | CSV | 231 | LLM responses filtered to this subset |

### CSV Schema: `final_items.csv`

| Column | Type | Description |
|--------|------|-------------|
| question_id | int | Unique Eedi question identifier |
| question_text | string | Full question text (may contain LaTeX) |
| correct_answer | string | A, B, C, or D |
| target_distractor | string | Misconception-tagged option (A, B, C, or D) |
| misconception_id | int | Eedi misconception ID (217, 1507, or 1597) |
| misconception_name | string | Human-readable misconception description |
| misconception_type | string | "procedural" or "conceptual" |
| human_target_rate | float | % of humans selecting target distractor |
| total_responses | int | Number of human responses |
| pct_A, pct_B, pct_C, pct_D | float | Human response distribution (%) |
| irt_difficulty | float | Proportion incorrect (0-1) |
| irt_discrimination | float | Point-biserial correlation (0-1) |
| irt_guessing | float | Estimated guessing parameter |
| option_A, option_B, option_C, option_D | string | Answer option text |

### CSV Schema: `final_responses.csv`

| Column | Type | Description |
|--------|------|-------------|
| response_id | int | Unique response identifier |
| model | string | "gpt-4o" or "claude-sonnet-4" |
| model_tier | string | "frontier" |
| spec_level | string | S1, S2, S3, or S4 |
| misconception_id | int | Target misconception ID |
| misconception_name | string | Human-readable misconception description |
| misconception_type | string | "procedural" or "conceptual" |
| question_id | int | Eedi question identifier |
| correct_answer | string | A, B, C, or D |
| target_distractor | string | Misconception-tagged option |
| llm_answer | string | LLM's selected answer |
| is_correct | bool | True if LLM answered correctly |
| hit_target | bool | True if LLM selected target distractor |
| human_target_rate | float | Human baseline for this item (0-1) |
| rep | int | Repetition number |
| timestamp | string | ISO timestamp of response |

---

## Misconception Descriptions

### 1507: Order of Operations (Procedural)

**Description:** Carries out operations from left to right regardless of priority order

**Example:**
- Question: Calculate 3 + 4 × 2
- Correct: 3 + (4 × 2) = 3 + 8 = **11**
- Misconception: (3 + 4) × 2 = 7 × 2 = **14**

**Cognitive basis:** Students apply operations sequentially as they appear rather than following BIDMAS/PEMDAS precedence rules. This is a procedural error—students know there are rules but apply the wrong procedure.

### 1597: Negative Multiplication (Conceptual)

**Description:** Believes multiplying two negatives gives a negative answer

**Example:**
- Question: Calculate (-3) × (-4)
- Correct: **+12**
- Misconception: **-12**

**Cognitive basis:** Students have an incorrect mental model where "negative times anything stays negative." This is a conceptual error—a fundamental misunderstanding of how negative numbers behave under multiplication.

### 217: Fraction Addition (Conceptual)

**Description:** When adding fractions, adds the numerators and denominators

**Example:**
- Question: Calculate 1/2 + 1/3
- Correct: 3/6 + 2/6 = **5/6**
- Misconception: (1+1)/(2+3) = **2/5**

**Cognitive basis:** Students incorrectly generalize the pattern of multiplication (multiply across) to addition. This reflects a flawed mental model of what fractions represent and how addition operates on them.

---

## LLM Experiment Results (Filtered Dataset)

### Response Summary

After filtering to this 36-item subset, the LLM experiment data contains:

| Statistic | Value |
|-----------|-------|
| Total responses | 231 |
| Models | GPT-4o (144), Claude Sonnet 4 (87) |
| Spec levels | S1, S2, S3, S4 |
| Responses per spec level | 57-58 |

### Error Rates by Specification Level

| Spec Level | Error Rate | Errors | Description |
|------------|------------|--------|-------------|
| S1 (Persona only) | 5.2% | 3/58 | "Student who struggles with math" |
| S2 (Knowledge state) | 6.9% | 4/58 | ZPD-style description |
| S3 (Mental model) | 56.9% | 33/58 | Explicit belief description |
| S4 (Production rules) | 45.6% | 26/57 | Step-by-step incorrect procedure |

### Target Rates (Among Errors)

| Spec Level | Target Rate | Hits | Interpretation |
|------------|-------------|------|----------------|
| S1 | 0.0% | 0/3 | Errors are random, not misconception-aligned |
| S2 | 0.0% | 0/4 | Errors are random, not misconception-aligned |
| S3 | 87.9% | 29/33 | Errors strongly align with target misconception |
| S4 | 84.6% | 22/26 | Errors strongly align with target misconception |

### Target Rates by Misconception (S3 + S4 Combined)

| Misconception | Target Rate | Errors | Interpretation |
|---------------|-------------|--------|----------------|
| 1507: Order of operations | 87.1% | 31 | Successfully induced |
| 1597: Neg × Neg = Neg | 84.6% | 13 | Successfully induced |
| 217: Fraction addition | 86.7% | 15 | Successfully induced |

### Key Findings

1. **Mental model prompts (S3) produce highest misconception alignment** at 87.9%, slightly outperforming production rules (S4) at 84.6%

2. **All three misconceptions successfully induced** with target rates of 85-87%

3. **Vague prompts (S1, S2) do not induce misconceptions** - the few errors that occur are random (0% target rate)

4. **LLM target rates substantially exceed human baselines** - LLMs hit the target distractor 85-88% of the time vs. humans at 27-46%

---

## Limitations

### 1. Single Misconception Per Item

Each item has only ONE tagged misconception. Other distractors may reflect different misconceptions, but these mappings are not provided. When analyzing LLM responses, we can only determine if errors match the *tagged* misconception.

### 2. Observational Human Data

Students were not randomly assigned to items. Assignment was based on curriculum, teacher choices, and prior performance. This may create confounds in human baseline estimates.

### 3. Slip Rate Interpretation

With slip rates of 19-42%, a substantial portion of human errors are likely careless mistakes rather than misconception-driven. LLM errors induced by prompting may be more purely misconception-aligned than the mixed human error distribution.

### 4. Text-Only Items

Some Eedi items originally included diagrams or images. We selected items that are comprehensible from text alone, but some mathematical notation may not render perfectly in all contexts.

---

## Citation

If using this dataset, please cite:

1. **Eedi NeurIPS 2020 Education Challenge** (source data)
   - Wang, Z., et al. (2020). Diagnostic Questions: The NeurIPS 2020 Education Challenge. Kaggle.

2. **This study** (item selection methodology and LLM response data)
   - [Authors]. (2026). Inducing Authentic Misconceptions in LLM Synthetic Students: The Role of Prompt Specification. AIED 2026.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial 36-item subset based on psychometric filtering |
| 1.1 | 2026-01-27 | Added filtered LLM responses (231 rows), experiment results summary |
