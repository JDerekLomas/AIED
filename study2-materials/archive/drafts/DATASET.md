# Dataset Description

## Overview

This dataset examines whether large language models (LLMs) can simulate students with specific mathematical misconceptions. It combines diagnostic math items from the Eedi platform with LLM response data collected under different prompting strategies.

---

## Data Sources

### 1. Eedi Diagnostic Items (Source: NeurIPS 2020 Education Challenge)

**Origin:** Eedi is a UK-based educational technology company that provides diagnostic assessments for mathematics. Their data was released for the NeurIPS 2020 Education Challenge on Kaggle.

**Structure:** Each diagnostic item is a multiple-choice question with:
- One correct answer
- Three distractors (wrong answers)
- One distractor tagged with a specific misconception

**Key limitation:** Only ONE misconception is tagged per item. Other distractors may reflect different misconceptions, but these mappings are not provided in the dataset.

### 2. Human Response Data (Source: NeurIPS 2020 Education Challenge)

Real student response data from Eedi, providing:
- Which answer each student selected
- Aggregate statistics on distractor selection rates

We use the **human target rate** (proportion of students who selected the misconception-tagged distractor) as our baseline for comparison.

---

## Items Selected for This Study

We selected **113 diagnostic items** covering **4 common misconceptions**:

| Misconception ID | Name | Type | Items |
|-----------------|------|------|-------|
| 1507 | Carries out operations from left to right regardless of priority order | Procedural | 33 |
| 1214 | When solving an equation, uses the same operation rather than the inverse | Procedural | 40 |
| 1597 | Believes multiplying two negatives gives a negative answer | Conceptual | 26 |
| 217 | When adding fractions, adds the numerators and denominators | Conceptual | 14 |

**Selection criteria:**
- Items where the misconception-tagged distractor had measurable human selection rates
- Items spanning a range of difficulty levels
- Two procedural misconceptions (rule-following errors) and two conceptual misconceptions (belief-based errors)

---

## Experimental Design

### Independent Variables

**1. Specification Level (S1-S4):** How explicitly the misconception is described in the prompt

| Level | Name | Description |
|-------|------|-------------|
| S1 | Persona Only | "You are a student who sometimes struggles with math" |
| S2 | Knowledge State | Describes what the student knows/doesn't know (ZPD-style) |
| S3 | Mental Model | Describes the student's underlying belief about how math works |
| S4 | Production Rules | Gives an explicit step-by-step incorrect procedure |

**2. Model:** GPT-4o and Claude Sonnet 4 (frontier-tier models)

**3. Misconception:** The target misconception being induced

### Dependent Variables

**Error Rate:** Proportion of responses where the LLM answered incorrectly
```
error_rate = incorrect_responses / total_responses
```

**Target Rate:** Among errors only, proportion that selected the misconception-tagged distractor
```
target_rate = target_hits / incorrect_responses
```

This distinguishes "random errors" from "misconception-aligned errors."

---

## Output Data

### File: `results/tidy_responses.csv`

**Format:** Tidy data with one row per LLM response (541 rows, 16 columns)

| Column | Type | Description |
|--------|------|-------------|
| response_id | int | Unique identifier |
| model | string | gpt-4o or claude-sonnet-4 |
| model_tier | string | frontier |
| spec_level | string | S1, S2, S3, or S4 |
| misconception_id | int | Eedi misconception ID |
| misconception_name | string | Human-readable description |
| misconception_type | string | procedural or conceptual |
| question_id | int | Eedi question ID |
| correct_answer | string | A, B, C, or D |
| target_distractor | string | Misconception-tagged option |
| llm_answer | string | LLM's selected answer |
| is_correct | bool | True if correct |
| hit_target | bool | True if selected target distractor |
| human_target_rate | float | Proportion of humans who selected target (0-1) |
| rep | int | Repetition number |
| timestamp | string | ISO timestamp |

---

## Key Findings Summary

| Spec Level | Error Rate | Target Rate (among errors) |
|------------|------------|---------------------------|
| S1 | 5% | 0% |
| S2 | 6% | 0% |
| S3 | 53% | 87% |
| S4 | 52% | 81% |

- **S3 (Mental Model) produces highest misconception alignment** (87%)
- **3 of 4 misconceptions successfully induced** (1507, 1597, 217)
- **1 misconception resistant** (1214: same operation instead of inverse) - 0% errors
- **No item-level correlation with human data** (r = -0.07)

---

## Interpreting the "Target Distractor"

Each Eedi item has ONE distractor tagged with a misconception. For example:

**Question:** Calculate 3 + 4 × 2
**A)** 14 ← Target distractor (misconception 1507: left-to-right)
**B)** 11 ← Correct
**C)** 9
**D)** 10

A student with misconception 1507 would compute: 3+4=7, then 7×2=14, selecting A.

**Important:** Options C and D may also reflect misconceptions, but the dataset only provides the mapping for ONE misconception per item. When an LLM selects C or D, we code it as "error but not target" - we cannot determine if it reflects a coherent different misconception.

---

## Student Population

The human baseline data comes from naturalistic Eedi platform usage:

- **~119,000 students** from UK schools
- **Data collection period:** September 2018 - May 2020
- **Ages:** Primarily 10-17 years old (born 2003-2008)
- **15.8 million total responses** across 27,613 questions
- **~133 responses per student** on average

Students did NOT take all items at once. Different students answered different items based on their curriculum and teacher assignments. The `human_target_rate` represents the proportion of students *who answered that specific item* who selected the misconception-tagged distractor.

---

## Statistical Limitations

### 1. Highly Skewed Response Counts

| Statistic | Value |
|-----------|-------|
| Range | 41 - 4,825 responses per item |
| Median | 281 |
| Mean | 650 |
| Items with n < 100 | 28 (25%) |

The 117x variation in response counts means human target rate estimates have very different precision across items. For items with n=50, a 20% target rate has 95% CI of roughly ±11%.

**Recommendation:** Consider weighting analyses by response count, or conduct sensitivity analyses excluding low-n items.

### 2. Non-Representative Difficulty Sample

| Statistic | Value |
|-----------|-------|
| Difficulty range | 5% - 100% incorrect |
| Median difficulty | 88% incorrect |
| Mean difficulty | 74% incorrect |
| Items with >95% incorrect | 15 |

Our items skew HARD compared to typical diagnostic items. They were selected because they have misconception-tagged distractors, not because they represent typical classroom assessments. Some extremely difficult items may be flawed or depend on images not captured in the text.

### 3. Observational Data, Not Random Assignment

Students were not randomly assigned to items. Items were assigned based on:
- Curriculum sequence
- Teacher/school choices
- Student prior performance

This creates potential confounds: harder items may be systematically given to more advanced students, affecting the human target rate baseline.

**Note:** No significant correlation between difficulty and response count (r = 0.04), suggesting "hard items get abandoned" is not a major issue.

### 4. Single Misconception Mapping

Each item has only ONE misconception tagged, but:
- Other distractors likely reflect different misconceptions (unmapped)
- When LLMs select unmapped distractors, we cannot determine if this reflects a coherent misconception
- This may underestimate LLM misconception behavior for non-target misconceptions

### 5. Misconception 1214 Anomaly

LLMs showed 0% errors on misconception 1214 ("uses same operation instead of inverse"), despite:
- Normal human target rates (mean 22%, range 3-89%)
- Full difficulty range (11-100% incorrect)
- Similar item structure to other misconceptions

This resistance appears to be a genuine LLM behavior pattern, not a data artifact. Possible explanations:
- Inverse operations are heavily represented in LLM training data
- The misconception conflicts with LLMs' learned equation-solving procedures
- The belief framing doesn't override procedural knowledge for this domain

### 6. Low Item-Level Correlation with Humans

The near-zero correlation (r = -0.07) between LLM and human target rates at the item level may reflect:
- Measurement noise in low-n human estimates
- LLMs applying misconceptions uniformly rather than matching human item-difficulty patterns
- Fundamental differences in how LLMs and humans process these items

---

## Psychometric Properties

### Item Response Theory (IRT) Parameters

Classical IRT parameters were calculated from 73,340 student responses across 112 items.

| Parameter | Mean | SD | Description |
|-----------|------|-----|-------------|
| Difficulty (Classical) | 0.31 | 0.14 | Proportion incorrect (0-1) |
| Discrimination | 0.73 | 0.16 | Point-biserial correlation with total score |
| Guessing Estimate | 0.25 | - | P(correct) for bottom quartile students |

**Interpretation:**
- Mean difficulty of 0.31 indicates these items are relatively easy for this student population (69% correct on average)
- High mean discrimination (0.73) suggests items effectively distinguish between high and low ability students
- Guessing estimates near 0.25 align with 4-choice multiple choice chance level

### Bayesian Knowledge Tracing (BKT) Parameters

BKT parameters were estimated per misconception using an EM algorithm on sequential student response data.

| Misconception | P(L₀) | P(T) | P(G) | P(S) | Sequences |
|---------------|-------|------|------|------|-----------|
| 1507: Left-to-right operations | 68.0% | 10.0% | 33.9% | 18.9% | 2,216 |
| 1214: Same operation (not inverse) | 96.7% | 10.0% | 1.0% | 31.9% | 2,582 |
| 1597: Neg × Neg = Neg | 90.9% | 10.0% | 7.9% | 34.3% | 3,240 |
| 217: Add numerators and denominators | 78.3% | 10.0% | 1.0% | 42.2% | 280 |

**Parameter definitions:**
- **P(L₀)**: Initial probability of mastery before seeing items
- **P(T)**: Probability of learning (transitioning from not-known to known)
- **P(G)**: Probability of guessing correctly when skill not mastered
- **P(S)**: Probability of slipping (error when skill is mastered)

**Key observations:**
- High P(L₀) values (68-97%) indicate most students already understood these concepts
- Elevated slip rates (19-42%) suggest significant error rates even among students who understand the material
- Misconception 1214 shows near-ceiling initial mastery (97%), consistent with LLMs' resistance to this misconception
- Low learning rates (10%) reflect that these are diagnostic items, not instructional sequences

### Implications for Item Selection

The psychometric properties suggest several criteria for selecting a high-quality subset:

**1. Exclude near-ceiling items (P(L₀) > 0.95)**
Misconception 1214 shows 97% initial mastery—students rarely exhibit this misconception even when they make errors. This aligns with LLMs' complete resistance to this misconception (0% errors). Items targeting misconceptions that students don't actually hold are poor candidates for synthetic student simulation.

**2. Prefer high-discrimination items (r > 0.6)**
Items with discrimination > 0.6 effectively separate students who understand from those who don't. These items are more likely to elicit misconception-aligned errors rather than random guessing.

**3. Consider slip rates when interpreting results**
With slip rates of 19-42%, a substantial portion of human errors may be careless mistakes rather than misconception-driven. LLM "errors" induced by prompting may more closely resemble misconception-driven errors than the mixed human error distribution.

**4. Recommended subset criteria:**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Misconception | ≠ 1214 | Near-ceiling mastery (97% P(L₀)) |
| Discrimination | r > 0.5 | Excludes poorly-discriminating items |
| Response count | n > 100 | Ensures stable human baseline estimates |
| Human target rate | > 10% | Confirms misconception is observable |
| Difficulty | 20-80% incorrect | Excludes floor/ceiling effects |

**5. Recommended subset: 36 items**

Applying all filters yields 36 high-quality items:

| Misconception | Items | Type |
|---------------|-------|------|
| 1507: Left-to-right operations | 15 | Procedural |
| 1597: Neg × Neg = Neg | 13 | Conceptual |
| 217: Add numerators and denominators | 8 | Conceptual |
| **Total** | **36** | 21 conceptual, 15 procedural |

Subset statistics:
- Discrimination: M=0.72, SD=0.14 (all items effectively distinguish ability levels)
- Difficulty: M=41%, SD=12% (moderate difficulty, good variance)
- Human target rate: M=33%, SD=23% (misconceptions clearly observable)
- Response count: M=845, range 103-4,825 (stable estimates)

**Alternative subsets:**
- Relaxed (exclude 1214 only): 72 items
- Moderate (exclude 1214, disc > 0.4): 70 items
- Strict (all criteria above): 36 items

See `results/recommended_items.csv` for the full recommended subset.

### Files

IRT and BKT statistics are available in:
- `results/irt_bkt_statistics.json` - Full parameter estimates
- `results/item_details.json` - IRT parameters merged with item metadata
- `results/item_statistics.html` - Interactive visualization with IRT & BKT tab

---

## Prompts Used

See `results/prompts_used.json` for machine-readable prompt templates and `results/CODEBOOK.md` for full documentation.

---

## Citation

If using this data, please cite:

1. The Eedi NeurIPS 2020 Education Challenge (source data)
2. This study (methodology and LLM response data)

---

## Files

```
study2-materials/
├── DATASET.md                      # This file
├── ABSTRACT.md                     # Paper abstract
├── results/
│   ├── tidy_responses.csv          # Main analysis dataset (541 rows)
│   ├── CODEBOOK.md                 # Variable definitions and analysis examples
│   ├── prompts_used.json           # Prompt templates by spec level and misconception
│   ├── irt_bkt_statistics.json     # IRT and BKT parameter estimates
│   ├── item_statistics.html        # Interactive visualization
│   ├── item_details.json           # Item data with IRT and tier responses
│   ├── recommended_items.csv       # Recommended 36-item subset
│   └── recommended_items_summary.json  # Subset criteria and statistics
├── data/
│   └── eedi/                       # Source Eedi data from Kaggle
└── scripts/
    ├── run_conceptual.py           # Experiment script for conceptual misconceptions
    ├── run_full_experiment.py      # Full experiment script
    └── calculate_irt_bkt.py        # IRT/BKT calculation script
```
