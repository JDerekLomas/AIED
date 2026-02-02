# Data Dictionary

## Source Datasets

### Kaggle Eedi Dataset
- **URL**: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics
- **Contents**: 1,869 diagnostic math questions with misconception labels
- **Key fields**: QuestionId, ConstructName, MisconceptionId, MisconceptionName, CorrectAnswer, Answer[A-D]Text

### NeurIPS 2020 Eedi Challenge
- **URL**: https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip
- **Contents**: 15,867,851 student responses
- **Key fields**: QuestionId, UserId, AnswerValue, IsCorrect, DateAnswered

---

## Experiment Data Files

### `experiment_items.json`

113 items selected for the experiment.

| Field | Type | Description |
|-------|------|-------------|
| question_id | int | Unique question identifier (from Eedi) |
| misconception_id | int | Target misconception ID |
| misconception_name | string | Human-readable misconception description |
| misconception_type | string | "procedural" or "conceptual" |
| construct_name | string | Math skill being assessed |
| question_text | string | Question stem (may contain LaTeX) |
| answer_a | string | Option A text |
| answer_b | string | Option B text |
| answer_c | string | Option C text |
| answer_d | string | Option D text |
| correct_answer | string | Correct option (A/B/C/D) |
| target_distractor | string | Option associated with misconception |
| human_responses | int | Number of student responses |
| human_distribution | object | {A: rate, B: rate, C: rate, D: rate} |
| human_correct_rate | float | Proportion answering correctly |
| human_target_rate | float | Proportion selecting target distractor |

### `pilot_items.json`

12-item random sample (3 per misconception) for pilot studies.

Same schema as `experiment_items.json`.

---

## Results Data Files

### `results/full_experiment/responses_*.jsonl`

JSONL format, one result per line.

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model name (e.g., "gpt-4o") |
| model_tier | string | "frontier" or "mid" |
| misconception_id | int | Target misconception |
| misconception_name | string | Misconception description |
| misconception_type | string | "procedural" or "conceptual" |
| spec_level | string | Specification level (S1/S2/S3/S4) |
| question_id | int | Question identifier |
| correct_answer | string | Correct option |
| target_distractor | string | Misconception-associated option |
| human_target_rate | float | Human baseline for this item |
| parsed_answer | string | LLM's selected answer (A/B/C/D or null) |
| is_correct | bool | Whether LLM answered correctly |
| hit_target | bool | Whether LLM selected target distractor |
| rep | int | Repetition number |
| raw_response | string | Full LLM response text |
| timestamp | string | ISO timestamp |

---

## Misconceptions

| ID | Name | Type | Items |
|----|------|------|-------|
| 1507 | Carries out operations from left to right regardless of priority order | procedural | 32 |
| 1214 | When solving an equation, uses the same operation rather than the inverse | procedural | 41 |
| 1597 | Believes multiplying two negatives gives a negative answer | conceptual | 26 |
| 217 | When adding fractions, adds the numerators and denominators | conceptual | 14 |

---

## Specification Levels

| Level | Name | Prompt Strategy |
|-------|------|-----------------|
| S1 | Persona | Generic "struggling student" roleplay |
| S2 | Knowledge State | ZPD-style knowledge description |
| S3 | Mental Model | Explicit belief about how math works |
| S4 | Production Rules | Step-by-step incorrect procedure |

---

## Data Processing Pipeline

### Step 1: Source Data
- **Kaggle Eedi** (downloaded 2026-01-26): 1,869 questions with misconception labels, answer texts, and `CorrectAnswer` (A/B/C/D in Kaggle ordering)
- **NeurIPS 2020 Eedi** (downloaded 2026-01-26): 15,867,851 student responses with `AnswerValue` (1-4 positional) and `IsCorrect`

### Step 2: Response Aggregation (`prepare_eedi_data.py`)
1. Aggregate 15.8M NeurIPS responses by QuestionId × AnswerValue
2. Pivot to `resp_A/B/C/D` and `pct_A/B/C/D` (NeurIPS positional ordering)
3. Extract NeurIPS `CorrectAnswer` (1→A, 2→B, 3→C, 4→D) as `neurips_correct_pos`
4. Join with Kaggle data on QuestionId → `eedi_with_student_data.csv`
5. Filter to 4 target misconceptions → `curated_eedi_items.csv`

### Step 3: IRT Analysis (`calculate_irt_proper.py`)
1. Estimate student ability (theta) from full 27K-item NeurIPS dataset
2. Fit 2PL and 3PL IRT models per item using MLE
3. Output: `results/irt_proper_statistics.json`

---

## Critical Data Issue: Kaggle vs NeurIPS Answer Ordering

### The Problem

The Kaggle and NeurIPS Eedi datasets use **completely independent** answer orderings:

- **Kaggle**: `CorrectAnswer` A/B/C/D refers to `AnswerAText`, `AnswerBText`, etc.
- **NeurIPS**: `AnswerValue` 1-4 mapped positionally to A/B/C/D; `pct_A/B/C/D` and `resp_A/B/C/D` use this ordering

Cross-tabulation of Kaggle `CorrectAnswer` vs NeurIPS `neurips_correct_pos` shows a **uniform distribution** — 24.5% match rate (pure chance for 4 options). The orderings are unrelated.

### Impact

Any code computing `pct_{Kaggle_CorrectAnswer}` looked up a **random wrong NeurIPS position** for 76% of items (1,412/1,869). This affected:

| What | How it was wrong |
|------|-----------------|
| Error rate ground truth | `100 - pct_{wrong_letter}` → off by 56-63pp for calibration anchors |
| Difficulty scoring | All experiments scored against garbage ground truth |
| Target distractor pct | `pct_{Kaggle_distractor_letter}` → random NeurIPS option |
| Student selection rates | Computed from wrong distractor position |
| Distribution comparisons | LLM options in Kaggle order, student pcts in NeurIPS order |

### What Was Fixed (2026-01-30)

**Correct answer lookups**: All scripts now use `neurips_correct_pos` for `pct_` column lookups.

**Data files updated**:
- `eedi_with_student_data.csv`: Added `neurips_correct_pos` column
- `curated_eedi_items.csv`: `correct_answer` = NeurIPS position; added `correct_answer_kaggle`
- `final_items.json`: Added `correct_answer_neurips` field

### What Cannot Be Fixed

The per-option Kaggle↔NeurIPS mapping only exists for the **correct answer** (both datasets identify it). Individual distractor mappings cannot be recovered because:
- NeurIPS data has no answer texts, only positional values 1-4
- No shared key to align individual options across datasets

This means:
- `target_distractor_kaggle`: Cannot be mapped to NeurIPS position
- `student_selection_pct_kaggle`: Computed from wrong position (renamed to flag this)
- Full response distribution comparisons: LLM sees Kaggle-ordered options, student pcts are NeurIPS-ordered — per-option correlations are scrambled

### Verification

After fix, Q189: `correct_answer=C`, `pct_C=63.34` → 37% error rate.
Before fix: `correct_answer=A`, `pct_A=9.46` → 90.5% error rate.
The corrected values align with expectations for diagnostic math items.

---

## Second Data Issue: Task 1+2 vs Task 3+4 Population Mismatch

The NeurIPS Eedi dataset has two response files:
- `train_task_1_2.csv`: 15.9M responses (used for `pct_A/B/C/D`)
- `train_task_3_4.csv`: Additional responses for a subset of items

**Key finding**: Task 3+4 represents a different population:
- Task 3+4 students score **16 percentage points lower** on the same items
- Cross-task p_correct correlation is **r ≈ 0** (essentially independent)
- 59/113 curated items have task 3+4 data; those with it have M=1,450 additional responses

**Resolution**: IRT analysis now uses task 1+2 only, matching the CSV ground truth. IRT classical difficulty correlates r=1.000 with corrected CSV error rates, confirming alignment.

**Impact of task 1+2 restriction on IRT**:
- Adaptive testing signal stronger: r=0.472 (vs r=0.306 with both tasks)
- Classical vs 2PL correlation: r=0.688 (vs r=0.525)
- Discrimination (rpb): M=0.370 (vs M=0.253)
- All 105 items well-fit (no bound-hitters)

---

## Column Reference

### `eedi_with_student_data.csv`

| Column | Source | Description |
|--------|--------|-------------|
| QuestionId | Both | Shared item identifier |
| CorrectAnswer | Kaggle | Correct option in Kaggle ordering (A/B/C/D) |
| neurips_correct_pos | NeurIPS | Correct option in NeurIPS ordering (A/B/C/D) |
| Answer[A-D]Text | Kaggle | Option texts in Kaggle ordering |
| Misconception[A-D]Id | Kaggle | Misconception IDs in Kaggle ordering |
| resp_A/B/C/D | NeurIPS | Response counts in NeurIPS ordering |
| pct_A/B/C/D | NeurIPS | Response percentages in NeurIPS ordering |
| total_responses | Derived | Sum of resp_A through resp_D |

### `curated_eedi_items.csv`

| Column | Description |
|--------|-------------|
| correct_answer | NeurIPS position — use with `pct_` columns |
| correct_answer_kaggle | Kaggle letter — use for answer text lookup and LLM prompts |
| target_distractor_kaggle | Kaggle letter for misconception distractor (cannot map to NeurIPS) |
| student_selection_pct_kaggle | **UNRELIABLE** — computed from Kaggle distractor in NeurIPS pct columns |
| pct_A/B/C/D | NeurIPS positional ordering |

---

*Last updated: 2026-01-30*
