# Eedi Data for Study 2: Reasoning Authenticity Gap

## Overview

This directory contains math misconception items from the Eedi platform, augmented with real student response distributions.

**Key insight**: We can compare LLM behavior to actual student behavior because we have:
1. Expert-labeled misconceptions per distractor (Kaggle 2024)
2. Student response distributions (NeurIPS 2020 - 15.8M responses)

---

## Data Sources

| Dataset | Source | Records | Purpose |
|---------|--------|---------|---------|
| Kaggle 2024 | [Competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) | 1,869 questions | Misconception labels |
| NeurIPS 2020 | [Challenge](https://eedi.com/projects/neurips-education-challenge) | 15.8M responses | Student distributions |

**Citations**:
```
Eedi. (2024). Eedi - Mining Misconceptions in Mathematics. Kaggle.

Wang, Z. et al. (2020). Diagnostic Questions: The NeurIPS 2020 Education Challenge.
arXiv:2007.12061.
```

---

## Files

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 1,869 | Kaggle source - questions + misconception labels |
| `misconception_mapping.csv` | 2,587 | Kaggle source - misconception definitions |
| `eedi_with_student_data.csv` | 1,869 | Joined dataset with student response %s |
| `curated_eedi_items.csv` | 113 | Items for experiment (4 target misconceptions) |

---

## Target Misconceptions

| ID | Type | Name | Items | Avg Selection |
|----|------|------|-------|---------------|
| 1507 | Procedural | Carries out operations left to right regardless of priority | 32 | 24.8% |
| 1214 | Procedural | Uses same operation rather than inverse when solving equations | 41 | 21.9% |
| 1597 | Conceptual | Believes multiplying two negatives gives negative | 26 | 22.4% |
| 217 | Conceptual | When adding fractions, adds numerators and denominators | 14 | 22.6% |

---

## Schema: `eedi_with_student_data.csv`

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `QuestionId` | int | Kaggle | Unique identifier |
| `ConstructName` | str | Kaggle | Math topic being tested |
| `QuestionText` | str | Kaggle | Question stem (contains LaTeX) |
| `CorrectAnswer` | str | Kaggle | A, B, C, or D |
| `Answer[A-D]Text` | str | Kaggle | Option text |
| `Misconception[A-D]Id` | float | Kaggle | Misconception ID per distractor (NaN if correct/unlabeled) |
| `resp_[A-D]` | int | NeurIPS | Student response counts |
| `total_responses` | int | Derived | Sum of resp_A through resp_D |
| `pct_[A-D]` | float | Derived | Selection percentage per option |

## Schema: `curated_eedi_items.csv`

| Column | Type | Description |
|--------|------|-------------|
| `target_key` | str | Our misconception key (ORDER_OPS, etc.) |
| `target_type` | str | "procedural" or "conceptual" |
| `QuestionId` | int | Links to eedi_with_student_data.csv |
| `question_text` | str | Full question |
| `correct_answer` | str | A, B, C, or D |
| `target_distractor` | str | Option with target misconception |
| `target_answer` | str | Text of target distractor |
| `misconception_id` | int | Eedi MisconceptionId |
| `misconception_name` | str | Misconception description |
| `student_selection_pct` | float | % students selecting target |
| `total_responses` | int | Total student responses |
| `pct_[A-D]` | float | Full response distribution |

---

## Reproducibility

To regenerate processed files from raw data:

```bash
# Requires raw NeurIPS data in data/eedi/data/train_data/train_task_1_2.csv
python scripts/prepare_eedi_data.py

# To skip the join step if eedi_with_student_data.csv exists:
python scripts/prepare_eedi_data.py --skip-join
```

**Raw data locations** (not committed to git):
- `data/eedi/data/` - Unzipped NeurIPS 2020 data
- `data/eedi/neurips_data.zip` - Original download

---

## Example Item

**Question 86** (Misconception 217: Adding fractions by adding parts)

```
Question: Work out: 4/11 + 7/11. Write your answer in its simplest form.

A) 11/22  <- TARGET DISTRACTOR (51% of students)
B) 1/2
C) 11/11
D) 1      <- CORRECT (35% of students)

Misconception: Student adds numerators (4+7=11) AND denominators (11+11=22)
```

---

## Data Quality Notes

1. **LaTeX content**: Question/answer text contains LaTeX markup
2. **Image references**: Some questions reference images not in text
3. **Response reliability**: Items have 41-5,892 responses (median 274)
4. **Population**: UK students, Sept 2018 - May 2020

---

*Last updated: 2026-01-26*
