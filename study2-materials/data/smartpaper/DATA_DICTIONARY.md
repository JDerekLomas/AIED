# SmartPaper Dataset — Data Dictionary

## Source

Open-ended assessments administered via SmartPaper (AI-graded paper assessments) in government schools in Jodhpur, Rajasthan, India. Students in Grades 6, 7, 8 assessed on English, Mathematics, Science, and Social Science.

## Raw Data

### `export_item_responses.csv` (~120K rows)

| Column | Type | Description |
|--------|------|-------------|
| `OrganizationName` | string | School/organization name |
| `OrganizationId` | string | Organization identifier |
| `SubOrganizationId` | string | Sub-organization identifier |
| `SubOrganizationName` | string | Sub-organization name |
| `GroupId` | int | Class group identifier |
| `GradeLevel` | int | Grade level (6, 7, 8) |
| `Section` | string | Class section |
| `RollNumber` | int | Student roll number within group |
| `TargetId` | string | Assessment target identifier (12 unique = 12 assessments) |
| `AssessmentName` | string | e.g., "Grade 6 — English" |
| `QuestionNumber` | int | Question number within assessment |
| `Skills[0-2].skillId` | string | Skill tag IDs (up to 3 per item) |
| `Skills[0-2].skillName` | string | Skill tag names |
| `MaxScore` | float | Maximum possible score (0.5, 1.0, or 1.5) |
| `StudentScore` | float | AI-assigned score |
| `StudentAnswer` | string | Student's written response text |

**Student identification**: Unique students identified by (SubOrganizationId, GroupId, RollNumber) — 2,936 unique students total.

### `export_assessment_metadata.csv` (143 rows)

| Column | Type | Description |
|--------|------|-------------|
| `AssessmentName` | string | Assessment name |
| `QuestionNumber` | int | Question number |
| `Subject[0].subjectName` | string | Subject name |
| `GradeLevel` | int | Grade level |
| `ContentSubType` | string | Content type |
| `QuestionText` | string | Question text |
| `Rubrics[0-3].score` | float | Rubric score levels |
| `Rubrics[0-3].criteria` | string | Rubric criteria text |
| `Skills[0-2].skillName` | string | Skill tags |

## Processed Item Statistics

### `item_statistics.json` (140 items)

| Field | Type | Description |
|-------|------|-------------|
| `assessment` | string | Assessment name (e.g., "Grade 6 — English") |
| `question_number` | int | Question number within assessment |
| `subject` | string | English, Mathematics, Science, Social Science |
| `grade` | string | "6", "7", "8" |
| `content_type` | string | "openEnded" (all items) |
| `question_text` | string | Full question text |
| `rubric` | string | Scoring rubric with acceptable answers |
| `max_score` | string | Maximum score ("0.5", "1", "1.5") |
| `skill` | string | Primary skill tag (33 unique skills) |
| `n_responses` | int | Number of student responses |
| `classical_difficulty` | float | Proportion correct (p-value) |
| `item_total_correlation` | float | Point-biserial correlation with total score |
| `discrimination_27` | float | Upper-lower 27% discrimination index |
| `score_sd` | float | Standard deviation of scores |

## Coverage

### Items per Subject-Grade

| Subject | Grade 6 | Grade 7 | Grade 8 | Total |
|---------|---------|---------|---------|-------|
| English | 15 | 14 | 12 | 41 |
| Mathematics | 12 | 11 | 10 | 33 |
| Science | 10 | 11 | 11 | 32 |
| Social Science | 12 | 11 | 11 | 34 |
| **Total** | **49** | **47** | **44** | **140** |

### Student Counts per Assessment

| Assessment | Students |
|------------|----------|
| Grade 6 — English | 1,023 |
| Grade 6 — Maths | 1,025 |
| Grade 6 — Science | 1,067 |
| Grade 6 — Social Science | 1,044 |
| Grade 7 — English | 976 |
| Grade 7 — Maths | 1,023 |
| Grade 7 — Science | 1,009 |
| Grade 7 — Social Science | 1,004 |
| Grade 8 — English | 568 |
| Grade 8 — Maths | 618 |
| Grade 8 — Science | 621 |
| Grade 8 — Social Science | 585 |

Total unique students: 2,936. Total response rows: 120,133.

## Exclusion Criteria

143 items in raw metadata → 140 in item_statistics.json.

**3 items excluded** (all English fill-in-the-blank verb tense items):
1. Grade 6 — English Q7: "She __________ (go) to school every morning."
2. Grade 7 — English Q8: "She _______ (live) in this city since 2010."
3. Grade 7 — English Q9: "By next week, I _______ (complete) my science project."

Exclusion rationale: These are essentially single-word cloze items with near-deterministic answers, more similar to MCQ than open-ended items. They do not represent the open-ended construct being studied.

**No items excluded for**:
- Low discrimination (all items disc_27 ≥ 0.101, item-total ≥ 0.265)
- Low response count (minimum 539 responses)
- Image dependency (all items are text-only)

## Item Quality Summary

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Classical difficulty (p) | 0.293 | 0.037 | 0.825 |
| Discrimination (27%) | 0.475 | 0.101 | 0.860 |
| Item-total correlation | 0.556 | 0.265 | 0.770 |
| n responses | 921 | 539 | 1,067 |

## IRT Analysis

No IRT parameters (Rasch or 2PL) have been computed for this dataset. Classical difficulty (p-value) is used as ground truth. Per the cross-dataset IRT analysis in the DBE-KT22 validation, 1PL difficulty is a monotonic transform of p-correct (ρ > 0.98 across all tested datasets), so classical p-correct is equivalent to Rasch difficulty for ranking purposes.

## Notes

- All items are **open-ended** (free response, AI-graded), not MCQ
- Scoring is binary or partial credit (max_score 0.5, 1, or 1.5)
- For difficulty estimation, `classical_difficulty` is computed as mean(score / max_score)
- Rubrics are available for all items, providing the LLM with scoring criteria
- 33 unique skill tags across 4 subjects
