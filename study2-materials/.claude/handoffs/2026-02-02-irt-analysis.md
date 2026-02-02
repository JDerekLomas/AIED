# IRT / Item Analysis on Indian Ed Assessment Data

## Dataset
- **Source**: `item_response_df_all_data.csv` (16.1M rows, 728K students, 61K schools)
- **Subjects**: Maths, English, Hindi
- **Grades**: Class 3, 4, 6, 7
- **Items**: 20 per subject-grade = 240 total
- **Date range**: April 24–May 1, 2025
- **Competency tables**: `{Maths,English,Hindi}_competency_table_complete.xlsx`

## Approach
1. **Identified image-dependent items** by scanning Question Text for picture/image/figure keywords (20 items)
2. **Classical item analysis** on 10K student sample:
   - P(correct) = item difficulty
   - Point-biserial correlation = item discrimination
   - Rasch difficulty = -log(p/(1-p))
3. **Flagged 10 low-discrimination items** (rpb < 0.2) — mostly geometry, sequence-of-events, and morphology items
4. **Removed 30 total items** (20 image + 10 low-disc), leaving 210 text-based items
5. **80/20 train/test split** by student (no student overlap)

## Excluded Items

### Image-dependent (20)
- Maths Class 3: Q1,2,3,12,17,19
- Maths Class 4: Q20
- Maths Class 6: Q4
- Maths Class 7: Q16
- English Class 3: Q4,7,10
- English Class 4: Q3,19
- English Class 6: Q1
- Hindi Class 3: Q2,3,4,10
- Hindi Class 4: Q1

### Low discrimination, rpb < 0.2 (10)
- Maths Class 3: Q6 (numberline subtraction, rpb=0.15), Q20 (books covering tablecloth, rpb=0.17)
- Maths Class 4: Q16 (fractions, rpb=0.19), Q19 (perimeter, rpb=0.17)
- Maths Class 6: Q8 (largest number, rpb=0.20)
- Maths Class 7: Q2 (place value, rpb=0.11)
- English Class 4: Q15 (sequence of events, rpb=0.20), Q18 (sequence, rpb=0.15)
- Hindi Class 6: Q17 (poetry inference, rpb=0.14)
- Hindi Class 7: Q19 (morphology, rpb=0.05)

## Output Files
- `item_analysis_results.csv` — full item stats (220 items, before bad-item removal)
- `train.csv` — 11.3M rows, 582K students (80%)
- `test.csv` — 2.8M rows, 145K students (20%)

## Summary Stats
| Subject | Items | Mean P(correct) | Mean rpb | Mean Rasch Diff |
|---------|-------|-----------------|----------|-----------------|
| Maths   | 71→65 | 0.52            | 0.36     | -0.08           |
| English | 74→72 | 0.59            | 0.38     | -0.37           |
| Hindi   | 75→73 | 0.62            | 0.41     | -0.49           |

## Next Steps
- Replicate findings from original study (TBD — user hasn't specified which paper)
- Competency-level analysis
- Grade-level comparisons
- LLM-based distractor generation or misconception analysis (connects to study2 work)
