# Structured IRT Estimation Experiment — Handoff

**Date**: 2026-01-30
**Status**: Pipeline working, answer bug fixed, Gemini 3 Flash baseline complete

## Current State

### What's Working
- `scripts/run_structured_estimation.py` — Multi-provider (Gemini/Opus) structured estimation with XML extraction, resume support, 100% extraction rate
- `scripts/train_irt_prediction.py` — ML evaluation pipeline (Ridge/Lasso/ElasticNet/RF × 4 feature sets × 4 targets × holdout + LOO-CV)
- Gemini 3 Flash run complete (500 calls, ~$1.58): `pilot/structured_estimation_gemini3flash/`

### Key Bug Found and Fixed
The `eedi_with_student_data.csv` file has TWO different answer orderings:
- `AnswerAText`-`DText` → **Kaggle order**
- `pct_A`-`pct_D` → **Student-facing display order**

The prompt was using `correct_answer` (display order) with answer texts in Kaggle order, giving the LLM the wrong correct answer for **76% of items**. Fixed by switching to `correct_answer_kaggle`.

**The IRT parameters themselves are correct** — computed from raw `IsCorrect` flags, no letter-order dependency.

### Results So Far (Gemini 3 Flash, fixed answers, holdout n=20)

| Target | Best ML model | r | Direct LLM r |
|--------|--------------|---|-------------|
| a_2pl (discrimination) | Lasso + all_28 | **0.471** | -0.297 |
| b_2pl (difficulty) | RF + ratings_mean_std | 0.350 | 0.273 |
| c_3pl_guess | Ridge + all_28 | 0.192 | 0.019 |
| difficulty_classical | Ridge + ratings_mean_std | 0.169 | -0.286 |

Key finding: ML on structured ratings substantially outperforms direct LLM predictions. But holdout is only 20 items — high variance.

## Next Steps

1. **Fix remaining prompt issues** (already in code, not yet run):
   - `classical_difficulty` label says "proportion correct" but target is proportion **incorrect**
   - Range hint "-5 to 2" should be "typically -1 to 1, extreme range -5 to 2"
2. **Run Opus** (`--model opus`, ~$13.75 or $6.88 batch) — expect better calibration
3. **Consider**: more reps, different temperature, or LOO-CV-only evaluation (more stable than 20-item holdout)
4. **Compare models**: Gemini 3 Flash vs Opus side-by-side on same items/split

## Key Decisions Made
- 5 calibration items at b_2pl percentiles (10/30/50/70/90), excluded from train/test
- 80/20 train/holdout split stratified by `target_type`
- 5 replications per item, temp=0.7
- Thinking disabled for Gemini 3 Flash (thinking tokens ate output budget)
- 12 rating dimensions + 4 direct predictions → 28 aggregated features (means + stds)

## Files Modified This Session
- `scripts/run_structured_estimation.py` (new) — multi-provider LLM estimation
- `scripts/train_irt_prediction.py` (new) — ML evaluation pipeline
- `.env` — added GOOGLE_API_KEY

## Files Created (outputs)
- `pilot/structured_estimation_gemini3flash/` — features.csv, raw_results.csv, split_info.json, evaluation_report.md, evaluation_results.csv, raw_responses/

## Data Architecture Reference
```
eedi_with_student_data.csv:
  AnswerAText-DText     → Kaggle order
  pct_A-pct_D           → Student display order
  CorrectAnswer         → Kaggle letter (matches AnswerXText)
  neurips_correct_pos   → Student display letter (matches pct_X)

curated_eedi_items.csv:
  correct_answer        → Student display order (= neurips_correct_pos)
  correct_answer_kaggle → Kaggle order (= CorrectAnswer) ← USE THIS IN PROMPTS

irt_proper_statistics.json:
  difficulty_classical  → proportion INCORRECT (not correct!)
  b_2pl, a_2pl         → from raw IsCorrect, ordering-independent
```
