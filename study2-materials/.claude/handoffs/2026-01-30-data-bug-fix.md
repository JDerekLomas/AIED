# Data Bug Fix: Kaggle vs NeurIPS Answer Ordering

## Bug Description
The Eedi dataset has TWO independent answer orderings:
- **Kaggle**: `CorrectAnswer` A/B/C/D with `AnswerAText`, `AnswerBText`, etc.
- **NeurIPS**: `AnswerValue` 1-4 mapped to positional A/B/C/D, with `pct_A/B/C/D` and `resp_A/B/C/D`

These orderings are **completely independent** (24.5% match = chance). Only 457/1869 items happen to have the same correct answer letter in both systems.

## Impact
Every script that computed `pct_{correct_answer}` using Kaggle's `CorrectAnswer` was looking up the wrong NeurIPS position. For 76% of items, the "actual error rate" was computed from a random wrong option.

Calibration anchors given to LLMs were off by 56-63pp. All difficulty estimation experiments were scored against garbage ground truth.

## What Was Fixed

### Data files
- `data/eedi/eedi_with_student_data.csv` — regenerated with `neurips_correct_pos` column
- `data/eedi/curated_eedi_items.csv` — `correct_answer` now = NeurIPS position; added `correct_answer_kaggle`; renamed `target_distractor` → `target_distractor_kaggle`; renamed `student_selection_pct` → `student_selection_pct_kaggle`
- `results/final_items.json` — added `correct_answer_neurips` field

### Scripts (pct lookup now uses neurips_correct_pos)
- `scripts/prepare_eedi_data.py` — step1 extracts NeurIPS correct, step3 uses it
- `scripts/run_opus_difficulty_estimation.py` — reads fixed curated CSV
- `scripts/analyze_opus_estimates.py` — reads fixed curated CSV
- `scripts/analyze_student_simulation.py` — uses `correct_answer_neurips`
- `scripts/replicate_direct_difficulty.py` — uses `neurips_correct_pos`
- `scripts/replicate_feature_extraction.py` — uses `neurips_correct_pos`
- `scripts/replicate_uncertainty_difficulty.py` — uses `neurips_correct_pos`
- `scripts/replicate_classroom_simulation.py` — uses `neurips_correct_pos`
- `scripts/replicate_confusion_tuples.py` — uses `neurips_correct_pos` for correct, target still Kaggle
- `scripts/replicate_error_alignment.py` — added `correct_neurips`, fixed student accuracy

## What CANNOT Be Fixed
The Kaggle→NeurIPS mapping only exists for the **correct answer** (via cross-referencing both datasets' CorrectAnswer fields). We cannot map individual distractor positions because:
- NeurIPS data has no answer texts, only position 1-4
- Kaggle data has answer texts but in its own ordering
- No shared key to align individual options

This means:
1. **`target_distractor`** — still uses Kaggle letter; pct lookup is wrong
2. **`student_selection_pct`** — computed from wrong Kaggle distractor position
3. **Full distribution comparisons** (error_alignment, confusion_tuples) — LLM dist uses Kaggle order but student dist uses NeurIPS order; per-option comparisons are scrambled
4. **Error-only correlations** — partially fixable (correct answer split is fixed, but option-level comparison still misaligned)

## Experiment Implications
All experiments that computed **item difficulty / error rate** are now fixed and should be re-run.

Experiments comparing **full response distributions** (error alignment, confusion tuples) have a fundamental data alignment problem that cannot be resolved with available data.
