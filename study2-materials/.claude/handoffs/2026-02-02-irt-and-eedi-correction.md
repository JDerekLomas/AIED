# IRT Analysis & Eedi Ground Truth Correction

**Date**: 2026-02-02
**Status**: Complete

## What Was Done

### 1. Cross-Dataset IRT Analysis (R mirt package)

Fit 1PL and 2PL models on 500-student subsamples for DBE-KT22, SmartPaper, and Eedi.

| Dataset | Items | 1PL b vs p-correct | 2PL b vs p-correct | 2PL a mean |
|:---|:---:|:---:|:---:|:---:|
| DBE-KT22 | 212 | 1.000 | 0.800 | 0.80 |
| SmartPaper | 135 | 0.996 | 0.622 | 1.11 |
| Eedi | — | not fittable (3% density) | — | — |

**Key insight**: 1PL difficulty ≈ p-correct (ρ > 0.99). Classical p-correct is equivalent to Rasch difficulty and is the appropriate benchmark for the paper (LLM estimates p-correct, not discrimination).

### 2. DBE-KT22 Validation

- Author-assigned difficulty (1/2/3) does NOT correlate with empirical difficulty (ρ = 0.061, ns)
- Direct estimation (Gemini 2.5 Flash): ρ = 0.342
- Contrastive pipeline (Gemini 3 Flash, 3 reps, T=1.5): ρ = 0.440
- Falls between Eedi (null) and SmartPaper (0.547) — consistent with complexity-selectivity framework
- Generalizes findings from math to computer science domain

### 3. Eedi Ground Truth Correction

**Discovery**: NeurIPS competition shuffled answer positions per student. `CorrectAnswer` varies per student to track the shuffle. `IsCorrect` is valid. But aggregate `pct_A/B/C/D` in `eedi_with_student_data.csv` uses Kaggle ordering — unreliable for p-correct.

**Impact**: Recomputed all Eedi experiments against NeurIPS IsCorrect-derived p-correct. Full-set result unchanged (ρ ≈ 0). Small curated subsets shifted but are within sampling noise at n=20.

## Key Files

- `pilot/dbe_kt22_validation/RESULTS.md` — Full results with all sections
- `data/dbe-kt22/` — DBE-KT22 dataset
- `data/eedi/data/train_data/` — NeurIPS student-level responses (correct ground truth)
- `/tmp/irt_results_*.csv` — IRT parameter estimates from R mirt (temporary)

## What's Next

- SmartPaper optimized pipeline result still TBD in cross-dataset table
- Consider archiving old Eedi experiments that used aggregate-derived p-correct
- Paper draft should note that NeurIPS IsCorrect is the valid ground truth for Eedi
