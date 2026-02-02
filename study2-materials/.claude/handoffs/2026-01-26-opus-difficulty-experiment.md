# Opus Difficulty Estimation Experiment - Handoff

**Date:** 2026-01-26
**Status:** Complete

## What We Did

Ran an experiment to test whether Claude Opus 4.5 can estimate item difficulty for EEDI math assessment items without seeing actual student data.

## Key Results

- **Consistency:** Excellent (within-item std = 2.5pp across 10 calls)
- **Accuracy:** Poor (r = 0.19, MAE = 39pp)
- **Ranking:** Essentially random (Spearman ρ = 0.12)

**Main finding:** Opus systematically underestimates difficulty. It knows misconceptions exist but treats them as exceptions rather than default student behavior.

## Files Created

```
pilot/opus_difficulty_estimation/
├── EXPERIMENT_REPORT.md          # Full documentation
├── config_20260126_175729.json   # Experiment config
├── results_20260126_175729.jsonl # 250 estimates
├── summary_20260126_175729.json  # Metadata
└── analysis_20260126_175729.csv  # Comparison data

scripts/
├── run_opus_difficulty_estimation.py
└── analyze_opus_estimates.py
```

## Key Decisions

1. Used calibrated one-shot design (3 examples with actual difficulty)
2. Fixed 25 items × 10 calls for consistency measurement
3. Stratified sampling across 4 misconception types

## Next Steps (if continuing)

1. Test with more calibration examples (10-20)
2. Try chain-of-thought student simulation before estimating
3. Compare other models (GPT-4o, Sonnet)
4. Test if providing misconception prevalence rates helps

## Important Insight

> "Opus knows misconceptions exist but doesn't understand they're the default behavior, not exceptions."

This may be a fundamental limitation for LLM-based synthetic students.
