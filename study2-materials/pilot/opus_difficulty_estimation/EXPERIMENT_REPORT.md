# Opus 4.5 Difficulty Estimation Experiment

**Date:** 2026-01-26
**Experimenter:** Derek Lomas + Claude
**Model:** claude-opus-4-5-20251101

## Research Question

Can Claude Opus 4.5 estimate the difficulty of math assessment items for UK secondary students without access to actual student performance data?

## Experimental Design

### Setup
- **25 test items** stratified across 4 misconception types (~6 each)
- **10 independent calls** with the same 25 items
- **10 estimates per item** to measure consistency
- **3 calibration examples** with actual difficulty revealed (anchors at 39%, 89%, 93% error rates)

### Information Available to Model
- Question text and answer options
- Correct answer indicated
- Target misconception name for one distractor
- Population context: UK secondary students (ages 11-16), 2018-2020

### Information Withheld (to prevent "cheating")
- Actual student response percentages (pct_A, pct_B, pct_C, pct_D)
- Total response counts
- Any empirical difficulty statistics

### Misconception Types Tested
| ID | Type | Misconception |
|----|------|---------------|
| 1507 | Procedural | Carries out operations left to right regardless of priority |
| 1214 | Procedural | Uses same operation rather than inverse when solving equations |
| 1597 | Conceptual | Believes multiplying two negatives gives negative |
| 217 | Conceptual | When adding fractions, adds numerators and denominators |

## Results

### Consistency: Excellent
| Metric | Value |
|--------|-------|
| Mean within-item std | 2.5 percentage points |
| Range of std | 1.2 - 4.8 pp |

Opus gives nearly identical estimates across all 10 calls. The model is highly reliable in its judgments.

### Accuracy: Poor
| Metric | Value | p-value |
|--------|-------|---------|
| Pearson r | 0.192 | 0.36 |
| MAE | 38.9 pp | - |
| RMSE | 41.6 pp | - |

### Relative Ranking: Also Poor
| Metric | Value | p-value |
|--------|-------|---------|
| Spearman ρ | 0.119 | 0.57 |
| Kendall τ | 0.077 | 0.59 |
| Quartile agreement | 28% | (chance = 25%) |

Even after normalization, Opus cannot rank which items are harder than others.

### By Misconception Type
| Type | MAE | Correlation |
|------|-----|-------------|
| Conceptual | 33.0 pp | r = 0.39 |
| Procedural | 44.3 pp | r = -0.03 |

### Systematic Bias: Underestimation
Opus consistently predicts items are **easier** than they actually are.

| Item | Predicted | Actual | Error |
|------|-----------|--------|-------|
| QID 644 | 35% | 100% | -65 pp |
| QID 1430 | 24% | 94% | -70 pp |
| QID 1493 | 30% | 92% | -62 pp |
| QID 243 | 60% | 12% | +48 pp |

## Qualitative Analysis: Why Opus Fails

### Case Study 1: QID 1430 (Fraction Addition)
**Question:** Calculate 2/7 + 1/7

**Opus reasoning:** "Basic fraction addition with same denominators is foundational. However, distractor D (adding numerators and denominators) is a common misconception that catches some students."

**Opus estimate:** 25% error
**Actual:** 94% error (only 5.8% correct)

**Diagnosis:** Opus assumes "same denominator = easy." But students apply the wrong procedure (add tops, add bottoms) **automatically**, regardless of whether denominators already match. The misconception isn't conditional.

### Case Study 2: QID 1493 (Order of Operations)
**Question:** 8 - 7 ÷ 6³ = ? Which calculation first?

**Opus reasoning:** "Powers (indices) come before division, which many students know, but some may choose left-to-right."

**Opus estimate:** 30% error
**Actual:** 92% error (only 8.1% identified exponents first)

**Diagnosis:** Opus assumes students "know" BIDMAS. They've heard of it but cannot apply it. 69.4% chose division first—they know division before subtraction but miss that exponents come before everything.

### Case Study 3: QID 243 (Inequalities) - Opposite Error
**Question:** Solve 3n + 2 ≥ 4n - 5

**Opus reasoning:** "Solving inequalities with variables on both sides is challenging. Students often make errors with inverse operations."

**Opus estimate:** 65% error
**Actual:** 12% error (87.9% correct)

**Diagnosis:** Opus equates mathematical complexity with difficulty. But this problem has clear structure that students can follow. The "inverse operations" misconception doesn't trigger here.

## Key Finding: The Expert Blindspot

Opus judges difficulty by:
1. Mathematical complexity of the operation
2. Intellectual awareness that misconceptions exist

Opus fails to model:
1. **Automaticity**: Misconceptions are default procedures, not occasional errors
2. **Procedure dominance**: Students execute learned (wrong) procedures without thinking
3. **Context insensitivity**: Wrong procedures fire regardless of problem features
4. **Learnability**: Some "complex" problems have patterns students successfully learn

**Opus knows misconceptions exist but doesn't understand they're the default behavior, not exceptions.**

## Implications

### For LLM-Based Assessment
- Frontier models cannot substitute for empirical difficulty data
- Even with calibration examples, absolute estimates are unreliable
- Relative rankings are no better than random
- The "expert blindspot" may be fundamental to how LLMs process math

### For Synthetic Student Research
- LLMs simulating student errors face the same blindspot
- Models may produce errors that "look like" misconceptions but don't match real frequencies
- Validating synthetic students requires empirical benchmarks like EEDI

### Possible Improvements to Test
1. **More calibration examples** (10-20 instead of 3)
2. **Explicit misconception prevalence data** ("70% of students have this misconception")
3. **Chain-of-thought with student simulation** (have Opus role-play as student first)
4. **Fine-tuning on difficulty prediction task**

## Files

```
pilot/opus_difficulty_estimation/
├── EXPERIMENT_REPORT.md          # This document
├── config_20260126_175729.json   # Experiment configuration
├── results_20260126_175729.jsonl # Raw estimates (250 rows)
├── summary_20260126_175729.json  # Call metadata
└── analysis_20260126_175729.csv  # Comparison with actual
```

## Scripts

```
scripts/
├── run_opus_difficulty_estimation.py  # Main experiment
└── analyze_opus_estimates.py          # Analysis script
```

## Reproduction

```bash
cd /Users/dereklomas/AIED/study2-materials
python3 scripts/run_opus_difficulty_estimation.py
python3 scripts/analyze_opus_estimates.py
```

## Conclusion

**Opus 4.5 cannot estimate item difficulty for student populations, even with calibration.**

The model is highly consistent (low variance across calls) but systematically wrong (no correlation with actual difficulty). This suggests a fundamental limitation: LLMs judge difficulty by mathematical complexity, not by how automatically students apply misconceptions.

This finding has implications for the broader project of using LLMs as synthetic students—if models can't predict which items are hard, they may not accurately simulate which errors students make.
