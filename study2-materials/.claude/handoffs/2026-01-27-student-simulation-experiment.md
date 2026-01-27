# Student Simulation Experiment - Handoff

**Date:** 2026-01-27
**Status:** Complete

## What We Did

Ran a student simulation experiment using the teacher-prediction prompting approach from the literature, with:
- 12 UK student profiles with bidirectional confusion examples
- 36 final items across 3 misconceptions
- Multi-model ensemble matched to proficiency levels:
  - Struggling → Llama 3.1 8B (Groq)
  - Developing → GPT-3.5-turbo
  - Secure → GPT-4o-mini
  - Confident → Claude Sonnet 4

Total: 432 API calls (12 students × 36 items)

## Key Results

### Internal Validity: Good
The models differentiate proficiency levels as expected:

| Proficiency | Correct | Target Misconception |
|-------------|---------|---------------------|
| Struggling | 35.2% | 27.8% |
| Developing | 30.6% | 36.1% |
| Secure | 47.2% | 23.6% |
| Confident | 93.5% | 1.9% |

### External Validity: Poor
No correlation with actual human data:

| Metric | Pearson r | p-value |
|--------|-----------|---------|
| Correct rate | -0.185 | 0.28 |
| Target rate | -0.092 | 0.59 |

**MAE:** 35.3pp for correct rate, 26.2pp for target rate

### Worst Divergences
Items where LLMs diverge most from humans:

| Question | LLM Correct | Human Correct | Diff |
|----------|-------------|---------------|------|
| Q1718 | 100% | 10.1% | 89.9pp |
| Q1430 | 91.7% | 5.8% | 85.8pp |
| Q525 | 100% | 22.3% | 77.7pp |

## Conclusion

**The expert blindspot persists even with:**
- Weaker models for struggling students
- Teacher-prediction framing (not direct role-play)
- Bidirectional confusion examples
- Named UK students with realistic backgrounds

The models can rank students internally (confident > secure > developing > struggling) but this ranking doesn't map onto which items are actually hard for real students.

**Root cause:** LLMs judge difficulty by mathematical complexity, not by how automatically misconceptions activate. When simulating Callum (struggling), the LLM knows he "has trouble with fractions" but still recognizes 2/7 + 1/7 as "simple same-denominator addition" and answers correctly.

## Files Created

```
pilot/student_simulation/
├── results_20260127_001931.jsonl  # 432 raw responses
├── summary_20260127_001931.json   # Metadata
└── comparison_with_human.csv       # Item-level comparison

scripts/
├── run_student_simulation.py       # Main experiment
└── analyze_student_simulation.py   # Analysis
```

## Comparison with Previous Experiment

| Approach | Correct r | Target r |
|----------|-----------|----------|
| Opus difficulty estimation | 0.19 | n/a |
| Student simulation (multi-model) | -0.19 | -0.09 |

Both approaches fail to predict human difficulty. The simulation approach is actually slightly worse (negative correlation).

## Case Study: Q1430 (2/7 + 1/7)

This item reveals exactly how the expert blindspot works:

**Human data:**
- 5.8% correct (C: 3/7)
- 64.7% chose target misconception (D: 3/14)

**LLM data:**
- 91.7% correct (all proficiency levels chose C)
- 0% chose target misconception

**The model's reasoning for struggling student Callum:**
> "Callum might add the numerators (2 + 1 = 3) and keep the original denominator (7)"

But that's the **correct method**! The actual misconception is adding BOTH numerators AND denominators:
- Wrong: 2/7 + 1/7 = (2+1)/(7+7) = 3/14
- Right: 2/7 + 1/7 = (2+1)/7 = 3/7

The model knows "fraction addition misconception exists" but doesn't correctly model how it actually fires. It describes the correct procedure while labeling it as a misconception.

## Implications for AIED Research

1. **LLMs cannot substitute for empirical student data** - not even with sophisticated prompting
2. **Internal consistency ≠ external validity** - models can self-consistently simulate different proficiency levels, but the pattern doesn't match reality
3. **The expert blindspot may be fundamental** - training on expert-written text means models don't internalize how misconceptions actually fire
4. **EEDI's value proposition is confirmed** - real student response data is irreplaceable

## Next Steps (if continuing)

1. Try fine-tuning on actual student response data
2. Test with explicit misconception prevalence rates ("70% of students make this error")
3. Compare against random baseline to quantify how much worse than chance
4. Document findings for paper
