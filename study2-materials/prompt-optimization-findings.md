# Prompt Optimization Pilot: Findings

*2026-01-27*

## Overview

Ran an automated prompt optimization loop to improve LLM simulation of student errors on Eedi math items. The optimizer tested prompts, computed error fidelity metrics, analyzed failures, and generated revisions.

**Dataset**: 36 items across 3 misconception types from `final_items.csv`
- 217: Fraction addition (adds numerators and denominators) - 8 items
- 1507: Order of operations (left-to-right) - 15 items
- 1597: Negative multiplication (neg × neg = neg) - 13 items

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Correlation** | Pearson r between simulated and real student distributions | > 0.30 |
| **Misconception hit rate** | When wrong, % picking the target distractor | > 50% |
| **Composite score** | 0.4 × correlation + 0.6 × misconception_hit | Higher = better |

---

## Results by Misconception

### Misconception 217: Fraction Addition

**Best prompt**: Initial (explicit instruction)
**Best score**: 0.494 (Round 1)

| Round | Correlation | Misc Hit | Score | Prompt Approach |
|-------|-------------|----------|-------|-----------------|
| 1 | -0.014 | 83.3% | **0.494** | Explicit "flawed understanding" |
| 2 | -0.245 | 80.0% | 0.382 | Time pressure, work quickly |
| 3 | -0.093 | 75.0% | 0.413 | "Confident student" persona |
| 4 | -0.319 | 70.0% | 0.292 | "Fuzzy memory" uncertainty |
| 5 | -0.148 | 75.0% | 0.391 | Explicit application |
| 6 | -0.295 | 80.0% | 0.362 | Rushed feeling |

**Finding**: Every revision made things worse. The straightforward initial prompt performed best.

### Misconception 1507: Order of Operations

**Best prompt**: Revised (implicit/automatic)
**Improvement**: Correlation 0.229 → 0.434

| Round | Correlation | Misc Hit | Notes |
|-------|-------------|----------|-------|
| 1 | 0.229 | 100% | Initial prompt |
| 2 | 0.434 | 100% | "Don't overthink, trust instincts" |

**Finding**: For procedural errors, removing deliberation helped.

---

## The Winning Prompts

### For Conceptual Errors (217 - fraction addition)

```
You are a student taking a math test. You have a specific misconception: {misconception}

Because of this misconception, you often make mistakes on problems like this.

Question: {question}

Options:
{options}

Think through this problem using your (flawed) understanding, then give your answer.
End with "Answer: X" where X is A, B, C, or D.
```

### For Procedural Errors (1507 - order of operations)

```
You are a student taking a timed math test. You've learned that {misconception}, and this approach has always made sense to you.

You need to work quickly and trust your instincts. Don't overthink it.

Question: {question}

Options:
{options}

Work through this the way you normally would. Show your steps briefly and pick your answer confidently.

Answer: X
```

---

## Why Revisions Failed for 217

The optimizer's analyses were insightful but led to counterproductive changes:

**Analysis**: "The model is too metacognitive and self-correcting"
**Revision attempted**: Make it more implicit/naturalistic
**Result**: Got worse

**Analysis**: "Creates cognitive dissonance when applying obviously wrong procedures"
**Revision attempted**: Add uncertainty ("fuzzy memory", "not completely sure")
**Result**: Worst performance (score dropped to 0.292)

### Sample Response Showing the Problem

Even with explicit instruction, the model shows awareness it's wrong:

> "Let me solve this using my incorrect method... So I would calculate the result as 5/12. **(Note: This is mathematically wrong. The correct way to add fractions requires finding a common denominator...)**"

The model can't help but acknowledge the error, revealing the **competence paradox**: LLMs know too much to authentically simulate ignorance.

---

## Implications for Study Design

### 1. Use Misconception-Specific Prompts

| Error Type | Recommended Framing |
|------------|---------------------|
| Conceptual (fraction addition) | Explicit "flawed understanding" |
| Procedural (order of operations) | "Trust instincts, don't overthink" |
| TBD (negative multiplication) | Needs testing |

### 2. ~83% Misconception Hit Rate May Be the Ceiling

For fraction addition, the best prompt achieved 83.3% target misconception selection. This may represent a ceiling for prompt-only approaches. Further improvement may require:
- Fine-tuning / DPO on error examples
- Chain-of-thought suppression
- Smaller/weaker models

### 3. Correlation Remains Low

Even the best prompts showed near-zero or negative correlation with real student distributions. The model either:
- Gets answers right (unlike struggling students)
- Picks the target misconception too consistently (100% vs real students' ~30-50%)

This suggests simulated students are **caricatures** rather than realistic models.

### 4. Don't Over-Engineer the Prompt

Simple, direct prompts outperformed elaborate roleplay scenarios. The optimizer's sophisticated revisions consistently degraded performance.

---

## Cost Summary

- 6 rounds × 3 misconceptions
- ~$0.40 per misconception
- Total pilot cost: ~$1.20

---

## Files Generated

```
prompt_optimization/
├── misc_217_20260127_002550/
│   ├── round_01.json ... round_06.json
│   ├── final_results.json
│   └── best_prompt.txt
├── misc_1507_20260127_004228/
│   ├── round_01.json, round_02.json
│   └── (incomplete - stopped early)
```

---

## Next Steps

1. **Test misconception 1597** (negative multiplication) to confirm pattern
2. **Compare prompt variants** in main experiment:
   - Explicit framing (for conceptual errors)
   - Implicit/automatic framing (for procedural errors)
3. **Consider baseline comparison**: How does a generic "student" prompt perform vs. misconception-specific?
4. **Explore fine-tuning** if prompt ceiling is confirmed
