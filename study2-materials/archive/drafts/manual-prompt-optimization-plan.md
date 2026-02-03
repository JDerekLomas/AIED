# Automated Prompt Optimization for Student Error Simulation

*Started: 2026-01-27*

## Overview

Automated prompt optimization loop that uses Claude to:
1. Test prompts on Eedi items
2. Compute error fidelity metrics
3. Analyze failures
4. Generate revised prompts
5. Repeat until convergence

---

## Data Available

**Eedi Dataset** (`data/eedi/curated_eedi_items.csv`):
- ~120 items across 4 misconception types
- Real student response distributions (pct_A, pct_B, pct_C, pct_D)
- Misconception-tagged distractors

| Misconception ID | Name | Items |
|------------------|------|-------|
| 217 | Adds fractions by adding num & denom | ~15 |
| 1507 | Left-to-right order of operations | ~30 |
| 1214 | Uses same op instead of inverse | ~35 |
| 1597 | Negative × negative = negative | ~25 |

---

## Optimization Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROUND N                                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Sample 10 items from training set                           │
│  2. Run current prompt 5× per item (Haiku, temp=0.8)            │
│  3. Compute metrics:                                             │
│     - Distribution correlation with real students                │
│     - Accuracy calibration (|sim - real|)                        │
│     - Misconception hit rate (when wrong, picked target?)        │
│  4. Sonnet analyzes: Why did it fail? What to change?            │
│  5. Sonnet generates revised prompt                              │
│  6. Track best prompt by composite score                         │
│  7. → Next round                                                 │
└─────────────────────────────────────────────────────────────────┘

After N rounds:
  → Evaluate best prompt on held-out test set (Sonnet)
  → Save results
```

---

## Metrics

| Metric | Computation | Target |
|--------|-------------|--------|
| **Distribution correlation** | Pearson r between sim & real distributions | > 0.70 |
| **Accuracy calibration** | mean |sim_correct% - real_correct%| | < 0.10 |
| **Misconception hit rate** | When wrong, % picking target distractor | > 50% |

**Composite score** = 0.4 × correlation + 0.6 × misconception_hit

---

## Cost Estimate

Per round:
- 10 items × 5 runs × Haiku = 50 calls ≈ $0.01
- 2 Sonnet calls (analysis + revision) ≈ $0.02
- **Per round: ~$0.03**

Full run (8 rounds + final eval):
- 8 rounds × $0.03 = $0.24
- Final eval (10 items × 5 runs × Sonnet) ≈ $0.15
- **Total per misconception: ~$0.40**

**All 4 misconceptions: ~$1.60**

---

## Running the Optimizer

```bash
cd /Users/dereklomas/AIED/study2-materials/scripts

# Single misconception (fraction addition)
python prompt_optimizer.py --misconception 217 --rounds 8

# All misconceptions
for m in 217 1507 1214 1597; do
    python prompt_optimizer.py -m $m -r 8
done
```

---

## Output Structure

```
prompt_optimization/
├── misc_217_20260127_143022/
│   ├── round_01.json      # Results + prompt for each round
│   ├── round_02.json
│   ├── ...
│   ├── final_results.json # Best prompt + test metrics
│   └── best_prompt.txt    # The winning prompt
```

---

## Initial Prompt (Starting Point)

```
You are a student taking a math test. You have a specific misconception: {misconception}

Because of this misconception, you often make mistakes on problems like this.

Question: {question}

Think through this problem using your (flawed) understanding, then give your answer.
End with "Answer: X" where X is A, B, C, or D.
```

---

## What We Expect to Learn

**If it works:**
- Optimized prompts will contain specific phrasing that elicits authentic errors
- We'll see what kinds of prompt features matter (role framing, examples, etc.)
- Can publish the prompts for others to use

**If it struggles:**
- Indicates the competence paradox may be fundamental
- Would motivate fine-tuning / DPO approaches
- Still valuable as a negative result

---

## Comparison to DSPy

| Aspect | This Approach | DSPy |
|--------|---------------|------|
| Search method | LLM-guided revision | Systematic enumeration |
| Interpretability | High (we see analysis) | Lower (black box) |
| Sample efficiency | Good (10 items/round) | Needs more data |
| Flexibility | Can add custom metrics | Metric must be numeric |
| Cost | ~$0.40 per misconception | ~$5-50 depending on config |

This approach is essentially **manual DSPy with LLM-in-the-loop** - cheaper and more interpretable for exploration.

---

## Next Steps

1. [x] Create optimizer script
2. [ ] Run on fraction addition misconception (217) as pilot
3. [ ] Analyze: What prompt changes helped?
4. [ ] Run on remaining misconceptions
5. [ ] Compare optimized vs baseline prompts
6. [ ] Document findings for paper

---

## Running Now

```bash
python prompt_optimizer.py -m 217 -r 8
```

This will:
- Optimize for ~25 minutes
- Cost ~$0.40
- Output to `prompt_optimization/misc_217_*/`
