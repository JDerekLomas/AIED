# Study 2: Ready to Run Experiment - Session Handoff

## Current State

**Experiment designed and scripts ready.** Ready to run the S1-S4 specification level experiment.

---

## Key Files

| File | Purpose |
|------|---------|
| `KCexperimentaldesign.md` | Full experimental design with theoretical grounding |
| `paper_draft_v2.md` | Complete paper draft with placeholders for results |
| `execution_plan.md` | Step-by-step execution plan |
| `scripts/run_experiment.py` | Main experiment runner (API calls, checkpointing) |
| `scripts/test_specification_levels.py` | Alternative runner (also updated) |

---

## Design Summary

### 4 × 3 × 2 Factorial

| Factor | Levels |
|--------|--------|
| Specification Level | S1 (Persona), S2 (Knowledge State), S3 (Mental Model), S4 (Production Rules) |
| Model Capability | Frontier, Mid, Weak |
| Misconception Type | Procedural, Conceptual |

### Models

| Tier | Models |
|------|--------|
| Frontier | GPT-4o, Claude 3.5 Sonnet |
| Mid | GPT-3.5-turbo, Claude 3 Haiku |
| Weak | Llama-3.1-8B, Mistral-7B |

### Misconceptions

| Type | ID | Description |
|------|-----|-------------|
| Procedural | PROC_ORDER_OPS | Left-to-right instead of PEMDAS |
| Procedural | PROC_SUBTRACT_REVERSE | Smaller-from-larger in each column |
| Conceptual | CONC_FRAC_DENOM | Larger denominator = larger fraction |
| Conceptual | CONC_MULT_INCREASE | Multiplication always increases |

### Total API Calls

- 4 specs × 6 models × 4 misconceptions × 7 items × 3 reps = **2,016 calls**
- Estimated cost: ~$20
- Estimated time: ~70 minutes

---

## To Run Experiment

```bash
cd /Users/dereklomas/AIED/study2-materials
source .venv/bin/activate

# Full experiment
python scripts/run_experiment.py --output pilot/spec_experiment

# Quick test (1 model, 1 misconception)
python scripts/run_experiment.py \
  --models gpt-3.5-turbo \
  --misconceptions PROC_ORDER_OPS \
  --reps 1 \
  --output pilot/quick_test
```

---

## Key Findings from Pilot

- **Reasoning Authenticity Gap**: 41.3 percentage points
- Target distractor rate: 50.0%
- Misconception alignment rate: 8.7%

---

## Hypotheses

1. **H1**: Alignment increases S1 → S4
2. **H2**: S4 works for procedural, S3 for conceptual
3. **H3**: At S4, capability differences diminish
4. **H4**: Gap shrinks with specification level

---

## Pre-registered Outcomes

| Outcome | Pattern | Implication |
|---------|---------|-------------|
| A | S4 >> S1 | Specification works, use KC profiles |
| B | Type × Spec interaction | Match spec type to error type |
| C | Gap persists | Fundamental limitation |
| D | Only weak models work | Capability paradox is real |

---

## Timeline

| Task | Date |
|------|------|
| Run experiment | Jan 28-29 |
| Misconception coding | Jan 29-30 |
| Analysis | Jan 30-31 |
| Write-up | Feb 1-2 |
| **Submit** | **Feb 2** |

---

## Next Steps

1. [ ] Verify API keys (`python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY')[:10])"`)
2. [ ] Run quick test with 1 model
3. [ ] Run full experiment
4. [ ] Code misconceptions
5. [ ] Run analysis
6. [ ] Fill in paper results
