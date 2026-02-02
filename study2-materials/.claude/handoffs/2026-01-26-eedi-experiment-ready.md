# Study 2: Eedi-Based Experiment Ready

## Current State

**Experiment uses real Eedi items with human response distributions.** Ready to run S1-S4 specification level experiment with ground truth comparison.

---

## Key Files

| File | Purpose |
|------|---------|
| `data/experiment_items.json` | 113 Eedi items with human response data |
| `data/DATA_DICTIONARY.md` | Full schema and provenance documentation |
| `scripts/run_eedi_experiment.py` | Main experiment runner |
| `KCexperimentaldesign.md` | Experimental design document |
| `paper_draft_v2.md` | Paper draft with placeholders |

---

## Data Sources

| Source | Location | Contents |
|--------|----------|----------|
| Eedi Kaggle | `data/eedi/train.csv` | 1,869 questions with misconception labels |
| NeurIPS 2020 | `data/neurips_eedi/` | 15.8M student responses |
| Merged | `data/experiment_items.json` | 113 items with human distributions |

---

## 4 Target Misconceptions

| ID | Type | Name | Items | Human Target Rate |
|----|------|------|-------|-------------------|
| 1507 | Procedural | Left-to-right operations | 32 | 24.8% |
| 1214 | Procedural | Same op instead of inverse | 41 | 21.9% |
| 1597 | Conceptual | neg × neg = neg | 26 | 22.4% |
| 217 | Conceptual | Add fractions by adding num+denom | 14 | 22.6% |

---

## Alignment Verified

Spot-checked items confirm target distractors match misconception logic:

- **1507**: 1-3n with n=15 → Left-to-right gives (1-3)×15 = -30 ✓
- **1214**: 5x+3y=15 → Same-op gives gradient 5/3 instead of -5/3 ✓
- **1597**: 10-2d where d=-2 → neg×neg=neg gives 6 instead of 14 ✓
- **217**: 3/(2x)+4/x → Add num+denom gives 7/(3x) ✓

---

## Quick Test Results (Misconception 217)

| Spec | Error Rate | Target Rate | Human Target |
|------|------------|-------------|--------------|
| S1 | 0% | N/A | 31.6% |
| S2 | 50% | 0% | 31.6% |
| S3 | 100% | **100%** | 31.6% |
| S4 | 100% | 50% | 31.6% |

**S3 produced authentic misconception reasoning:**
> "add the fractions by combining the numerators and the denominators separately"

---

## To Run Full Experiment

```bash
cd /Users/dereklomas/AIED/study2-materials
source .venv/bin/activate

# Full experiment (4 models × 113 items × 4 specs × 3 reps = 5,424 calls)
python scripts/run_eedi_experiment.py --output pilot/eedi_full

# Quick test (1 model, 5 items per misconception)
python scripts/run_eedi_experiment.py \
  --models gpt-3.5-turbo \
  --sample 5 \
  --reps 1 \
  --output pilot/eedi_quick
```

---

## Experiment Design

### Factors

| Factor | Levels |
|--------|--------|
| Specification Level | S1, S2, S3, S4 |
| Model Tier | Frontier (GPT-4o, Claude 3.5 Sonnet), Mid (GPT-3.5, Haiku) |
| Misconception Type | Procedural (1507, 1214), Conceptual (1597, 217) |

### Primary DV

**Gap = Human Target Rate - LLM Target Rate**

- Positive gap: LLM underperforms human misconception pattern
- Zero gap: LLM matches human misconception pattern
- Negative gap: LLM overproduces misconception pattern

### Secondary DVs

1. **Reasoning alignment**: Does LLM explanation match misconception?
2. **Consistency**: Does LLM give same answer across reps?
3. **Discriminant validity**: Does LLM avoid unrelated misconceptions?

---

## Hypotheses

1. **H1**: Target rate increases S1 → S4
2. **H2**: S4 optimal for procedural, S3 for conceptual
3. **H3**: Gap shrinks with specification level
4. **H4**: Frontier models need more specification than mid-tier

---

## Next Steps

1. [x] Verify API keys
2. [x] Verify misconception-item alignment
3. [x] Run quick test
4. [ ] Run full experiment
5. [ ] Code misconception reasoning
6. [ ] Statistical analysis
7. [ ] Fill in paper results

---

*Updated: 2026-01-26*
