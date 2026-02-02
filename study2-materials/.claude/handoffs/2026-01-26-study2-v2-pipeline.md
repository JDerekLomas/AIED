# Study 2 v2: Reasoning Authenticity Gap - Pipeline Ready

## Current State

**Pipeline ready with real Eedi items and student data.** Data dictionary and provenance documented.

### Key Insight from Pilot
- Pilot: 50% target distractor rate but only 8.7% misconception alignment
- **41.3 percentage point Reasoning Authenticity Gap**
- LLMs select "right wrong answers" without authentic misconception reasoning

## Major Update: Real Student Data

Now using Eedi items with actual student response distributions:
- **Kaggle 2024**: 1,869 items with expert misconception labels
- **NeurIPS 2020**: 15.8M student responses with selection percentages
- **Joined on QuestionId**: Both misconception labels AND student baselines

### Curated Items: 58 items across 6 misconceptions

| Misconception | Type | Items | Avg Student Selection |
|--------------|------|-------|----------------------|
| PROC_ORDER_OPS | procedural | 12 | 49% |
| PROC_BORROW_SUBTRACT | procedural | 5 | 52% |
| CONC_FRAC_DENOM | conceptual | 5 | 71% |
| CONC_NEG_MULTIPLY | conceptual | 16 | 49% |
| PROC_SQUARE_DOUBLE | procedural | 15 | 54% |
| CONC_DIV_COMMUTATIVE | conceptual | 5 | 68% |

## Spot Checks Completed

Verified alignment of items to misconceptions:
- Q0 (Order of Ops): 73.3% select "doesn't need brackets" - students who think addition before multiplication
- Q351 (Square=Double): 78.9% select "32" for "16² = ?" - students thinking squaring = doubling
- Q529 (Frac Denom): 74.2% order by denominator size
- Q1252 (Neg Multiply): 22.1% select "-12" for "-3 × -4"

## Scripts Created

### 1. Data Collection: `collect_eedi_study.py` (NEW)
- Uses real Eedi items from `curated_eedi_items.csv`
- S1-S4 prompts for 6 misconceptions
- Compares LLM responses to student baselines

### 2. Original Collection: `test_specification_levels.py`
- 6 models across 3 tiers:
  - Frontier: `claude-sonnet-4`, `gpt-4o`
  - Mid: `gpt-4o-mini`, `claude-3-haiku`
  - Weak: `llama-3.1-8b`, `mixtral-8x7b`

### 3. LLM-Assisted Coding: `code_misconceptions_llm.py`
- Uses Claude Haiku to code CoT reasoning
- Codes: FULL_MATCH, PARTIAL_MATCH, DIFFERENT_ERROR, UNCLEAR

### 4. Statistical Analysis: `analyze_study2.py`
- 3-way ANOVA (Spec × Tier × Misconception Type)
- Planned contrasts (S4 vs S1, type interactions)

### 5. Pipeline Runner: `run_study2_pipeline.py`
```bash
# Quick pilot (mid tier, 1 rep)
python run_study2_pipeline.py --pilot

# Full experiment (all tiers, 3 reps)
python run_study2_pipeline.py --full
```

## Data Files

```
data/eedi/
├── eedi_with_student_data.csv   # 1,869 items: misconceptions + student %s
├── curated_eedi_items.csv       # 58 items: filtered for target misconceptions
├── target_misconception_items.csv # Alternative with original 4 types
├── DATA_DICTIONARY.md           # Field definitions
└── DATA_PROVENANCE.md           # Data sources, joins, transformations
```

## Design: 4×3×2 Factorial

**Specification Level (S1-S4)**
| Level | Name | What's Specified |
|-------|------|-----------------|
| S1 | Persona | "Struggling student" |
| S2 | Knowledge State | Know/Learning/Unknown |
| S3 | Mental Model | Flawed belief system |
| S4 | Production Rules | Step-by-step algorithm |

**Model Tier**
- Frontier (~95% GSM8K): Claude Sonnet 4, GPT-4o
- Mid (~75% GSM8K): GPT-4o-mini, Claude Haiku
- Weak (~50% GSM8K): Llama-3.1-8B, Mixtral-8x7B

**Misconception Type** (updated with better coverage)
- Procedural: PROC_ORDER_OPS, PROC_BORROW_SUBTRACT, PROC_SQUARE_DOUBLE
- Conceptual: CONC_FRAC_DENOM, CONC_NEG_MULTIPLY, CONC_DIV_COMMUTATIVE

## Next Steps

1. [ ] Run pilot with Eedi items: `python collect_eedi_study.py --pilot`
2. [ ] Compare LLM target distractor rate vs student baseline
3. [ ] Code CoT reasoning for misconception alignment
4. [ ] Calculate Reasoning Authenticity Gap per condition
5. [ ] Run full experiment if pilot looks good

## Files Modified This Session

- `scripts/test_specification_levels.py` - Updated with Claude Sonnet 4, tier support
- `scripts/code_misconceptions_llm.py` - NEW: LLM-assisted reasoning coding
- `scripts/analyze_study2.py` - NEW: ANOVA, contrasts, figures
- `scripts/run_study2_pipeline.py` - NEW: Pipeline runner
- `scripts/collect_eedi_study.py` - NEW: Eedi-based collection
- `data/eedi/eedi_with_student_data.csv` - NEW: Joined Eedi data
- `data/eedi/curated_eedi_items.csv` - NEW: 58 curated items
- `data/eedi/DATA_DICTIONARY.md` - NEW: Field documentation
- `data/eedi/DATA_PROVENANCE.md` - NEW: Data lineage

## Key Research Questions

**RQ1**: Does the Reasoning Authenticity Gap persist across prompting conditions?
**RQ2**: Do different specs work better for different misconception types?
**RQ3**: Does model capability interact with specification level?
**RQ4**: Do LLM target distractor rates correlate with actual student rates?

## Paper Status

- Paper draft: `paper_draft_v2.md` - Complete with [PLACEHOLDER] for results
- Design: `experimental-design-v2.md` - Finalized
- Prompts: `prompt-strategies-v2.md` - All S1-S4 templates ready
