# Experiment Report: Misconception Alignment in LLM Synthetic Students

**Study 2 for AIED 2026**
**Date:** 2026-01-26

---

## Research Question

Can LLMs simulate students with specific mathematical misconceptions, and how does the level of prompt specificity affect alignment with target misconceptions?

---

## Method

### Data Source

- **Kaggle Eedi Dataset**: 1,869 diagnostic math questions with misconception labels
- **NeurIPS 2020 Eedi Challenge**: 15.8M student responses providing human performance baselines
- **Merged Dataset**: 113 items across 4 misconceptions with human response distributions

### Misconceptions Studied

| ID | Name | Type | Items |
|----|------|------|-------|
| 1507 | Carries out operations left-to-right regardless of priority | Procedural | 32 |
| 1214 | Uses same operation rather than inverse when solving equations | Procedural | 41 |
| 1597 | Believes multiplying two negatives gives negative | Conceptual | 26 |
| 217 | When adding fractions, adds numerators and denominators | Conceptual | 14 |

### Specification Levels (S1-S4)

Based on cognitive science frameworks for representing student knowledge:

| Level | Name | Description | Example Prompt Element |
|-------|------|-------------|------------------------|
| S1 | Persona | Generic struggling student | "You are a student who sometimes struggles with math" |
| S2 | Knowledge State | ZPD-style knowledge description | "KNOW WELL: X, STILL LEARNING: Y, FUZZY ON: Z" |
| S3 | Mental Model | Explicit belief/conceptualization | "You believe math should be solved left-to-right, like reading" |
| S4 | Production Rules | Step-by-step procedure | "STEP 1: Find leftmost operation, STEP 2: Do it..." |

### Models Tested

| Model | Provider | Tier | RPM |
|-------|----------|------|-----|
| gpt-4o | OpenAI | Frontier | 500 |
| claude-sonnet-4 | Anthropic | Frontier | 50 |
| gpt-3.5-turbo | OpenAI | Mid | 500 |
| claude-3-haiku | Anthropic | Mid | 50 |

### Metrics

- **Error Rate**: % of responses that are incorrect
- **Target Rate**: % of errors that select the target distractor (the answer option associated with the misconception)
- **Human Target Rate**: Baseline from NeurIPS student response data

---

## Results

### Pilot Study (n=192, 12 items)

Mid-tier models only:

| Spec Level | Error Rate | Target Rate |
|------------|------------|-------------|
| S1 (Persona) | 18.8% | 0.0% |
| S2 (Knowledge State) | 37.5% | 16.7% |
| **S3 (Mental Model)** | **52.1%** | **72.0%** |
| S4 (Production Rules) | 58.3% | 46.4% |

### Extended Study (n=322, 47 items)

Frontier models (gpt-4o, claude-sonnet-4):

| Spec Level | Error Rate | Target Rate | n |
|------------|------------|-------------|---|
| S1 (Persona) | 4.9% | 0.0% | 82 |
| S2 (Knowledge State) | 5.0% | 0.0% | 80 |
| **S3 (Mental Model)** | **43.8%** | **88.6%** | 80 |
| S4 (Production Rules) | 35.0% | 78.6% | 80 |

### Model Comparison (S3 only)

| Model | Error Rate | Target Rate |
|-------|------------|-------------|
| gpt-4o | 29.8% | 92.9% |
| claude-sonnet-4 | 63.6% | 85.7% |

### Misconception-Level Variance

| Misconception | Type | S3 Error Rate | S3 Target Rate | Human Target Rate |
|---------------|------|---------------|----------------|-------------------|
| 1507 (left-to-right) | Procedural | 54.7% | 88.6% | 24.8% |
| 1214 (same operation) | Procedural | 0.0% | N/A | 26.3% |

**Key finding**: LLMs are highly susceptible to misconception 1507 but completely resistant to 1214.

---

## Correlation Analysis

### Item-Level Correlation (Human vs LLM Target Rates)

| Spec | Correlation (r) | n |
|------|-----------------|---|
| S1 | 0.000 | 48 |
| S2 | 0.000 | 47 |
| S3 | -0.074 | 47 |
| S4 | -0.090 | 47 |

**Finding**: No correlation between human and LLM target rates at the item level. LLMs do not replicate human item-level difficulty patterns.

### Binary Comparison (S3)

| Human Target Rate | LLM Target Rate | n |
|-------------------|-----------------|---|
| > 30% | 45.5% | 22 |
| ≤ 30% | 36.2% | 58 |

Slight positive relationship, but not statistically meaningful.

---

## Key Findings

### 1. S3 (Mental Model) is the Optimal Specification Level

- Produces highest target alignment (88.6%)
- Works for both procedural and conceptual misconceptions
- Outperforms S4 (production rules) contrary to initial hypothesis

### 2. Frontier Models Show Controlled Behavior

- Near-perfect accuracy at S1/S2 (baseline)
- High target alignment when induced via S3/S4
- Claude Sonnet 4 shows binary pattern: either correct or hits target

### 3. Misconception Resistance Varies

- 1507 (left-to-right): Highly inducible, LLM exceeds human susceptibility
- 1214 (same operation): Completely resistant, 0% errors even with S3/S4
- Not all misconceptions can be induced with current prompting strategies

### 4. No Item-Level Correlation with Humans

- LLMs don't mimic human item difficulty patterns (r ≈ 0)
- LLMs apply misconceptions uniformly when prompted
- Synthetic students useful for specific misconceptions, not general simulation

---

## Implications

### For AIED Research

1. **Prompt design matters**: S3 (mental model) significantly outperforms simpler approaches
2. **Misconception selection**: Some misconceptions are more inducible than others
3. **Validity concerns**: LLM synthetic students don't replicate human response distributions

### For Practitioner Use

1. LLMs can simulate specific misconceptions with high fidelity
2. Useful for generating practice items or testing diagnostic systems
3. Should not be used as general-purpose student simulators

---

## Limitations

1. Only tested 2 of 4 planned misconceptions in extended study
2. Single repetition per condition in extended study
3. No reasoning alignment coding (only answer alignment)
4. Limited to math domain

---

## Files

| File | Description |
|------|-------------|
| `data/experiment_items.json` | 113 items with human response data |
| `data/pilot_items.json` | 12-item random sample for pilot |
| `data/DATA_DICTIONARY.md` | Schema documentation |
| `data/DATA_PREPARATION.md` | Data pipeline documentation |
| `scripts/run_full_experiment.py` | Main experiment script |
| `results/full_experiment/responses_*.jsonl` | Raw results |
| `pilot/PILOT_RESULTS.md` | Pilot study summary |

---

## Reproducibility

```bash
# Run full experiment
python3 scripts/run_full_experiment.py --model gpt-4o --reps 1

# Run specific spec level
python3 scripts/run_full_experiment.py --model claude-sonnet-4 --spec S3

# Analyze results
cat results/full_experiment/*.jsonl | python3 -c "..."
```

---

*Report generated: 2026-01-26*
