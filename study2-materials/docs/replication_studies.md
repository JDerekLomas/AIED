# Eedi Dataset Replication Studies

This document describes five replication studies from the difficulty estimation literature, implemented using the Eedi dataset (113 curated items, 1,869 full items, ~17M student responses).

## Quick Reference

| # | Study | Script | Key Metric | Benchmark |
|---|-------|--------|------------|-----------|
| 1 | Error Alignment | `replicate_error_alignment.py` | Error correlation | r=0.73-0.80 |
| 2 | Classroom Simulation | `replicate_classroom_simulation.py` | Difficulty correlation | r=0.75-0.82 |
| 3 | Confusion Tuples | `replicate_confusion_tuples.py` | Target rate improvement | +10-15% |
| 4 | Feature Extraction | `replicate_feature_extraction.py` | Test correlation | r=0.62-0.87 |
| 5 | Uncertainty | `replicate_uncertainty_difficulty.py` | RMSE improvement | >10% |

---

## Replication 1: Error Alignment (Index-Based)

**Paper:** "Do LLMs Make Mistakes Like Students?" (arXiv:2502.15140)

**Research Question:** When LLMs make errors, do they select the same wrong answers as students?

### Method
1. Present items with A/B/C/D options to models (no student prompt)
2. Run N=10 samples per item with temperature=1.0
3. Compute probability distribution over options
4. Compare LLM error distribution to student error distribution

### Usage
```bash
# Quick test (10 items)
python scripts/replicate_error_alignment.py --items 10 --samples 5 --models gpt-4o-mini

# Full run (all items, multiple models)
python scripts/replicate_error_alignment.py --samples 10 --models gpt-4o-mini gpt-4o claude-3-haiku

# Analyze existing results
python scripts/replicate_error_alignment.py --analyze-only pilot/replications/error_alignment/all_results.json
```

### Key Metric
- **Error selection correlation**: Pearson r between LLM error distribution and student error distribution
- Benchmark from paper: r=0.73-0.80

### Expected Output
```
Distribution Correlations:
  Overall (all options): r=0.XXX, p=X.XXXX
  Errors only (wrong answers): r=0.XXX, p=X.XXXX

Benchmark from paper: r=0.73-0.80
```

---

## Replication 2: Classroom Simulation + Aggregation

**Paper:** "Take Out Your Calculators" (Kröger et al., arXiv:2601.09953)

**Research Question:** Can aggregating responses from simulated students at different ability levels predict item difficulty?

### Method
1. Define ability levels with population proportions:
   - Below Basic (25%): Struggles significantly
   - Basic (35%): Handles simple problems
   - Proficient (25%): Generally performs well
   - Advanced (15%): Excels, rarely makes errors
2. Simulate N students at each level
3. Aggregate responses with population weighting
4. Compare predicted difficulty to actual student difficulty

### Usage
```bash
# Quick test (10 items, 3 students per level)
python scripts/replicate_classroom_simulation.py --items 10 --students 3

# Full run (all items, 5 students per level)
python scripts/replicate_classroom_simulation.py --students 5 --model gpt-4o-mini

# Analyze existing results
python scripts/replicate_classroom_simulation.py --analyze-only pilot/replications/classroom_simulation/gpt-4o-mini_results.json
```

### Key Metric
- **Difficulty correlation**: Pearson r between simulated difficulty and actual student difficulty
- Benchmark from Kröger et al.: r=0.75-0.82

### Expected Output
```
Difficulty Prediction (Weighted by ability level proportions):
  Pearson r: 0.XXX (p=X.XXXX)
  Spearman ρ: 0.XXX (p=X.XXXX)
  RMSE: 0.XXX
```

---

## Replication 3: Confusion Tuples Validation

**Paper:** "Towards Valid Student Simulation" (arXiv:2601.05473)

**Research Question:** Does explicit misconception specification improve alignment with student errors?

### Method
1. Map Eedi misconceptions to confusion tuples
2. Compare two prompt conditions:
   - **P1 (Generic):** "You are a struggling student"
   - **P3 (Confusion Tuple):** "You confuse {KC_A} with {KC_B}"
3. Measure target distractor selection rate
4. Test whether P3 produces "causally attributable" errors

### Usage
```bash
# Quick test (10 items, 3 reps)
python scripts/replicate_confusion_tuples.py --items 10 --reps 3

# Full run
python scripts/replicate_confusion_tuples.py --reps 5 --model gpt-4o-mini

# Analyze existing results
python scripts/replicate_confusion_tuples.py --analyze-only pilot/replications/confusion_tuples/gpt-4o-mini_results.json
```

### Key Metric
- **Target rate improvement**: Difference between P3 and P1 target distractor selection
- Expected: P3 should show 10-15 percentage points higher target rate

### Expected Output
```
KEY FINDING: Target Distractor Selection
P1 (generic persona): XX.X%
P3 (confusion tuple): XX.X%
Improvement: +XX.X percentage points
```

---

## Replication 4: Feature Extraction + ML Baseline

**Paper:** Razavi & Powers (arXiv:2504.08804)

**Research Question:** Can LLM-extracted features + tree model beat direct estimation?

### Method
1. Extract 7 features per item via LLM:
   - Vocabulary complexity (1-5)
   - Syntax complexity (1-5)
   - Conceptual complexity (1-5)
   - Cognitive load (1-5)
   - DOK level (1-4)
   - Skill difficulty (1-5)
   - Distractor quality (1-5)
2. Train GBM on 80% of items
3. Evaluate correlation on 20% holdout
4. Compare to direct LLM difficulty estimation

### Usage
```bash
# Quick test (50 items)
python scripts/replicate_feature_extraction.py --items 50

# Full run (needs 100+ items for reliable ML)
python scripts/replicate_feature_extraction.py --items 200

# Skip direct baseline (faster)
python scripts/replicate_feature_extraction.py --items 100 --skip-direct

# Analyze existing features
python scripts/replicate_feature_extraction.py --analyze-only pilot/replications/feature_extraction/features.json
```

### Key Metric
- **Test set correlation**: Pearson r between predicted and actual difficulty
- Benchmark: r=0.62-0.87 (domain dependent; reading=0.87, math=0.62-0.82)

### Expected Output
```
GBM MODEL RESULTS
Train (n=XX):
  Pearson r: 0.XXX
  RMSE: 0.XXX

Test (n=XX):
  Pearson r: 0.XXX (p=X.XXXX)
  RMSE: 0.XXX

Feature Importance:
  cognitive_load: 0.XXX
  skill_difficulty: 0.XXX
  ...
```

---

## Replication 5: Model Uncertainty as Difficulty

**Paper:** EDM 2025 proceedings

**Research Question:** Does LLM uncertainty correlate with item difficulty?

### Method
1. For each item, get logprobs of answer tokens
2. Compute 1st-token probability (confidence)
3. Randomize option order, measure consistency
4. Use uncertainty features to predict difficulty:
   - Answer probability
   - Answer entropy
   - Permutation consistency

### Usage
```bash
# Quick test (20 items, 3 permutations)
python scripts/replicate_uncertainty_difficulty.py --items 20 --permutations 3

# Full run
python scripts/replicate_uncertainty_difficulty.py --items 100 --permutations 4 --model gpt-4o-mini

# Analyze existing results
python scripts/replicate_uncertainty_difficulty.py --analyze-only pilot/replications/uncertainty_difficulty/gpt-4o-mini_results.json
```

### Key Metric
- **RMSE improvement**: Reduction in RMSE compared to mean baseline
- Expected: >10% improvement

### Expected Output
```
Correlations with Difficulty:
  Answer probability: r=-0.XXX (p=X.XXXX)
  Answer entropy: r=0.XXX (p=X.XXXX)
  Permutation consistency: r=-0.XXX (p=X.XXXX)

Using (1-prob) as difficulty predictor:
  RMSE: 0.XXX

Improvement over baseline: XX.X%
```

---

## Running All Replications

### Quick Validation (30 mins, ~$5)
```bash
# Test all scripts work with small samples
python scripts/replicate_error_alignment.py --items 10 --samples 3 --models gpt-4o-mini
python scripts/replicate_classroom_simulation.py --items 10 --students 2
python scripts/replicate_confusion_tuples.py --items 5 --reps 2
python scripts/replicate_feature_extraction.py --items 20 --skip-direct
python scripts/replicate_uncertainty_difficulty.py --items 10 --permutations 2
```

### Full Experiment (~2-4 hours, ~$20-50)
```bash
# Phase 1: Error Alignment + Confusion Tuples
python scripts/replicate_error_alignment.py --samples 10 --models gpt-4o-mini gpt-4o
python scripts/replicate_confusion_tuples.py --reps 5

# Phase 2: Classroom Simulation
python scripts/replicate_classroom_simulation.py --students 5

# Phase 3: ML Baselines
python scripts/replicate_feature_extraction.py --items 200
python scripts/replicate_uncertainty_difficulty.py --items 100 --permutations 4
```

---

## Data Requirements

All scripts use the same Eedi data file:
- Default: `data/eedi/eedi_with_student_data.csv`
- Override with `--data /path/to/file.csv`

Required columns:
- `QuestionId`, `QuestionText`
- `AnswerAText`, `AnswerBText`, `AnswerCText`, `AnswerDText`
- `CorrectAnswer`
- `pct_A`, `pct_B`, `pct_C`, `pct_D`
- `total_responses`

For Replication 3, additional columns needed:
- `MisconceptionAId`, `MisconceptionBId`, `MisconceptionCId`, `MisconceptionDId`

---

## Results Location

All results are saved to `pilot/replications/{study_name}/`:
- `{model}_results.json` - Raw experimental data
- `analysis.json` - Computed metrics
- `{model}_intermediate.json` - Checkpoint during runs

---

## Environment Setup

```bash
# Required packages
pip install openai anthropic pandas numpy scipy scikit-learn python-dotenv

# Optional (for some models)
pip install groq together

# API keys in .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
