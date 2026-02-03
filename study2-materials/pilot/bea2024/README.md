# BEA 2024 Shared Task Analysis

## Overview

We ran our difficulty estimation prompts on the BEA 2024 shared task dataset (USMLE medical MCQs) to validate cross-domain transfer and compare against published baselines.

## Dataset

- **Source**: BEA 2024 Shared Task on Automated Prediction of Difficulty and Response Time
- **Items**: 667 total, 595 text-only (72 excluded for requiring images)
- **Split**: 466 train, 201 test
- **Ground truth**: IRT difficulty parameter (b), NOT p-correct
  - Higher b = harder item
  - Range: 0.02 to 1.38
  - Convert to p-correct: `P = 1 / (1 + exp(b))`

## Experimental Conditions

| Condition | Model | Prompt | Reps | Items with predictions |
|-----------|-------|--------|------|------------------------|
| teacher_t1.0 | Gemini 3 Flash | teacher | 3 | 595 |
| prerequisite_chain_t1.0 | Gemini 3 Flash | prerequisite_chain | 3 | 595 |
| buggy_rules_t1.0 | Gemini 3 Flash | buggy_rules | 2 | 595 |
| scout_teacher_t1.0 | Llama-4-Scout | teacher | 3 | 595 |
| scout_prerequisite_chain_t1.0 | Llama-4-Scout | prerequisite_chain | 3 | 580 |
| scout_buggy_rules_t1.0 | Llama-4-Scout | buggy_rules | 3 | 399 |

## Key Results

### Test Set Performance (n=201)

| Method | RMSE (raw) | RMSE (calibrated) | Spearman ρ | Kendall τ |
|--------|------------|-------------------|------------|-----------|
| BEA baseline (mean) | 0.311 | — | — | — |
| BEA winner (EduTec) | 0.299 | — | — | — |
| BEA post-comp best (UnibucLLM) | 0.281 | — | — | ~0.28 |
| **Our Gemini ensemble** | 1.56 | **0.280** | 0.46 | 0.32 |
| Our teacher-only | 1.80 | 0.284 | 0.45 | — |

### Calibration

Raw predictions have systematic bias: models overestimate student performance (predict items are easier than they are).

**Calibration learned from train set:**
```
b_calibrated = 0.258 × b_predicted + 0.768
```

This linear scaling corrects the bias while preserving rank order.

### Missing Items

22 of 201 test items had no valid predictions across any condition. These were imputed with the train set mean difficulty. Missing items: 24, 52, 117, 167, 180, 188, 258, 290, 312, 319, 350, 395, 396, 406, 409, 421, 537, 566, 601, 620, 648, 663.

## Comparison Notes

1. **Our approach**: Zero-shot prompting with prompts designed for K-8 Indian education
2. **BEA winners**: Supervised ML (BERT embeddings + SVR, gradient boosting, etc.)
3. **Fair comparison?**: Both use train set - they for feature extraction/training, we for calibration
4. **Key finding**: Zero-shot + calibration matches supervised approaches

## Files

- `submission_gemini_ensemble.csv` - Raw predictions (uncalibrated)
- `submission_gemini_calibrated.csv` - Calibrated predictions
- `teacher_t1.0/`, `prerequisite_chain_t1.0/`, etc. - Raw response files

## Reproduction

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress

# Load data
train = pd.read_excel('~/Downloads/train_final.xlsx')
test = pd.read_excel('~/Downloads/test_final.xlsx')
gold = pd.read_excel('~/Downloads/gold_final.xlsx')

# Aggregate predictions across conditions and reps
# ... (see analysis scripts)

# Convert p-correct prediction to IRT scale
pred['b_pred'] = -np.log(pred['p_correct'] / (1 - pred['p_correct']))

# Calibrate on train set
slope, intercept, _, _, _ = linregress(train_pred['b_pred'], train['Difficulty'])
pred['b_calibrated'] = slope * pred['b_pred'] + intercept

# Evaluate
rmse = np.sqrt(((gold['Difficulty'] - pred['b_calibrated'])**2).mean())
```

## References

- Yaneva et al. (2024). Findings from the First Shared Task on Automated Prediction of Difficulty. BEA Workshop, ACL 2024.
- Rogoz & Ionescu (2024). UnibucLLM: Harnessing LLMs for Automated Prediction of Item Difficulty. BEA Workshop, ACL 2024.
- BEA 2024 Shared Task: https://sig-edu.org/sharedtask/2024
