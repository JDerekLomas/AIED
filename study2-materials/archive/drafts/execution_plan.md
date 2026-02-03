# Experiment Execution Plan
## Running the S1-S4 Specification Level Experiment

---

## Overview

| Parameter | Value |
|-----------|-------|
| Specification Levels | 4 (S1, S2, S3, S4) |
| Model Tiers | 3 (Frontier, Mid, Weak) |
| Models per Tier | 2 |
| Misconceptions | 4 (2 procedural, 2 conceptual) |
| Items per Misconception | 5 near-transfer + 2 discriminant = 7 |
| Repetitions | 3 |
| **Total API Calls** | 4 × 6 × 4 × 7 × 3 = **2,016** |

---

## Phase 1: Preparation (Jan 27)

### 1.1 Finalize Prompts

**Status**: S1-S4 prompts drafted for 4 misconceptions in `test_specification_levels.py`

**TODO**:
- [ ] Review prompts for theoretical consistency
- [ ] Ensure S4 prompts don't directly instruct the answer
- [ ] Add any missing prompt variations

### 1.2 Finalize Test Items

**Current**: 5 near-transfer + 1-2 discriminant per misconception

**TODO**:
- [ ] Verify all items have clear correct answers and target distractors
- [ ] Ensure discriminant items test different misconceptions
- [ ] Add 1-2 "trick" items where misconception gives correct answer (validity check)

### 1.3 API Setup

**Available APIs** (from .env):
| Provider | Models | Rate Limit | Cost Estimate |
|----------|--------|------------|---------------|
| OpenAI | GPT-4o, GPT-3.5-turbo | 500 RPM | ~$15-20 |
| Anthropic | Claude 3.5 Sonnet, Claude 3 Haiku | 50 RPM | ~$10-15 |
| Groq | Llama-3.1-8B | 30 RPM | Free |
| Together | Llama, Mistral | 60 RPM | ~$2-3 |

**TODO**:
- [ ] Verify all API keys are active
- [ ] Test each model with one prompt
- [ ] Set up rate limiting in script

---

## Phase 2: Data Collection (Jan 28-29)

### 2.1 Collection Strategy

**Approach**: Run sequentially by model to manage rate limits

```
Day 1 (Jan 28):
  Morning: GPT-4o, GPT-3.5-turbo (OpenAI - fast)
  Afternoon: Claude 3.5 Sonnet, Claude 3 Haiku (Anthropic - slower)

Day 2 (Jan 29):
  Morning: Llama-3.1-8B (Groq)
  Afternoon: Mistral-7B (Together/OpenRouter)
```

### 2.2 Per-Model Collection

For each model:
```
For each misconception (4):
  For each spec level (S1, S2, S3, S4):
    For each item (7):
      For each rep (3):
        Call API
        Parse response
        Save to JSONL
```

**Calls per model**: 4 × 4 × 7 × 3 = 336
**Time per model** (at 30 RPM): ~12 minutes
**Total time**: ~70 minutes of API calls + overhead

### 2.3 Data Storage

**Output file**: `pilot/spec_experiment/responses.jsonl`

**Schema per response**:
```json
{
  "response_id": "uuid",
  "timestamp": "ISO8601",
  "model": "gpt-3.5-turbo",
  "model_tier": "mid",
  "misconception_id": "PROC_ORDER_OPS",
  "misconception_type": "procedural",
  "spec_level": "S3",
  "item_id": "OPS_01",
  "item_type": "near_transfer",
  "question": "Calculate: 5 + 3 × 2",
  "options": "A) 16  B) 11  C) 13  D) 10",
  "correct_answer": "B",
  "target_distractor": "A",
  "prompt": "[full prompt text]",
  "raw_response": "[full model response]",
  "parsed_answer": "A",
  "is_correct": false,
  "hit_target": true,
  "rep": 1
}
```

### 2.4 Checkpointing

- Save after each model completes
- If interrupted, resume from last completed model
- Log all errors with full context

---

## Phase 3: Misconception Coding (Jan 29-30)

### 3.1 Automated Coding

Use regex patterns from `code_misconceptions.py`:

```python
MISCONCEPTION_PATTERNS = {
    "PROC_ORDER_OPS": {
        "indicators": [
            r"left.*to.*right",
            r"first.*add|add.*first",
            r"5.*\+.*3.*=.*8",
        ]
    },
    # ... etc
}
```

**Output codes**:
- FULL_MATCH: 2+ indicators
- PARTIAL_MATCH: 1 indicator
- DIFFERENT_ERROR: Has reasoning, no indicators
- UNCLEAR: Minimal reasoning

### 3.2 Manual Validation

- Sample 10% of responses (~200)
- Two coders independently code
- Compute Cohen's kappa
- Resolve disagreements

---

## Phase 4: Analysis (Jan 30-31)

### 4.1 Compute Metrics

For each cell (Spec × Model × Misconception):
```python
error_rate = 1 - accuracy
target_rate = target_hits / errors
alignment_rate = (full_match + partial_match) / errors
gap = target_rate - alignment_rate
consistency = within_misconception_agreement
```

### 4.2 Statistical Tests

**Primary**: 3-way ANOVA
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('alignment ~ C(spec_level) * C(model_tier) * C(misc_type)', data=df)
anova_table = sm.stats.anova_lm(model, typ=2)
```

**Contrasts**:
```python
# S4 vs S1
# S4 vs S3 within procedural
# S3 vs S4 within conceptual
# Frontier-S4 vs Weak-S1
```

### 4.3 Visualizations

1. **Gap by Specification Level** (main finding)
2. **Spec × Type Interaction** (H2)
3. **Spec × Capability Interaction** (H3)
4. **Consistency by Spec Level**
5. **Example responses** (qualitative)

---

## Phase 5: Write-up (Feb 1-2)

- Fill in Results section of `paper_draft_v2.md`
- Generate figures
- Update Discussion based on outcome
- Final proofread
- Submit Feb 2

---

## Scripts Needed

### 1. `run_experiment.py` (main runner)
```
- Load prompts and items
- Iterate through design matrix
- Call APIs with rate limiting
- Save responses to JSONL
- Checkpoint after each model
```

### 2. `code_responses.py` (misconception coding)
```
- Load responses
- Apply regex patterns
- Output coded data
- Generate coding report
```

### 3. `analyze_results.py` (statistical analysis)
```
- Load coded data
- Compute cell-level metrics
- Run ANOVA and contrasts
- Generate figures
- Output analysis report
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Sequential by provider, built-in delays |
| API failures | Retry logic, checkpoint after each model |
| Frontier models 100% accurate | Report as finding, analyze mid/weak |
| Low inter-rater reliability | Use conservative coding, report kappa |
| Timeline slip | Prioritize core analysis, defer robustness checks |

---

## Estimated Costs

| Provider | Calls | Est. Cost |
|----------|-------|-----------|
| OpenAI (GPT-4o) | 336 | ~$8 |
| OpenAI (GPT-3.5) | 336 | ~$1 |
| Anthropic (Sonnet) | 336 | ~$8 |
| Anthropic (Haiku) | 336 | ~$1 |
| Groq (Llama) | 336 | Free |
| Together (Mistral) | 336 | ~$1 |
| **Total** | 2,016 | **~$20** |

---

## Command to Run

```bash
cd /Users/dereklomas/AIED/study2-materials
source .venv/bin/activate
python scripts/run_experiment.py --output pilot/spec_experiment
```

---

## Checklist

### Before Running
- [ ] All API keys verified
- [ ] Prompts reviewed for all 4 × 4 = 16 conditions
- [ ] Test items complete (28 total)
- [ ] Test script with 1 call per model
- [ ] Output directory created

### During Collection
- [ ] Monitor for errors
- [ ] Check parse rates periodically
- [ ] Verify checkpoints saving

### After Collection
- [ ] Verify response count (2,016)
- [ ] Run misconception coding
- [ ] Spot-check 20 responses manually
- [ ] Run analysis pipeline
