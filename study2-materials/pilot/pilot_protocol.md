# Pilot Study Protocol

## Objectives

1. **Validate prompts** — Do all three conditions produce usable responses?
2. **Test coding rubric** — Can coders reliably apply the misconception match scheme?
3. **Estimate costs** — Validate budget projections before full run
4. **Identify issues** — Surface problems with items, parsing, or procedure

---

## Pilot Design

### Sample

| Component | N | Selection |
|-----------|---|-----------|
| Items | 20 | 7 procedural, 7 conceptual, 6 interpretive (from probe items) |
| Models | 1 | gpt-4o-mini (cheapest, fastest) |
| Conditions | 3 | answer_only, explain, persona |
| Responses/condition | 5 | Enough to see variability |

**Total responses:** 20 × 1 × 3 × 5 = 300 responses

### Selected Pilot Items

From `probe_items.json`, select:

**Procedural (7 items):**
- PROC_ORDER_OPS_01, PROC_ORDER_OPS_02
- PROC_SUBTRACT_REVERSE_01, PROC_SUBTRACT_REVERSE_02
- PROC_NEGATIVE_SUBTRACT_01
- PROC_ALGEBRA_COMBINE_01
- PROC_DECIMAL_PLACE_01

**Conceptual (7 items):**
- CONC_MULT_INCREASE_01, CONC_MULT_INCREASE_02
- CONC_FRAC_DENOM_01, CONC_FRAC_DENOM_02
- CONC_AREA_PERIMETER_01
- CONC_PERCENT_BASE_01
- CONC_PERCENT_BASE_02

**Interpretive (6 items):**
- INTERP_GRAPH_AXES_01, INTERP_GRAPH_AXES_02
- INTERP_GRAPH_AXES_03, INTERP_GRAPH_AXES_04
- INTERP_GRAPH_AXES_05
- (Only 1 interpretive misconception in probes; supplement from Eedi if available)

---

## Procedure

### Step 1: Setup (30 min)

```bash
# Create pilot directory
mkdir -p study2-materials/pilot/results

# Create pilot items file (subset of probe_items.json)
python scripts/create_pilot_items.py

# Verify API access
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Step 2: Collect Responses (1-2 hours)

```bash
# Run collection script
python scripts/collect_responses.py \
    --items pilot/pilot_items.json \
    --output pilot/results \
    --models gpt-4o-mini \
    --conditions answer_only explain persona \
    --n 5

# Expected: ~300 API calls
# Estimated cost: ~$0.50-1.00
# Estimated time: 15-30 minutes (with rate limiting)
```

### Step 3: Initial Data Check (30 min)

Review `pilot/results/responses.jsonl`:

- [ ] All 300 responses collected
- [ ] No systematic API errors
- [ ] Parse success rate > 90%
- [ ] Responses readable and coherent

```python
import pandas as pd
import json

# Load responses
records = [json.loads(line) for line in open("pilot/results/responses.jsonl")]
df = pd.DataFrame(records)

# Basic checks
print(f"Total: {len(df)}")
print(f"Parse success: {df['parsed_answer'].notna().mean():.1%}")
print(f"Accuracy: {df['is_correct'].mean():.1%}")
print(f"By condition:\n{df.groupby('condition')['is_correct'].mean()}")

# Check for issues
print(f"\nUnparseable responses: {df['parsed_answer'].isna().sum()}")
print(f"API errors: {df['error'].notna().sum()}")

# Sample responses
for cond in ['answer_only', 'explain', 'persona']:
    print(f"\n=== {cond} ===")
    sample = df[df['condition'] == cond].iloc[0]
    print(sample['raw_response'][:500])
```

### Step 4: Coding Practice (2-3 hours)

**Coder preparation:**
1. Read coding rubric thoroughly
2. Review all examples
3. Code 10 practice items together (discuss each)

**Materials needed:**
- `rubrics/misconception_coding_rubric.md`
- Coding spreadsheet template (see below)

### Step 5: Independent Coding (2 hours)

**Select coding sample:**
- 50 responses from explain + persona conditions
- Exclude answer_only (no reasoning to code)
- Balance across misconception types

**Each coder independently codes:**
- match_code (1-6)
- confidence (high/medium/low)
- notes (for ambiguous cases)

### Step 6: Reliability Check (1 hour)

```python
from sklearn.metrics import cohen_kappa_score

# Load coding data
coder1 = pd.read_csv("pilot/coding_coder1.csv")
coder2 = pd.read_csv("pilot/coding_coder2.csv")

# Merge on response_id
merged = coder1.merge(coder2, on="response_id", suffixes=("_1", "_2"))

# Full kappa (6 categories)
kappa_full = cohen_kappa_score(merged['match_code_1'], merged['match_code_2'])
print(f"Cohen's κ (6 categories): {kappa_full:.3f}")

# Collapsed kappa (Match vs No Match vs Other)
def collapse(code):
    if code in [1, 2]:
        return "Match"
    elif code in [3, 4]:
        return "NoMatch"
    else:
        return "Other"

merged['collapsed_1'] = merged['match_code_1'].apply(collapse)
merged['collapsed_2'] = merged['match_code_2'].apply(collapse)

kappa_collapsed = cohen_kappa_score(merged['collapsed_1'], merged['collapsed_2'])
print(f"Cohen's κ (collapsed): {kappa_collapsed:.3f}")

# Identify disagreements
disagreements = merged[merged['match_code_1'] != merged['match_code_2']]
print(f"\nDisagreements: {len(disagreements)} / {len(merged)} ({len(disagreements)/len(merged):.1%})")
```

### Step 7: Disagreement Resolution (1 hour)

For each disagreement:
1. Review response and both codings
2. Discuss reasoning
3. Reach consensus
4. Note if rubric needs clarification

### Step 8: Pilot Report (1 hour)

Document:
- [ ] Response collection success rate
- [ ] Parse success rate
- [ ] Accuracy by condition
- [ ] Inter-rater reliability (κ)
- [ ] Common coding disagreements
- [ ] Rubric revisions needed
- [ ] Cost estimate for full study
- [ ] Go/no-go recommendation

---

## Coding Spreadsheet Template

| Column | Description |
|--------|-------------|
| response_id | Unique identifier |
| item_id | Item identifier |
| condition | answer_only / explain / persona |
| raw_response | Full LLM response |
| parsed_answer | A/B/C/D |
| correct_answer | Ground truth |
| is_correct | TRUE/FALSE |
| misconception_name | From item data |
| match_code | 1-6 (see rubric) |
| confidence | high/medium/low |
| notes | Free text |

---

## Decision Criteria

### Proceed to Full Study if:

- [ ] Response collection: > 95% success
- [ ] Parse success: > 90%
- [ ] Inter-rater reliability: κ ≥ 0.60
- [ ] No fundamental issues with prompts or coding scheme

### Revise and Re-pilot if:

- Parse success 80-90% → Improve parsing logic
- κ = 0.40-0.60 → Clarify rubric, retrain
- Systematic prompt issues → Revise prompts

### Major Pivot if:

- Parse success < 80%
- κ < 0.40 after revision
- LLMs produce unusable responses

---

## Timeline

| Day | Activity | Hours |
|-----|----------|-------|
| 1 | Setup + response collection | 2-3 |
| 1 | Initial data check | 1 |
| 2 | Coding practice | 2-3 |
| 2 | Independent coding | 2 |
| 3 | Reliability check + resolution | 2 |
| 3 | Pilot report | 1 |

**Total: 10-12 hours over 2-3 days**

---

## Files to Create

```
pilot/
├── pilot_items.json          # 20 selected items
├── results/
│   ├── responses.jsonl       # Raw API responses
│   └── summary.json          # Collection stats
├── coding/
│   ├── coding_sample.csv     # 50 responses to code
│   ├── coding_coder1.csv     # Coder 1 results
│   ├── coding_coder2.csv     # Coder 2 results
│   └── disagreements.csv     # Resolution notes
└── pilot_report.md           # Final pilot report
```

---

## Quick Start Commands

```bash
# 1. Create pilot items (manual or script)
# For now, manually copy 20 items from probe_items.json

# 2. Run pilot collection
cd /Users/dereklomas/AIED/study2-materials
python scripts/collect_responses.py \
    --items data/probe_items.json \
    --output pilot/results \
    --models gpt-4o-mini \
    --n 5

# 3. Check results
python -c "
import json
import pandas as pd
records = [json.loads(l) for l in open('pilot/results/responses.jsonl')]
df = pd.DataFrame(records)
print(df.groupby(['condition'])['is_correct'].agg(['count', 'mean']))
"
```

---

*Version 1.0 - January 26, 2026*
