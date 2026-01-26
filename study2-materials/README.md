# Study 2: Misconception Alignment Study Materials

## Overview

This directory contains all materials for **Study 2: Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?**

The study tests whether LLM-simulated students exhibit the same underlying misconceptions as real students when answering math problems incorrectly.

---

## Directory Structure

```
study2-materials/
├── README.md                           # This file
├── COMPLETION_CRITERIA.md              # Success criteria for each phase
│
├── rubrics/
│   └── misconception_coding_rubric.md  # Detailed coding scheme with examples
│
├── prompts/
│   └── prompt_templates.md             # Three prompting conditions
│
├── data/
│   ├── eedi_data_guide.md              # How to access and process Eedi data
│   └── probe_items.json                # 50 researcher-generated items
│
├── scripts/
│   └── collect_responses.py            # Data collection pipeline
│
└── pilot/
    └── pilot_protocol.md               # 20-item pilot study procedure
```

---

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install openai anthropic together pandas tqdm

# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"  # for Llama
```

### 2. Run Pilot (Recommended First Step)

```bash
cd /Users/dereklomas/AIED/study2-materials

# Run pilot with probe items
python scripts/collect_responses.py \
    --items data/probe_items.json \
    --output pilot/results \
    --models gpt-4o-mini \
    --conditions answer_only explain persona \
    --n 5
```

### 3. Check Results

```python
import json
import pandas as pd

records = [json.loads(l) for l in open('pilot/results/responses.jsonl')]
df = pd.DataFrame(records)

print(f"Total responses: {len(df)}")
print(f"Accuracy: {df['is_correct'].mean():.1%}")
print(df.groupby('condition')['is_correct'].mean())
```

---

## Study Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0. Setup | **COMPLETE** | Materials prepared |
| 1. Pilot | **READY** | Run 20-item pilot, validate coding |
| 2. Data Collection | PENDING | Full 400-item collection |
| 3. Coding & Analysis | PENDING | Misconception match analysis |
| 4. Writing | PENDING | Paper drafting |

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `COMPLETION_CRITERIA.md` | When is each phase "done"? |
| `rubrics/misconception_coding_rubric.md` | How to code LLM reasoning |
| `prompts/prompt_templates.md` | The three prompting conditions |
| `pilot/pilot_protocol.md` | Step-by-step pilot procedure |

---

## Next Steps (For You)

### Immediate (Today/Tomorrow)

1. **Run the pilot**
   ```bash
   python scripts/collect_responses.py \
       --items data/probe_items.json \
       --output pilot/results \
       --models gpt-4o-mini \
       --n 5
   ```

2. **Review responses** — Are they usable? Does parsing work?

3. **Practice coding** — Code 10 responses using the rubric

### This Week

4. **Download Eedi dataset** from Kaggle (if not already)

5. **Complete pilot coding** — Get IRR estimate

6. **Decide: Proceed or revise?**

### Before Feb 9 (EDM deadline)

7. **Full data collection** — 400 Eedi items + 50 probes

8. **Analysis** — Misconception match rates

9. **Write paper**

---

## Budget Estimate

| Item | Cost |
|------|------|
| Pilot (300 calls, gpt-4o-mini) | ~$1 |
| Full study (81K calls, mixed models) | ~$500-800 |
| **Total** | ~$500-800 |

---

## Target Venues

| Venue | Deadline | Status |
|-------|----------|--------|
| EDM 2026 | Feb 9 | Primary target |
| L@S 2026 | Feb 16 | Backup |
| IJAIED | Rolling | If results warrant journal |

---

## Contact

Questions about this study design? The materials are self-contained but may need iteration based on pilot results.

---

*Created: January 26, 2026*
