# Study 2: Handoff Summary

*January 26, 2026 — End of Session*

---

## Current State: READY TO PILOT

All materials prepared. Eedi dataset downloaded and validated.

### Completed
- [x] Study design document
- [x] Coding rubric with examples
- [x] Prompt templates (3 conditions)
- [x] Data collection script (Python)
- [x] Probe items (50 items, 10 misconceptions)
- [x] Pilot protocol
- [x] Eedi dataset downloaded (1,869 items, 2,587 misconceptions)
- [x] Abstract drafted (247 words)

### Files Created
```
study2-materials/
├── README.md
├── COMPLETION_CRITERIA.md
├── abstract_draft.md              # SUBMIT TODAY
├── rubrics/misconception_coding_rubric.md
├── prompts/prompt_templates.md
├── data/
│   ├── eedi/                      # Downloaded dataset
│   │   ├── train.csv              # 1,869 items
│   │   └── misconception_mapping.csv
│   ├── eedi_data_guide.md
│   └── probe_items.json           # 50 researcher-generated items
├── scripts/
│   ├── collect_responses.py
│   └── explore_eedi.py
└── pilot/
    └── pilot_protocol.md
```

---

## URGENT: AIED Abstract Due TODAY (Jan 26)

**Submit to:** https://easychair.org/conferences/?conf=aied2026

**Title:** Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?

**Track:** Human Aspects of AIED (recommended)

**Abstract:** See `abstract_draft.md` (247 words, ready to submit)

**Full paper deadline:** February 2, 2026

---

## Open Design Questions

### 1. Model Selection
**Current plan:** GPT-4o, Claude 3.5 Sonnet, Llama-3.1-70B

**Open questions:**
- Include smaller models (GPT-4o-mini, Llama-8B) to test capability effect?
- Drop Llama for simplicity (only 2 models)?
- Add Gemini?

### 2. Prompting Conditions
**Current plan:** answer_only, explain, persona

**Open questions:**
- Add ability levels within persona? ("struggling student" vs "average")
- Add demographic personas? (risky—bias concerns)
- Is answer_only useful if we can't code misconceptions from it?

### 3. Scale
**Current plan:** 400 items × 3 models × 3 prompts × 20 responses = 72K responses

**Open questions:**
- Is 20 responses per cell necessary? Could do 10 for 36K total
- Is 400 items necessary? Could do 200 for faster turnaround
- Cost at full scale: ~$500-800. Acceptable?

### 4. Misconception Coding
**Current plan:** Code reasoning against Eedi's labeled misconceptions

**Open questions:**
- Code all responses or sample? (Full = expensive; Sample = less power)
- Use LLM-assisted coding? (Fast but validity concerns)
- What if IRR is low? Fallback to coarser categories?

### 5. Analysis
**Current plan:** Mixed-effects logistic regression

**Open questions:**
- Primary DV: Full match only, or Full + Partial?
- How to handle items with missing misconception labels (~40%)?
- Report distractor match as secondary, or separate analysis?

---

## To Run Pilot

```bash
cd /Users/dereklomas/AIED/study2-materials

# Set API key
export OPENAI_API_KEY="your-key"

# Run pilot (300 responses, ~$1, ~15 min)
python3 scripts/collect_responses.py \
    --items data/probe_items.json \
    --output pilot/results \
    --models gpt-4o-mini \
    --conditions answer_only explain persona \
    --n 5
```

**After pilot:**
1. Check `pilot/results/responses.jsonl`
2. Verify parse rate > 90%
3. Code 20 responses manually using rubric
4. Decide: proceed or revise

---

## Timeline to Feb 2 (Full Paper)

| Day | Task |
|-----|------|
| Jan 26 (today) | Submit abstract; run pilot |
| Jan 27 | Review pilot; finalize design decisions |
| Jan 28-30 | Full data collection (72K responses) |
| Jan 31 | Code sample; run analysis |
| Feb 1 | Write paper |
| Feb 2 | Submit full paper |

**Risk:** Very tight. Consider targeting EDM (Feb 9) or L@S (Feb 16) if AIED feels rushed.

---

## Key References (for paper)

1. **SMART** (Lan et al., EMNLP 2025) — IRT-aligned simulation
2. **Generative Students** (Lu & Wang, L@S 2024) — KC-based profiles
3. **Do LLMs Make Mistakes Like Students** (Liu et al., AIED 2025) — Distractor alignment
4. **Can LLMs Reliably Simulate** (arXiv 2025) — NAEP grade-level prompting
5. **Simulating Students Review** (arXiv Nov 2025) — Comprehensive taxonomy

See: `llm-synthetic-students-literature-review.md`

---

## Resume Command

```
Continue from @.claude/handoffs/2026-01-26-misconception-study.md

Status: Ready to pilot. Abstract needs submission TODAY.
Next: Run pilot, then decide on full study parameters.
```

---

*End of session*
