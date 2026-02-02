# Handoff: Eedi Dropped, Data Audit Complete — 2026-02-02

## Key Decision

**Eedi data removed from all analyses.** Too many data integrity issues to use reliably.

## What Happened This Session

### 1. 3-Rep Fullset Experiment Completed
- Ran Llama-4-Scout on all 1,869 Eedi items, 3 reps, contrastive prompt @ t=1.5
- Result: **ρ = 0.008**, 95% CI [-0.037, 0.054] — null
- The ρ=0.624 from 20-item RSM experiments does not scale

### 2. Eedi Data Bug Recurred (Again)
- `run_3rep_fullset.py` used `CorrectAnswer` (Kaggle ordering) to parse LLM predictions
- p_value computed from `neurips_correct_pos` (NeurIPS ordering)
- Initially looked like scoring bug causing negative ρ on filtered items
- On closer inspection, the original scoring was actually correct: LLM sees Kaggle-ordered options, so `CorrectAnswer` is the right key for LLM predictions
- The two-ordering system is so confusing that even after documenting it extensively, it keeps causing errors and confusion

### 3. Eedi Dropped
- All results null regardless of scoring method, filtering, or averaging
- Adaptive testing means sparse responses per item
- Selectivity-driven difficulty (which distractor traps students) is inherently illegible from item text
- The Kaggle/NeurIPS ordering mismatch makes every analysis error-prone
- Not worth continued investment

### 4. Data Audit of Other Datasets
- **SmartPaper**: Clean. Open-ended items with AI grading, no ordering issue. 99.98% of data valid.
- **DBE-KT22**: Ground truth (p_correct from `answer_state`) is safe. Found minor fragility: A/B/C/D labels assigned from CSV row order, not sorted by database ID. Fixed in both `build_dbe_kt22_item_statistics.py` and `run_dbe_kt22_validation.py`. Preventive fix — existing results unaffected.

### 5. Dashboard Partially Updated
- Updated 3-rep fullset entry from PLANNED to NULL in `replication_dashboard.html`
- Added CANCELLED badge for DeepSeek/Gemini runs
- Dashboard update was interrupted mid-edit — may need review

## Current Paper Datasets

| Dataset | Domain | n items | ρ | Status |
|---------|--------|---------|---|--------|
| SmartPaper | Math (open-ended) | 140 | 0.547 | Primary dataset |
| DBE-KT22 | Database systems (MCQ) | 168 | 0.342 | Out-of-sample validation |
| ~~Eedi~~ | ~~Math (MCQ)~~ | ~~1,869~~ | ~~≈0~~ | ~~Dropped~~ |

## Files Modified
- `data/DATA_DICTIONARY.md` — added Eedi removal notice, DBE-KT22 label fix documentation
- `scripts/build_dbe_kt22_item_statistics.py` — sort choices by ID before label assignment
- `scripts/run_dbe_kt22_validation.py` — same fix
- `results-site/replication_dashboard.html` — partial update (3-rep null result)
- `.claude/handoffs/2026-02-02-dashboard-and-3rep.md` — updated with null results and ground truth investigation

## Raw Data Preserved
- `pilot/replications/3rep_fullset/groq_llama4scout/` — 5,607 cached LLM responses (3 reps × 1,869 items)
- Can be re-analyzed if Eedi ever becomes useful again
