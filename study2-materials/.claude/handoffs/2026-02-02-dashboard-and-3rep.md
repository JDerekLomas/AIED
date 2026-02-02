# Handoff: Dashboard Update + 3-Rep Fullset Experiment — 2026-02-02

## What Was Done This Session

### 1. Unified Dashboard (`results-site/replication_dashboard.html`)
Major update to the replication dashboard:
- **Nav bar** linking to RSM/Optimization dashboard (`rsm_analysis.html`)
- **Cross-model ranking section** — 15-model horizontal bar chart with error bars from the RSM experiment
- **Expanded results table** — new columns: Temp, N reps, Target metric, Bootstrap 95% CI; sortable by clicking headers
- **Cost estimation stat card** (~$3 for all single-pass experiments)
- **3-Rep Scaling placeholder section** with PLANNED badge for incoming results
- RSM exception card updated to show "n=20 items, 3 reps"

### 2. 3-Rep Fullset Experiment Script (`scripts/run_3rep_fullset.py`)
New script that bridges the gap between:
- RSM experiments: 20 items, 3 reps → ρ=0.57-0.67 (signal)
- Single-pass experiments: 1,869 items, 1 rep → ρ≈0 (null)

Features:
- Runs contrastive prompt at t=1.5 on all 1,869 items with 3 reps
- Models: groq_llama4scout, deepseek_chat, gemini_flash
- Correlates against p_value (available for all items)
- Bootstrap 95% CIs, cost tracking, resume support
- `--reps` flag for parallel execution (e.g., `--reps 1,2`)
- Saves raw responses to `pilot/replications/3rep_fullset/{model}/rep{N}/qid{id}.txt`
- Summary JSON to `pilot/replications/3rep_fullset/summary.json`

### 3. Handoff Updates
- Updated `2026-02-02-replication-rerun.md` with consolidated findings, 15-model ranking, and 3-rep experiment plan
- Research journey file (`2026-02-02-research-journey.md`) updated with full prompt text for Phase 10 two-stage experiments

### 4. Bug Fixes in Script
- `neurips_correct_pos` contains letters (A/B/C/D), not numbers — fixed `get_p_value()` to use `pct_{letter}`
- Column name: `QuestionText` not `question_text`

## Currently Running

**Llama-4-Scout 3-rep fullset** — two parallel processes with two Groq API keys:
- Process 1 (task `b17bebf`): rep 0 — ~907/1,869
- Process 2 (task `bbf7342`): reps 1,2 — rep 1 at ~655/1,869, rep 2 not started

Both have resume support. Safe to interrupt and restart:
```bash
# Resume rep 0 (uses default GROQ_API_KEY)
python3 scripts/run_3rep_fullset.py groq_llama4scout --reps 0

# Resume reps 1,2 (uses second key)
GROQ_API_KEY=[REDACTED] python3 scripts/run_3rep_fullset.py groq_llama4scout --reps 1,2
```

ETA: ~20-25 min remaining.

## When Results Come In

Each process will print per-rep rho and averaged-prediction rho at the end. To manually compute after all 3 reps finish:
```bash
python3 scripts/run_3rep_fullset.py groq_llama4scout --reps 0,1,2
```
(Will skip all cached items and just compute correlations + save summary.json)

### What to look for:
- **Averaged rho > 0.3**: Averaging scales — strong paper finding
- **Averaged rho ≈ 0**: 20-item RSM result was overfitting
- **Averaged rho 0.1-0.3**: Partial signal, weaker at scale

### Next models to run (if Scout shows signal):
```bash
python3 scripts/run_3rep_fullset.py deepseek_chat
python3 scripts/run_3rep_fullset.py gemini_flash
```

## Files Modified/Created
- `results-site/replication_dashboard.html` — major update
- `scripts/run_3rep_fullset.py` — new script
- `.claude/handoffs/2026-02-02-replication-rerun.md` — updated
- `.claude/handoffs/2026-02-02-dashboard-and-3rep.md` — this file
- `pilot/replications/3rep_fullset/groq_llama4scout/` — raw responses (accumulating)

## Key Context for Next Session
- The "honest numbers" from 10-rep validation: true per-rep ρ≈0.50, averaged ρ≈0.57 (on 20 items)
- All prior results are on 20 probe items only. This fullset experiment is the first test at scale.
- Second Groq API key: `[REDACTED]`
- Full research journey documented in `2026-02-02-research-journey.md`
