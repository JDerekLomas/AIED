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

### 2. 3-Rep Fullset Experiment (`scripts/run_3rep_fullset.py`)
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

### 3. 3-Rep Fullset Results: NULL

**Llama-4-Scout on 1,869 items, 3 reps, contrastive prompt, t=1.5:**

| Rep | Spearman ρ vs p_value | p-value |
|-----|----------------------|---------|
| 0   | -0.013               | 0.587   |
| 1   | -0.005               | 0.832   |
| 2   | 0.029                | 0.211   |
| **3-rep averaged** | **0.008** | **0.714** |

95% CI: [-0.037, 0.054]

**Interpretation**: The 3-rep averaging that produced ρ=0.624 on 20 probe items produces ρ=0.008 on 1,869 items. The RSM signal was an artifact of small sample size (n=20). Multi-sample averaging does not rescue LLM difficulty estimation at scale.

### 4. Bug Fixes
- `neurips_correct_pos` contains letters (A/B/C/D), not numbers — fixed `get_p_value()` to use `pct_{letter}`
- Column name: `QuestionText` not `question_text`

### 5. Ground Truth Quality Investigation

Checked whether the null result could be due to noisy p_values from adaptive testing (few students per item):

| Min total_responses | n items | Spearman ρ |
|---------------------|---------|------------|
| 100                 | 1,436   | 0.007      |
| 250                 | 977     | 0.020      |
| 500                 | 691     | 0.020      |
| 1,000               | 360     | 0.038      |
| 2,000               | 87      | -0.006     |

No signal at any threshold — even items with 1,000+ student responses show zero correlation.

**However**: The Eedi dataset has a deeper problem. With adaptive testing, ~500 students are spread across 1,532 items, meaning most items have very few responses. IRT estimation is unreliable under these conditions — only 14 items survived strict filtering in a separate IRT analysis. The p_value ground truth itself may be too noisy for meaningful correlation, regardless of LLM performance.

**Next step**: Rerunning IRT with more students to get reliable ground truth before drawing conclusions.

## Key Findings (Consolidated)

1. **The Null Wall is real**: All methods produce ρ≈0 on the full 1,869-item dataset, including 3-rep averaging
2. **Small-sample overfitting**: The ρ=0.57-0.67 results on 20 probe items do not generalize. With n=20, bootstrap CIs were wide but looked promising. At n=1,869, the true signal is indistinguishable from zero.
3. **Averaging doesn't scale**: The "averaging is magic" finding from 20-item experiments was illusory
4. **DeepSeek/Gemini runs skipped**: Given the null result for Scout, running additional models on the full set is not justified
5. **Eedi ground truth is suspect**: Adaptive testing means sparse responses per item. IRT parameters are unreliable for most items. Need to rerun IRT with larger student sample before concluding the null is real vs. an artifact of noisy ground truth.

## Files Modified/Created
- `results-site/replication_dashboard.html` — major update
- `scripts/run_3rep_fullset.py` — new script
- `.claude/handoffs/2026-02-02-replication-rerun.md` — updated
- `.claude/handoffs/2026-02-02-dashboard-and-3rep.md` — this file
- `pilot/replications/3rep_fullset/groq_llama4scout/` — 5,607 raw responses (3 reps × 1,869 items)
- `pilot/replications/3rep_fullset/summary.json` — final results

## Key Context for Next Session
- The full research journey is documented in `2026-02-02-research-journey.md`
- All "signal" findings were on 20 probe items only. At scale (1,869 items), everything is null.
- **Critical open question**: Is the null result real, or an artifact of unreliable Eedi ground truth? Adaptive testing with ~500 students across 1,532 items means most items have sparse data. IRT is being rerun with more students.
- The 3-rep fullset raw responses (5,607 files) are cached and can be re-correlated against improved IRT estimates without re-running the LLM calls.
