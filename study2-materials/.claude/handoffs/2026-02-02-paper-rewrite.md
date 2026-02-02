# Handoff: Paper Rewrite Session (2026-02-02)

## What happened this session

### 105-item confirmation experiment completed
- Script: `scripts/run_105_b2pl.py`
- Output: `pilot/b2pl_105_experiment/summary.json`
- **Result: NULL.** ρ=0.114 (p=0.246, 95% CI [-0.072, 0.297])
- Per-rep ρ: 0.160, 0.073, 0.079 (mean 0.104 ± 0.039)
- The ρ≈0.50 on 20 probe items does NOT generalize to 105 items
- This is the definitive Eedi result — no more experiments needed

### Paper rewrite
- Rewrote `results-site/paper.html` with updated narrative
- Initial rewrite framed it as "MCQs don't work, open-ended does"
- **User pushed back**: the MCQ vs open-ended claim is overclaiming. We only tested ONE MCQ dataset (Eedi) which has known data quality issues. Kröger et al. got r=0.75-0.82 on NAEP MCQs. Can't generalize from one dataset.
- Changed title to "When Does LLM Difficulty Estimation Work? Successes, Failures, and Boundary Conditions Across Two Datasets"
- User then pointed to `paper_sections_draft.md` which has better intro/method sections already written

### Key editorial direction from user
The honest framing is:
- LLMs succeed on SmartPaper, fail on Eedi
- We don't know if it's MCQ vs open-ended, or Eedi-specific data problems, or item homogeneity
- The "selectivity-driven vs complexity-driven" framework is a hypothesis, not a proven mechanism
- The datasets differ on too many dimensions (format, population, subject breadth, difficulty metric) to isolate the cause

## Current state of paper.html
- Has updated results (105-item null, all tables correct)
- Title updated to the neutral framing
- But intro/method sections are my version, NOT the user's preferred `paper_sections_draft.md` version
- **Next step: integrate `paper_sections_draft.md` into paper.html** — replace sections 1-2 with that content, keep sections 3-6 results/discussion/conclusion but update framing to match

## Key files
- `results-site/paper.html` — current paper draft (partially updated)
- `paper_sections_draft.md` — user's preferred intro + method text (NOT yet integrated)
- `.claude/handoffs/2026-02-02-hypothesis-registry.md` — updated with 105-item result and confirmed Path A
- `pilot/b2pl_105_experiment/summary.json` — raw 105-item experiment results
- `scripts/run_105_b2pl.py` — the experiment script

## What still needs doing
1. **Integrate `paper_sections_draft.md` into paper.html** — replace intro/method with user's version
2. **Update discussion/conclusion framing** to avoid MCQ overclaim, match the more careful tone of paper_sections_draft.md
3. **Figures may need updating** — current Fig 3 (generalization bar chart) frames it as "Eedi MCQs" which user may want changed
4. **(Optional) DeepSeek on 140 SmartPaper items** — second model for open-ended finding
5. **LaTeX conversion** for AIED 2026 submission
