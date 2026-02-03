# Documentation & Cleanup Handoff — 2026-02-03

## What was done

Reorganized the study2-materials repo: archived stale files, wrote experiment docs, updated README.

### Archive moves
- **22 .md files** → `archive/drafts/` (old abstracts, paper drafts, design docs, handoffs)
- **59 scripts** → `archive/scripts/` (replicate_*, old test_*, old run_* from earlier phases)
- **~40 pilot dirs** → `archive/pilot/` (provider tests, quick_tests, old phase dirs)
- Nothing deleted — all moved to archive/

### New files created
- `pilot/prompt_framing_experiment/README.md` — standalone experiment doc with full results table, 5 key findings, viz guide, reproduction instructions
- `paper/sections/prompt_framing_results.md` — paper-ready RSM results section with temperature x framing interaction, category comparison table

### Path fixes
- `analyze_probe_inflation.py:32` — updated `pilot/b2pl_105_experiment/` → `archive/pilot/`
- `analyze_results.py:545,547` — updated `pilot/full_collection/` → `archive/pilot/full_collection/`

### README.md rewritten
- Current key results, directory structure, reproduction instructions, datasets, citation

## What remains

### Active scripts (27 in scripts/)
analyze_probe_inflation, analyze_results, analyze_smartpaper, build_dbe_kt22_item_statistics, prompts_mcq, rsm_analysis, run_bea2024, run_confirmation, run_dbe_kt22_hypotheses, run_dbe_kt22_validation, run_indian_state_assessment, run_model_survey, run_prompt_screening_g3f, run_rsm_experiment, run_smartpaper_direct, run_smartpaper_expansion, run_smartpaper_pilot, run_smartpaper_rsm, run_smartpaper_rsm_v2, run_transition_zone, test_buggy_reasoning, test_cognitive_modeling, test_cross_model, test_improved_prompts, test_metaprompt_sweep, test_specification_levels, test_two_stage

### Active pilot dirs (14 in pilot/)
bea2024, confirmation_experiment, dbe_kt22_validation, indian_state_assessment, model_survey, prompt_framing_experiment, replications, rsm_experiment, smartpaper_direct_estimation, smartpaper_expansion, smartpaper_rsm, smartpaper_rsm_v2, smartpaper_structured_pilot, smartpaper_two_stage

## Deferred work
- **Shared LLM runner extraction** — most scripts duplicate API call logic. Worth extracting a shared module to reduce tech debt. Separate session.
- **Uncommitted changes** — all changes are local, not committed. Review with `git status` before committing.
