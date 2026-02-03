# Archived Eedi Experiments

Archived 2026-02-02. These experiments are not used in the paper.

## Why archived

The Eedi dataset produces ρ ≈ 0 for LLM difficulty estimation across all methods and subsets (n=1,869 items). This is consistent with the paper's framework — Eedi items are selectivity-driven (difficulty comes from distractor attractiveness, not item complexity), making difficulty illegible from item text alone.

Additionally, the NeurIPS competition shuffled answer positions per student. The aggregate `pct_A/B/C/D` columns in `eedi_with_student_data.csv` use Kaggle ordering and are unreliable for computing p-correct. The NeurIPS `IsCorrect` field is the valid ground truth. Recomputation confirmed the null result is genuine.

## Contents

- `eedi_pilot/` — Initial Eedi pilot responses
- `eedi_pilot_frontier/` — Frontier model pilot (Claude + Gemini)
- `eedi_test/` — Small test run
- `rsm_v2_eedi_prompts/` — RSM prompt sweep on Eedi items
- `scripts/` — Dedicated Eedi experiment scripts

## Data retained

`data/eedi/` is kept for reference (NeurIPS student-level data is valid and may be useful for future work).
