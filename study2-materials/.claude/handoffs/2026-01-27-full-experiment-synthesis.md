# Full Experiment Synthesis - Handoff

**Date:** 2026-01-27
**Status:** Complete - Abstract ready for submission

## Summary

Ran 6 experiments testing whether LLMs can serve as synthetic students for item calibration. All converge on the same finding: the **selectivity problem**.

## Experiments Completed

| # | Experiment | Method | Key Result | Human Correlation |
|---|-----------|--------|------------|-------------------|
| 1 | S1-S4 Specification | Mental model prompting (S3) | 88.6% target alignment | r = -0.07 |
| 2 | Reasoning Authenticity | CoT reasoning analysis | 41.3pp authenticity gap | 8.7% reasoning match |
| 3 | Student Simulation | Multi-model + UK profiles | Internal differentiation works | r = -0.19 |
| 4 | Direct Difficulty | 4 prompting styles (GPT-4o-mini) | Best: basic (r=0.14), worst: IRT (r=-0.27) | r = -0.27 to 0.14 |
| 5 | Classroom Simulation | Weighted student responses | Near-zero | r = -0.03 |
| 6 | Opus Estimation | Calibrated one-shot | MAE = 39pp | r = 0.19 |

## Key Findings

1. **Misconception induction works**: S3 prompts produce 88.6% target alignment
2. **Conceptual > Procedural**: 68.2% vs 54.8% target rate
3. **No item-level correlation**: r ≈ 0 across ALL methods
4. **Authenticity gap**: LLMs select right distractors (50%) but for wrong reasons (8.7%)
5. **Expert blindspot**: Models describe correct procedures while labeling them misconceptions (Q1430 case study)

## The Selectivity Problem

**Definition:** LLMs apply misconceptions uniformly across items rather than selectively like students.

**Root cause:** LLMs judge difficulty by mathematical complexity. Humans apply misconceptions based on surface features that trigger automatic (wrong) procedures.

**Example:** Q1430 (2/7 + 1/7) — 91.7% LLM accuracy vs 5.8% human accuracy. The model knows the misconception exists but can't model when it fires.

## Viable vs Non-Viable Use Cases

| Use Case | Viable? |
|----------|---------|
| Generating misconception examples | Yes |
| Testing tutor error detection | Yes |
| Item difficulty calibration | No |
| Replacing student pilots | No |
| Adaptive system calibration | No |

## Files

### Abstracts
- `ABSTRACT_v3.md` — Comprehensive 6-experiment abstract (latest)
- `ABSTRACT_v2.md` — Earlier 2-experiment abstract

### Experiment Results
- `pilot/PILOT_RESULTS.md` — S1-S4 factorial results
- `pilot/full_collection/figures/analysis_report.md` — Full collection analysis
- `pilot/student_simulation/` — Student simulation results (432 responses)
- `pilot/replications/` — Direct difficulty, classroom sim, error alignment
- `results/misconception_coding/coding_summary.json` — Reasoning authenticity coding

### Scripts
- `scripts/run_student_simulation.py` — Multi-model student simulation
- `scripts/analyze_student_simulation.py` — Simulation analysis
- `scripts/run_full_experiment.py` — S1-S4 factorial experiment
- `scripts/run_experiment.py` — Core experiment runner

### Data
- `results/final_items.json` — 36 items across 3 misconceptions
- `data/uk_student_profiles.json` — 12 UK student profiles
- `data/eedi/curated_eedi_items.csv` — Full EEDI dataset

## Misconceptions Studied

| ID | Misconception | Type | Target Rate (S3) |
|----|--------------|------|------------------|
| 217 | Adds numerators and denominators in fraction addition | Conceptual | ~68% |
| 1507 | Carries out operations from left to right (BODMAS) | Procedural | ~55% |
| 1597 | Believes multiplying two negatives gives a negative | Conceptual | ~68% |

## Models Used

- **Claude Opus 4.5**: Difficulty estimation (Exp 6)
- **Claude Sonnet 4**: S1-S4 experiments, confident student simulation
- **GPT-4o-mini**: Direct difficulty estimation, secure student simulation
- **GPT-3.5-turbo**: Developing student simulation
- **Llama 3.1 8B (Groq)**: Struggling student simulation

## Future Directions

1. Fine-tuning on student response distributions
2. Explicit difficulty priors in prompts
3. Hybrid LLM + misconception library approaches
4. Neuro-symbolic systems with rule-based triggering
5. Feature-based triggering (identify surface features that activate misconceptions)

## Next Steps

- Write full paper draft from ABSTRACT_v3.md
- Consider additional experiments on feature-based triggering
- Potential replication with different subject domains (not just math)
