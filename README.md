# When Does LLM Difficulty Estimation Work?

**A Systematic Search Across Methods, Models, and Datasets**

Derek Lomas | Submission to AIED 2026 | February 2026

---

## Overview

Can LLMs estimate how difficult a test item is for students? The literature reports correlations from r=0 to r=0.83. We conducted a megaexperiment — ~200 experimental conditions across 15+ models, 20+ prompt configurations, 6 temperature settings, and 2 datasets — to map the full parameter landscape and find out why the literature disagrees.

**The answer:** it depends on what makes items difficult.

- **Complexity-driven difficulty** (content complexity, prerequisite knowledge, cognitive demand) — LLMs can estimate this. SmartPaper: **ρ=0.55** on 140 items, generalises to held-out items.
- **Selectivity-driven difficulty** (which specific distractors trigger which specific misconceptions) — LLMs fail. Eedi: **ρ=0.11** on 105 items, non-significant.

## Key Findings

| Finding | Evidence |
|---------|----------|
| Prompt design is the primary variable on MCQs | Plain prediction ρ=0.12 → contrastive ρ=0.45 (same temp) |
| More cognitive scaffolding hurts | Cognitive modeling ρ=0.17, buggy reasoning ρ=0.19, direct prediction ρ=0.50 |
| Temperature is roughly flat once prompt is good | ρ≈0.35–0.58 across t=0.3–2.0 (Eedi, contrastive prompt) |
| Model × prompt × temperature interaction on SmartPaper | Gemini improves with temp (0.81→0.88); DeepSeek degrades (0.80→0.68) |
| Signal generalises on open-ended, collapses on MCQ | SmartPaper ρ=0.52 on 120 held-out items; Eedi ρ=−0.18 on 30 new items |
| Model size/benchmarks don't predict task performance | Scout 17B (ρ=0.37) > GPT-4o (ρ=0.17) |

## Theoretical Framing

We tested predictions from four learning sciences traditions:

- **Knowledge Component theory** (Koedinger et al., 2012) — teacher prediction framing
- **Buggy Procedures** (Brown & Burton, 1978) — production rule prompts
- **Conceptual Change** (Chi et al., 1994) — mental model prompts
- **Epistemic State Specification** (Tack & Piech, 2024) — E3-level misconception-structured simulation

All four predict that richer cognitive specification should improve estimation. **Our central finding is that it doesn't** — the binding constraint is not what the model knows about misconceptions but what it cannot know about their population prevalence.

---

## Repository Structure

```
study2-materials/
├── results-site/
│   └── paper.html              # Full paper with interactive figures (open in browser)
│
├── data/
│   ├── eedi/
│   │   ├── eedi_with_student_data.csv    # 1,869 items with IRT parameters
│   │   └── curated_eedi_items.csv        # Probe + confirmation items
│   ├── smartpaper/                        # 140 open-ended items, 4 subjects
│   └── DATA_DICTIONARY.md
│
├── scripts/                    # All experiment scripts (~50 files)
│   ├── run_rsm_experiment.py             # Phase 1: RSM screening
│   ├── test_metaprompt_sweep.py          # Phase 2: Prompt optimisation
│   ├── test_cross_model.py              # Phase 3: 15-model comparison
│   ├── test_cognitive_modeling.py        # Phase 3: Structured elicitation
│   ├── test_buggy_reasoning.py
│   ├── test_two_stage.py
│   ├── run_confirmation.py              # Phase 4: 105-item confirmation
│   ├── run_smartpaper_expansion.py      # Phase 4: 140-item SmartPaper
│   ├── run_transition_zone.py           # Phase 5: Temperature sweeps
│   ├── replicate_*.py                   # Literature replication attempts
│   └── analyze_*.py                     # Analysis scripts
│
├── pilot/                      # Raw experiment outputs (~25k files)
│   ├── rsm_experiment/                   # RSM screening + temperature sweeps
│   ├── replications/                     # Literature replications
│   │   ├── 3rep_fullset/                # Large-scale confirmation
│   │   ├── classroom_simulation*/       # Kröger et al. replication
│   │   ├── error_alignment/             # Liu et al. replication
│   │   ├── feature_extraction*/         # Razavi & Powers replication
│   │   └── uncertainty_difficulty*/     # EDM 2025 replication
│   ├── smartpaper_rsm_v2/              # SmartPaper RSM + temp sweeps
│   ├── smartpaper_expansion/            # 140-item SmartPaper results
│   ├── smartpaper_two_stage/            # Two-stage diversity experiments
│   └── structured_estimation*/          # Structured elicitation results
│
├── docs/
│   ├── difficulty_estimation_prompts.md  # Literature review (13 papers)
│   └── prompt-strategies-v2.md           # Prompt design rationale (S1–S4)
│
├── .claude/handoffs/           # Session-by-session research documentation
│   ├── 2026-02-02-research-journey.md   # Full research arc (14 phases)
│   ├── 2026-02-02-temperature-reframing.md
│   ├── 2026-02-02-model-selection-experiment.md
│   └── ...
│
├── paper_sections_draft.md     # Polished intro + method draft
├── paper_draft_v3.md           # Earlier full paper draft
└── EXPERIMENT_REPORT.md        # Detailed experiment log
```

## Quick Start

### View the paper

```bash
open results-site/paper.html
```

The paper includes interactive Plotly figures showing RSM heatmaps, temperature curves, and prompt comparisons.

### Run an experiment

```bash
# Install dependencies
pip install openai google-generativeai groq scipy numpy pandas tqdm

# Set API keys
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"

# Example: test contrastive prompt on Eedi probe items
python scripts/test_improved_prompts.py
```

### Inspect raw results

Each experiment in `pilot/` contains:
- Per-item response files (`.txt`) with full model output
- `results.json` or CSV summary with correlations
- Per-rep breakdowns where applicable

---

## Datasets

### Eedi (UK, MCQ)
- 1,869 diagnostic maths questions targeting specific misconceptions
- 4-option MCQ; each distractor maps to a known error pattern
- 73,000+ UK students, ages 11–16
- Difficulty: IRT b₂PL parameters (MLE on 15.8M responses)
- Source: [Kaggle competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

### SmartPaper (India, open-ended)
- 140 questions across English, Maths, Science, Social Science
- Open-ended with rubric scoring, Grades 6–8
- Indian government schools
- Difficulty: classical proportion correct

---

## Experimental Phases

| Phase | Description | Key Result |
|-------|-------------|------------|
| 1. RSM Screening | 46 configs on 20 probe items | One viable region: teacher prediction framing |
| 2. Metaprompt Sweep | 63 configs, prompt optimisation | Error analysis ρ=0.50, contrastive ρ=0.51 (10-rep) |
| 3. Boundary Testing | 15 models, structured elicitation | More scaffolding hurts; model tier structure |
| 4. Generalisation | 105 Eedi items, 140 SmartPaper | Eedi: ρ=0.11 (null); SmartPaper: ρ=0.55 (holds) |
| 5. Mechanistic | Temperature sweeps, two-stage tests | Prompt > temperature; two-stage destroys signal |

Total cost: ~$50 in API calls across all phases.

---

## Citation

```
@inproceedings{lomas2026difficulty,
  title={When Does LLM Difficulty Estimation Work? A Systematic Search Across Methods, Models, and Datasets},
  author={Lomas, Derek},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence in Education (AIED)},
  year={2026}
}
```

---

## License

Research materials provided for review and reproducibility. Data usage subject to original dataset licenses (Eedi: Kaggle competition terms; SmartPaper: provided with permission).

---

*Last updated: February 2, 2026*
