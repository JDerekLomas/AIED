# Prompt Framing Experiment

**Dataset:** SmartPaper (140 open-ended items, Grades 6-8, India)
**Model:** Gemini 3 Flash
**Reps:** 3 per configuration
**Metric:** Spearman rho vs. classical proportion-correct difficulty

## Design

15 prompt framings x 2-4 temperature settings x 3 reps x 140 items = ~25,000 API calls.

Each framing asks the model to estimate item difficulty (0-1 scale) but structures the reasoning differently. Framings range from structural analysis (prerequisite chains, cognitive load) to simulation-based (classroom simulation, synthetic students) to analytical (contrastive, error affordance).

## Results

| Framing | Temp | rho | p | 95% CI | MAE | N |
|---------|------|-----|---|--------|-----|---|
| **prerequisite_chain** | 0.25 | **0.666** | 3e-19 | [0.563, 0.749] | 0.155 | 140 |
| prerequisite_chain | 0.0 | 0.661 | 6e-19 | [0.557, 0.739] | 0.160 | 140 |
| prerequisite_chain | 0.5 | 0.658 | — | — | — | 140 |
| prerequisite_chain | 1.0 | 0.645 | — | — | — | 140 |
| prerequisite_chain | 2.0 | 0.627 | — | — | — | 140 |
| **cognitive_load** | 2.0 | **0.673** | 9e-20 | [0.550, 0.766] | 0.190 | 140 |
| cognitive_load | 0.5 | 0.623 | 2e-16 | [0.494, 0.729] | 0.200 | 140 |
| cognitive_load | 1.0 | 0.617 | 5e-16 | [0.474, 0.727] | 0.192 | 140 |
| familiarity_gradient | 1.0 | 0.609 | — | — | — | 140 |
| familiarity_gradient | 2.0 | 0.601 | — | — | — | 140 |
| cognitive_profile | 1.0 | 0.586 | 3e-14 | [0.456, 0.693] | 0.214 | 140 |
| cognitive_profile | 2.0 | 0.584 | 4e-14 | [0.449, 0.694] | 0.214 | 140 |
| contrastive | 1.0 | 0.572 | — | — | — | 140 |
| contrastive | 2.0 | 0.554 | — | — | — | 140 |
| error_affordance | 1.0 | 0.493 | 6e-10 | [0.348, 0.607] | 0.179 | 140 |
| teacher_decomposed | 1.0 | 0.487 | 8e-09 | [0.355, 0.609] | 0.273 | 125 |
| error_affordance | 2.0 | 0.478 | 2e-09 | [0.327, 0.601] | 0.184 | 140 |
| teacher_decomposed | 2.0 | 0.458 | 1e-08 | [0.306, 0.582] | 0.269 | 140 |
| error_analysis | 1.0 | ~0.52 | — | — | — | 140 |
| error_analysis | 2.0 | ~0.50 | — | — | — | 140 |

Additional framings tested: teacher, imagine_classroom, misconception_holistic, buggy_rules, classroom_sim, devil_advocate, verbalized_sampling, irt_sim, kc_mastery, synthetic variants. See `results.json` for full data.

## Key Findings

1. **Structural framings dominate.** Prerequisite chain (rho=0.67) and cognitive load (rho=0.67) outperform all simulation-based approaches. Framings that ask "what must students know?" beat framings that ask "simulate how students respond."

2. **Temperature interacts with framing type.** Structural framings (prerequisite_chain) work best at low temp (0.0-0.25). Cognitive load peaks at high temp (2.0). This interaction is the core RSM finding.

3. **Simulation framings underperform.** Classroom simulation, synthetic students, and IRT simulation framings all produce lower correlations than direct structural analysis.

4. **Parse reliability varies.** Teacher_decomposed had 12% parse failures at t=1.0. Structural framings achieved 100% parse rates.

5. **Effect is robust across items.** All top framings achieve N=140 with p < 1e-15. The signal is not driven by outliers (confirmed via probe inflation analysis).

## Visualizations

- `rsm_extended.html` — Interactive 3D response surface (framing x temperature x rho)
- `rsm_dashboard.html` — Dashboard view with all framings
- `rsm_3d.html` — 3D surface plot
- `rsm_viz.html` — Basic RSM visualization

Open any HTML file in a browser; they use Plotly.js (loaded from CDN).

## Reproduction

```bash
# Requires GOOGLE_API_KEY for Gemini 3 Flash
export GOOGLE_API_KEY="your-key"
python scripts/run_prompt_screening_g3f.py
```

Results are saved per-framing in subdirectories (e.g., `prerequisite_chain_t0.25/`) with per-item `.txt` files containing full model output, plus aggregated `results.json`.

## Data Files

- `results.json` — All configurations with rho, p, CI, MAE, bias, N, parse stats
- `rsm_data.json` — Data formatted for RSM visualization
- `*/rep0/*.txt` — Raw model outputs per item per rep
