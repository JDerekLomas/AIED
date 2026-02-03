# Prompt Framing Results

## Response Surface Methodology

We screened 15 prompt framings across 2-4 temperature settings using Gemini 2.5 Flash on 140 open-ended items (SmartPaper, India, Grades 6-8). Each configuration was evaluated with 3 replications (420 API calls per config). The response surface reveals a clear interaction between framing category and temperature.

## Temperature x Framing Interaction

The central finding is that optimal temperature depends on framing type:

- **Structural framings** (prerequisite chain, cognitive load) that decompose items by knowledge requirements show a temperature-dependent split: prerequisite chain peaks at t=0.0-0.25 (rho=0.666), while cognitive load peaks at t=2.0 (rho=0.673).
- **Analytical framings** (contrastive, error affordance) show moderate performance (rho=0.49-0.57) with weak temperature sensitivity.
- **Simulation framings** (classroom sim, synthetic students) consistently underperform structural approaches regardless of temperature.

This interaction explains why single-temperature studies disagree: a method that works at t=0.25 may fail at t=2.0, and vice versa.

## Category Comparison

Framings that ask "what structural properties make this item hard?" consistently outperform framings that ask "how would students respond to this item?"

| Category | Best rho | Representative framing |
|----------|----------|----------------------|
| Structural analysis | 0.673 | cognitive_load (t=2.0) |
| Prerequisite mapping | 0.666 | prerequisite_chain (t=0.25) |
| Cognitive profiling | 0.586 | cognitive_profile (t=1.0) |
| Contrastive analysis | 0.572 | contrastive (t=1.0) |
| Error analysis | ~0.52 | error_analysis (t=1.0) |
| Decomposed teaching | 0.487 | teacher_decomposed (t=1.0) |
| Simulation-based | <0.45 | classroom_sim, synthetic |

## Top Configurations

| Rank | Config | rho | 95% CI | MAE |
|------|--------|-----|--------|-----|
| 1 | cognitive_load t=2.0 | 0.673 | [0.550, 0.766] | 0.190 |
| 2 | prerequisite_chain t=0.25 | 0.666 | [0.563, 0.749] | 0.155 |
| 3 | prerequisite_chain t=0.0 | 0.661 | [0.557, 0.739] | 0.160 |
| 4 | cognitive_load t=0.5 | 0.623 | [0.494, 0.729] | 0.200 |
| 5 | cognitive_load t=1.0 | 0.617 | [0.474, 0.727] | 0.192 |

The top two configurations achieve comparable correlation (overlapping CIs) but prerequisite_chain has lower MAE (0.155 vs 0.190), indicating better-calibrated absolute predictions.

## Implications

These results suggest that LLMs estimate difficulty best when prompted to analyze item structure (prerequisite knowledge, cognitive demands) rather than simulate student behavior. This aligns with the broader finding that LLMs succeed on complexity-driven difficulty but fail on selectivity-driven difficulty: structural analysis captures complexity, while simulation attempts to capture selectivity and fails.
