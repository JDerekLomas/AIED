# Handoff: RSM Experiment & Prompt Optimization

**Date:** 2026-02-02
**Status:** Metaprompt sweep running, cross-model test in second window

## Current State

### What's Working
- **Best single-run:** Config 7 (teacher_prediction, temp=1.5, partial hint, Gemini Flash) ρ=0.673 (p=0.001)
- **Best stable (3-rep):** v3_contrastive prompt at temp=1.5: ρ=0.577±0.075 (all 3 reps significant)
- **Metaprompt sweep in progress:** v5_error_analysis at t=1.8 hit ρ=0.650 (single rep so far)

### Key Findings

**1. RSM Sweep (11/46 configs, then killed for speed)**
- Only teacher_prediction × temp=1.5 achieved significance (ρ=0.67, p=0.001)
- Sharp interaction: same prompt at temp=0.3 → ρ=0.12
- classroom_batch and individual_roleplay never exceeded ρ=0.41
- Full hints hurt (ρ=0.12) vs hidden (ρ=0.41) or partial (ρ=0.26)
- Results: `pilot/rsm_experiment/results.csv`

**2. Prompt Variant Stability Test (4 variants × 3 reps)**
- v3_contrastive: 0.549±0.017 (most stable, all reps p<0.05)
- v0_baseline: 0.370±0.097
- v1_anchored: 0.376±0.068
- v4_contrastive+anchored: 0.315±0.097 (anchoring hurts!)
- Results: `pilot/rsm_experiment/prompt_variants_reps/`

**3. Cross-Model Test (contrastive prompt, 3 reps)**
- Gemini Flash: ρ=0.549±0.017 (strong)
- Llama-70B (Groq): ρ=0.272±0.126 (weak signal, high variance)
- Qwen-32B (Groq): ρ=0.005±0.110 (zero)
- GPT-4o-mini: ρ=-0.092±0.266 (zero, very unstable)
- **Finding is Gemini Flash-specific**, not a general LLM capability
- Results: `pilot/rsm_experiment/cross_model/`

**4. Metaprompt Sweep (IN PROGRESS — b4b756c)**
- 7 prompts × 3 temps (1.5, 1.8, 2.0) × 3 reps × 20 items = 1260 calls
- Partial results so far:
  - v3_contrastive t=1.5: 0.577±0.075
  - v3_contrastive t=1.8: 0.518±0.061
  - v3_contrastive t=2.0: ~0.51 (1 rep)
  - v5_error_analysis t=1.5: 0.451±0.049
  - v5_error_analysis t=1.8: 0.650 (1 rep — promising!)
  - v6_distractor_first: PARSING FAILURE (two-step format breaks regex)
- **Higher temp does NOT help contrastive** — t=1.5 > t=1.8 > t=2.0
- v6 needs parser fix (looks for the 4-level format but the two-step prompt puts reasoning first)
- Results: `pilot/rsm_experiment/metaprompt_sweep/`

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/run_rsm_experiment.py` | Full Box-Behnken RSM sweep | Killed at 11/46 configs |
| `scripts/test_improved_prompts.py` | Single-rep prompt test | Complete |
| `scripts/test_improved_prompts_reps.py` | 3-rep stability test | Complete |
| `scripts/test_cross_model.py` | Cross-model generalization | Complete (may be re-running in 2nd window) |
| `scripts/test_metaprompt_sweep.py` | 7 prompts × 3 temps sweep | Running (background b4b756c) |

## Probe Items
- 20 items from `pilot/rsm_experiment/probe_items.csv`
- Stratified by b_2pl quintile (4 per quintile)
- All correlations are Spearman ρ against IRT b_2pl

## Objective Function
`Spearman rho(weighted_p_incorrect, b_2pl)` on 20 probe items.
- weighted_p_incorrect = 1 - Σ(weight_level × p_correct_level)
- Weights: below_basic=0.25, basic=0.35, proficient=0.25, advanced=0.15

## API Keys Available
- GOOGLE_API_KEY (Gemini) — in .env
- GROQ_API_KEY — paid tier now, in .env
- OPENAI_API_KEY — in .env
- ANTHROPIC_API_KEY — in .env

## Next Steps

1. **Check metaprompt sweep results** when it finishes — especially v5_error_analysis at t=1.8 and the remaining variants (v7-v10)
2. **Fix v6_distractor_first parser** — needs to handle the two-step output format (skip reasoning, find the distribution lines)
3. **Two-stage prompting experiment** (Derek's idea):
   - Stage 1: Generate diverse student backstories at high temp
   - Stage 2: Use backstories as context for teacher predictions
   - Hypothesis: adds variance without the full simulation failure mode
4. **Cognitive modeling experiment** (Derek's idea):
   - Generate 10 incomplete chains of thought per item (where students stop/err at different points)
   - Aggregate into response distributions
   - Related work: Brown & Burton BUGGY, process mining in education
5. **Cross-model re-run with paid Groq** — higher rate limits, cleaner data (original had 429 errors)
6. **Cross-validate on held-out items** — remaining 85 items in the calibrated set
7. **Update paper** — `paper_draft_v3.md` has RSM + contrastive results, needs metaprompt and cross-model findings

## Key Decisions Made
- Killed RSM sweep at 11/46 — the interaction finding was clear and remaining configs were slow
- Teacher prediction > simulation — reframing from "be a student" to "predict as a teacher" is the breakthrough
- Contrastive prompt addresses selectivity problem explicitly — forces item-specific reasoning
- Anchoring/calibration anchors consistently hurt — model has better internal calibration than our provided anchors
- Temperature sweet spot appears to be 1.5 (not higher) for contrastive, but other prompts may peak higher

## Paper
- `paper_draft_v3.md` — new paper incorporating RSM results
- Framing: "From Zero to Moderate" — most configs fail, one specific approach works
- Needs updating with metaprompt sweep and cross-model results

## Visualization
- `results-site/rsm_analysis.html` — interactive Plotly dashboard with 3D surface, heatmaps, etc.
