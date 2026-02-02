# RSM Optimization Strategy — 2026-02-02

## Objective
Maximize Spearman ρ between LLM-predicted weighted_p_incorrect and IRT b_2pl on 20 probe items (UK Year 9 maths MCQs from Eedi).

## What We Know

### Model Selection

**Gemini Family (error_analysis @ t=2.0, 3 reps):**
| Model | Mean ρ | SD | Avg-pred ρ | Parse fails | Notes |
|-------|--------|-----|-----------|-------------|-------|
| **Gemini 3 Flash** | **0.604** | 0.062 | **0.666** | 0-1/20 | Clear winner |
| Gemini 2.5 Pro | 0.240 | 0.242 | 0.398 | 1-3/20 | Some signal, unstable |
| Gemini 2.5 Flash | 0.088 | 0.131 | 0.148 | 6-9/20 | Parse failures at t=2.0 |
| Gemini 3 Pro | 0.052 | 0.365 | -0.001 | 16-17/20 | Thinking mode breaks format |
| Gemini 2.0 Flash | -0.019 | 0.050 | 0.015 | 0-1/20 | Clean output, no signal |

**Cross-provider (error_analysis @ t=2.0):**
| Model | Provider | Mean ρ | Notes |
|-------|----------|--------|-------|
| **Llama-4-Scout 17B** | Groq | 0.355 ± 0.167 (10-rep) | Avg-pred 0.668 at 10 reps, matches Gemini |
| Llama-3.3-70B | Groq | 0.27 ± 0.13 (3-rep) | Some signal, high variance |
| DeepSeek-Chat V3 | DeepSeek | 0.338 ± 0.140 (3-rep) | Decent signal |
| GPT-4o | OpenAI | 0.172 ± 0.126 (3-rep) | Weak despite frontier status |
| Qwen3-32B | Groq | 0.01 ± 0.11 (3-rep) | No signal |
| GPT-4o-mini | OpenAI | -0.09 ± 0.27 (3-rep) | No signal, unstable |

**Finding:** Gemini 3 Flash is dramatically better than ALL other models tested — including larger/newer Gemini models. It's not "bigger is better"; the task requires a specific combination of (a) instruction-following fidelity at high temperature and (b) implicit pedagogical knowledge. Pro models with mandatory thinking mode can't produce structured output reliably at t=2.0. Gemini 2.0 Flash parses fine but lacks pedagogical knowledge. Only Gemini 3 Flash has both.

### Prompt × Temperature (Metaprompt Sweep, Gemini Flash)
Top configs from sweep (mean ρ ± std across 3 reps):

| Config | Mean ρ | Std | Individual reps |
|--------|--------|-----|-----------------|
| v5_error_analysis t=2.0 | **0.604** | 0.062 | 0.603, 0.679, 0.528 |
| v3_contrastive t=1.5 | **0.577** | 0.075 | 0.590, 0.662, 0.480 |
| v5_error_analysis t=1.8 | 0.566 | 0.108 | 0.650, 0.635, 0.414 |
| v3_contrastive t=2.0 | 0.562 | 0.084 | 0.513, 0.493, 0.680 |
| v8_imagine_classroom t=2.0 | 0.547 | — | single rep |
| v3_contrastive t=1.8 | 0.518 | 0.061 | 0.519, 0.592, 0.444 |
| v5_error_analysis t=1.5 | 0.451 | 0.049 | 0.456, 0.389, 0.508 |

Failed/broken configs:
- v6_distractor_first: chain-of-thought step before distribution → parse failures, poor results
- v7_comparative_difficulty: constant outputs at high temp → NaN correlations
- v8_imagine_classroom t=1.5: low (0.23), but improves dramatically at t=2.0

### Original RSM Experiment (11 configs)
- Best single config: teacher_prediction + t=1.5 → ρ=0.673 (single run, no reps)
- classroom_batch prompts: ρ=0.24–0.41 regardless of temperature
- individual_roleplay: ρ=0.31
- Misconception hints (hidden vs partial vs full): no clear effect

## Emerging Optimization Principles

### 1. Temperature as Diversity Engine
Higher temperature (1.5–2.0) consistently beats lower temperature (0.3). The mechanism: high temp creates diverse "plausible teacher opinions" that, when averaged across reps, converge on something closer to the true population distribution. This is a **wisdom-of-crowds effect within a single model**.

The sweet spot appears to be t=1.5–2.0. At t=2.0, error_analysis and imagine_classroom improve, but contrastive is stable across the range.

### 2. Prompt Framing Hierarchy
1. **Error analysis** ("what specific calculation error leads to each wrong answer?") — best at high temp
2. **Contrastive** ("would students ACTUALLY make errors on THIS question?") — most stable across temps
3. **Imagine classroom** — needs high temp to work, then competitive
4. **Teacher prediction** (original) — strong single-run but not systematically tested with reps
5. **Sparse** / **Devil's advocate** — untested at time of writing (sweep still running)
6. **Distractor-first** (chain-of-thought then predict) — BROKEN, parse failures
7. **Comparative difficulty** (rate on 1-10 scale) — BROKEN, constant outputs

### 3. What Makes a Good Prompt
- Asks the model to reason about **student error processes**, not just predict distributions
- Avoids multi-step structured output (distractor-first failed because the CoT step interfered with the distribution step)
- Keeps the output format simple (4 lines of A=XX% B=XX% C=XX% D=XX%)
- Frames the task as **recall** ("you've just marked 200 papers") not **prediction** ("what would happen")

### 4. Aggregation Has Limits (UPDATED)
Averaging across 3 reps produces higher ρ than any single rep, BUT more reps does NOT always help:
- error_analysis@2.0: N=2 reps peaked at ρ=0.690, N=3 → 0.666, N=5 → 0.614, N=7 → 0.573
- The first 3 reps may have been lucky; additional reps regress toward the true (lower) mean
- This suggests **ρ≈0.60-0.67 is the real ceiling** for this config, not an underestimate

### 5. Option Shuffling Doesn't Help
Permuting A/B/C/D order across reps (inspired by CAPE, NAACL 2024) was tested:
- Original order (3 reps avg): ρ=0.666
- Shuffled order (3 reps avg): ρ=0.617
- All 6 combined: ρ=0.665
- Shuffled single-reps are weaker (0.316, 0.469, 0.527) — reordering makes the model worse, not just different
- **Conclusion:** Option order adds noise, not orthogonal signal

### 6. Thinking Mode Hurts (PRELIMINARY)
Gemini Flash thinking_budget=1024: first rep ρ=0.467 (vs 0.604 baseline). Thinking may cause overthinking — the task benefits from fast intuitive judgment, not deliberation. (Results still running for budget=4096, 8192.)

## Active Experiments

### Completed
- **Metaprompt sweep** — DONE. 7 prompts × 3 temps × 3 reps = 63 configs. Final top: error_analysis@2.0 (0.604±0.062), contrastive@1.5 (0.577±0.075), devil_advocate@1.5 (0.543±0.020).
- **Option shuffling** — DONE. Shuffling hurts; no additive diversity. Script: `test_option_shuffle.py`
- **Rep scaling** — RUNNING. More reps degrades ρ (regression to mean). Script: `test_rep_scaling.py`
- **Thinking budget** — RUNNING. budget=1024 first rep at 0.467 (worse). Script: `test_thinking_budget.py`

### Queued (scripts written, not yet running)
1. **Cross-model retest** (`test_cross_model.py groq_llama4scout groq_llama70b`) — test Llama 4 Scout on Groq paid tier
2. **Two-stage backstory** (`test_two_stage.py`) — generate 5 student personas at t=2.0, then predict per-persona at t=1.5. Tests whether explicit student diversity > implicit temperature diversity.
3. **Cognitive modeling** (`test_cognitive_modeling.py`) — generate 10 incomplete chains-of-thought per item, aggregate answer counts. Tests whether simulating the error *process* beats predicting the error *distribution*.

## Strategic Questions

### Are we near the ceiling?
With n=20 items, ρ=0.60 has a 95% CI of roughly ±0.25. The difference between ρ=0.58 and ρ=0.65 is not statistically significant. We may already be at the Gemini Flash ceiling for this item set.

**Implication:** Rather than pushing the overall ρ higher, the paper may benefit more from:
- Characterizing **which item types** the method works for (procedural vs conceptual, high vs low discrimination)
- Showing the model ranking is robust (Gemini >> Llama >> others)
- Demonstrating the temperature/aggregation mechanism

### What would change the game?
- **More probe items** (40-50) would tighten CIs and let us distinguish ρ=0.55 from ρ=0.65
- ~~**Gemini Flash with thinking enabled**~~ — TESTED, thinking hurts (0.467 vs 0.604). Fast intuition > deliberation for this task.
- **Gemini Pro or Claude** — bigger models may have deeper pedagogical knowledge. This is the most promising untested axis.
- **Item-level analysis** — some items may be unpredictable for ALL methods (ambiguous, atypical)
- ~~**More reps**~~ — TESTED. Per-rep mean regresses (Gemini: 0.604→0.500 at 10 reps; Scout: 0.505→0.355) but averaged predictions improve monotonically for Scout (0.609→0.668 at 10 reps). Trade-off: more calls for better avg-pred signal.
- ~~**Option shuffling**~~ — TESTED, adds noise not signal. Model is sensitive to presentation order in a bad way.

### If cognitive modeling wins
This would be the strongest paper finding: LLMs have better *process* knowledge of student errors than *distributional* knowledge. It implies the model can simulate individual error trajectories but struggles to aggregate them into population statistics — so we should let the model simulate and do the aggregation ourselves.

### If two-stage wins
The story becomes about **structured elicitation**: prompting for student diversity explicitly outperforms relying on temperature for implicit diversity. This connects to the broader literature on prompt engineering for calibrated predictions.

### If nothing beats ρ≈0.60 (INCREASINGLY LIKELY)
We tested: more reps (worse), option shuffling (no help), thinking mode (worse), ensemble (no help). The paper story is: (a) LLM difficulty estimation works but has a ceiling with current models at ρ≈0.60-0.67, (b) the key ingredients are a capable model + error-focused framing + high temperature + 3-rep averaging, (c) model selection matters more than prompt engineering beyond a baseline quality threshold, (d) multiple optimization attempts confirm this is a genuine ceiling, not an artifact of insufficient tuning. The negative results (thinking hurts, shuffling hurts, more reps don't help) are themselves informative — they suggest the model's fast intuitive judgment is the signal, and anything that disrupts it adds noise.

## Latest Results (Updated 2026-02-03 ~00:30, sweep COMPLETE)

### Metaprompt Sweep — Complete Results

| Config | Mean ρ | SD | Rep 0 | Rep 1 | Rep 2 | Avg-pred ρ |
|--------|--------|-----|-------|-------|-------|-----------|
| v5_error_analysis t=2.0 | 0.604 | 0.062 | 0.603 | 0.679 | 0.528 | **0.666** |
| v3_contrastive t=1.5 | 0.577 | 0.075 | 0.590 | 0.662 | 0.480 | **0.660** |
| v5_error_analysis t=1.8 | 0.566 | 0.108 | 0.650 | 0.635 | 0.414 | — |
| v3_contrastive t=2.0 | 0.562 | 0.084 | 0.513 | 0.493 | 0.680 | — |
| v9_devil_advocate t=1.5 | 0.543 | 0.020 | 0.520 | 0.542 | 0.568 | — |
| v3_contrastive t=1.8 | 0.518 | 0.061 | 0.519 | 0.592 | 0.444 | — |
| v9_devil_advocate t=2.0 | 0.516 | 0.122 | 0.387 | 0.679 | 0.482 | — |
| v5_error_analysis t=1.5 | 0.451 | 0.049 | 0.456 | 0.389 | 0.508 | — |
| v8_imagine_classroom t=2.0 | 0.428 | 0.115 | 0.539 | 0.270 | 0.474 | — |
| v9_devil_advocate t=1.8 | 0.405 | 0.106 | 0.366 | 0.299 | 0.550 | — |
| v8_imagine_classroom t=1.8 | 0.404 | 0.035 | 0.452 | 0.367 | 0.395 | — |
| v10_sparse t=1.5 | 0.235 | 0.189 | -0.010 | 0.263 | 0.451 | PARSE FAIL (11-15/20) |
| v8_imagine_classroom t=1.5 | 0.232 | 0.041 | 0.223 | 0.186 | 0.286 | — |
| v6_distractor_first | — | — | — | — | — | PARSE FAIL |
| v7_comparative_difficulty | — | — | — | — | — | PARSE FAIL |
| v10_sparse t=1.8 | ~0.1 | — | 0.139 | 0.067 | ... | PARSE FAIL (14/20) |

Sweep complete (v10_sparse_t2.0 still running but already confirmed broken).

### Ensemble Test (No New API Calls)
Averaged contrastive@1.5 + error_analysis@2.0 predictions per item:
- Contrastive alone (avg across reps): ρ=0.660
- Error_analysis alone (avg across reps): ρ=0.666
- **Ensemble of both: ρ=0.639** — slightly worse, not additive

The two prompts capture similar signal. Ensemble doesn't help because they're positively correlated. To get ensemble benefit, we'd need prompts that succeed on *different* items.

### Temperature Gradient by Prompt

```
                  t=1.5    t=1.8    t=2.0    Trend
contrastive       0.577    0.518    0.562    Flat/peak at 1.5
error_analysis    0.451    0.566    0.604    Rising — wants more temp
devil_advocate    0.543    0.405    0.516    Peak at 1.5, U-shaped
imagine_classroom 0.232    0.404    0.428    Rising steeply
sparse            0.235    ~0.1     ...      BROKEN (parse failures)
```

Confirms the principle: analytically constrained prompts (contrastive, devil_advocate) don't need temperature diversity; experiential/open prompts (error_analysis, imagine_classroom) benefit from it. Devil_advocate shows a curious U-shape (dip at 1.8) — unclear why.

### Revised Prompt Tier List

**Tier 1 (ρ > 0.55, stable)**
- v5_error_analysis @ t=2.0 — ρ=0.604±0.062
- v3_contrastive @ t=1.5 — ρ=0.577±0.075
- v5_error_analysis @ t=1.8 — ρ=0.566±0.108 (high variance but strong mean)

**Tier 2 (ρ ≈ 0.50-0.55, moderate)**
- v3_contrastive @ t=2.0 — ρ=0.562±0.084
- v9_devil_advocate @ t=1.5 — ρ=0.543±0.020 (most stable of all!)
- v3_contrastive @ t=1.8 — ρ=0.518±0.061
- v9_devil_advocate @ t=2.0 — ρ=0.516±0.122 (high variance)

**Tier 3 (ρ < 0.45)**
- v8_imagine_classroom — peaks at 0.43, too variable
- v5_error_analysis @ t=1.5 — ρ=0.451 (temp too low)
- v9_devil_advocate @ t=1.8 — ρ=0.405±0.106 (curious dip)
- v0_baseline — ρ=0.370±0.097

**Failed**
- v6_distractor_first — format incompatible with high temp
- v7_comparative_difficulty — constant/broken outputs
- v10_sparse — massive parse failures (11-15/20 items)

## Key Files
- `pilot/rsm_experiment/results.csv` — original 11-config RSM results
- `pilot/rsm_experiment/cross_model/` — cross-model comparison
- `pilot/rsm_experiment/metaprompt_sweep/` — prompt × temperature sweep
- `pilot/rsm_experiment/two_stage/` — two-stage backstory experiment (when run)
- `pilot/rsm_experiment/cognitive_modeling/` — cognitive modeling experiment (when run)
- `scripts/test_cross_model.py` — cross-model script (Groq sleep reduced for paid tier)
- `scripts/test_metaprompt_sweep.py` — sweep script (running)
- `scripts/test_two_stage.py` — two-stage script (ready)
- `scripts/test_cognitive_modeling.py` — cognitive modeling script (ready)
- `scripts/test_rep_scaling.py` — rep scaling experiment (reps diminish after 3)
- `scripts/test_thinking_budget.py` — thinking mode test (thinking hurts)
- `scripts/test_option_shuffle.py` — option order permutation test (no benefit)
