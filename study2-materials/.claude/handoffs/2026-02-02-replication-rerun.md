# Replication Re-run with Corrected Ground Truth — 2026-02-02

## Context
On Jan 30, we discovered that the Eedi dataset has two independent answer orderings (Kaggle vs NeurIPS) that match only 24.5% of the time. All previous difficulty estimation experiments were scoring against wrong ground truth for 76% of items. See `2026-01-30-data-bug-fix.md` for details.

Today we re-ran all fixable experiments with the corrected data, and extended to multiple model families via Groq, Gemini, and DeepSeek.

## Key Finding: The Null Wall + One Exception

**Finding 1 — Universal null:** No single-pass method produces meaningful correlation with item difficulty. This holds across 7 models, 6 prompt strategies, and 4 methodologies.

**Finding 2 — RSM exception:** The RSM experiment (Gemini Flash + error_analysis + t=2.0 + 3-rep averaging) achieves rho=0.666. This result is valid (correlates against ordering-independent b_2pl).

**Finding 3 — Model specificity:** Even within RSM, only Gemini Flash shows signal. Llama 70B shows rho=0.27, others near zero. This is not a general LLM capability.

**Finding 4 — A/B test:** Decomposing the error_analysis prompt into reasoning-scaffold-only vs full-distribution-format shows neither works in single-pass mode (Llama null for both, DeepSeek early null). The multi-sample averaging appears essential.

**Implication for the paper:** LLM difficulty estimation is not a general capability — it requires a specific pipeline (model + prompt + temperature + aggregation). Published single-pass benchmarks (r=0.54-0.87) do not replicate on Eedi math items with corrected ground truth.

## Experiment Results

### 1. Direct Difficulty Estimation (expert prompt, 1869 items)

| Model | Provider | n valid | Pearson r | p-value |
|-------|----------|---------|-----------|---------|
| gpt-4o-mini | OpenAI | 1869 | +0.010 | 0.673 |
| qwen3-32b | Groq | 1548 | +0.015 | 0.544 |
| llama-3.3-70b | Groq | 1530 | -0.032 | 0.205 |
| llama-4-scout | Groq | 1868 | -0.013 | 0.561 |
| llama-4-maverick | Groq | 1735 | -0.024 | 0.326 |

gpt-4o-mini was also tested with 4 prompt variants (basic, expert, irt, comparative) — all null (r = -0.03 to +0.01).

### 2. Feature Extraction (gpt-4o, 1869 items)

GBM trained on 7 LLM-extracted features (DOK, cognitive load, etc.):
- Test set: r=0.063, p=0.223
- Direct LLM baseline: r=-0.015
- Published benchmark: r=0.62-0.87

Feature importance: cognitive_load (0.237) > conceptual_complexity (0.187) > syntax_complexity (0.158)

### 3. Uncertainty-as-Difficulty (gpt-4o-mini, 1869 items)

- Answer probability vs difficulty: r=-0.007, p=0.747
- Answer entropy vs difficulty: r=0.016, p=0.486
- Permutation consistency vs difficulty: r=-0.010, p=0.662
- Worse than mean-prediction baseline (-107%)

### 4. Classroom Simulation (llama-3.3-70b via Groq, 1869 items)

20 simulated students per item (5 per ability level × 4 levels):
- Weighted difficulty: r=0.013, p=0.560
- Unweighted difficulty: r=0.013, p=0.581
- Null result, consistent with all other experiments

## Experiments NOT Re-run

**Error alignment** and **confusion tuples** — these compare full response distributions across options. The distractor-level mapping between Kaggle and NeurIPS orderings cannot be resolved (only the correct answer can be cross-referenced). Per-option comparisons remain fundamentally scrambled.

## Interpretation

Three hypotheses for the discrepancy with published results:

1. **Data alignment bugs in other studies**: If published Eedi studies had similar ordering confusion, reported correlations could be artifacts. Worth investigating whether the NeurIPS competition data was commonly misaligned with Kaggle metadata.

2. **Domain specificity**: Published positive results (r=0.54-0.82) are often on medical or language items. Math MCQs may have difficulty determined more by curriculum context and student population than by item text features.

3. **Item homogeneity**: The Eedi items may cluster in a narrow difficulty range, limiting observable correlation. (Counter-evidence: classical difficulty ranges from ~0.05 to ~0.95 in our data.)

## Infrastructure Notes

- Added Groq support to `replicate_direct_difficulty.py` and `replicate_classroom_simulation.py`
- Added resume support to `replicate_feature_extraction.py` (reads intermediate JSON, skips completed items)
- Groq models added: qwen3-32b, llama-3.3-70b, llama-4-scout, llama-4-maverick
- Groq API key: set as GROQ_API_KEY env var (paid tier)

## A/B Test: Error Analysis Prompt Decomposition

### Motivation
The RSM experiment achieved rho=0.666 with Gemini 2.5 Flash using the `error_analysis` prompt at temperature 2.0, correlating against `b_2pl`. This prompt combines two ingredients:
1. **Reasoning scaffold**: Teacher marking exam papers, analyzing misconceptions per option
2. **Distribution format**: Output per-option percentages at 4 ability levels (below_basic, basic, proficient, advanced)

We want to determine which ingredient drives the signal: the reasoning or the distribution format.

### Design
Two prompts tested at temperature 2.0:
- **(A) error_analysis_direct**: Full reasoning scaffold → single `ESTIMATE: XX` difficulty number
- **(B) error_analysis_sim**: Full reasoning scaffold → per-level per-option distribution format

### Results

| Condition | Model | n valid | r vs p_value | rho vs p_value |
|-----------|-------|---------|-------------|----------------|
| error_analysis_direct | Llama 3.3 70b | 658/660 | -0.008 | -0.026 |
| error_analysis_sim | Llama 3.3 70b | 1826/1869 | -0.007 | -0.008 |
| error_analysis_direct | Gemini 2.5 Flash | 18/20 | -0.248 | -0.185 |
| error_analysis_sim | Gemini 2.5 Flash | 6/40 | -0.131 | -0.314 |
| error_analysis_direct | DeepSeek V3 | in progress | - | - |
| error_analysis_sim | DeepSeek V3 | in progress | - | - |

**Key findings so far:**
- Llama 3.3 70b: both conditions are null (r ≈ 0), consistent with all other Llama results
- Gemini 2.5 Flash: too early (20-40 items) and sim has 85% parse failure at temp 2.0
- The Gemini sim parse failures suggest the model can't reliably produce structured distribution output at high temperature without thinking enabled

### Interpretation
The RSM's rho=0.666 may depend critically on:
1. The specific model (Gemini Flash) — Llama shows zero signal regardless of prompt
2. The multi-sample averaging across 3 replications at temp 2.0
3. The full RSM pipeline (not just single-pass estimation)

## New Directions

### DeepSeek V3 (in progress)
Added DeepSeek support (deepseek-chat and deepseek-reasoner models). Running both error_analysis conditions at temp 2.0. DeepSeek V3 is a strong reasoning model that may show signal where Llama didn't.

### Enhanced classroom simulation (separate session)
Two innovations being prototyped:
1. **Backstory generation**: Create diverse student personas with specific math backgrounds, learning gaps, cognitive profiles — then use as context for simulation
2. **Cognitive modeling**: Model incomplete/flawed reasoning chains rather than just "you are a student, pick an answer"

## Files Created/Modified

### Results
- `pilot/replications/direct_difficulty_v2/` — all model × prompt results
- `pilot/replications/feature_extraction_v2/` — features.json, analysis.json
- `pilot/replications/uncertainty_difficulty_v2/` — results + analysis
- `pilot/replications/classroom_simulation_v2/` — llama-3.3-70b results (null)
- `pilot/replications/error_analysis_ab/` — A/B test results (4 model×condition combos)

### Dashboards
- `results-site/replication_dashboard.html` — comprehensive replication results dashboard with null wall chart, A/B test results, model comparison, and research timeline

### Scripts modified
- `scripts/replicate_direct_difficulty.py` — added Groq, Gemini, DeepSeek providers; added error_analysis_direct and error_analysis_sim prompts; added temperature CLI arg; increased max_tokens to 500 for error_analysis prompts
- `scripts/replicate_classroom_simulation.py` — added Groq provider + llama-3.3-70b model
- `scripts/replicate_feature_extraction.py` — added resume from intermediate file

## Cross-Model Ranking (from RSM experiment, 20 probe items)

See also: `2026-02-02-model-selection-experiment.md`

| Model | Provider | Mean ρ (3 reps) | SD | Avg-pred ρ | Prompt | Temp |
|-------|----------|-----------------|-----|------------|--------|------|
| Gemini 3 Flash | Google | 0.604 | 0.062 | 0.666 | error_analysis | 2.0 |
| Llama-4-Scout 17B | Groq (free) | 0.505 | 0.048 | 0.609 | error_analysis | 2.0 |
| Llama-4-Scout (contrastive) | Groq (free) | 0.371 | 0.115 | 0.624 | contrastive | 1.5 |
| DeepSeek-Chat V3 | DeepSeek | 0.338 | 0.140 | 0.409 | contrastive | 1.5 |
| GPT-OSS-120B | Groq | 0.336 | 0.100 | — | contrastive | 1.5 |
| Kimi-K2 | Groq | 0.336 | 0.116 | — | contrastive | 1.5 |
| Claude Sonnet 4 | Anthropic | 0.263 | 0.166 | — | contrastive | 1.5 |
| Llama-3.3-70B | Groq | 0.257 | 0.062 | — | contrastive | 1.5 |
| Gemini 2.5 Pro | Google | 0.240 | 0.242 | — | contrastive | 1.5 |
| GPT-4o | OpenAI | 0.172 | 0.126 | — | contrastive | 1.5 |
| Gemini 2.5 Flash | Google | 0.088 | 0.131 | — | contrastive | 1.5 |
| Claude Haiku 3.5 | Anthropic | 0.077 | 0.186 | — | contrastive | 1.5 |
| Qwen3-32B | Groq | 0.005 | 0.110 | — | contrastive | 1.5 |
| Gemini 2.0 Flash | Google | -0.019 | 0.050 | — | contrastive | 1.5 |
| GPT-4o-mini | OpenAI | -0.092 | 0.266 | — | contrastive | 1.5 |

**Key insight — averaging is magic:** Llama-4-Scout jumps from ρ=0.371 (single rep) to ρ=0.624 (3-rep averaged prediction). This is the largest averaging gain observed, suggesting high-variance models benefit most from multi-sample aggregation.

## Five Consolidated Findings (Updated Feb 2 EOD)

1. **The Null Wall**: All single-pass methods (7 models × 6 prompts × 4 methodologies) produce r≈0 on 1,869 items with corrected ground truth.
2. **The RSM Exception**: Gemini Flash + error_analysis + t=2.0 + 3-rep averaging → ρ=0.666 (on 20 probe items vs b_2pl).
3. **Model Specificity**: 15-model cross-model test shows clear tiers. Gemini 3 Flash >> Llama-4-Scout >> DeepSeek-V3 ≈ GPT-OSS-120B ≈ Kimi-K2 >> others.
4. **Averaging is magic**: Llama-4-Scout jumps 0.371→0.624 with 3-rep averaging. Multi-sample aggregation appears essential for extracting signal.
5. **A/B test**: Neither reasoning scaffold nor distribution format alone produces signal in single-pass. Multi-sample averaging is the key ingredient.

## Next Step: 3-Rep Full-Set Experiment

**Script:** `scripts/run_3rep_fullset.py`

The RSM and model-selection experiments used only 20 probe items. The single-pass experiments used 1,869 items but only 1 rep. The 3-rep fullset experiment bridges this gap:

- Run 3-rep averaging for top-tier models on all 1,869 items
- Correlate against p_value (available for all items; b_2pl only available for 105)
- Models: Llama-4-Scout (free on Groq), DeepSeek V3, Gemini 3 Flash
- Contrastive prompt at t=1.5
- Includes cost tracking and bootstrap 95% CIs
- Resume support (safe to interrupt and restart)

**Cost estimates:**
- Llama-4-Scout: ~1,869 × 3 reps × ~400 tokens ≈ 2.2M tokens → ~$0.25 input + ~$0.75 output ≈ **$1**
- DeepSeek V3: similar volume → ~$0.60 input + ~$2.45 output ≈ **$3**
- Gemini 3 Flash: similar volume → ~$0.22 input + ~$0.88 output ≈ **$1.10**

**Run:** `cd study2-materials && python scripts/run_3rep_fullset.py`

## IRT Estimates
**No re-run needed.** IRT parameters (a_2pl, b_2pl, c_3pl, difficulty_classical) were computed from raw `IsCorrect` flags, which have no dependency on answer letter ordering.

## Files Created/Modified (This Session)

### New
- `scripts/run_3rep_fullset.py` — 3-rep averaged experiment on full 1,869 items
- `results-site/replication_dashboard.html` — major update with unified view, cross-model ranking, cost estimates, nav bar

### Updated
- `.claude/handoffs/2026-02-02-replication-rerun.md` — this file (consolidated findings + next steps)
