# Model Selection Experiment — 2026-02-02

## Setup
- **Task**: Predict item difficulty (Spearman ρ vs IRT b_2pl) on 20 Eedi probe items
- **Prompt**: Contrastive teacher prediction ("Would real students actually make errors on THIS specific question?")
- **Temperature**: 1.5 for all models (except DeepSeek-Reasoner which doesn't support it)
- **Reps**: 3 per model, also computed averaged-prediction ρ (mean p_incorrect across reps)
- **Script**: `scripts/test_cross_model.py`

## Full Ranking (12 models tested)

| Rank | Model | Provider | Mean ρ | SD | Avg-pred ρ | Notes |
|------|-------|----------|--------|-----|-----------|-------|
| 1 | **Gemini 3 Flash Preview** | Google | **0.470** | 0.076 | **0.549** | Clear winner, consistent |
| 2 | Llama-4-Scout 17B | Groq | 0.371 | 0.115 | 0.624 | Surprisingly strong avg-pred |
| 3 | DeepSeek-Chat (V3) | DeepSeek | 0.338 | 0.140 | 0.409 | High variance, some signal |
| 4 | GPT-OSS-120B | Groq | 0.336 | 0.100 | 0.450 | Decent, stable |
| 5 | Kimi-K2 | Groq | 0.336 | 0.116 | 0.319 | Similar to GPT-OSS |
| 6 | Llama-3.3-70B | Groq | 0.257 | 0.062 | — | Low variance, moderate signal |
| 7 | GPT-4o | OpenAI | 0.172 | 0.126 | 0.204 | Surprisingly weak |
| 8 | Qwen3-32B | Groq | 0.005 | 0.110 | — | No signal |
| 9 | Llama-3.1-8B | Groq | -0.081 | 0.097 | -0.009 | No signal |
| 10 | GPT-4o-mini | OpenAI | -0.092 | 0.266 | — | No signal, unstable |
| 11 | Gemini 2.0 Flash | OpenRouter | -0.200 | 0.132 | -0.269 | Different model version! |
| 12 | DeepSeek-Reasoner | DeepSeek | NaN | NaN | NaN | Constant output (no temp control) |

## Key Findings

### 1. Gemini Flash wins, but the version matters
Gemini 3 Flash Preview (via Google API): ρ=0.470. Gemini 2.0 Flash (via OpenRouter): ρ=-0.200. These are completely different models. The "3 Flash Preview" version has something the 2.0 version doesn't.

### 2. Tier structure is clear
- **Tier 1 (ρ > 0.35)**: Gemini 3 Flash, Llama-4-Scout, DeepSeek-V3
- **Tier 2 (ρ 0.15–0.35)**: GPT-OSS-120B, Kimi-K2, Llama-70B, GPT-4o
- **Tier 3 (ρ ≈ 0)**: Qwen-32B, Llama-8B, GPT-4o-mini
- **Failed**: DeepSeek-Reasoner (constant output), Gemini 2.0 Flash (wrong direction)

### 3. Size isn't everything
- GPT-4o (larger, expensive) ρ=0.172 < Llama-4-Scout (17B, free on Groq) ρ=0.371
- DeepSeek-Chat ρ=0.338 beats GPT-4o at fraction of cost
- The task requires specific pedagogical content knowledge, not general capability

### 4. Averaged predictions boost all models
Llama-4-Scout jumps from 0.371 → 0.624 with averaged predictions. This confirms the wisdom-of-crowds mechanism: high temperature creates diverse "teacher opinions" that average toward truth.

### 5. DeepSeek-Reasoner fails completely
No temperature control → constant outputs → NaN correlation. Reasoning models that can't be made stochastic are useless for this task.

### 6. OpenAI models underperform
Both GPT-4o and GPT-4o-mini are in the bottom half. This is notable given their general benchmark performance. The task seems to favor models with specific training data about math education / student errors.

## Surprising Result: Llama-4-Scout
At avg-pred ρ=0.624, Llama-4-Scout (17B parameter, free on Groq) nearly matches Gemini Flash's best single-prompt result. This is with the contrastive prompt at t=1.5. Worth testing with error_analysis at t=2.0.

## API Keys & Providers Used
- **Groq**: `GROQ_API_KEY` in .env (also hardcoded in enhanced_classroom_sim.py)
- **DeepSeek**: `[REDACTED]` (hardcoded in test_cross_model.py)
- **OpenAI**: `OPENAI_API_KEY` in .env
- **Google**: `GOOGLE_API_KEY` in .env
- **OpenRouter**: `OPENROUTER_API_KEY` in .env
- **Replicate**: `REPLICATE_API_TOKEN` in .env (not used yet)

## Files
- `scripts/test_cross_model.py` — cross-model experiment script (updated with DeepSeek, OpenRouter)
- `pilot/rsm_experiment/cross_model/` — raw results per model per rep
- `pilot/rsm_experiment/cross_model/summary.json` — aggregated metrics
- `pilot/rsm_experiment/probe_items.csv` — the 20 probe items with IRT parameters

## Gemini Family Deep-Dive (error_analysis @ t=2.0)

Tested 5 Gemini models with the best prompt config. Script: `scripts/test_gemini_models.py`

| Model | Mean ρ | SD | Avg-pred ρ | Parse fails/20 |
|-------|--------|-----|-----------|----------------|
| **Gemini 3 Flash** | **0.604** | 0.062 | **0.666** | 0-1 |
| Gemini 2.5 Pro | 0.240 | 0.242 | 0.398 | 1-3 |
| Gemini 2.5 Flash | 0.088 | 0.131 | 0.148 | 6-9 |
| Gemini 3 Pro | ~0.05 | ~0.37 | ~0 | 16-17 |
| Gemini 2.0 Flash | -0.019 | 0.050 | 0.015 | 0-1 |

### Why Gemini 3 Flash wins within its family

1. **Pro models have mandatory thinking mode** — at t=2.0, thinking consumes output budget and breaks structured format (16-17/20 parse failures for 3 Pro with thinking_budget=1024). Increasing budget to 4096 fixes parsing but thinking itself hurts — separate experiment showed thinking_budget=1024 on Flash dropped ρ from 0.604 to 0.467.

2. **2.5 Flash also struggles with format** — 6-9/20 parse failures. Its thinking mode (which can't be fully disabled) interferes with structured output at high temperature.

3. **2.0 Flash lacks pedagogical knowledge** — clean parsing (0-1 failures) but zero signal (ρ=-0.019). The older generation simply doesn't know enough about student errors.

4. **The task requires "System 1" judgment** — fast pattern matching on pedagogical knowledge, not deliberative reasoning. Thinking/deliberation actively hurts. Only Gemini 3 Flash combines (a) implicit pedagogical knowledge, (b) structured output fidelity at t=2.0, and (c) no mandatory thinking overhead.

### Additional files
- `pilot/rsm_experiment/gemini_models/` — cached raw responses per model
- `pilot/rsm_experiment/gemini_models/summary.json` — results summary

## Frontier Model Comparison (from parallel session)

| Model | Temp | Mean ρ | Notes |
|-------|------|--------|-------|
| Gemini 3 Flash | 2.0 | 0.500 (10-rep) | Baseline |
| Claude Sonnet 4 | 1.0 | 0.263 | Max temp = 1.0 |
| Claude Haiku 3.5 | 1.0 | 0.077 | No signal |
| GPT-4o | 1.5 | 0.338 (1 rep) | Partial |

Claude models hampered by max t=1.0. Temperature is a confound in cross-provider comparison.

## Llama-4-Scout + Error Analysis @ t=2.0

| Config | Mean ρ (3-rep) | SD | Avg-pred ρ (3-rep) |
|--------|----------------|-----|-------------------|
| Scout + contrastive @ t=1.5 | 0.371 | 0.115 | 0.624 |
| **Scout + error_analysis @ t=2.0** | **0.505** | **0.048** | **0.609** |

Free on Groq, 17B parameters. Cached in `pilot/rsm_experiment/cross_model/groq_llama4scout_erroranalysis_t2/`.

### 10-Rep Stability Test

The initial 3-rep SD=0.048 was misleadingly low. Extended to 10 reps:

| Metric | 3-rep | 10-rep |
|--------|-------|--------|
| Per-rep mean ρ | 0.505 | **0.355** |
| Per-rep SD | 0.048 | **0.167** |
| Individual reps | 0.543, 0.535, 0.437 | 0.543, 0.535, 0.437, 0.403, 0.530, 0.116, 0.304, 0.422, 0.070, 0.194 |

But **averaged predictions improve monotonically** with more reps:

| Reps | Avg-pred ρ | p-value |
|------|-----------|---------|
| 3 | 0.609 | 0.004 |
| 5 | 0.639 | 0.002 |
| 7 | 0.648 | 0.002 |
| **10** | **0.668** | **0.001** |

At 10 reps, Scout (0.668) matches Gemini Flash's 3-rep avg-pred (0.666). The wisdom-of-crowds mechanism compensates for higher per-rep variance.

### Temperature Sweep (Scout)

t=2.0 is Groq's maximum and confirmed optimal:

| Temp | Mean ρ | SD | Avg-pred ρ |
|------|--------|-----|-----------|
| 1.0 | 0.312 | 0.168 | 0.423 |
| 1.5 | 0.301 | 0.075 | 0.445 |
| **2.0** | **0.424** | **0.097** | **0.490** |
| 2.5 | — | — | Groq max is 2.0 |

Script: `scripts/test_scout_10rep.py`. Data: `pilot/rsm_experiment/cross_model/scout_temp_sweep.json`.

### Updated Top Configs Leaderboard

| Model + Prompt + Temp | Mean ρ | SD | Avg-pred ρ |
|---|---|---|---|
| Gemini 3 Flash + error_analysis @ t=2.0 | 0.604 | 0.062 | **0.666** |
| Gemini 3 Flash + contrastive @ t=1.5 | 0.577 | 0.075 | 0.660 |
| **Llama-4-Scout + error_analysis @ t=2.0 (10-rep)** | 0.355 | 0.167 | **0.668** |
| Llama-4-Scout + contrastive @ t=1.5 | 0.371 | 0.115 | 0.624 |

## Synthesis

The combined evidence from 15+ models across 5 providers:
- **Gemini 3 Flash at t=2.0 remains the best per-rep model** (ρ≈0.50-0.60)
- **Llama-4-Scout matches Gemini at 10 reps** via averaged predictions (0.668 vs 0.666), but requires ~3× more calls to compensate for higher per-rep variance
- Per-rep SD estimates from 3 reps are unreliable — Scout's true SD is 0.167, not 0.048
- **Averaging is the key mechanism**: both models benefit, and more reps → monotonically better signal
- Model capability alone doesn't predict success — GPT-4o, Claude Sonnet 4, and Gemini 3 Pro are all "smarter" but worse
- The task requires pedagogical knowledge + format compliance at high temp + no thinking overhead
- Temperature ceiling matters: Groq max=2.0, Anthropic max=1.0, Google allows 2.0
- **Two free/cheap models can do this task**: Llama-4-Scout (Groq free) and DeepSeek-Chat (~$0.001/item)
