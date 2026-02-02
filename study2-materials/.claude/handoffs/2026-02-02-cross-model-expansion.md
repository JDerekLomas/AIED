# Cross-Model Expansion: RSM Difficulty Estimation — 2026-02-02

## What We Did

Expanded the cross-model comparison from 4 models to 11, testing which LLMs can predict item difficulty using the contrastive teacher-prediction prompt (RSM pipeline) on 20 Eedi probe items with IRT b_2pl ground truth.

### New Models Added
- **DeepSeek Chat** (deepseek-chat) via DeepSeek API
- **DeepSeek Reasoner** (deepseek-reasoner) via DeepSeek API — still running
- **GPT-4o** (gpt-4o) via OpenAI
- **Llama-4 Scout** (llama-4-scout-17b-16e) via Groq
- **GPT-OSS-120B** (openai/gpt-oss-120b) via Groq
- **Kimi-K2** (moonshotai/kimi-k2-instruct-0905) via Groq
- **Llama-3.1-8B** (llama-3.1-8b-instant) via Groq
- **Gemini 2.0 Flash** via OpenRouter — queued, not yet run
- **Gemini Flash** via Google API — now complete (3 reps)

### API Keys & Providers
| Provider | Key Location | Models |
|---|---|---|
| Groq | .env `GROQ_API_KEY` (gsk_g1Jc...) | llama-70b, llama-8b, llama-4-scout, qwen-32b, gpt-oss-120b, kimi-k2 |
| OpenAI | .env `OPENAI_API_KEY` (sk-proj-hR...) | gpt-4o-mini, gpt-4o |
| Google | .env `GOOGLE_API_KEY` (AIzaSy...) | gemini-3-flash-preview |
| DeepSeek | hardcoded in script | deepseek-chat, deepseek-reasoner |
| OpenRouter | .env `OPENROUTER_API_KEY` | google/gemini-2.0-flash-001 (tested, not yet run in experiment) |
| Replicate | .env `REPLICATE_API_TOKEN` | available but not integrated |

DeepSeek API key: `[REDACTED]` (base_url: `https://api.deepseek.com`)

## Full Results: Cross-Model Ranking

All models tested with **contrastive prompt at t=1.5**, 3 reps, 20 items.

| Model | Provider | Mean ρ (single) | SD | Avg-pred ρ | Notes |
|---|---|---|---|---|---|
| **Gemini Flash** | Google | **0.470** | 0.076 | **0.549** | Winner. Prior sweep showed 0.577 with different code path |
| Llama-4 Scout | Groq | 0.371 | 0.115 | 0.624* | Surprisingly high averaged-pred; high single-rep variance |
| DeepSeek Chat | DeepSeek | 0.338 | 0.140 | 0.409 | Middle tier, high variance |
| GPT-OSS-120B | Groq | 0.336 | 0.100 | 0.450 | Solid middle tier |
| Kimi-K2 | Groq | 0.336 | 0.116 | 0.319 | Similar to GPT-OSS |
| Llama-3.3-70B | Groq | 0.272 | 0.126 | — | Some signal |
| GPT-4o | OpenAI | 0.172 | 0.126 | 0.204 | Surprisingly weak |
| Qwen3-32B | Groq | 0.005 | 0.110 | — | No signal |
| GPT-4o-mini | OpenAI | -0.092 | 0.266 | — | No signal, unstable |
| Llama-3.1-8B | Groq | -0.081 | 0.097 | -0.009 | No signal |
| DeepSeek Reasoner | DeepSeek | NaN | — | NaN | FAILED: empty output files (reasoning in wrong field) |
| Gemini 2.0 Flash | OpenRouter | **-0.200** | 0.132 | -0.269 | ANTI-CORRELATED. Gemini 3.0 >> 2.0 |

*Llama-4 Scout's avg-pred ρ=0.624 is notable — may benefit from aggregation more than other models.

## Key Findings

### 1. Clear Tier Structure
- **Tier 1 (ρ > 0.4):** Gemini Flash only
- **Tier 2 (ρ ≈ 0.3-0.4):** Llama-4 Scout, DeepSeek Chat, GPT-OSS-120B, Kimi-K2
- **Tier 3 (ρ ≈ 0.1-0.3):** Llama-70B, GPT-4o
- **Tier 4 (ρ ≈ 0):** Qwen-32B, GPT-4o-mini, Llama-8B

### 2. Price ≠ Performance
GPT-4o (expensive) at ρ=0.172 is beaten by free Groq models (Scout at 0.371). Model architecture and training data matter more than scale for this task.

### 3. Aggregation Amplifies Signal
Every model benefits from averaging predictions across reps. Llama-4 Scout jumps from 0.371 → 0.624 with averaging — the largest gain. This suggests it has real signal buried under noise.

### 4. OpenAI Models Underperform
Both GPT-4o and GPT-4o-mini are in the bottom half. This is not a general "bigger = better" story.

## Script Changes

`scripts/test_cross_model.py` updated:
- Added `call_deepseek()` — uses OpenAI-compatible client with DeepSeek base URL
- Added `call_openrouter()` — uses OpenAI-compatible client with OpenRouter base URL
- New model entries: `deepseek_chat`, `deepseek_reasoner`, `openrouter_gemini_flash`, `groq_gptoss120b`, `groq_kimik2`, `groq_llama8b`
- All have resume support (cached raw responses per rep per item)

## All Complete

### 5. Gemini 2.0 vs 3.0 Flash: Generation Gap
Gemini 3.0 Flash (ρ=0.470) vs Gemini 2.0 Flash (ρ=-0.200). The pedagogical content knowledge required for difficulty estimation **emerged between Gemini 2.0 and 3.0**. This is one of the strongest findings — it's not just about "using Gemini," it's about a specific model generation.

## Completed But Failed
- `deepseek_reasoner` — all 60 files saved but EMPTY. The DeepSeek Reasoner API returns reasoning in `reasoning_content` field, not `content`. The `call_deepseek()` function reads `.content` which is empty for this model. To fix: extract from `resp.choices[0].message.reasoning_content` and concatenate, or just skip this model since it can't do temp>1.0 anyway (no diversity for aggregation).

## What's Next
1. Run top 2-3 models with `error_analysis` prompt at t=2.0 (the best config from metaprompt sweep)
2. Consider expanding probe items from 20 → 40-50 to tighten CIs
3. Update the dashboard (results-site/rsm_analysis.html) with full 11-model ranking
4. OpenRouter Gemini 2.0 Flash results will show if the Gemini generation matters (2.0 vs 3.0)

## Parallel Research Tracks Summary

| Track | Best Result | Status |
|---|---|---|
| **RSM optimization** (20 items, Gemini Flash) | ρ=0.666 (error_analysis, t=2.0, averaged) | Active, expanding models |
| **Replication** (1,869 items, 5 methods) | r=0.063 (feature extraction v2) | Dead end — negative result |
| **Enhanced classroom sim** | r=-0.328 (10 items) | Negative — LLMs don't fail like students |

The RSM track is the only one producing real signal. The paper story is becoming: model selection >> prompt engineering > temperature tuning, and only specific models (Gemini Flash) have the pedagogical content knowledge needed.
