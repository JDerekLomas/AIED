# Benchmark Methodology & Model Specifications

## GSM8K Standard Evaluation Protocol

Standard GSM8K benchmark evaluation uses **single-item prompts** (one problem per API call):

- **Few-shot setting**: 0-8 exemplars (commonly 3-8)
- **Chain-of-Thought (CoT)**: Enabled by default
- **Self-consistency**: Multiple independent API calls, majority vote
- **No batching**: Each problem evaluated independently

**Sources:**
- [DeepEval GSM8K Documentation](https://deepeval.com/docs/benchmarks-gsm8k)
- [Klu GSM8K Benchmark](https://klu.ai/glossary/GSM8K-eval)
- [GSM8K Platinum Analysis](https://gradientscience.org/gsm8k-platinum/)

**Note:** Batching (presenting multiple items in a single prompt) is not standard practice. Our batched condition would be a novel extension to test sequencing effects.

---

## Model GSM8K Benchmarks

| Model | Tier | GSM8K | Evaluation | Source |
|-------|------|-------|------------|--------|
| Claude 3.5 Sonnet | Frontier | 96.4% | Standard | [Klu Leaderboard](https://klu.ai/llm-leaderboard) |
| GPT-4o | Frontier | 89.8% | Standard | [Klu Leaderboard](https://klu.ai/llm-leaderboard) |
| Claude 3 Haiku | Mid | 88.9% | Standard | [Klu Leaderboard](https://klu.ai/llm-leaderboard) |
| GPT-4o-mini | Mid | ~85% | Estimated | OpenAI |
| Gemini 1.5 Flash | Mid | 68.8% | Standard | [Klu Leaderboard](https://klu.ai/llm-leaderboard) |
| GPT-3.5-turbo | Weak | 57.1% | 5-shot | [HuggingFace Discussion](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/30) |
| Mistral-7B | Weak | 45.2% | 8-shot | [Mistral AI](https://mistral.ai/news/announcing-mistral-7b/) |

**Capability spread:** 51.2 percentage points (96.4% → 45.2%)

---

## Context Window Sizes

| Model | Context | Notes |
|-------|---------|-------|
| GPT-4o | 128K tokens | |
| GPT-4o-mini | 128K tokens | |
| GPT-3.5-turbo | 16K tokens | Limiting for batch mode |
| Claude 3.5 Sonnet | 200K tokens | |
| Claude 3 Haiku | 200K tokens | |
| Mistral-7B | 32K tokens | |
| Gemini 1.5 Flash | 1M tokens | Largest context |

**Batch mode estimate:** 50-item test ≈ 15-25K tokens (questions + full reasoning output)

---

## Our Evaluation Design

### Standard Mode (Current)
- 1 item per API call
- Matches standard benchmark methodology
- Clean isolation of item-level effects

### Batch Mode (Planned Extension)
- N items per call (e.g., 10 or 50)
- Tracks position/sequence in batch
- Tests for:
  - Fatigue effects (performance degradation)
  - Learning effects (improvement over sequence)
  - Context interference
- Novel contribution beyond standard benchmarks

### Variables to Document
- Temperature: 0.7 (allows variation)
- Max tokens: 500 (sufficient for CoT)
- System prompt: None (model default)
- N responses per item: 3-5 (for variance estimation)

---

*Created: January 26, 2026*
