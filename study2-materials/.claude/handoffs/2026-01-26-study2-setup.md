# Study 2: Misconception Alignment - Session Handoff

## Current State

API keys configured and tested. All models working. Ready for full data collection.

### API Keys (in `.env`)
- OPENAI_API_KEY - GPT-3.5, GPT-4o-mini
- ANTHROPIC_API_KEY - Claude 3 Haiku, Claude 3.5 Haiku
- MISTRAL_API_KEY - Mistral 7B
- GROQ_API_KEY - Llama 3.1 8B (fast)
- TOGETHER_API_KEY - Llama 3.2 3B, Llama 3.1 8B
- OPENROUTER_API_KEY - Multi-model fallback
- REPLICATE_API_TOKEN - Llama 2 7B (slow)

### Working Models by Capability Tier
| Tier | Model | GSM8K | Provider |
|------|-------|-------|----------|
| Very Weak | llama-3.2-3b-together | ~25% | Together |
| Weak | mistral-7b | ~45% | Mistral |
| Mid-Weak | gpt-3.5-turbo | ~57% | OpenAI |
| Mid | claude-3-haiku | ~88% | Anthropic |

### Conditions
1. `explain` - baseline problem-solving
2. `persona` - role-play as struggling student
3. `diagnose_simulate` - identify misconception then exhibit it

## Next Steps

1. Run full data collection across all models and conditions:
   ```bash
   python3 scripts/collect_batch.py --items data/probe_items.json \
     --output pilot/full_collection \
     --models llama-3.2-3b-together mistral-7b gpt-3.5-turbo claude-3-haiku \
     --conditions explain persona diagnose_simulate \
     --batch-size 10 --n-batches 5
   ```

2. Analyze target distractor rates by model/condition

3. Complete paper draft (currently ~1,200 words, target 8-10 pages)

## Key Decisions Made

- Focus on RQs 1-3 (validation, prompting, capability) not RQ4 (transfer)
- Target distractor rate as primary DV (chance = 33%)
- "Sweet spot" hypothesis: mid-capability models (~57% GSM8K) show best alignment
- diagnose_simulate prompting shows 60-75% hit rate in preliminary tests

## Files Modified This Session

- `.env` - All API keys
- `.gitignore` - Excludes .env
- `scripts/collect_batch.py` - Added dotenv, Groq, Together, OpenRouter support
- `paper/draft.md` - Complete paper draft
- `paper/sections/*.md` - Individual sections

## Key Findings (Pilot Data)

- Overall target rate: 50.5% vs 33% baseline (p<.001)
- GPT-3.5 sweet spot: 72.7% target rate
- diagnose_simulate: 60-75% hit rate, best for procedural misconceptions
