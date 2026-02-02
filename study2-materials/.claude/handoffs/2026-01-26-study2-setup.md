# Study 2: Misconception Alignment - Session Handoff

## Current State

**Data collection complete.** 1,050 responses collected across 7 models, 3 conditions, 50 items.

### Models Tested
| Tier | Model | Accuracy | GSM8K | Errors | Target Rate |
|------|-------|----------|-------|--------|-------------|
| Frontier | claude-3.5-sonnet | ~100% | 96% | 2 | 100.0% |
| Frontier | gpt-4o | 100% | 92% | 0 | N/A |
| Mid | claude-3-haiku | 86.7% | 89% | 20 | 50.0% |
| Mid | gpt-3.5-turbo | 83.3% | 57% | 25 | **76.0%** |
| Weak | llama-3.1-8b-groq | 87.3% | 57% | 19 | 52.6% |
| Weak | mistral-7b | 55.3% | 45% | 67 | 53.7% |
| Very Weak | llama-3.2-3b-together | 72.7% | 25% | 41 | **65.9%** |

### Conditions
1. `answer_only` - baseline (52.6% target rate)
2. `explain` - chain-of-thought (**78.4%** target rate)
3. `persona` - struggling student role-play (65.2% target rate)

## Key Findings

### Overall
- **59.8% target distractor rate** (95% CI: 52.3%-66.8%)
- χ² = 54.72, p < .0001 (vs 33.3% chance)
- 174 total errors available for analysis
- 96.8% parse success rate

### Capability Paradox Confirmed
- Frontier models (GPT-4o, Claude 3.5 Sonnet) = ~100% accuracy, minimal errors
- **Best alignment: GPT-3.5-turbo** at 76.0% target rate
- **Best volume + alignment: Llama-3.2-3B** at 65.9% with 41 errors
- Weaker models provide more errors for analysis

### Misconception Categories
- **Conceptual: 68.2%** - strongest alignment (area/perimeter, fractions)
- **Procedural: 54.8%** - moderate alignment
- **Interpretive: 50.0%** - limited data (n=4)

### Prompting Effects
- Chain-of-thought (`explain`) significantly outperforms answer_only
- ~26 percentage point improvement with reasoning prompts

## Analysis Outputs

### Figures Generated
All in `pilot/full_collection/figures/`:
- `fig1_model_comparison.png/pdf` - Accuracy and target rates by model
- `fig2_condition_effect.png/pdf` - Prompting condition effects
- `fig3_category_effect.png/pdf` - Misconception category effects
- `fig4_model_condition_heatmap.png/pdf` - Model × condition interaction
- `fig5_capability_paradox.png/pdf` - GSM8K vs alignment scatter
- `fig6_misconception_breakdown.png/pdf` - Individual misconception rates
- `analysis_report.md` - Full statistics report with CIs

## Next Steps

1. [ ] Complete misconception coding on errors (N=174)
2. [x] Added Llama-3.1-8B and Llama-3.2-3B for weak-tier data
3. [x] Run chi-square tests: capability x misconception type
4. [x] Add formal statistical significance tests (χ² = 54.72, p < .0001)
5. [ ] Upload figures to Google Doc for submission

## Files Modified This Session

- `scripts/analyze_results.py` - NEW: Comprehensive analysis with 6 figures
- `pilot/full_collection/batch_responses.jsonl` - All 1,050 responses (7 models)
- `pilot/full_collection/analysis_summary.json` - Summary statistics
- `pilot/full_collection/figures/` - 6 publication-ready figures (PNG + PDF)
- Google Doc updated with new results and exact prompt templates

## API Keys (in `.env`)
All working: OPENAI, ANTHROPIC, MISTRAL, GROQ, TOGETHER, OPENROUTER, REPLICATE

## Paper Status
Google Doc: https://docs.google.com/document/d/1hX1uoC8i2kU9QHCIfIp7nSEv9uQs-cT7WfdQMweaNws/edit

- Results section updated with 7-model data
- Exact prompt templates now documented
- Discussion section updated with new findings
- Abstract deadline: January 26, 2026 (today)
- Full paper deadline: February 2, 2026

## Prompts Used (Exact Templates)

### Answer-Only
```
You are taking a math test with {n_items} questions.
Select the best answer for each question (A, B, C, or D).
Format each answer as:
**Question N:** X
```

### Explain (Chain-of-Thought)
```
You are taking a math test with {n_items} questions.
Answer each question by showing your reasoning, then stating your final answer.
Format each answer as:
**Question N:**
[Your reasoning]
**Answer: X**
```

### Persona (Struggling Student)
```
You are a middle school student taking a math test.
Students at your level sometimes make mistakes - that's okay and normal.
Work through each problem the way a real student would.
Format each answer as:
**Question N:**
[Your thinking/work]
**Answer: X**
```
