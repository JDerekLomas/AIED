# Handoff: Indian State Assessment Experiments — 2026-02-02

## What was done

### Data Preparation (5 tasks, all complete)
1. **Probe set**: 20 stratified items in `data/indian_state_assessment/probe_set_20.json`
2. **Ground truth verification**: Train-test ρ ≥ 0.9955 across all 12 subject-grade groups
3. **Text quality audit**: ~53% self-contained, ~33% partial, ~15% label-only
4. **MCQ prompt templates**: `scripts/prompts_mcq.py` — direct, contrastive, error_analysis
5. **DBE-KT22 item_statistics.json**: 168 items in consistent format at `data/dbe-kt22/item_statistics.json`

### Experiments (complete)
- **Gemini 3 Flash** (direct, t=2.0, 3 reps): ρ = **0.410** (p < 0.001)
- **Llama-4 Scout** (direct, t=2.0, 3 reps): ρ = **0.319** (p < 0.001)

### Documentation
- `data/smartpaper/DATA_DICTIONARY.md` — full SmartPaper data dictionary (was missing)
- `pilot/indian_state_assessment/RESULTS.md` — complete results with breakdowns
- `pilot/indian_state_assessment/results_overview.html` — shareable HTML for test creators

### Key findings
- **By subject**: English best (ρ=0.480), Hindi weak (0.264), Maths surprisingly flat (0.125)
- **By grade**: G7 best (0.572), G6 (0.532), G3 (0.527), **G4 dead zone (0.045)**
- **Grade 4 investigation**: LLM predictions collapse to narrow 0.55-0.80 band. Not a text quality issue. Likely developmental/population factors driving difficulty invisible to LLM.

## Cross-dataset picture

| Dataset | Domain | Items | Best ρ | Prompt |
|---------|--------|-------|--------|--------|
| Eedi | Math MCQ | 105 | 0.114 (ns) | Direct |
| Indian State | Math/Eng/Hindi MCQ | 210 | 0.410 | Direct 3-rep |
| DBE-KT22 | Database MCQ | 168 | 0.440 | Contrastive 3-rep |
| SmartPaper | Math/Eng/Sci/SS open | 140 | 0.547 | Direct 3-rep |

## Background processes
- `run_dbe_kt22_hypotheses.py` (PID 6182) was running — check if complete
- All Indian state experiments finished

## What's next
- [ ] By-subject breakdown for self-contained items only (subset analysis)
- [ ] Contrastive prompt on Indian state (template ready in `prompts_mcq.py`)
- [ ] Error analysis prompt comparison
- [ ] Investigate why Maths is weak (expected strongest given text quality)
- [ ] SmartPaper: compute IRT parameters if needed for paper
- [ ] Paper integration: update `paper_sections_draft.md` with new results

## Key files
- `pilot/indian_state_assessment/gemini_flash/summary.json`
- `pilot/indian_state_assessment/llama4_scout/summary.json`
- `pilot/indian_state_assessment/RESULTS.md`
- `pilot/indian_state_assessment/results_overview.html`
- `data/indian_state_assessment/item_statistics.json` (210 items)
- `data/smartpaper/DATA_DICTIONARY.md`
- `scripts/run_indian_state_assessment.py`
- `scripts/prompts_mcq.py`
