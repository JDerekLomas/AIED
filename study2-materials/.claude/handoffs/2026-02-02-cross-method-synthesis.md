# Cross-Method Synthesis: LLM Difficulty Estimation Experiments

## Date: 2026-02-02

## What We Did Today

### Enhanced Classroom Simulation
Built `scripts/run_enhanced_classroom_sim.py` — extends Kröger et al. replication with:
1. **Backstory generation**: Rich student personas per ability level (strengths, weaknesses, error patterns, emotional profiles)
2. **Cognitive simulation**: Persona-based reasoning chains instead of generic "you are a struggling student"
3. **Built-in A/B comparison**: Runs both enhanced and baseline conditions with the same model

Tested with multiple Groq models:
- `llama-3.3-70b-versatile` — best quality but hit daily TPD limits (100k)
- `meta-llama/llama-4-scout-17b-16e-instruct` — too collapsed, unanimous answers
- `qwen/qwen3-32b` — works but `<think>` tags need parsing (handled)
- `openai/gpt-oss-120b` — actually made realistic student errors on manual test
- `llama-3.1-8b-instant` — available as fallback

Script supports `--model`, `--fresh`, `--skip-baseline`, resume from intermediate files.

### Results: Enhanced Classroom Sim (10 items, llama-3.3-70b)
- Enhanced: r=-0.328, RMSE=0.408
- Baseline: r=-0.309, RMSE=0.520
- Enhanced slightly better on RMSE but both negatively correlated with actual difficulty
- Core issue: LLMs converge on correct answers regardless of persona

## Full Cross-Method Comparison

| Method | n (test) | Pearson r | RMSE | Status |
|---|---|---|---|---|
| Feature extraction v1 (50 items) | 10 | **0.770** | 0.204 | Overfitting? |
| Feature extraction v2 (full) | 374 | 0.063 | 0.161 | Not significant |
| Direct difficulty (GPT-4o-mini, basic) | 1869 | 0.007 | 0.213 | Chance |
| Direct difficulty (GPT-4o-mini, expert) | 1869 | 0.010 | 0.246 | Chance |
| Direct difficulty (Llama-70b, expert) | 1869 | -0.032 | 0.239 | Chance |
| Uncertainty difficulty v1 (30) | 30 | -0.261 | 0.798 | Worse than baseline |
| Uncertainty difficulty v2 (full) | 1868 | 0.007 | 0.327 | Worse than baseline |
| Classroom sim v1 (30 items) | 30 | -0.028 | 0.719 | Chance |
| Enhanced classroom sim (10 items) | 10 | -0.328 | 0.408 | Negative |
| Error alignment (50 items) | 50 | -0.019 | — | Chance |

**Benchmark from Kröger et al.: r=0.75-0.82**

## Key Finding

**No method reliably predicts item difficulty at scale.** The only positive result (feature extraction v1, r=0.77) was on 10 test items from a 50-item subset and did not replicate when scaled to 1,869 items (r=0.06).

## Emerging Pattern

LLMs fail at difficulty estimation because:
1. **They don't fail like students** — simulation approaches (classroom, uncertainty) produce too-accurate responses
2. **They can't introspect on difficulty** — direct estimation gives near-zero correlation
3. **Feature extraction overfits on small samples** — the r=0.77 was likely fitting noise
4. **The Eedi items may be too homogeneous** — most are middle-school math with difficulty 0.0-0.5, limited variance to predict

## Possible Directions

1. **Negative result paper**: Document the systematic failure to replicate Kröger et al. across 5 methods, 6+ models, 1,869 items
2. **Distractor-focused approach**: Instead of predicting overall difficulty, predict which distractors attract students (the model CAN identify plausible wrong answers)
3. **Hybrid human-AI**: Use LLM features as inputs to a model trained on a labeled subset
4. **Different item types**: Test on items with more difficulty variance (not just middle-school math)

## Cognitive Chain Experiments (from parallel session)

A parallel session tested whether adding explicit reasoning steps improves the RSM pipeline. **Every attempt hurt performance.**

| Experiment | Best ρ | vs Baseline (0.60) | What it tested |
|---|---|---|---|
| Cognitive modeling | 0.165 | -0.44 | Explicit student reasoning chains |
| Two-stage (analyze → predict) | 0.408 | -0.20 | Generate error analysis, then predict |
| Buggy reasoning with hints | NaN | — | Broken parsing at high temp |
| Thinking mode (budget=1024) | ~0.467 | -0.14 | Gemini Flash with thinking enabled |
| Calibration anchors | 0.445 | -0.16 | Provide IRT anchor values in prompt |
| Option shuffle | TBD | TBD | Shuffle MCQ option order for diversity |

### The "System 1" Finding

The dominant pattern across ALL experiments: **deliberation hurts, intuition helps.**

- Thinking mode on Gemini Flash: ρ drops from 0.604 → 0.467
- Cognitive chains: ρ=0.165 vs contrastive baseline ρ=0.583
- Calibration anchors: providing explicit numerical guidance makes it worse (0.445 vs 0.525 uncalibrated)
- Two-stage reasoning: extra analysis step drops ρ from 0.60 to 0.41

The task requires **fast pattern matching on pedagogical content knowledge** — recognizing which errors students make from training data, not reasoning about why. This explains why:
- Gemini 3 Flash (no mandatory thinking) wins over Gemini 3 Pro (mandatory thinking)
- Higher temperature helps (more diverse pattern samples to average)
- Reasoning models (DeepSeek-Reasoner) fail completely
- "Smarter" models (GPT-4o, Claude Sonnet 4) don't outperform cheaper ones

### Uncalibrated > Calibrated

Providing IRT anchor values in the prompt degraded performance (ρ=0.445 vs 0.525 without). The model's implicit sense of difficulty — baked into weights from training on education content — is better calibrated than explicit numerical anchors. This is evidence that the signal comes from latent pedagogical knowledge, not numerical reasoning.

### Outstanding Questions

1. **Option shuffle results not yet computed** — data exists (6 reps, 3 shuffled) but no final comparison. Could be a free diversity boost.
2. **two_stage_diversity** has raw Gemini responses but no summary — worth computing correlations to confirm the pattern.
3. **Why Gemini 3 Flash specifically?** Hypothesis: training data included disproportionate math education content (teacher forums, misconception databases). This is testable — do items where common misconceptions are well-documented show higher model accuracy?

## Updated Synthesis: What Actually Works

| What | ρ range | Key insight |
|---|---|---|
| Model selection | 0.00–0.60 | Biggest lever. Gemini 3 Flash >> all others |
| Prompt variant | 0.40–0.60 | error_analysis and contrastive work; cognitive chains don't |
| Temperature | 0.30–0.60 | Higher = more diversity = better averaged predictions |
| Averaging reps | +0.05–0.15 | Consistent boost, "wisdom of crowds" |
| Thinking/reasoning | -0.05–-0.15 | **Hurts.** System 1 > System 2 for this task |
| Calibration anchors | -0.08 | Hurts. Implicit > explicit calibration |
| Model size/price | no effect | GPT-4o < Llama-4-Scout (17B, free) |

## Files
- `scripts/run_enhanced_classroom_sim.py` — new enhanced simulation script
- `pilot/replications/enhanced_classroom_sim/` — results, backstories, analysis
- `pilot/rsm_experiment/cognitive_modeling/` — cognitive chain results
- `pilot/rsm_experiment/two_stage_diversity/` — two-stage with diverse seeds
- `pilot/rsm_experiment/calibration_anchors/` — anchor calibration experiment
- `pilot/rsm_experiment/option_shuffle/` — option order shuffling
- `pilot/rsm_experiment/thinking_test/` — thinking budget experiments
- All prior experiments in `pilot/replications/*/analysis.json`

## Running Experiments
- Groq API key: in script (paid tier, 100k TPD on llama-3.3-70b)
- The enhanced sim can be re-run with `--items 50` for a bigger sample
- Existing baseline classroom sim was at ~43% (801/1869) when last checked
