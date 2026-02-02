# Handoff: Two-Stage Diversity Experiments — 2026-02-02

## TL;DR
Two-stage diversity injection (generate diverse content at high temp, use as context for prediction) **does not improve difficulty estimation** and actively destroys signal on full item sets. Direct prediction remains best. This is consistent across both Eedi (MCQ) and SmartPaper (open-ended) datasets.

## What Was Tested

**Hypothesis:** Generate diverse student perspectives at high temperature (t=2.0), then feed them as context to a predictor at lower temperature. Should combine the diversity benefit of high temp with the stability of lower temp.

**Conditions tested:**
- **cognitive_chains** — 5 simulated student working-out attempts
- **buggy_analyses** — 5 misconception/error analyses
- **error_perspectives** — 5 "why this is hard" researcher analyses
- **direct_baseline** — standard prediction without two-stage

**Pipelines tested:**
- Gemini→Gemini (t=2.0 seeds, t=1.5 prediction)
- Gemini→Scout (t=2.0 seeds, t=2.0 prediction)
- Scout→Scout (t=2.0 seeds, t=2.0 prediction)

## Results

### Eedi (20 probe items) — Two-stage matches but doesn't beat direct

| Condition (Gemini→Gemini) | Mean ρ | SD |
|--------------------------|--------|-----|
| cognitive_chains | 0.508 | 0.067 |
| **direct_baseline** | **0.502** | 0.125 |
| buggy_analyses | 0.268 | 0.061 |
| error_perspectives | 0.241 | 0.061 |

Cognitive chains matches direct with lower variance. But on only 20 items, this is a wash.

### SmartPaper (134 items) — Two-stage destroys signal

| Method | ρ | p | n |
|--------|---|---|---|
| **Direct prediction (Phase 12)** | **0.547** | <0.0001 | 140 |
| Two-stage cognitive chains | 0.059 | 0.50 | 134 |

By subject: worst damage where direct was strongest (Science: 0.734→0.173, Social Science: 0.702→-0.011).

### Cross-pipeline (Eedi)

Same-model pipelines beat cross-model:
- Gemini→Gemini: 0.508 (cognitive_chains)
- Scout→Scout: 0.401
- Gemini→Scout: 0.356

## Why It Fails

The seeds are high quality (realistic student writing with grade-appropriate errors). But:

1. **Direct prediction is a judgment task** — the model accesses implicit pedagogical knowledge directly
2. **Two-stage is an integration task** — the model must infer population statistics from 5 samples, which is lossy
3. **Context pollutes** — 2000 tokens of student attempts dilute the model's own knowledge
4. **5 samples can't represent a population** — the model anchors on specific seed content rather than generalizing

## Key Scripts
- `scripts/test_two_stage_diversity.py` — Eedi experiment (4 conditions × 3 pipelines × 3 reps)
- `scripts/test_two_stage_smartpaper.py` — SmartPaper experiment (134 items)
- Both support `--scout-seeds` flag for Scout-generated seeds
- CLI: `python3 scripts/test_two_stage_diversity.py [conditions...] [models...] [--scout-seeds]`

## Cached Data
- `pilot/rsm_experiment/two_stage_diversity/` — Eedi seeds and predictions
- `pilot/smartpaper_two_stage/` — SmartPaper seeds and predictions
- All cached — rerunning reads from disk

## Running Experiments
- SmartPaper Gemini cognitive_chains: reps 1-2 + direct_baseline still running (task b7807c5)
- SmartPaper Scout: restarted with new Groq key (task bb08525)

## Groq API Keys (rotated due to 503s)
Current key in `.env`. Backup: [REDACTED]

## Implications for Paper

The two-stage results reinforce the Phase 11/12 story:
1. **LLMs have implicit pedagogical knowledge** best accessed via direct prompting
2. **Any intermediary representation corrupts the signal** — personas, chains, misconceptions, error analyses
3. **The "System 1" hypothesis is confirmed** — fast judgment > deliberative integration
4. **Open-ended items generalize (ρ=0.55), MCQ items don't (ρ≈0)** — this is the paper's main finding
5. **Small probe sets (n=20) can make any method look good** — generalization testing is essential
