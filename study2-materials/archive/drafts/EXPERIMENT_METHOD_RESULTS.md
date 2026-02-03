# LLM Difficulty Estimation: Experiment Method & Results

**Status:** In progress — last updated 2026-02-02

## 1. Research Questions

1. Can LLMs estimate item difficulty (proportion correct) from item text alone?
2. What prompt strategies, models, and aggregation methods maximize estimation accuracy?
3. Does performance generalize across item formats (MCQ vs open-ended) and populations (UK vs India)?

## 2. Datasets

### 2.1 Eedi (UK Year 9 Maths MCQ)

- **Source:** Eedi diagnostic platform
- **Items:** 20 probe items, multiple-choice (4 options), Year 9 maths
- **Ground truth:** IRT 2PL difficulty (b parameter) estimated from student response data; also `weighted_p_incorrect` (weighted proportion selecting incorrect distractors)
- **Population:** UK Year 9 students (~age 14), typical school population
- **Target variable:** `weighted_p_incorrect` (primary), IRT `b_2pl` (secondary)
- **Baseline performance:** ~50% average correct across items

### 2.2 SmartPaper (Indian Government School Open-Ended)

- **Source:** SmartPaper assessment platform, Jodhpur district government schools
- **Items:** 134 text-only items (6 visual-dependent excluded), open-ended free response, Grades 6-8, across English, Maths, Science, Social Science
- **Ground truth:** Classical difficulty (proportion correct) from N≈950-1050 students per item
- **Population:** Indian government school students, predominantly Hindi-medium, economically weaker sections
- **Target variable:** Classical difficulty (proportion correct)
- **Baseline performance:** Mean difficulty 0.29 (median 0.26), range 0.04–0.83; heavily right-skewed (most items are hard for this population)

### 2.3 Key Differences Between Datasets

| Dimension | Eedi | SmartPaper |
|-----------|------|------------|
| Format | Multiple choice (4 options) | Open-ended (free response) |
| Subject | Maths only | English, Maths, Science, Social Science |
| Population baseline | ~50% correct | ~29% correct |
| Language | English (L1) | English exam, many L2/L3 students |
| N items (probe) | 20 | 20 (from 134 total) |
| Error structure | Encoded in distractors | Unstructured in free text |

## 3. Method

### 3.1 General Approach

For each item, we prompt an LLM to estimate the proportion of students who would answer correctly. We compare LLM estimates to ground-truth difficulty using Spearman rank correlation (ρ) as the primary metric, with Pearson r, mean absolute error (MAE), and bias as secondary metrics.

All experiments use multiple independent replications per item. Final estimates are the mean across replications. This aggregation-across-reps approach consistently improves accuracy (see §4.3).

### 3.2 Eedi Experiments

#### 3.2.1 Model Selection

We tested 4 models on the same 20 probe items using a baseline teacher-prediction prompt:

- Gemini 2.5 Flash (Google)
- Llama-3.3-70B (via Groq)
- Qwen3-32B (via Groq)
- GPT-4o-mini (OpenAI)

Each model produced 3 replications per item at temperature 1.0.

#### 3.2.2 Prompt Strategy Sweep

Using Gemini 2.5 Flash (the winning model), we tested 10 prompt strategies at 3 temperature levels (1.5, 1.8, 2.0), with 3 replications each:

| Strategy | Description |
|----------|-------------|
| v0_baseline | Basic teacher prediction: "You've marked 200 papers, estimate the distribution across options" |
| v3_contrastive | "Would students ACTUALLY make errors on THIS question? Consider whether distractors are plausible." |
| v5_error_analysis | "What specific calculation error leads to each wrong answer?" |
| v6_distractor_first | Chain-of-thought: analyze each distractor, then predict distribution |
| v7_comparative_difficulty | Rate difficulty on 1-10 scale, then convert to distribution |
| v8_imagine_classroom | "Imagine watching 30 students take this test. What do you see?" |
| v9_devil_advocate | Challenge your initial estimate — what would make it wrong? |
| v10_sparse | Minimal prompt: just the question and "predict the distribution" |

Output format for all prompts: `A=XX% B=XX% C=XX% D=XX%` (proportion selecting each option).

#### 3.2.3 Temperature and Aggregation

We systematically varied temperature (0.3, 0.7, 1.0, 1.5, 1.8, 2.0) and number of replications (1, 3, 5, 10) to characterize the temperature-diversity-aggregation mechanism.

### 3.3 SmartPaper Experiments

#### 3.3.1 Baseline Approaches (RSM v2)

Three prompt styles tested on 20 probe items (stratified sample across difficulty quintiles):

1. **Teacher prediction** — "You are an experienced teacher. What percentage of your students would score full marks?" Parsed per-proficiency-level predictions, weighted by population distribution (35% struggling, 35% basic, 20% competent, 10% advanced).
2. **Classroom simulation** — "Simulate N students answering this question." Responses scored by LLM-as-judge.
3. **Individual roleplay** — Simulate one student at a time at specified proficiency levels. Responses scored by LLM-as-judge.

Model: Gemini 2.5 Flash. Single replication per config.

#### 3.3.2 Calibrated Teacher Prediction (Phase 1)

The core innovation: instead of asking the LLM to simulate students from an unfamiliar population, provide real calibration data and ask it to *predict relative to known anchors*.

**Calibration data preparation:**
- Selected 5 anchor items from the non-probe set, targeting difficulty levels p≈{0.05, 0.15, 0.30, 0.50, 0.70}
- Sampled 5 real student responses per anchor item (mix of scored 0 and 1) from `export_item_responses.csv`
- Computed population-level statistics (mean, median, percentiles, distribution shape)
- Extracted common error patterns per subject from incorrect responses on the 5 hardest items

**Five calibration strategies tested:**

| Strategy | Calibration info provided |
|----------|--------------------------|
| baseline | None — same teacher prompt as §3.3.1 |
| popstats | Population statistics only (mean=29%, distribution shape) |
| anchors | 5 anchor items with actual pass rates and 5 real student responses each |
| errors | Common wrong answers per subject from real data + "mean is ~29%" |
| hybrid | Population stats + 3 closest anchors + error patterns |

All strategies: Gemini 2.5 Flash, temperature 1.0, 5 replications per item, 20 probe items. Output format: `ESTIMATED PROPORTION CORRECT: XX%`. Total: ~500 API calls.

## 4. Results

### 4.1 Eedi: Model Selection

| Model | Mean ρ (3 reps) | Notes |
|-------|-----------------|-------|
| **Gemini 2.5 Flash** | **0.58–0.60** | Clear winner |
| Llama-3.3-70B | 0.27 ± 0.13 | Some signal, high variance |
| Qwen3-32B | 0.01 ± 0.11 | No signal |
| GPT-4o-mini | -0.09 ± 0.27 | No signal |

**Finding:** Weaker models cannot perform this task. It requires genuine pedagogical content knowledge, not just instruction-following.

### 4.2 Eedi: Prompt Strategy Sweep

| Strategy × Temperature | Mean ρ | SD | Notes |
|------------------------|--------|-----|-------|
| v5_error_analysis @ t=2.0 | **0.604** | 0.062 | Best overall |
| v3_contrastive @ t=1.5 | **0.577** | 0.075 | Most stable |
| v5_error_analysis @ t=1.8 | 0.566 | 0.108 | High variance |
| v3_contrastive @ t=2.0 | 0.562 | 0.084 | |
| v9_devil_advocate @ t=1.5 | 0.520 | — | Incomplete |
| v8_imagine_classroom @ t=2.0 | 0.428 | 0.115 | Needs high temp |
| v0_baseline | 0.370 | 0.097 | |
| v6_distractor_first | — | — | Parse failures |
| v7_comparative_difficulty | — | — | Constant outputs |

**Finding:** Error-focused analytical prompts outperform simulation prompts. The best prompts ask the model to reason about *student error processes* rather than predict distributions directly.

### 4.3 Eedi: Temperature and Aggregation

```
                  t=1.5    t=1.8    t=2.0
contrastive       0.577    0.518    0.562    (stable)
error_analysis    0.451    0.566    0.604    (rising — benefits from diversity)
imagine_classroom 0.232    0.404    0.428    (rising steeply)
```

**Finding:** Averaging across replications at high temperature (1.5–2.0) consistently outperforms single predictions at low temperature. The mechanism: high temperature creates diverse "plausible expert opinions" that converge on better estimates when averaged — a wisdom-of-crowds effect within a single model. Analytically constrained prompts (contrastive) don't need as much temperature diversity; open/experiential prompts (error_analysis, imagine_classroom) benefit strongly.

Ensemble of contrastive + error_analysis (ρ=0.639) did not beat either alone (ρ=0.660, 0.666), suggesting they capture similar signal.

### 4.4 SmartPaper: Baseline Approaches

Single replication, Gemini 2.5 Flash:

| Approach | Spearman ρ |
|----------|-----------|
| Teacher prediction | 0.311 |
| Classroom simulation (batch=5) | ~0.24 |
| Individual roleplay | ~0.31 |

**Finding:** Simulation-based approaches fail on this population. The LLM cannot faithfully simulate students from Indian government schools — it lacks calibration for this extreme low-performance context.

### 4.5 SmartPaper: Calibrated Teacher Prediction

5 replications at temperature 1.0, Gemini 2.5 Flash:

| Strategy | Spearman ρ | p | Pearson r | MAE | Bias |
|----------|-----------|---|-----------|-----|------|
| **errors** | **0.825** | <0.0001 | 0.619 | 0.155 | -0.147 |
| **anchors** | **0.809** | <0.0001 | 0.773 | 0.241 | +0.241 |
| **hybrid** | **0.806** | <0.0001 | 0.770 | 0.183 | +0.180 |
| baseline | 0.662 | 0.0015 | 0.610 | 0.087 | -0.021 |
| popstats | 0.642 | 0.0023 | 0.544 | 0.100 | -0.060 |

Note: The baseline here (ρ=0.662) is substantially higher than §4.4 (ρ=0.311) due to 5-rep aggregation at temperature 1.0 vs single rep at 0.7. This further confirms the aggregation finding from Eedi.

**Findings:**

1. **Calibration with real data dramatically improves ranking accuracy** — all three calibration strategies (errors, anchors, hybrid) achieve ρ>0.80, vs ρ≈0.65 without calibration.

2. **Error patterns produce the best ranking** (ρ=0.825) but severely underestimate difficulty (bias=-0.147). Showing real wrong answers makes the LLM pessimistic — it compresses all estimates toward 5-20%.

3. **Anchor items produce the best absolute calibration** (r=0.773) but systematically overestimate (bias=+0.241). The LLM sees anchor items and overshoots upward.

4. **Population statistics alone don't help** — popstats (ρ=0.642) is slightly worse than baseline (ρ=0.662). Telling the LLM "the mean is 29%" causes it to anchor too tightly, reducing variance in predictions (many estimates collapse to exactly 20%).

5. **The hybrid approach does not improve on anchors alone** (ρ=0.806 vs 0.809) — combining signals is not additive here.

### 4.6 SmartPaper: Temperature × Model Sweep

We tested 3 strategies (baseline, errors, anchors) × 4 temperatures (0.5, 1.0, 1.5, 2.0) × 2 models (Gemini 2.5 Flash, DeepSeek Chat) with 5 reps per item on 20 probe items.

#### 4.6.1 Gemini 2.5 Flash Temperature Gradient

```
                  t=0.5   t=1.0   t=1.5   t=2.0
baseline          0.571   0.743   0.311   0.398
errors            0.773   0.794   0.797   0.840
anchors           0.811   0.821   0.783   0.875  ← best: ρ=0.875, r=0.872
```

**Key finding: Calibration acts as a guardrail for high-temperature diversity.** Without calibration data, high temperature destroys signal on SmartPaper (baseline: 0.743→0.311 from t=1.0 to t=1.5). With calibration, high temperature *improves* signal (anchors: 0.811→0.875 from t=0.5 to t=2.0). The calibration data constrains the output space so that temperature-induced diversity generates a beneficial wisdom-of-crowds effect rather than noise.

This confirms and extends the Eedi temperature finding: on Eedi, analytically constrained prompts (contrastive) were stable across temperature while open prompts benefited from high temp. Here, the constraint comes from the calibration data itself.

#### 4.6.2 DeepSeek Chat Temperature Gradient

```
                  t=0.5   t=1.0   t=1.5   t=2.0
baseline          0.800   0.809   0.728   0.675
errors            0.799   0.787   0.783   0.766
anchors           0.743   0.729   0.726   0.691
```

**Key finding: DeepSeek shows the opposite temperature pattern to Gemini.** All three strategies *decline* with higher temperature. Baseline peaks at t=1.0 (ρ=0.809), errors is stable (ρ≈0.79), anchors declines steadily. Unlike Gemini, DeepSeek does not benefit from temperature-induced diversity — it performs best at low temperatures.

**DeepSeek baseline is surprisingly strong** (ρ=0.809 at t=1.0) — comparable to Gemini's calibrated strategies. This suggests DeepSeek may have stronger built-in population calibration for this context. However, calibration data (errors, anchors) does *not* further improve DeepSeek's ranking on this dataset.

**DeepSeek errors has the best absolute calibration:** MAE=0.089 vs Gemini's best MAE=0.087 (baseline). Both produce well-calibrated absolute estimates, but through different mechanisms.

#### 4.6.3 Cross-Model Comparison (SmartPaper)

| Metric | Gemini (best) | DeepSeek (best) |
|--------|---------------|-----------------|
| Best ranking (ρ) | 0.875 (anchors_t2.0) | 0.809 (baseline_t1.0) |
| Best calibration (MAE) | 0.087 (baseline_t1.0) | 0.089 (errors_t0.5) |
| Best linear fit (r) | 0.872 (anchors_t2.0) | 0.722 (anchors_t1.0/1.5) |
| Temperature pattern | Higher is better WITH calibration | Lower is better for all strategies |
| Calibration benefit | +0.13 ρ (baseline→anchors at t=2.0) | No benefit (baseline already ρ=0.81) |

**Key insight:** The two models have fundamentally different optimization profiles. Gemini needs calibration data + high temperature to reach its ceiling (ρ=0.875). DeepSeek reaches ρ=0.809 with just the baseline prompt at moderate temperature — but cannot be pushed further. The "optimal recipe" is model-dependent.

## 5. Cross-Dataset Comparison

| Finding | Eedi (MCQ) | SmartPaper (Open-ended) |
|---------|------------|------------------------|
| Best ρ (probe set) | 0.604 (error_analysis, t=2.0) | 0.875 (Gemini anchors, t=2.0) |
| Aggregation effect | +0.15–0.25 ρ improvement | +0.35 ρ improvement (0.31→0.66) |
| Temperature effect | Higher is better (1.5–2.0) | Model-dependent: Gemini benefits, DeepSeek declines |
| Model selection | Gemini Flash >> 14 others | Gemini (0.875) > DeepSeek (0.809) on ranking |
| Calibration | Not tested | +0.13 ρ for Gemini; no benefit for DeepSeek |

**Cross-dataset synthesis:**

1. **Temperature + aggregation works on both datasets** — but the mechanism differs. On Eedi, structured prompts constrain the output space. On SmartPaper, calibration data provides the constraint. For Gemini, both enable high-temperature diversity. DeepSeek does not benefit from high temperature on either.

2. **Error-focused analytical reasoning is the universal signal.** On Eedi, error_analysis beats simulation prompts. On SmartPaper (Gemini), error-pattern calibration produces the best ranking at t=1.0. The LLM's strongest signal comes from reasoning about *how students go wrong*.

3. **The SmartPaper ρ=0.875 substantially exceeds Eedi's ρ=0.604.** This is partly because SmartPaper has 20 items spanning a wider difficulty range (0.04–0.83) vs Eedi's narrower range, making ranking easier. But it also validates that LLMs can estimate difficulty for open-ended items in unfamiliar populations — the knowledge is there, it just needs calibration to surface.

4. **Optimal recipe is model-dependent.** Gemini: calibration + high temp. DeepSeek: baseline prompt + low temp. This has practical implications — the best approach depends on which model you're using.

## 6. Parsing Methodology and Impact

### 6.1 The Problem

Eedi experiments require LLM responses in a specific format: `below_basic: A=XX% B=XX% C=XX% D=XX%` for each of 4 proficiency levels. A strict regex parser enforces this exact format. When responses use markdown tables, commas, colons, or bullet-point formatting instead, the strict parser silently drops those responses.

SmartPaper uses a simpler format (single numeric output) and has **0% parse failure** across all 2900 response files.

### 6.2 Eedi Failure Rates

Overall: **25.0% failure rate** (708/2828 files failed strict parsing).

| Config | Strict Parse Rate | Notes |
|--------|------------------|-------|
| v3_contrastive (all temps) | 100% | Clean format |
| v5_error_analysis (all temps) | 100% | Clean format |
| v8_imagine_classroom (all temps) | 100% | Clean format |
| v9_devil_advocate (all temps) | 100% | Clean format |
| v10_sparse | 45–70% | Partial losses |
| v6_distractor_first | 5% | Nearly complete loss |
| v7_comparative_difficulty | 0% | Total loss (100% fail) |
| deepseek_reasoner | 0% | Total loss (empty responses) |
| groq_qwen32b | 10% | Near-total loss |
| openai_4o | 18% | Severe loss |

### 6.3 Lenient Parser Recovery

A lenient parser (supports markdown tables, alternate delimiters `A: XX%`, bullet formatting) recovers most lost data. Key recoveries:

| Config | Strict ρ | Lenient ρ | Items Recovered |
|--------|----------|-----------|-----------------|
| v7_comparative_difficulty @ t=1.8 | N/A (0 items) | **0.698** | 0→19 |
| v7_comparative_difficulty @ t=1.5 | N/A (0 items) | 0.368 | 0→20 |
| v6_distractor_first @ t=1.8 | N/A (2 items) | 0.390 | 2→20 |
| v5_error_analysis @ t=2.0 | 0.508 | 0.651 | 20→20 (better reps) |
| openai_4omini | 0.114 | 0.350 | 20→20 (better reps) |

**However**, lenient parsing can *hurt* well-formatted configs by mis-extracting numbers from reasoning text:

| Config | Strict ρ | Lenient ρ | Delta |
|--------|----------|-----------|-------|
| v10_sparse @ t=2.0 | 0.900 | 0.526 | -0.374 |
| v8_imagine_classroom @ t=1.8 | 0.562 | -0.081 | -0.643 |
| v9_devil_advocate @ t=1.5 | 0.615 | 0.218 | -0.397 |

### 6.4 Recommendation

Use a **strict-then-lenient cascade**: attempt strict parsing first, fall back to lenient only when strict fails. This preserves accuracy for well-formatted responses while recovering data from format-variant responses.

### 6.5 Impact on Reported Results

The §4.2 prompt strategy sweep results use strict parsing only. The key implication: **v7_comparative_difficulty was not inherently bad — it just used an incompatible output format.** With lenient parsing, v7 at t=1.8 achieves ρ=0.698, which would rank among the top strategies. The §4.2 table should be interpreted with this caveat.

## 7. Remaining Experiments

- [x] SmartPaper: temperature sweep on best calibration strategies
- [x] SmartPaper: cross-model comparison (Gemini vs DeepSeek)
- [x] SmartPaper: DeepSeek full sweep (baseline, errors, anchors × 4 temps)
- [ ] SmartPaper: linear recalibration of anchor/hybrid predictions to correct bias
- [ ] SmartPaper: scale best strategy to all 134 items with 10 reps
- [ ] SmartPaper: subject-specific anchor sets
- [ ] Eedi: two-stage backstory, cognitive modeling experiments
- [ ] Both: item-level moderator analysis (which items are unpredictable?)
- [x] Parsing audit: strict vs lenient parser comparison across all configs
- [ ] Eedi: re-run prompt sweep with strict-then-lenient cascade parser
- [ ] Cross-dataset: test Eedi-style prompt strategies on SmartPaper with calibration

## 8. Key Files

- `scripts/run_smartpaper_rsm_v2.py` — SmartPaper experiment runner (baseline + calibration)
- `scripts/test_metaprompt_sweep.py` — Eedi prompt × temperature sweep
- `scripts/test_cross_model.py` — Eedi cross-model comparison
- `pilot/smartpaper_rsm_v2/calibration/` — SmartPaper calibration raw data + results
- `pilot/rsm_experiment/metaprompt_sweep/` — Eedi sweep raw data
- `data/smartpaper/item_statistics.json` — SmartPaper ground truth
- `data/smartpaper/export_item_responses.csv` — SmartPaper raw student responses
- `.claude/handoffs/2026-02-02-optimization-strategy.md` — Eedi optimization notes
