# Draft: Full Paper

## Abstract

Can LLMs estimate how difficult a test item is for students? Published correlations range from r≈0 to r=0.83, but each study tests one method on one dataset. We map the full parameter landscape using sequential design of experiments across two datasets — 1,869 misconception-targeted maths MCQs (Eedi, UK) and 140 open-ended items across four subjects (SmartPaper, India) — testing 15+ models, 20+ prompt configurations, and multiple structured elicitation strategies grounded in four learning science traditions.

The central finding is a dissociation by item type. On open-ended items, even simple direct estimation achieves ρ=0.55 (n=140, p<0.0001), generalising to 120 held-out items (ρ=0.52). On misconception-targeted MCQs, the best optimised configuration achieves ρ=0.50 on 20 probe items but collapses to ρ=0.11 (n=105, non-significant, 95% CI [−0.07, 0.30]) on confirmation items. No combination of model, prompt, temperature, or elicitation strategy overcomes this boundary.

We report a registry of 18 tested hypotheses, most producing null or negative results: cognitive scaffolding hurts (misconception hints: ρ=0.19 vs. 0.50 baseline), deliberation hurts (reasoning models: ρ=0.05 vs. 0.50), two-stage diversity injection destroys signal at scale (ρ=0.06 vs. 0.55), and model size does not predict success (GPT-4o ρ=0.17 vs. Llama-4-Scout 17B ρ=0.36). These negative results, together with a framework distinguishing complexity-driven from selectivity-driven difficulty, provide actionable guidance for researchers and practitioners.

**Keywords:** item difficulty estimation, LLM evaluation, design of experiments, mathematics education, psychometrics


## 1. Introduction

Can LLMs estimate how difficult a test item is for students? The answer in the literature ranges from "not at all" to "very well." Direct estimation studies report correlations from r≈0 (Kröger et al., 2025) to r=0.83 (Razavi & Powers, 2025). Classroom simulation approaches range from r=0.01 (our replication) to r=0.82 (Kröger et al., 2025). Feature extraction ranges from r=0.06 (our replication at scale) to r=0.87 (Razavi & Powers, 2025). These are not minor discrepancies — they span the entire range from chance to near-perfect prediction.

Why the disagreement? Each study tests one method on one dataset and reports a single correlation. But difficulty estimation involves a large parameter space: the choice of model, prompt strategy, sampling temperature, aggregation method, and — critically — the type of items being estimated. A single-configuration study cannot distinguish between "this method works" and "this method works on these items, with this model, at this temperature." The result is a literature of point estimates that cannot be compared or reconciled.

We take a different approach. Rather than testing one pipeline and reporting a correlation, we conduct a systematic search across the parameter landscape using sequential design of experiments (DOE) — beginning with a Box-Behnken response surface design (Box & Behnken, 1960) for screening, then progressing through optimisation, boundary testing, and confirmation phases. DOE was developed precisely for situations like this: the outcome depends on multiple continuous and categorical factors whose interactions are unknown, individual experiments are cheap, and the goal is to map the response surface rather than test a single hypothesis.

We apply this methodology to two datasets: 1,869 misconception-targeted maths MCQs from Eedi (UK) and 140 open-ended items across four subjects from SmartPaper (India). Across these two datasets, we test 15+ models, 20+ prompt configurations, 6 temperature settings, and multiple structured elicitation strategies grounded in four learning science traditions. The result is not a single correlation but a map — one that reveals a dissociation by item type that reconciles the conflicting findings in the literature, and a set of boundary conditions that narrow the space of viable methods.

The prompt strategies we test are grounded in four theoretical traditions, each offering a different account of why students make errors:

- **Knowledge Component theory** (Koedinger, Corbett & Perfetti, 2012): Difficulty reflects the knowledge components an item requires. Motivates teacher-prediction framing.
- **Buggy Procedures** (Brown & Burton, 1978): Students apply internally-consistent but flawed algorithms. Motivates production-rule prompts that specify step-by-step erroneous procedures.
- **Conceptual Change theory** (Chi, Slotta & de Leeuw, 1994): Students hold coherent-but-wrong mental models. Motivates prompts describing flawed beliefs and why they feel right.
- **Epistemic State Specification** (Tack & Piech, 2024): A taxonomy of student simulation prompts from unspecified (E0) to data-calibrated (E4). Motivates increasingly detailed cognitive specifications.

All four predict that more scaffolding should help. Our experiments test this prediction — and find it wrong.

### Methodological Note

This research was conducted through iterative optimisation with Claude Code (Anthropic), an AI-assisted programming environment. The experimental sequence was not pre-registered; each experiment informed the next in a sequential DOE pipeline. We address the resulting multiple-comparisons concern in three ways: (1) 10-rep stability testing showed our initial 3-rep estimates were inflated by ~0.08–0.10, and we report the corrected numbers; (2) the definitive test is a held-out 105-item confirmation on Eedi, which returned ρ=0.11 (null); (3) all prompts, scripts, raw model outputs, and analysis code are publicly available. We report both successes and failures transparently.


## 2. Method

### 2.1 Design Framework: Sequential Experimentation

We adopt a sequential design-of-experiments (DOE) pipeline, standard in industrial process optimisation but novel in LLM evaluation. The pipeline has five phases, each answering a question that determines the next:

**Phase 1 — Screening (Box-Behnken RSM).** A response surface design (Box & Behnken, 1960) varying five factors across 46 configurations: prompt style (individual roleplay / classroom batch / teacher prediction), temperature (0.3 / 0.9 / 1.5), batch size (1 / 5 / 30 simulated students per call), misconception hint level (hidden / partial / full), and model (Gemini 2.5 Flash / Gemini 3 Flash). Each configuration was evaluated on 20 probe items with 20 simulated responses per item (18,400 API calls in this phase). The objective function was Spearman ρ between simulated difficulty and IRT b₂PL. This phase identified one viable region: teacher-prediction framing at high temperature.

**Phase 2 — Optimisation (prompt sweep).** Within the viable region, we tested 7 prompt variants × 3 temperatures (1.5, 1.8, 2.0) × 3 replications = 63 configurations. Each configuration predicted difficulty for the same 20 probe items. Prompts ranged from sparse ("estimate difficulty") to structured ("identify specific calculation errors students would make"). This phase identified two top configurations: error analysis (ρ=0.50 per-rep, 10-rep validated) and contrastive prompting (ρ=0.51 per-rep).

**Phase 3 — Stability and boundary testing.** Three parallel experiments:
- *10-rep stability*: Top 2 configs run for 10 replications each, revealing that 3-rep estimates were optimistically biased by ~0.08–0.10.
- *Cross-model survey*: 15 models across 5 providers (Google, OpenAI, Anthropic, Groq, DeepSeek), revealing a three-tier structure with no correlation between model size/benchmark ranking and task performance.
- *Structured elicitation*: Cognitive modeling, buggy reasoning, student personas, two-stage diversity injection, and calibration anchors — testing predictions from four theoretical traditions. All matched or underperformed direct prediction.

**Phase 4 — Generalisation.** The critical test: does the signal hold beyond the 20-item probe set?
- *Eedi expansion*: 20 → 50 items (adding 30 random), then 105 items with IRT b₂PL parameters. Signal collapsed: ρ=0.04 (50 items), ρ=0.11 (105 items, non-significant).
- *SmartPaper expansion*: 20 → 140 items. Signal held: ρ=0.55 (140 items, p<0.0001), ρ=0.52 on 120 held-out items.

**Phase 5 — Mechanistic investigation.** Controlled temperature sweeps (6 temperatures × 3 reps, prompt held constant) to disentangle the prompt × temperature confound from Phase 1. Two-stage experiments on SmartPaper's full 134 items to test whether context-based elicitation generalises. Temperature sweeps on SmartPaper with 2 models × 3 strategies × 4 temperatures to characterise the three-way interaction.

The total experiment involved approximately 25,000 API calls across all phases, at a cost of approximately $50.

### 2.2 Datasets

**Eedi (UK, MCQ).** 1,869 diagnostic maths questions targeting specific misconceptions, each with 4 options where every distractor corresponds to a known error pattern (e.g., adding numerators and denominators separately when adding fractions). Student responses from 73,000+ UK students aged 11–16. Difficulty quantified as IRT b₂PL parameters fitted via MLE on the full response matrix. Probe set: 20 items stratified by difficulty quintile, initially selected to test five well-documented misconception areas (inverse operations, order of operations, fraction arithmetic, negative number operations, and place value). The probe set was selected before seeing any LLM estimation results, based on the availability of well-characterised misconceptions in the education literature. Confirmation set: 105 items with sufficient response volume for stable IRT parameter estimation.

**SmartPaper (India, open-ended).** 140 questions across English, Mathematics, Science, and Social Science for Grades 6–8 in Indian government schools. Open-ended format with rubric-based scoring. Difficulty quantified as classical proportion correct. Items span genuine cognitive complexity variation: recall vs. reasoning, concrete vs. abstract, single-step vs. multi-step. Probe set: 20 items; full set: 140 items (134 non-visual items used for most experiments).

### 2.3 Ground Truth and Data Quality

**Eedi answer ordering.** During this research, we discovered that the Eedi dataset contains two different answer orderings: the Kaggle ordering (how answer texts map to A/B/C/D labels in the data) and the NeurIPS competition ordering (how answers were presented to students on screen). These match only 24.5% of the time. All analyses in this paper use the NeurIPS ordering (student-facing), which is the correct basis for computing difficulty. We report this as a cautionary example of subtle alignment issues in secondary analysis of educational datasets.

**IRT parameter estimation.** For the 105-item confirmation set, we fitted 2PL IRT models using maximum likelihood estimation on the full response matrix (15.8M responses). Parameters were validated against the original Eedi-provided parameters at ρ=1.000 for the overlapping items.

**SmartPaper calibration.** Models systematically overestimate item easiness by approximately +0.40 (predicted mean p=0.73 vs. actual mean p=0.29), likely reflecting higher-performing populations in training data. All correlations are rank-order (Spearman ρ), which is invariant to this monotone miscalibration.

### 2.4 Prompt Architecture

The RSM screening phase (Phase 1) tested three prompt framings: *individual roleplay* (simulate a single student answering), *classroom batch* (simulate N students answering together), and *teacher prediction* (predict what proportion of students at each ability level would choose each option). Only teacher prediction produced significant signal.

All prompts in the viable region identified by Phase 1 share the teacher-prediction structure: the model is asked to predict what percentage of students at each of four ability levels (below basic, basic, proficient, advanced) would choose each answer option. The predicted difficulty is computed as a weighted average of predicted proportion incorrect, with weights reflecting a typical school population (below basic: 0.25, basic: 0.35, proficient: 0.25, advanced: 0.15).

In Phase 2, we tested seven variants of this teacher-prediction prompt, differing in what additional reasoning they request. The two best-performing were:

**Contrastive.** Asks the model to consider what makes *this specific item* harder or easier than similar items, then predict response distributions. Frames the task as comparative judgment.

**Error analysis.** Asks the model whether students would *actually make errors* on this item — specifically, whether the wrong answers represent errors that real students would plausibly commit. Frames the task as error plausibility assessment.

Both prompts share a key design principle: they redirect the model from generic difficulty judgment ("how hard is this?") toward specific error analysis ("would students actually get this wrong, and why?"). This framing accesses the model's implicit pedagogical content knowledge — absorbed from teacher forums, textbook discussions, and education research in training data — rather than asking it to reason about difficulty in the abstract.

Full prompt texts for all configurations are provided in the supplementary materials.

### 2.5 Models

We tested 15+ models across 5 providers. The primary models for each dataset:

- **Gemini 3 Flash** (Google): Best on Eedi probe items (ρ=0.50 per-rep), strong on SmartPaper (ρ=0.55). No mandatory thinking, clean structured output at high temperature.
- **Llama-4-Scout 17B** (Groq): Free inference, high per-rep variance (SD=0.17) but reaches ρ=0.67 with 10-rep averaging on Eedi probe items. Matches Gemini Flash's averaged performance despite being much smaller.
- **DeepSeek V3** (DeepSeek): Best intrinsic calibration on SmartPaper (ρ=0.80 at low temperature without structured prompting). Degrades at high temperature — opposite of Gemini.
- **GPT-4o** (OpenAI): Surprisingly weak on Eedi probe items (ρ=0.17) despite strong general benchmarks.

The full cross-model results are reported in Section 3.3.

### 2.6 Statistical Approach

All correlations are Spearman rank correlations (ρ), appropriate because (a) we care about rank ordering of items by difficulty, not absolute calibration, and (b) the relationship between predicted and actual difficulty may be monotone but nonlinear. Significance is assessed at α=0.05 (two-tailed). For the 105-item confirmation test, we report bootstrap 95% confidence intervals (10,000 resamples).

We do not correct for multiple comparisons across the screening and optimisation phases, because these phases are exploratory by design — they map the parameter landscape rather than test pre-specified hypotheses. The confirmatory test is the 105-item expansion (Phase 4), which is a single pre-planned comparison. This is standard DOE practice: screening experiments identify promising regions; confirmation experiments provide the inferential test.

For multi-rep experiments, we report both per-rep ρ (mean ± SD across replications) and averaged-prediction ρ (Spearman correlation between the mean prediction across reps and the ground truth). The latter is consistently higher because averaging reduces noise, but we treat the per-rep numbers as the honest estimate of single-pass performance.


### 2.7 Hypothesis Registry

We organize the investigation as a hypothesis table. Each row is drawn from a specific claim in the prior literature or from an intermediate finding in our own pipeline.

| ID | Hypothesis | Source | Eedi Result | SmartPaper Result |
|----|-----------|--------|-------------|-------------------|
| H1 | Direct estimation yields moderate-to-strong correlations | Attali (2024), Yaneva et al., Benedetto et al. | r≈0 (5 models × 4 prompts, n=1,869) | ρ=0.55–0.65 (n=140) |
| H2 | Student simulation recovers difficulty ordering | Lu & Wang (2024), SMART (Lan et al., 2025) | ρ≈0.1–0.3 (roleplay, classroom batch) | — |
| H3 | Teacher prediction framing outperforms student simulation | "Take Out Your Calculators" (2026) | ρ=0.50 per-rep; 0.57 averaged | ρ=0.66 (baseline calibration) |
| H4 | Misconception hints improve estimation | Eedi competition literature | Full hints ρ=0.12 (worse than hidden ρ=0.41) | — |
| H5 | Higher temperature improves difficulty discrimination | Novel (RSM screening) | Flat (ρ≈0.35–0.58 across t=0.3–2.0 with contrastive prompt); original "cliff" was prompt×temp confound | Model-dependent: Gemini ↑ (0.81→0.88); DeepSeek ↓ (0.80→0.68) |
| H6 | Contrastive prompting stabilizes estimates | Novel (metaprompt sweep) | ρ=0.577±0.075 (most stable) | — |
| H7 | Error analysis prompting yields highest mean correlation | Novel (metaprompt sweep) | ρ=0.604±0.062 (highest mean) | — |
| H8 | Error information in calibration boosts estimation | Novel | — | ρ=0.83 (errors strategy) |
| H9 | Model choice matters more than prompt choice | Cross-model comparison | Gemini Flash >> Llama-70B >> Qwen/GPT-4o-mini | DeepSeek ≥ Gemini at baseline; Gemini wins with structured prompts at high temp |
| H10 | Averaging across multiple runs improves estimates | Wisdom-of-crowds literature | Averaged ρ exceeds mean single-rep ρ consistently | — |
| H11 | Optimized configs generalize to held-out items | Confirmation experiment | **NO.** 105-item: ρ=0.114 (probe 20: ρ=0.46; new 85: null) | **YES.** 140-item: ρ=0.547 (probe 20: ρ=0.77; held-out 120: ρ=0.52) |

## 3. Results

We present results in the order of the DOE pipeline: screening, optimisation, boundary testing, and generalisation. The key result — the item-type dissociation — emerges in Section 3.4.

### 3.1 Screening: The Null Wall and One Viable Region

**Direct estimation at scale (Eedi, n=1,869).** Before the RSM experiment, we tested direct difficulty estimation — asking LLMs to predict what proportion of students would answer correctly — using 5 models × 4 prompt variants at temperature 0.

| Model | Best r | n valid |
|-------|--------|---------|
| GPT-4o-mini | +0.010 | 1,869 |
| Llama-3.3-70B | −0.032 | 1,530 |
| Llama-4-Scout | −0.013 | 1,868 |
| Llama-4-Maverick | −0.024 | 1,735 |
| Qwen3-32B | +0.015 | 1,548 |

All correlations are indistinguishable from zero. This is consistent with Kröger et al. (2025), who report r≈0 for direct estimation, and inconsistent with Razavi & Powers (2025), who report r=0.83 on K–5 items. The discrepancy is addressed in Section 4.

**Feature extraction + ML (Eedi, n=1,869).** GPT-4o extracted 7 features (DOK, cognitive load, syntax complexity, etc.) for all items; gradient boosted machine trained on 80% split. Test r=0.063. An initial pilot on 50 items had shown r=0.77 on 10 test items — classic small-sample overfitting.

**Classroom simulation (Eedi, n=1,869).** Llama-3.3-70B simulated 20 students × 4 ability levels per item. r=0.013. The model produces near-identical response distributions regardless of item difficulty — accuracy changes less than 1% across proficiency levels. This replicates the "Human–AI Difficulty Alignment" finding that LLMs don't exhibit the same difficulty ordering as human students.

**Box-Behnken screening (Eedi, n=20 probe items).** Having established the null on standard approaches, we ran the response surface design. Of 46 configurations, only one region produced significant signal: teacher-prediction framing at high temperature.

| Prompt style | t=0.3 | t=0.9 | t=1.5 |
|-------------|-------|-------|-------|
| Individual roleplay | 0.307 | — | 0.306 |
| Classroom batch | 0.211 | — | 0.325 |
| Teacher prediction | 0.119 | — | 0.673* |

*Single run; later validated at ρ≈0.50 (10-rep mean).

The apparent "temperature cliff" for teacher prediction (0.12 → 0.67) was later shown to be a confound: the t=1.5 configuration used a stronger prompt variant than the t=0.3 configuration (see Section 3.5). With prompt held constant, temperature is roughly flat on Eedi.

### 3.2 Optimisation: Prompt Design Is the Primary Variable

Within the viable region, we tested 7 prompt variants × 3 temperatures × 3 replications.

| Prompt | t=1.5 | t=1.8 | t=2.0 |
|--------|-------|-------|-------|
| Error analysis | 0.451 ± 0.049 | 0.566 ± 0.108 | **0.604 ± 0.062** |
| Contrastive | **0.577 ± 0.075** | 0.518 ± 0.061 | 0.562 ± 0.084 |
| Devil's advocate | 0.543 ± 0.020 | 0.405 ± 0.035 | 0.516 ± 0.122 |
| Imagine classroom | 0.232 ± 0.041 | 0.404 ± 0.035 | 0.428 ± 0.115 |
| Sparse | 0.235 ± 0.189 | 0.111 ± 0.032 | 0.232 ± 0.175 |

The range within a single temperature column (e.g., t=2.0: 0.23 to 0.60) far exceeds the range within a single prompt row (e.g., error analysis: 0.45 to 0.60). Prompt design is the dominant factor; temperature is secondary.

**10-rep validation.** The top two configurations were indistinguishable after 10 replications:

| Config | 3-rep estimate | 10-rep per-rep mean | 10-rep averaged ρ |
|--------|---------------|--------------------|--------------------|
| Error analysis t=2.0 | 0.604 ± 0.062 | 0.500 ± 0.111 | 0.573 |
| Contrastive t=1.5 | 0.577 ± 0.075 | 0.513 ± 0.097 | 0.571 |

The 3-rep estimates were optimistically biased by ~0.10. True per-rep performance is ρ≈0.50. Averaged predictions converge to ρ≈0.57, with diminishing returns beyond 7 reps.

### 3.3 Boundary Testing

#### 3.3.1 Cross-Model Survey

Using the contrastive prompt at t=1.5, we tested 15 models across 5 providers (3 reps each).

| Tier | Model | Provider | Per-rep ρ | SD | Avg-pred ρ |
|------|-------|----------|-----------|-----|-----------|
| 1 | Gemini 3 Flash | Google | 0.470 | 0.076 | 0.549 |
| 1 | Llama-4-Scout 17B | Groq | 0.355* | 0.167* | 0.668* |
| 1 | DeepSeek V3 | DeepSeek | 0.338 | 0.140 | 0.409 |
| 2 | GPT-OSS-120B | Groq | 0.336 | 0.100 | 0.450 |
| 2 | Kimi-K2 | Groq | 0.336 | 0.116 | 0.319 |
| 2 | Llama-3.3-70B | Groq | 0.257 | 0.062 | — |
| 2 | GPT-4o | OpenAI | 0.172 | 0.126 | 0.204 |
| 3 | Qwen3-32B | Groq | 0.005 | 0.110 | — |
| 3 | Llama-3.1-8B | Groq | −0.081 | 0.097 | −0.009 |
| 3 | GPT-4o-mini | OpenAI | −0.092 | 0.266 | — |
| 3 | Gemini 2.0 Flash | OpenRouter | −0.200 | 0.132 | −0.269 |

*Scout values from 10-rep validation with error analysis at t=2.0.

There is no correlation between model benchmark performance and task performance. GPT-4o (frontier) is outperformed by Llama-4-Scout (17B, free). Gemini 3 Pro, with mandatory thinking tokens, produces ρ=0.05 — worse than its Flash counterpart at ρ=0.47. This is the "System 1" pattern: models that are forced to deliberate perform worse.

#### 3.3.2 Deliberation Hurts

| Model | Thinking | ρ |
|-------|----------|---|
| Gemini 3 Flash (no thinking) | None | 0.500 |
| Gemini 3 Flash (thinking_budget=1024) | Optional | 0.467 |
| Gemini 3 Pro (mandatory thinking) | Required | 0.052 |
| Gemini 2.5 Pro (mandatory thinking) | Required | 0.240 |
| DeepSeek-Reasoner | Required | NaN (constant output) |

Every form of deliberation degrades performance. The task requires fast pattern matching on implicit pedagogical knowledge, not step-by-step reasoning.

#### 3.3.3 Structured Elicitation: Testing Theoretical Predictions

We tested predictions from four learning science traditions. Each predicts that richer specification of student cognition should improve estimation.

| Strategy | Theoretical basis | ρ | vs. baseline (0.50) |
|----------|------------------|---|---------------------|
| Direct prediction | KC theory | 0.500 ± 0.111 | — |
| Two-stage cognitive chains | Conceptual change | 0.508 ± 0.067 | ≈ 0 |
| Buggy reasoning (no hint) | BUGGY | 0.488 | −0.01 |
| Two-stage backstory | Generative students | 0.408 ± 0.096 | −0.09 |
| Buggy reasoning (with hint) | ESS E3 + confusion tuples | 0.193 | −0.31 |
| Cognitive modeling (10 CoT chains) | SMART / classroom sim | 0.165 ± 0.046 | −0.34 |

The prediction is wrong. Every structured approach matched or underperformed direct prediction. Providing the known misconception actively hurt (ρ=0.19 vs. 0.50) — the model over-anchors on the specified error rather than analysing overall item difficulty.

One nuanced finding: cognitive chains used as *context* for a teacher-prediction prompt (ρ=0.51) dramatically outperformed the same chains used as a *prediction mechanism* by counting simulated answers (ρ=0.17). The model integrates diverse perspectives better than it enacts individual student cognition. But context-based elicitation provides no improvement over direct prediction — it merely stabilises variance (SD 0.067 vs. 0.111).

#### 3.3.4 Calibration Anchors

Providing example items with known difficulty as in-context anchors:

| | SmartPaper (20 probe) | Eedi (4-fold CV) |
|---|---|---|
| With anchors | ρ=0.81 | ρ=0.445 |
| Without anchors | ρ=0.66 | ρ=0.525 |
| Effect | **+0.15** | **−0.08** |

Anchors help on SmartPaper but hurt on Eedi. The model's implicit calibration on Eedi is better than explicit numerical anchoring — likely because providing IRT values confuses the ability-stratified distribution format.

### 3.4 Generalisation: The Central Finding

This is the most important result.

#### Eedi: Signal Collapse

We expanded the probe set from 20 items to 50 (adding 30 randomly sampled items), then to 105 items with full 2PL IRT parameters.

| Set | n | ρ | p | 95% CI |
|-----|---|---|---|--------|
| Original 20 probe | 20 | 0.462 | 0.04 | — |
| New 30 random | 30 | −0.176 | 0.35 | — |
| Combined 50 | 50 | 0.039 | 0.79 | — |
| **105-item confirmation** | **105** | **0.114** | **0.246** | **[−0.072, 0.297]** |

The signal collapsed on non-probe items. To understand why, we analysed the probe and non-probe items within the 105-item experiment (same prompt, model, and reps for all items). The 20 probe items still show ρ=0.537 (p=0.015) while the 85 non-probe items show ρ=0.027 (p=0.81). The signal is item-specific.

**Diagnosis: difficulty-range inflation.** The probe set was stratified by difficulty quintile and contains two items with extreme IRT difficulty values (b=−5.18, −3.37) far outside the non-probe range ([−1.62, +1.85]). This gives the probe set 2.3× the difficulty SD of the non-probe items (1.46 vs. 0.64). Removing just these two outliers drops the probe correlation from ρ=0.537 to ρ=0.391 (p=0.11, non-significant). The model can distinguish "very easy" items (which nearly all students answer correctly) from items in the normal difficulty range, but cannot rank-order items within that range.

Three further checks confirm this interpretation: (1) all 105 items test the same four misconceptions as the probe set, ruling out misconception familiarity as an explanation; (2) prediction spread is identical across probe and non-probe items (SD=0.087 vs. 0.085), so the correlation comes from ground-truth spread, not better predictions; (3) bootstrap resampling (10,000 random 20-item subsets from the 105) shows the mean ρ=0.110 with only 2.5% of subsets reaching ρ≥0.50 — the probe result is a tail event explained by its extreme difficulty range.

The 105-item result is definitive: ρ=0.114, 95% CI [−0.072, 0.297]. The optimised prompt does not generalise. The probe-set correlations reported in Sections 3.1–3.3 are inflated by difficulty-range selection and should be interpreted as parameter-space exploration, not as evidence of genuine item-level discrimination.

#### SmartPaper: Signal Generalises

| Set | n | ρ | p |
|-----|---|---|---|
| Probe 20 | 20 | 0.768 | <0.001 |
| Held-out 120 | 120 | 0.518 | <0.0001 |
| **All 140** | **140** | **0.547** | **<0.0001** |

The signal holds on held-out items. By subject:

| Subject | ρ | n | p |
|---------|---|---|---|
| Science | 0.734 | 32 | <0.0001 |
| Social Science | 0.702 | 34 | <0.0001 |
| Mathematics | 0.600 | 33 | 0.0002 |
| English | 0.432 | 41 | 0.005 |

Signal is strongest in Science and Social Science (where content complexity variation is greatest) and weakest in English (where difficulty may depend more on population-specific language proficiency).

#### The Dissociation

| | Eedi (MCQ) | SmartPaper (open-ended) |
|---|---|---|
| Probe items | ρ=0.46 (n=20) | ρ=0.77 (n=20) |
| Non-probe items | ρ=−0.18 (n=30) | ρ=0.52 (n=120) |
| Full confirmation | **ρ=0.11** (n=105) | **ρ=0.55** (n=140) |
| Generalises? | **No** | **Yes** |

### 3.5 Mechanistic Investigation: Disentangling Temperature and Prompt

Controlled temperature sweeps with the contrastive prompt held constant (Gemini Flash, 3 reps per point):

| Temp | Eedi ρ (contrastive) |
|------|---------------------|
| 0.3 | 0.449 ± 0.118 |
| 0.6 | 0.458 ± 0.046 |
| 0.9 | 0.358 ± 0.070 |
| 1.2 | 0.354 ± 0.127 |
| 1.5 | 0.462 ± 0.031 |
| 2.0 | 0.580 ± 0.110 |

The curve is flat (ρ≈0.35–0.58) — no temperature cliff. The original RSM confounded prompt quality with temperature: it compared a weak prompt at t=0.3 (ρ=0.12) to a strong prompt at t=1.5 (ρ=0.67).

On SmartPaper, temperature interacts with model:

| Model | Strategy | t=0.5 | t=1.0 | t=1.5 | t=2.0 |
|-------|----------|-------|-------|-------|-------|
| Gemini | errors | 0.774 | 0.803 | 0.783 | **0.841** |
| Gemini | anchors | 0.811 | 0.821 | 0.783 | **0.877** |
| DeepSeek | baseline | **0.800** | **0.809** | 0.728 | 0.675 |
| DeepSeek | errors | **0.799** | 0.787 | 0.783 | 0.766 |

Temperature helps Gemini (monotonic increase with structured prompts) but hurts DeepSeek (monotonic decrease across all strategies). Structured prompts provide temperature-robustness for Gemini: its baseline collapses at t=1.5 (ρ=0.31) while errors/anchors stay in a tight 0.77–0.88 band.

#### Two-Stage Diversity at Scale

Two-stage diversity injection (generate 5 diverse student attempts at t=2.0, use as context for prediction) was tested on SmartPaper's full 134 items:

| Method | ρ | p | n |
|--------|---|---|---|
| Direct prediction (Phase 4) | **0.547** | <0.0001 | 140 |
| Two-stage cognitive chains | 0.059 | 0.50 | 134 |

Two-stage destroyed the signal. The damage was worst where direct prediction was strongest (Science: 0.734 → 0.173; Social Science: 0.702 → −0.011). The simulated student attempts, though realistic in quality, act as noise that dilutes the model's implicit knowledge. Five samples cannot represent a population distribution; the model anchors on seed content rather than generalising.


## 4. Discussion

### 4.1 Why the Literature Disagrees: Complexity-Driven vs. Selectivity-Driven Difficulty

The dissociation between Eedi and SmartPaper resolves the conflicting findings in the literature. We propose a framework distinguishing two sources of item difficulty:

**Complexity-driven difficulty.** The item requires more cognitive steps, deeper reasoning, or less familiar content. In Knowledge Component terms (Koedinger et al., 2012), difficulty reflects the number and type of KCs required. This is legible from item text — an LLM can detect that multi-step reasoning is harder than recall, or that abstract concepts are harder than concrete ones. SmartPaper items vary primarily on this dimension: "explain the water cycle" is harder than "name the capital of India" because it requires more cognitive operations, and this is evident from the text alone. Studies reporting high correlations (Razavi & Powers, 2025: r=0.83 on K–5 items spanning topics and cognitive demands; Kröger et al., 2025: r=0.82 via simulation on NAEP items spanning grade-level breadth) likely use item pools where difficulty is primarily complexity-driven.

**Selectivity-driven difficulty.** The item's difficulty depends on whether specific distractors trigger specific misconceptions in specific students — and how prevalent those misconceptions are in the target population. This is Brown & Burton's (1978) domain at the population level: not just which buggy procedures exist, but how many students in a given population apply each one. Two items testing the same concept (e.g., −1×−4 vs. 3×−2) can have very different error rates depending on how the numbers and context interact with common bugs. This information is not in the item text; it is an empirical property of the student–item interaction. Eedi items vary primarily on this dimension. Studies reporting low correlations (Kröger et al., 2025: r≈0 for direct estimation; our Eedi results) use items where difficulty is primarily selectivity-driven.

This framework explains three patterns:

1. **Variable correlations across studies.** Razavi & Powers (r=0.83 on diverse K–5 items) vs. Kröger et al. (r≈0 for direct estimation on NAEP) is not a methodological discrepancy — it reflects different difficulty sources in the item pools.

2. **Why simulation fails on Eedi.** Kröger et al. achieve r=0.82 via simulation on NAEP items that span grade-level breadth (complexity variation). Our simulation on Eedi (r=0.01) fails because Eedi items are narrowly targeted within a single misconception cluster (selectivity variation). The model produces near-identical simulated response distributions regardless of which specific item tests a given misconception.

3. **Why the 20-item success didn't generalise.** The probe set was stratified by difficulty quintile and contained two extreme-difficulty outliers (b=−5.2, −3.4) that inflated the difficulty SD to 2.3× the non-probe level. Removing these two items renders the probe correlation non-significant (ρ=0.39, p=0.11). The model can distinguish "very easy" items (which nearly all students answer correctly) from items in the normal difficulty range, but cannot rank-order within that range. This is a difficulty-range artifact, not genuine item-level discrimination.

**Important caveat.** This framework is a post-hoc interpretation. The two datasets differ in multiple ways beyond difficulty source: item format (MCQ vs. open-ended), population (UK vs. India), subject breadth (maths-only vs. 4 subjects), and difficulty metric (IRT vs. classical). We cannot isolate the causal factor without matched item sets varying only format or difficulty source.

### 4.2 The "System 1" Finding: Deliberation Hurts

Across every comparison, fast pattern-matching outperformed deliberative reasoning:

- Direct prediction (ρ=0.50) > cognitive modeling with 10 reasoning chains (ρ=0.17)
- Gemini Flash without thinking (ρ=0.50) > with thinking budget (ρ=0.47) > Pro with mandatory thinking (ρ=0.05)
- Simple teacher prediction (ρ=0.50) > buggy reasoning with misconception specification (ρ=0.19)
- Direct prediction on SmartPaper (ρ=0.55) > two-stage with 5 simulated attempts as context (ρ=0.06)

This is consistent with dual-process theory (Kahneman, 2011) applied to the model's knowledge retrieval. The model's implicit pedagogical knowledge — what proportion of Year 9 students would get confused by this fraction addition — is best accessed as a fast judgment, not a deliberative analysis. Forcing the model to "show its work" (reasoning chains, misconception specification, student simulation) introduces noise and anchoring effects that corrupt the implicit signal.

The practical implication is counterintuitive: for difficulty estimation, the optimal prompt is the simplest one that activates the right knowledge. Scaffolding — specifying misconceptions, generating student personas, requiring step-by-step analysis — actively degrades performance.

### 4.3 What Doesn't Work: A Registry of Null Results

We tested 18 hypotheses drawn from prior work. The following produced null or negative results and, to our knowledge, represent novel negative findings:

1. **Feature extraction + ML** collapses from r=0.77 (n=10 test) to r=0.06 (n=1,869) — small-sample overfitting.
2. **Classroom simulation** produces r=0.01 on misconception-targeted MCQs — simulated students converge on the correct answer regardless of persona.
3. **Misconception hints actively hurt** (ρ=0.19 vs. 0.50 without) — the model over-anchors on the specified error.
4. **Student personas add noise** (ρ=0.41 vs. 0.50 without) — backstories don't produce useful diversity.
5. **Deliberation hurts** across all models tested — reasoning tokens consume output budget and break format compliance at high temperature.
6. **Option shuffling adds noise** (ρ=0.62 vs. 0.67 original order) — presentation-order sensitivity degrades rather than diversifies predictions.
7. **Two-stage diversity injection destroys signal at scale** (ρ=0.06 vs. 0.55 direct on SmartPaper) — intermediary representations corrupt the model's implicit knowledge.
8. **Calibration anchors don't transfer** — helped on SmartPaper (+0.15) but hurt on Eedi (−0.08).
9. **Model size/benchmark performance doesn't predict success** — GPT-4o (frontier) at ρ=0.17 is outperformed by Llama-4-Scout 17B at ρ=0.36.
10. **Small evaluation sets inflate correlations** — ρ=0.50 on 20 curated items collapsed to ρ=0.11 on 105 items.

These null results are arguably more valuable to the field than our positive findings. Each eliminates a plausible approach and narrows the space of viable methods.

### 4.4 Averaging Helps, but Has Limits

Multi-sample averaging consistently improves predictions:

| Model | Per-rep ρ | 3-rep avg ρ | 10-rep avg ρ |
|-------|-----------|-------------|-------------|
| Gemini Flash | 0.500 ± 0.111 | 0.666 | 0.573 |
| Scout 17B | 0.355 ± 0.167 | 0.609 | 0.668 |

Scout's trajectory is instructive: high per-rep variance (SD=0.167) but monotonic improvement with averaging (3→0.609, 5→0.639, 7→0.648, 10→0.668). This is a wisdom-of-crowds effect — diverse stochastic predictions averaging toward truth. However, the improvement plateaus, and importantly, it only boosts signal that already exists. On the full 105-item Eedi set, where the per-rep signal is null, averaging cannot recover what isn't there.

### 4.5 Limitations

1. **Probe set size.** The screening and optimisation phases used 20 items per dataset. While the confirmation experiments (105 Eedi items, 140 SmartPaper items) provide the inferential tests, the RSM surface was estimated on a small basis.

2. **Two datasets, many differences.** Eedi and SmartPaper differ in item format, subject, population, country, difficulty metric, and difficulty range. We attribute the dissociation to difficulty source (selectivity vs. complexity), but other confounds exist. Matched item sets varying only format would be needed to isolate the causal factor.

3. **Not pre-registered.** The adaptive experimental sequence means that screening-phase correlations are exploratory. We mitigate this through the 105-item confirmation test (null) and transparent reporting of all conditions, including failures.

4. **Model-specific results.** The strongest Eedi probe-set results require Gemini 3 Flash or Llama-4-Scout specifically. Other models produce weaker or null correlations. Results are not model-general.

5. **Surface feature confound on SmartPaper.** Question text length correlates ρ=−0.44 with difficulty. The LLM's ρ=0.55 exceeds this, and within-subject correlations hold, but surface features contribute partially.

6. **Calibration offset.** Models overestimate easiness by +0.40 for Indian government school students. Rank ordering transfers but absolute calibration does not. Practitioners would need population-specific anchoring for any application requiring absolute difficulty estimates.


## 5. Conclusion

We mapped the parameter landscape for LLM item difficulty estimation across two datasets, 15+ models, and 18 hypotheses drawn from prior literature using a sequential design-of-experiments pipeline. The central finding is a dissociation: LLMs can rank-order difficulty for open-ended items where difficulty varies in cognitive demand (ρ=0.55, n=140, generalises to held-out items) but cannot do so for misconception-targeted MCQs where difficulty depends on distractor–misconception interactions (ρ=0.11, n=105, non-significant). No combination of model, prompt, temperature, structured elicitation, or aggregation strategy overcame this boundary.

Within each dataset, the factors that matter are not what the literature would predict. Temperature is flat on Eedi once prompt quality is controlled, and interacts with model on SmartPaper (helps Gemini, hurts DeepSeek). Prompt design is the dominant factor on Eedi, creating a 3–4× difference. Model size and benchmark performance do not predict task success. And every form of cognitive scaffolding — misconception hints, student personas, reasoning chains, two-stage elicitation — either matched or degraded performance relative to simple direct prediction.

The practical recommendation: characterise the difficulty source in your item pool before investing in LLM-based estimation, and always test generalisation beyond curated probe sets. The methodological recommendation: search systematically rather than testing one configuration and concluding.


## 6. Related Work

### 6.1 Direct Difficulty Estimation

Attali (2024) demonstrated that GPT-4 can estimate item difficulty for adaptive testing items, reporting moderate correlations. Yaneva et al. (2024) extended this to medical examination items. Razavi & Powers (2025) achieved the highest reported correlation (r=0.83) using GPT-4o on K–5 math and reading items, combined with tree-based ML on LLM-extracted features (r=0.62–0.87). Benedetto et al. (2023) applied LLMs to language teaching items. These studies share a common design: one model, one prompt, one dataset, one reported correlation.

Our contribution is methodological: by testing the same task across many configurations, we show that these point estimates are not comparable because the outcome depends on model × prompt × temperature × item-type interactions. The high correlations in Razavi & Powers likely reflect complexity-driven difficulty in their K–5 items; Attali's more moderate results may reflect a mix of difficulty sources.

### 6.2 Student Simulation

Kröger et al. (2025) simulate students using Gemma-2-27B at four NAEP ability levels, achieving r=0.75–0.82. The SMART framework (Lan et al., 2025) aligns simulated student responses with IRT parameters. Lu & Wang (2024) generate student profiles with specified knowledge states. These approaches share the assumption that simulating individual student cognition can recover population-level difficulty.

Our replication on Eedi (r=0.01) and structured elicitation experiments (Section 3.3.3) challenge this assumption for misconception-targeted items. The model produces near-identical simulated responses regardless of item difficulty because it converges on correct answers irrespective of persona specification. The one nuanced finding — that simulated student reasoning used as *context* (ρ=0.51) outperforms the same reasoning used as *mechanism* (ρ=0.17) — suggests the model has implicit pedagogical knowledge that is better accessed through judgment than simulation.

### 6.3 Reasoning and Scaffolding Approaches

The AIED 2025 paper (arXiv:2503.08551) reports 10–28% MSE reduction from reasoning augmentation (generating per-option analyses before predicting difficulty). The ESS framework (Tack & Piech, 2024; arXiv:2601.05473) proposes a taxonomy of student specification levels (E0–E4), predicting that richer specification improves simulation fidelity. The "confusion tuples" approach specifies exactly which concepts a student confuses.

Our experiments directly test these predictions (Section 3.3.3) and find them wrong for difficulty estimation (as opposed to response simulation). Every form of cognitive scaffolding — misconception hints, buggy procedure specification, student personas, reasoning chains — matched or degraded performance. This is consistent with our "System 1" interpretation: the task requires fast pattern matching, not deliberative analysis.

### 6.4 LLM Error Alignment

Liu, Sonkar & Baraniuk (2025) find that when LLMs make errors, they select the same wrong answers as students (r=0.73–0.80 error alignment). This suggests LLMs share some of students' misconception patterns. Our error alignment replication was inconclusive due to the Eedi answer ordering discrepancy (Section 2.3), but our broader finding is consistent: LLMs have implicit knowledge *about* common misconceptions (enabling ρ=0.50 on well-documented errors) but cannot simulate the *population prevalence* of those misconceptions.

### 6.5 Design of Experiments in AI Evaluation

Response surface methodology (Box & Behnken, 1960) and sequential DOE are standard in manufacturing, pharmaceuticals, and chemical engineering for optimising multi-factor processes. Their application to LLM evaluation is novel. The closest precedent is hyperparameter optimisation in ML (Bergstra & Bengio, 2012), but DOE differs in that it seeks to *understand the response surface* (which factors matter and how they interact) rather than simply find the optimum. This distinction is important for our purpose: we want to explain why the literature disagrees, not just find the best configuration.


## References

Attali, Y. (2024). Can language models estimate item difficulty? Findings from adaptive testing research. *Educational Measurement: Issues and Practice*.

Benedetto, L., et al. (2023). On the application of large language models for language teaching: Practice, evaluations, and challenges. *AAAI Workshop on AI for Education*.

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.

Box, G. E. P., & Behnken, D. W. (1960). Some new three level designs for the study of quantitative variables. *Technometrics*, 2(4), 455–475.

Brown, J. S., & Burton, R. R. (1978). Diagnostic models for procedural bugs in basic mathematical skills. *Cognitive Science*, 2(2), 155–192.

Chi, M. T. H., Slotta, J. D., & de Leeuw, N. (1994). From things to processes: A theory of conceptual change for learning science concepts. *Learning and Instruction*, 4(1), 27–43.

Eedi. (2024). Mining misconceptions in mathematics [Dataset]. Kaggle. https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Koedinger, K. R., Corbett, A. T., & Perfetti, C. (2012). The Knowledge-Learning-Instruction framework: Bridging the science-practice chasm to enhance robust student learning. *Cognitive Science*, 36(5), 757–798.

Kröger, B., et al. (2025). Take out your calculators: LLM-simulated classrooms for item difficulty estimation. arXiv:2601.09953.

Lan, A., et al. (2025). SMART: Simulating students aligned with item response theory. *EMNLP 2025*. arXiv:2507.05129.

Liu, Z., Sonkar, S., & Baraniuk, R. (2025). Do LLMs make mistakes like students? *AIED 2025*. arXiv:2502.15140.

Lu, Y., & Wang, S. (2024). Generative students: Using LLM-simulated student profiles for question item evaluation. *L@S 2024*. arXiv:2405.11591.

Razavi, R., & Powers, D. (2025). Estimating item difficulty using large language models and tree-based machine learning algorithms. arXiv:2504.08804.

Tack, A., & Piech, C. (2024). Harnessing LLMs for automated prediction of item difficulty and response time. *BEA Workshop, ACL 2024*.

Tack, A., et al. (2025). Towards valid student simulation with LLMs: The epistemic state specification framework. arXiv:2601.05473.

Yaneva, V., et al. (2024). Predicting item difficulty for medical examinations. *Journal of Educational Measurement*.

AIED 2025. Reasoning and sampling-augmented MCQ difficulty prediction. arXiv:2503.08551.
