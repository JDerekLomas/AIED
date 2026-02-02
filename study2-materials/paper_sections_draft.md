# Draft: Introduction and Method Sections

## 1. Introduction

Can LLMs estimate how difficult a test item is for students? The answer in the literature ranges from "not at all" to "very well." Direct estimation studies report correlations from r≈0 (Kröger et al., 2025) to r=0.83 (Razavi & Powers, 2025). Classroom simulation approaches range from r=0.01 (our replication) to r=0.82 (Kröger et al., 2025). Feature extraction ranges from r=0.06 (our replication at scale) to r=0.87 (Razavi & Powers, 2025). These are not minor discrepancies — they span the entire range from chance to near-perfect prediction.

Why the disagreement? Each study tests one method on one dataset and reports a single correlation. But difficulty estimation involves a large parameter space: the choice of model, prompt strategy, sampling temperature, aggregation method, and — critically — the type of items being estimated. A single-configuration study cannot distinguish between "this method works" and "this method works on these items, with this model, at this temperature." The result is a literature of point estimates that cannot be compared or reconciled.

We take a different approach. Rather than testing one pipeline and reporting a correlation, we conduct a systematic search across the parameter landscape using response surface methodology (RSM) — an experimental design framework from industrial process optimization (Box & Behnken, 1960). RSM was developed precisely for situations like this: the outcome (correlation with empirical difficulty) depends on multiple continuous and categorical factors whose interactions are unknown, individual experiments are cheap, and the goal is to map the response surface rather than test a single hypothesis.

We apply this methodology to two datasets that differ in a theoretically important way: 1,869 misconception-targeted maths MCQs from Eedi (UK), where difficulty depends on which specific distractors trigger which specific student errors; and 140 open-ended items across four subjects from SmartPaper (India), where difficulty depends on content complexity, prerequisite knowledge, and grade-level expectations. Across these two datasets, we test 15+ models, 20+ prompt configurations, 6 temperature settings, and multiple structured elicitation strategies — approximately 200 experimental conditions in total.

The result is not a single correlation but a map. The map reveals why the literature disagrees: LLM difficulty estimation works when difficulty is driven by content complexity (SmartPaper: ρ=0.55, n=140, generalises to held-out items) and fails when difficulty is driven by the interaction between specific distractors and specific student misconceptions (Eedi: ρ=0.11, n=105, non-significant). This dissociation holds across all models, prompts, and temperatures we tested. It is the primary finding.

The map also reveals what matters within each dataset. On Eedi, prompt design creates a 3–4× difference in correlation (ρ=0.12 for generic prediction vs. ρ=0.45 for contrastive error-focused prompting), while temperature is roughly flat once prompt quality is controlled. On SmartPaper, temperature interacts with model: it helps Gemini (ρ increases from 0.81 to 0.88) but hurts DeepSeek (ρ decreases from 0.80 to 0.68). No single factor dominates — the outcome is a model × prompt × temperature × item-type interaction.

Finally, the map reveals what doesn't work — and this may be the most useful contribution for the field. We tested predictions from four theoretical traditions in the learning sciences, each of which predicts that richer specification of student cognition should improve difficulty estimation:

- **Knowledge Component theory** (Koedinger, Corbett & Perfetti, 2012): Difficulty reflects the knowledge components an item requires. Motivates teacher-prediction framing.
- **Buggy Procedures** (Brown & Burton, 1978): Students apply internally-consistent but flawed algorithms. Motivates production-rule prompts that specify step-by-step erroneous procedures.
- **Conceptual Change theory** (Chi, Slotta & de Leeuw, 1994): Students hold coherent-but-wrong mental models. Motivates prompts describing flawed beliefs and why they feel right.
- **Epistemic State Specification** (Tack & Piech, 2024): A taxonomy of student simulation prompts from unspecified (E0) to data-calibrated (E4). Motivates increasingly detailed cognitive specifications.

All four predict that more scaffolding should help. Our central finding on Eedi is that it doesn't: every structured approach — cognitive modeling (ρ=0.17), buggy reasoning with misconception hints (ρ=0.19), student personas (ρ=0.41), two-stage diversity injection (ρ=0.06 at scale) — matched or underperformed simple direct prediction (ρ=0.50). The model already has implicit pedagogical knowledge about common misconceptions; explicit specification narrows its reasoning rather than enriching it. We call this the "System 1" finding: fast pattern matching on absorbed pedagogical knowledge outperforms deliberative cognitive simulation.

### Methodological Note

This research was conducted through iterative optimisation with Claude Code (Anthropic), an AI-assisted programming environment. The experimental sequence was not pre-registered; each experiment informed the next in a sequential design-of-experiments pipeline. We address the resulting multiple-comparisons concern in three ways: (1) 10-rep stability testing showed our initial 3-rep estimates were inflated by ~0.08–0.10, and we report the corrected numbers; (2) the definitive test is a held-out 105-item confirmation on Eedi, which returned ρ=0.11 (null); (3) all prompts, scripts, raw model outputs, and analysis code are publicly available. We report both successes and failures transparently.


## 2. Method

### 2.1 Design Framework: Sequential Experimentation

We adopt a sequential design-of-experiments (DOE) pipeline, standard in industrial process optimisation but novel in LLM evaluation. The pipeline has five phases, each answering a question that determines the next:

**Phase 1 — Screening (RSM).** A Box-Behnken response surface design (Box & Behnken, 1960) varying five factors across 46 configurations: prompt style (individual roleplay / classroom batch / teacher prediction), temperature (0.3 / 0.9 / 1.5), batch size (1 / 5 / 30 simulated students per call), misconception hint level (hidden / partial / full), and model (Gemini 2.5 Flash / Gemini 3 Flash). Each configuration was evaluated on 20 probe items with 20 simulated responses per item. The objective function was Spearman ρ between simulated difficulty and IRT b₂PL. This phase identified one viable region: teacher-prediction framing at high temperature.

**Phase 2 — Optimisation (metaprompt sweep).** Within the viable region, we tested 7 prompt variants × 3 temperatures (1.5, 1.8, 2.0) × 3 replications = 63 conditions. Prompts ranged from sparse ("estimate difficulty") to structured ("identify specific calculation errors students would make"). This phase identified two top configurations: error analysis (ρ=0.50 per-rep, 10-rep validated) and contrastive prompting (ρ=0.51 per-rep).

**Phase 3 — Stability and boundary testing.** Three parallel experiments:
- *10-rep stability*: Top 2 configs run for 10 replications each, revealing that 3-rep estimates were optimistically biased by ~0.08–0.10.
- *Cross-model survey*: 15 models across 5 providers (Google, OpenAI, Anthropic, Groq, DeepSeek), revealing a three-tier structure with no correlation between model size/benchmark ranking and task performance.
- *Structured elicitation*: Cognitive modeling, buggy reasoning, student personas, two-stage diversity injection, and calibration anchors — testing predictions from four theoretical traditions. All matched or underperformed direct prediction.

**Phase 4 — Generalisation.** The critical test: does the signal hold beyond the 20-item probe set?
- *Eedi expansion*: 20 → 50 items (adding 30 random), then 105 items with IRT b₂PL parameters. Signal collapsed: ρ=0.04 (50 items), ρ=0.11 (105 items, non-significant).
- *SmartPaper expansion*: 20 → 140 items. Signal held: ρ=0.55 (140 items, p<0.0001), ρ=0.52 on 120 held-out items.

**Phase 5 — Mechanistic investigation.** Controlled temperature sweeps (6 temperatures × 3 reps, prompt held constant) to disentangle the prompt × temperature confound from Phase 1. Two-stage experiments on SmartPaper's full 134 items to test whether context-based elicitation generalises. Temperature sweeps on SmartPaper with 2 models × 3 strategies × 4 temperatures to characterise the three-way interaction.

This pipeline consumed approximately $50 in API costs across all phases. Each API call costs $0.001–0.01; the total experiment represents roughly 20,000 calls across all conditions.

### 2.2 Datasets

**Eedi (UK, MCQ).** 1,869 diagnostic maths questions targeting specific misconceptions, each with 4 options where every distractor corresponds to a known error pattern (e.g., adding numerators and denominators separately when adding fractions). Student responses from 73,000+ UK students aged 11–16. Difficulty quantified as IRT b₂PL parameters fitted via MLE on the full response matrix. Probe set: 20 items stratified by difficulty quintile, selected to test five well-documented misconceptions. Confirmation set: 105 items with sufficient response volume for stable IRT parameter estimation.

**SmartPaper (India, open-ended).** 140 questions across English, Mathematics, Science, and Social Science for Grades 6–8 in Indian government schools. Open-ended format with rubric-based scoring. Difficulty quantified as classical proportion correct. Items span genuine cognitive complexity variation: recall vs. reasoning, concrete vs. abstract, single-step vs. multi-step. Probe set: 20 items; full set: 140 items (134 non-visual items used for most experiments).

The two datasets were chosen because they differ on a theoretically important dimension. In Eedi, item difficulty is primarily *selectivity-driven*: it depends on whether specific number choices and problem contexts trigger specific buggy procedures in specific students. Two items testing the same concept (e.g., −1×−4 vs. 3×−2) can have very different error rates depending on how the numbers interact with common bugs. This information is not in the item text — it is an empirical property of the student-item interaction. In SmartPaper, item difficulty is primarily *complexity-driven*: it depends on the number and type of cognitive operations required, the abstractness of the content, and the prerequisite knowledge assumed. An LLM can detect that "explain the water cycle" is harder than "name the capital of India" from the text alone.

### 2.3 Ground Truth and Data Quality

**Eedi answer ordering.** During this research, we discovered that the Eedi dataset contains two different answer orderings: the Kaggle ordering (how answer texts map to A/B/C/D labels in the data) and the NeurIPS competition ordering (how answers were presented to students on screen). These match only 24.5% of the time. All analyses in this paper use the NeurIPS ordering (student-facing), which is the correct basis for computing difficulty. We report this as a cautionary example of subtle alignment issues in secondary analysis of educational datasets.

**IRT parameter estimation.** For the 105-item confirmation set, we fitted 2PL IRT models using maximum likelihood estimation on the full response matrix (15.8M responses). Parameters were validated against the original Eedi-provided parameters at ρ=1.000 for the overlapping items.

**SmartPaper calibration.** Models systematically overestimate item easiness by approximately +0.40 (predicted mean p=0.73 vs. actual mean p=0.29), likely reflecting higher-performing populations in training data. All correlations are rank-order (Spearman ρ), which is invariant to this monotone miscalibration.

### 2.4 Prompt Architecture

All prompts in the viable region share a common structure: the model is asked to predict what percentage of students at each of four ability levels (below basic, basic, proficient, advanced) would choose each answer option. The predicted difficulty is computed as a weighted average of predicted proportion incorrect, with weights reflecting a typical school population (below basic: 0.25, basic: 0.35, proficient: 0.25, advanced: 0.15).

The two best-performing prompts on Eedi probe items were:

**Contrastive.** Asks the model to consider what makes *this specific item* harder or easier than similar items, then predict response distributions. Frames the task as comparative judgment.

**Error analysis.** Asks the model whether students would *actually make errors* on this item — specifically, whether the wrong answers represent errors that real students would plausibly commit. Frames the task as error plausibility assessment.

Both prompts share a key design principle: they redirect the model from generic difficulty judgment ("how hard is this?") toward specific error analysis ("would students actually get this wrong, and why?"). This framing accesses the model's implicit pedagogical content knowledge — absorbed from teacher forums, textbook discussions, and education research in training data — rather than asking it to reason about difficulty in the abstract.

Full prompt texts for all configurations are provided in the supplementary materials.

### 2.5 Models

We tested 15+ models across 5 providers. The primary models for each dataset:

- **Gemini 3 Flash** (Google): Best on Eedi (ρ=0.50 per-rep), strong on SmartPaper (ρ=0.55). No mandatory thinking, clean structured output at high temperature.
- **Llama-4-Scout 17B** (Groq): Free inference, high per-rep variance (SD=0.17) but reaches ρ=0.67 with 10-rep averaging on Eedi probe items. Matches Gemini Flash's averaged performance despite being much smaller.
- **DeepSeek V3** (DeepSeek): Best intrinsic calibration on SmartPaper (ρ=0.80 at low temperature without structured prompting). Degrades at high temperature — opposite of Gemini.
- **GPT-4o** (OpenAI): Surprisingly weak on Eedi (ρ=0.17) despite strong general benchmarks.
- **Claude Sonnet 4** (Anthropic): Limited by max temperature of 1.0 (ρ=0.26).

The full cross-model results are reported in Section 4.

### 2.6 Statistical Approach

All correlations are Spearman rank correlations (ρ), appropriate because (a) we care about rank ordering of items by difficulty, not absolute calibration, and (b) the relationship between predicted and actual difficulty may be monotone but nonlinear. Significance is assessed at α=0.05 (two-tailed). For the 105-item confirmation test, we report bootstrap 95% confidence intervals (10,000 resamples).

We do not correct for multiple comparisons across the screening and optimisation phases, because these phases are exploratory by design — they map the parameter landscape rather than test pre-specified hypotheses. The confirmatory test is the 105-item expansion (Phase 4), which is a single pre-planned comparison. This is standard DOE practice: screening experiments identify promising regions; confirmation experiments provide the inferential test.

For multi-rep experiments, we report both per-rep ρ (mean ± SD across replications) and averaged-prediction ρ (Spearman correlation between the mean prediction across reps and the ground truth). The latter is consistently higher because averaging reduces noise, but we treat the per-rep numbers as the honest estimate of single-pass performance.
