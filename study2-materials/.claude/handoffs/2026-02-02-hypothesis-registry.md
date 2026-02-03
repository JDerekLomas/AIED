# Hypothesis Registry: LLM Item Difficulty Estimation

Every hypothesis tested in this project, grounded in prior literature, with experimental verdicts.

---

## H1. Direct estimation produces meaningful correlations with item difficulty

**Literature basis:** Razavi & Powers (2025) report r=0.83 on K-5 math/reading with GPT-4o. Kröger et al. (2025) report r≈0 for direct estimation but r=0.75–0.82 for classroom simulation.

**Our test:** 7 models × 4 prompt variants × 1,869 Eedi items (corrected ground truth).

| Model | Prompt | n valid | r vs p_value |
|-------|--------|---------|-------------|
| GPT-4o-mini | basic | 1,869 | +0.007 |
| GPT-4o-mini | expert | 1,869 | +0.010 |
| GPT-4o-mini | irt | 1,867 | -0.026 |
| GPT-4o-mini | comparative | 1,869 | +0.006 |
| Llama-3.3-70B | expert | 1,530 | -0.032 |
| Llama-4-Scout | expert | 1,868 | -0.013 |
| Qwen3-32B | expert | 1,548 | +0.015 |
| Llama-4-Maverick | expert | 1,735 | -0.024 |

**Verdict: REJECTED.** All correlations ≈ 0. Consistent with Kröger et al. (r≈0 for direct), inconsistent with Razavi & Powers (r=0.83 on K-5). The discrepancy is likely domain-specific: K-5 items have more variance and simpler difficulty drivers than secondary maths MCQs.

---

## H2. Feature extraction + ML predicts difficulty

**Literature basis:** Razavi & Powers (2025) report r=0.62–0.87 using GBM on LLM-extracted features (DOK, cognitive load, syntax complexity, etc.). UnibucLLM (BEA 2024) achieved MSE=0.053 with SVR+BERT+LLM features.

**Our test:** GPT-4o extracted 7 features for 1,869 items → GBM.

| Metric | Our result | Published |
|--------|-----------|-----------|
| Test r | 0.063 | 0.62–0.87 |
| Feature importance | cognitive_load (0.237), conceptual_complexity (0.187) | Similar features |

**Initial result on 50 items:** r=0.770 on 10 test items — appeared promising but was overfitting to a tiny test set.

**Verdict: REJECTED at scale.** Feature extraction collapsed from r=0.77 (n=10 test) to r=0.06 (n=1,869). The small-sample result was noise. Eedi items may be too homogeneous in surface features for text-based feature extraction to discriminate difficulty.

---

## H3. Classroom simulation + IRT aggregation works

**Literature basis:** Kröger et al. (2025) achieve r=0.75–0.82 with Gemma-2-27b simulating students at 4 NAEP ability levels, 20 students per item. SMART framework (Scarlatos et al., 2025) also reports strong results.

**Our test:** Llama-3.3-70B via Groq, 20 simulated students × 4 ability levels × 1,869 items. Also enhanced version with rich backstories (10 items).

| Variant | n items | r |
|---------|---------|---|
| Standard classroom sim | 1,869 | +0.013 |
| Enhanced (backstory-augmented) | 10 | -0.328 |

**Verdict: REJECTED.** r≈0 at scale. The enhanced version with detailed student personas actually produced *negative* correlation. Core problem: LLMs converge on correct answers regardless of persona — accuracy changes <1% across proficiency levels (consistent with "Human-AI Difficulty Alignment" paper finding).

---

## H4. Model uncertainty signals difficulty

**Literature basis:** EDM 2025 reports SOTA RMSE using LLM uncertainty features (1st token probability, choice-order sensitivity). "Weaker models sometimes predict difficulty better" — uncertainty is more informative than accuracy.

**Our test:** GPT-4o-mini, 1,868 items. Three uncertainty metrics.

| Metric | r vs difficulty |
|--------|----------------|
| Answer probability | -0.007 |
| Answer entropy | +0.016 |
| Permutation consistency | -0.010 |

**Verdict: REJECTED.** All uncertainty metrics ≈ 0. The EDM 2025 paper used uncertainty as *features for ML* (combined with text features), not as direct predictors. Our zero-shot approach may fail because GPT-4o-mini solves these items too easily — there's no uncertainty to measure.

---

## H5. Reasoning-augmented prompts improve estimation

**Literature basis:** AIED 2025 (arXiv:2503.08551) reports 10–28% MSE reduction when generating reasoning about each option before predicting difficulty.

**Our test:** Multiple reasoning variants on Eedi probe items (20 items, Gemini Flash).

| Approach | ρ vs b_2pl | vs baseline (0.50) |
|----------|-----------|-------------------|
| Cognitive modeling (10 CoT chains) | 0.165 | -0.34 |
| Two-stage backstory | 0.408 | -0.09 |
| Buggy reasoning with hint | 0.193 | -0.31 |
| Buggy reasoning no hint | 0.488 | -0.01 |
| Two-stage cognitive chains | 0.508 | ≈ 0 |
| Two-stage error perspectives | 0.241 | -0.26 |
| Calibration anchors | 0.445 | -0.06 |

**Verdict: REJECTED.** Every structured reasoning approach matched or underperformed direct prediction (ρ≈0.50). The AIED 2025 paper used fine-tuned models on training data; zero-shot reasoning augmentation provides no benefit. The "System 1" finding: fast pattern matching outperforms deliberation for this task.

---

## H6. LLM errors naturally align with student errors

**Literature basis:** "Do LLMs Make Mistakes Like Students?" (arXiv:2502.15140) finds error selection alignment r=0.73–0.80. When LLMs err, they select the same wrong answers students select.

**Our test:** Error alignment replication on Eedi items (50 items).

| Metric | r |
|--------|---|
| Error alignment (corrected ground truth) | -0.019 |

**Verdict: INCONCLUSIVE.** Our replication used corrected answer ordering (the Kaggle vs NeurIPS ordering bug affected 76% of items). Per-option comparisons are fundamentally scrambled by the distractor ordering mismatch, so we cannot fully replicate this approach. The original paper's strong results may also reflect favorable item selection.

---

## H7. Confusion tuples / explicit misconception specification improves simulation

**Literature basis:** "Towards Valid Student Simulation" (arXiv:2601.05473) shows high correlation when explicitly telling the model which concepts a student confuses.

**Our test:** Buggy reasoning with known misconception hint (Gemini Flash, 20 items).

| Condition | ρ |
|-----------|---|
| With misconception hint | 0.193 |
| Without hint (control) | 0.488 |

**Verdict: REJECTED — hints actively hurt.** Providing the known misconception causes the model to over-anchor on one distractor rather than analyzing overall item difficulty. The model already has implicit knowledge of common misconceptions; explicit specification narrows its reasoning. This contradicts the confusion tuples paper, but that paper tested simulation accuracy (does the model pick the right wrong answer?), not difficulty estimation (does the model predict how many students err?).

---

## H8. Higher temperature improves predictions via diversity

**Literature basis:** Wisdom-of-crowds literature; high-temperature sampling creates diverse "expert opinions" that average toward truth. Our own Phase 2 RSM suggested a "temperature cliff" from t=0.3 (ρ=0.12) to t=1.5 (ρ=0.67).

**Our test:** Controlled sweeps with prompt held constant.

**Eedi (Gemini Flash, contrastive prompt, 3 reps × 6 temperatures):**

| Temp | Mean ρ |
|------|--------|
| 0.3 | 0.449 |
| 0.6 | 0.458 |
| 0.9 | 0.358 |
| 1.2 | 0.354 |
| 1.5 | 0.462 |
| 2.0 | 0.580 |

**Eedi (Scout, contrastive, 1024 tokens, 3 reps × 6 temperatures):**

| Temp | Mean ρ |
|------|--------|
| 0.3 | 0.398 |
| 0.6 | 0.466 |
| 0.9 | 0.552 |
| 1.2 | 0.508 |
| 1.5 | 0.372 |
| 2.0 | 0.499 |

**SmartPaper (Gemini 2.5 Flash, errors prompt, 5 reps × 4 temps):**

| Temp | ρ |
|------|---|
| 0.5 | 0.774 |
| 1.0 | 0.803 |
| 1.5 | 0.783 |
| 2.0 | 0.841 |

**SmartPaper (DeepSeek Chat, baseline, 5 reps × 4 temps):**

| Temp | ρ |
|------|---|
| 0.5 | 0.800 |
| 1.0 | 0.809 |
| 1.5 | 0.728 |
| 2.0 | 0.675 |

**Verdict: NUANCED.** The original "temperature cliff" was an artifact of confounding prompt × temperature. With prompt controlled:
- **Eedi + Gemini/Scout**: flat (ρ≈0.35–0.58). No temperature effect.
- **SmartPaper + Gemini**: temperature helps (ρ increases 0.77→0.84). Monotonic.
- **SmartPaper + DeepSeek**: temperature *hurts* (ρ decreases 0.80→0.68). Monotonic.

The interaction is model × prompt × dataset × temperature. No universal temperature rule.

---

## H9. Multi-sample averaging improves predictions

**Literature basis:** Ensemble/averaging literature. Kröger et al. average across simulated classrooms.

**Our test:** Systematic rep-scaling experiments.

| Model | 1-rep range | 3-rep avg | 10-rep avg |
|-------|------------|-----------|-----------|
| Gemini Flash (Eedi) | 0.24–0.68 | 0.666 | 0.573 |
| Scout (Eedi) | -0.02–0.54 | 0.609 | 0.668 |

**Verdict: SUPPORTED but with caveats.** Averaging consistently boosts correlations vs single reps. But: (a) the 3-rep averaged number (0.666) was inflated because those reps were the "lucky" cached ones from the sweep — fresh 10-rep averaging converges to ρ≈0.57 for Gemini; (b) diminishing returns after ~7 reps; (c) Scout benefits more from averaging (per-rep mean 0.355±0.167, but avg-pred improves monotonically: 3→0.609, 5→0.639, 7→0.648, 10→0.668) because it has higher per-rep variance. Temperature sweep confirmed t=2.0 is optimal for Scout (and is Groq's maximum).

---

## H10. Model capability (size/benchmark performance) predicts success

**Literature basis:** General expectation that "smarter" models do better. Kröger et al. note weaker models (Gemma) can outperform stronger ones.

**Our test:** 15+ models across 5 providers.

| Model | Size | Benchmark tier | ρ |
|-------|------|---------------|---|
| Gemini 3 Flash | Medium | High | **0.604** |
| Llama-4-Scout | 17B | Mid | 0.355 (per-rep mean; avg-pred 0.668 at 10 reps) |
| GPT-4o | Large | Very high | 0.172 |
| Claude Sonnet 4 | Large | Very high | 0.263 |
| GPT-4o-mini | Small | Mid | -0.092 |
| Gemini 3 Pro | Large | Very high | 0.052 |
| DeepSeek-Reasoner | Large | Very high | NaN |

**Verdict: REJECTED.** No correlation between model capability and task performance. GPT-4o and Claude Sonnet 4 (frontier models) are outperformed by Llama-4-Scout (17B, free) — Scout's per-rep mean is only 0.355 (SD=0.167), but with 10-rep averaging it reaches 0.668, matching Gemini Flash. Gemini 3 Pro (with mandatory thinking) is worse than Gemini 3 Flash. Consistent with Kröger et al.'s finding that weaker models can be better.

**Explanation:** The task requires (a) implicit pedagogical content knowledge from training data, (b) structured output compliance at high temperature, (c) no mandatory deliberation. Pro/frontier models fail on (b) and (c).

---

## H11. Deliberation/thinking improves predictions

**Literature basis:** Chain-of-thought generally helps on reasoning tasks. Reasoning models (o1, DeepSeek-Reasoner) outperform on math benchmarks.

**Our test:**

| Condition | ρ |
|-----------|---|
| Gemini Flash (no thinking) | 0.604 |
| Gemini Flash (thinking_budget=1024) | 0.467 |
| Gemini 3 Pro (mandatory thinking) | 0.052 |
| Gemini 2.5 Pro (mandatory thinking) | 0.240 |
| DeepSeek-Reasoner | NaN (constant output) |

**Verdict: REJECTED — deliberation actively hurts.** This is the "System 1" finding. The task requires fast pattern matching on pedagogical knowledge ("I've seen students make this mistake before"), not step-by-step reasoning about why students might err. Thinking consumes output budget, breaks format compliance at high temperature, and adds noise to intuitive judgments. Consistent with the "psychometrically plausible responses" paper finding that temperature scaling offers "minimal benefit" — reasoning models can't be made usefully stochastic.

---

## H12. Prompt design is the primary lever

**Literature basis:** Prompt engineering literature broadly. Our Phase 2 RSM showed teacher_prediction at t=0.3 (ρ=0.12) vs t=1.5 (ρ=0.67), originally attributed to temperature.

**Our test:** Controlled experiments holding temperature constant.

| Prompt | Temp | ρ |
|--------|------|---|
| Plain teacher_prediction | 0.3 | 0.119 |
| Contrastive | 0.3 | 0.449 |
| Error analysis | 2.0 | 0.604 |
| Contrastive | 1.5 | 0.577 |
| Imagine classroom | 1.5 | 0.232 |
| Sparse | 1.5 | 0.235 |
| Comparative difficulty | any | NaN (broken) |

**Verdict: SUPPORTED.** At the same temperature, prompt design creates a 3–4× difference (0.12→0.45). The two best prompts (error_analysis, contrastive) share a key feature: they ask the model to reason about *specific errors students would make on THIS question*, not generic difficulty judgment. "Would real students actually make errors?" outperforms "estimate the difficulty."

---

## H13. Signal generalizes beyond probe items

**Literature basis:** Standard psychometric concern about generalizability. Our probe set was 20 hand-picked items testing 5 well-known misconceptions.

**Our test:** Expanded from 20 → 50 Eedi items (adding 30 random). Also tested SmartPaper 20 → 140.

**Eedi:**

| Set | n | ρ |
|-----|---|---|
| Original 20 probe | 20 | 0.462 |
| New 30 random | 30 | -0.176 |
| Combined 50 | 50 | 0.039 |

**SmartPaper:**

| Set | n | ρ |
|-----|---|---|
| Original 20 probe | 20 | 0.768 |
| Remaining 120 | 120 | 0.518 |
| All 140 | 140 | 0.547 |

**Verdict: DATASET-DEPENDENT.**
- **Eedi MCQs: REJECTED.** Signal collapses on non-probe items. The ρ≈0.50 was specific to items testing well-documented misconceptions (inverse ops, order of operations, fraction addition, negative multiplication).
- **SmartPaper open-ended: SUPPORTED.** Signal holds at ρ=0.52 on 120 non-probe items. Open-ended item difficulty depends on general curriculum knowledge that LLMs have absorbed broadly from training data.

**This is arguably the most important finding of the project.**

---

## H14. Item type (MCQ vs open-ended) determines predictability

**Literature basis:** Published positive results span MCQs (Kröger, AIED 2025) and open-ended (various). No systematic comparison. Eedi items are MCQs with distractors; SmartPaper items are open-ended with rubrics.

**Our test:** Same pipeline (contrastive prompt, Gemini Flash, 3 reps) on both datasets.

| | Eedi (MCQ) | SmartPaper (open-ended) |
|---|---|---|
| Probe items | ρ=0.462 | ρ=0.768 |
| Full set | ρ=0.039 (50 items) | ρ=0.547 (140 items) |
| Generalizes? | No | Yes |
| Calibration offset | N/A | +0.43 (overestimates easiness) |

**Verdict: STRONGLY SUPPORTED.** MCQ difficulty depends on distractor quality — how well each wrong answer maps to a specific misconception. This requires knowledge of misconception-to-distractor mappings that LLMs only have for the most well-studied errors. Open-ended difficulty depends on general curriculum alignment, which LLMs have absorbed broadly. The models know *which* items are harder (rank order) but not *how hard* (absolute calibration offset of +0.40 for Indian student population).

---

## H15. Option shuffling adds diversity

**Literature basis:** CAPE (NAACL 2024) and others suggest randomizing answer order reduces position bias and increases prediction diversity.

**Our test:** 3 reps original order vs 3 reps shuffled (A→B/C/D permutations).

| Condition | Avg-pred ρ |
|-----------|-----------|
| Original 3 reps | 0.666 |
| Shuffled 3 reps | 0.617 |
| All 6 combined | 0.665 |

**Verdict: REJECTED.** Shuffling adds noise, not signal. The model is sensitive to presentation order in a way that degrades (not diversifies) predictions. More reps of the original order is preferable to shuffled reps.

---

## H16. Calibration anchors transfer across datasets

**Literature basis:** Providing example items with known difficulty as in-context anchors should help calibrate predictions. Showed +0.16 improvement on SmartPaper in prior session.

**Our test:** 4-fold CV on Eedi probe items (5 anchors / 15 test items per fold).

| Condition | CV mean ρ |
|-----------|----------|
| With anchors | 0.445 |
| Without anchors | 0.525 |

**Verdict: REJECTED for cross-dataset transfer.** Anchors hurt on Eedi despite helping on SmartPaper. The model's implicit calibration (from training data) is better than explicit numerical anchors. Providing IRT values may confuse the ability-stratified distribution format.

---

## H17. Two-stage elicitation (diverse context → prediction) outperforms single-stage

**Literature basis:** Inspired by debate/deliberation frameworks. Generate diverse perspectives first, then synthesize.

**Our test:** Stage 1 generates 5 diverse seeds at t=2.0, Stage 2 uses as context for prediction at t=1.5.

| Pipeline | Seed type | ρ |
|----------|-----------|---|
| Gemini→Gemini | cognitive_chains | 0.508 |
| Gemini→Gemini | buggy_analyses | 0.268 |
| Gemini→Gemini | error_perspectives | 0.241 |
| Gemini→Gemini | direct_baseline | 0.502 |

**Verdict: NO IMPROVEMENT but interesting mechanism.** Two-stage cognitive chains (0.508) ≈ direct baseline (0.502), but with lower variance (SD=0.067 vs 0.125). The two-stage approach *stabilizes* rather than *boosts*. Crucially: using diverse content as *context* (ρ=0.508) dramatically outperforms using it as *mechanism* (cognitive modeling, ρ=0.165). The model is better at integrating diverse perspectives than enacting them.

---

## H18. Same-model coherence matters in multi-stage pipelines

**Literature basis:** Novel finding, no direct literature precedent.

**Our test:** Cross-model two-stage experiments.

| Pipeline | cognitive_chains ρ | buggy_analyses ρ |
|----------|-------------------|-----------------|
| Gemini→Gemini | 0.508 | 0.268 |
| Gemini→Scout | 0.356 | -0.077 |
| Scout→Scout | 0.401 | 0.275 |

**Verdict: SUPPORTED.** Scout→Scout consistently outperforms Gemini→Scout. A model interprets its own outputs better than another model's outputs. But Gemini→Gemini still wins overall because Gemini produces higher-quality content at both stages.

---

## H19. Cognitive modeling prompts (enumerate-then-estimate) outperform direct and role-based prompts

**Literature basis:** Embretson (1998) cognitive design system approach — item difficulty is predictable from cognitive processing demands. Our own Phase 1 screening showed error-focused prompts outperform direct estimation (H12). New prompts test whether explicit cognitive feature enumeration before holistic estimation further improves predictions.

**Design space (Cognitive Scaffolding × Role Framing):**

|  | No Role | Teacher/Expert | Student Sim | Psychometrician |
|---|---|---|---|---|
| None (direct) | direct_clean: 0.365 (DBE) | teacher: 0.555 (SP) | simulation: 0.271 (Ph2) | — |
| Enumerate features | error_analysis: 0.596 (SP) | devil_advocate: 0.596 (SP) | contrastive: 0.584 (SP) | — |
| Cognitive modeling | misconception_holistic: 0.625 (SP) | prerequisite_chain: 0.686 (SP) | cognitive_profile: 0.599 (SP) | buggy_rules: 0.655 (SP) |
| Mechanistic computation | — | — | — | cognitive_load: 0.673 (SP) |

**Our test:** Phase 1, Gemini 3 Flash, 3 reps, SmartPaper 140 items.

| Prompt | Scaffolding level | Role | ρ | MAE | Bias |
|--------|------------------|------|---|-----|------|
| prerequisite_chain t=1.0 | cognitive modeling | teacher | 0.686 | 0.124 | -0.069 |
| cognitive_load t=2.0 | mechanistic | psychometrician | 0.673 | 0.152 | -0.074 |
| buggy_rules t=1.0 | cognitive modeling | psychometrician | 0.655 | 0.117 | +0.054 |
| misconception_holistic t=1.0 | cognitive modeling | none | 0.625 | 0.130 | -0.031 |
| cognitive_profile t=2.0 | cognitive modeling | student sim | 0.599 | 0.129 | -0.048 |
| error_analysis t=1.0 | enumerate features | none | 0.596 | — | — |
| devil_advocate t=1.0 | enumerate features | teacher | 0.596 | — | — |
| contrastive t=1.0 | enumerate features | none | 0.584 | — | — |
| teacher t=1.0 | none | teacher | 0.555 | — | — |

**Key findings:**

1. **Cognitive scaffolding is the dominant dimension.** Row averages: cognitive modeling ≈ 0.633, enumerate features ≈ 0.592, direct ≈ 0.555. Each step up adds ~0.04 ρ.

2. **Role framing is secondary.** Within the cognitive modeling row, spread is 0.599–0.686 — smaller than the row-to-row difference.

3. **buggy_rules achieves best calibration** (MAE=0.117, bias=+0.054) despite lower ρ than prerequisite_chain. The "enumerate specific buggy rules, then estimate holistically" mechanism produces well-calibrated predictions.

4. **Mechanistic computation is fragile.** cognitive_load's multiplicative cascade (ρ=0.673 on SP) assumes conditional independence between error sources — works on SmartPaper but cross-validation on DBE-KT22 pending.

5. **"Enumerate then estimate holistically" is the mechanism.** All cognitive modeling prompts share this structure: list specific cognitive features/errors/prerequisites first, then make a single holistic difficulty judgment. This consistently outperforms prompts that skip enumeration (direct) or that try to compute difficulty mechanistically.

**Cross-validation (DBE-KT22):**

| Prompt | SP ρ | DBE ρ | Δ |
|--------|------|-------|---|
| prerequisite_chain | 0.686 | 0.531 | -0.155 |
| cognitive_load | 0.673 | 0.469 | -0.204 |
| buggy_rules | — | PENDING | — |

**Verdict: SUPPORTED.** Cognitive modeling prompts consistently occupy the top ranks. The design space has two clear dimensions (scaffolding × role), with scaffolding being the primary lever. The "enumerate then estimate" mechanism is well-characterized and generalizes across the cognitive modeling row regardless of role framing.

---

## Summary: What Actually Predicts Item Difficulty?

### What works
1. **Item type**: Open-ended >> MCQ (ρ=0.55 vs ρ≈0)
2. **Cognitive scaffolding in prompts**: "Enumerate then estimate holistically" — cognitive modeling prompts (ρ≈0.63) > enumerate features (ρ≈0.59) > direct (ρ≈0.55). 3–4× improvement over generic prompts.
3. **Multi-sample averaging**: Consistent boost, especially for high-variance models
4. **Specific model knowledge**: Gemini 3 Flash and Llama-4-Scout have pedagogical content knowledge others lack
5. **Calibration via buggy_rules**: Best MAE (0.117) and minimal bias (+0.054) — psychometrician framing + cognitive feature enumeration

### What doesn't work
1. Direct estimation (r≈0 at scale)
2. Feature extraction + ML (r=0.06 at scale)
3. Classroom simulation (r=0.01)
4. Model uncertainty (r≈0)
5. Reasoning/deliberation (hurts)
6. Misconception hints (hurts)
7. Calibration anchors (hurt on Eedi)
8. Option shuffling (adds noise)
9. Student personas/backstories (add noise)
10. Bigger/smarter models (no correlation with success)

### The core finding
LLMs can estimate difficulty for open-ended items (ρ≈0.55) because it depends on general curriculum knowledge. They cannot estimate MCQ difficulty (ρ≈0) because it depends on distractor-specific misconception knowledge that only exists in training data for well-documented errors. Published positive results on MCQs likely reflect favorable item selection (items testing well-known misconceptions) or small evaluation sets.

### Running experiment
**105-item b_2pl Eedi test** (contrastive prompt, Gemini Flash, 3 reps, t=1.5): Tests whether signal holds across all 105 items with IRT parameters.

Script: `scripts/run_105_b2pl.py`. Output: `pilot/b2pl_105_experiment/`.

**RESULT (2026-02-02):**
| Metric | Value |
|--------|-------|
| Per-rep ρ | 0.160, 0.073, 0.079 |
| Per-rep mean | 0.104 ± 0.039 |
| Averaged (3-rep) ρ | 0.114 |
| p-value | 0.246 |
| 95% CI | [-0.072, 0.297] |
| n items | 105 |

**Verdict: CONFIRMED NULL.** ρ=0.114, CI includes zero. The signal on the original 36-item subset (ρ≈0.50–0.58) was driven by item selection, not the method. The contrastive prompt does not generalize to the full 105-item set. This confirms H13 and establishes the paper narrative as Path A (open-ended vs MCQ contrast).

---

## Paper Completion Plan

### Status: Experiments COMPLETE (2026-02-02)
The 105-item experiment returned ρ=0.114 (null). Paper narrative is Path A: open-ended vs MCQ contrast.

### What exists
- `results-site/paper.html` — HTML paper draft with Plotly figures, but narrative is a patchwork of old claims (temperature cliff, 4× increase) and corrected numbers. Needs clean rewrite.
- `results-site/replication_dashboard.html` — visualization of null wall results
- `results-site/rsm_analysis.html` — RSM analysis visualization
- `.claude/handoffs/2026-02-02-hypothesis-registry.md` — this file, all 18 hypotheses documented

### Paper narrative: CONFIRMED as Path A
- Title: "LLMs Can Rank Open-Ended but Not MCQ Item Difficulty: A Systematic Evaluation"
- Core story: Open-ended ρ=0.55 (n=140) vs MCQ ρ≈0 (n=105). The difference is item type, not method.
- SmartPaper vs Eedi contrast is the centerpiece.
- Negative results on MCQs are important and publishable — they contradict optimistic claims from studies with small/curated test sets.
- The System 1 finding and model survey are strong secondary results.
- The 36-item "success" dissolving at 105 items is itself an important finding about overfitting in small evaluation sets.

### Remaining work items
1. **(Optional) DeepSeek on 140 SmartPaper items** — would add a second model confirming the open-ended finding. Strengthens the SmartPaper story.
2. **Rewrite paper.html** — clean narrative for Path A
3. **Clean figures** — Eedi null wall, SmartPaper success, generalization contrast, small-set overfitting illustration
4. **Convert to LaTeX** — AIED 2026 submission format

### What is NOT needed
- More Eedi experiments (18 hypotheses is comprehensive)
- More models (15+ tested)
- More temperature sweeps (covered)
- More prompt variants (covered)
- Held-out Eedi confirmation beyond the 105 — that IS the confirmation
