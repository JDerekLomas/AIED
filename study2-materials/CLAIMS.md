# Paper Claims → Evidence Map

**Paper:** "It's Hard to Know How Hard It Is: Mapping the Design Space of LLM Item Difficulty Estimation"
**Venue target:** AIED 2026
**Source of truth:** `paper/main.tex` (LaTeX version)
**Last updated:** 2026-02-03 (opus45/batching removed, gemini per-item runs completed)

---

## The Argument in Five Sentences

LLMs can rank-order item difficulty for open-ended educational items (ρ=0.55–0.69, n=140–168) but the advantage of structured prompts attenuates on MCQs. The difference is not prompt, model, or method — we tested 13 hypotheses across 8 models and 15 prompt framings. Prompt framing (especially "enumerate structural features then estimate") is the primary lever on datasets where estimation works at all. **On the BEA 2024 benchmark (595 medical MCQs), our zero-shot approach with linear calibration achieves RMSE=0.280 and τ=0.31 — matching or slightly exceeding the best supervised system (UnibucLLM: RMSE=0.281, τ=0.28).** Methodologically, small samples (n<50) are unreliable for validation, and unbalanced hyperparameter sweeps produce Simpson's paradox artifacts — we recommend two-stage sequential DOE.

---

## Methodological Contributions (§6.4 in paper)

The paper now explicitly claims three methodological contributions, consolidated in §6.4 "Methodological Recommendations":

| Contribution | Evidence | Paper Location | Recommendation |
|--------------|----------|----------------|----------------|
| **Simpson's paradox in unbalanced sweeps** | η²=0.74 (unbalanced) → η²=0.001 (balanced) | §6.4 ¶1 | Report within-config contrasts, not marginals |
| **Two-stage sequential DOE** | Retrospective efficiency analysis | §6.4 ¶2 | Screen cheap (1 rep), then balanced factorial on survivors |
| **≥100 items for validation** | Bootstrap: 2.5% of n=20 subsets reach ρ≥0.50 | §6.4 ¶3 | Treat n<50 as exploratory only |

These are also summarized in the **Conclusion** as finding #5: "Small samples mislead."

**What we did right (DOE perspective):**
- Full coverage before optimization (revealed temperature doesn't matter)
- Multiple replications (detected instability)
- Cross-dataset confirmation (caught attenuation)
- Correct outcome metric (Spearman's ρ for rank-order)

**What we did wrong (DOE perspective):**
- Unbalanced temperature sweeps (created Simpson's paradox)
- No stopping rule (wasted ~12% of budget on poor prompts)
- Single model in screening (prompt rankings partially reverse across models)
- Underpowered feature regression (N=15 prompts, 6 predictors)

---

## Claim 1: Prompt framing is the dominant factor

**Strength: STRONG**

Counting prompts (prerequisite_chain rho=0.686, cognitive_load rho=0.673) outperform direct teacher judgment (rho=0.555) by 0.12–0.13 on SmartPaper. This exceeds the meta-analytic average for expert teacher judgment (r=0.63–0.66; Sudkamp et al., 2012).

| Evidence | Script | Output | Key numbers |
|----------|--------|--------|-------------|
| Phase 1 screening: 7 framings × 2 temps × 3 reps × 140 items | `scripts/run_prompt_screening_g3f.py` | `pilot/prompt_framing_experiment/` | Table in paper §3.1 |
| Extended screening: 15 framings × 2-4 temps | same | same | `pilot/prompt_framing_experiment/README.md` |
| RSM visualization | `scripts/rsm_analysis.py` | `pilot/prompt_framing_experiment/rsm_extended.html` | Interactive 3D surface |

**Caveat that must be stated clearly:** The counting advantage attenuates from 0.131 (SmartPaper) to 0.010 (DBE-KT22) in Phase 3. The claim should be: "structured prompts beat direct judgment" (robust), not "counting prompts are best" (dataset-dependent).

---

## Claim 2: The method transfers across datasets — but not to all item types

**Strength: STRONG (the most important finding)**

| Dataset | Type | n | Best ρ | τ | Significant? |
|---------|------|---|--------|---|-------------|
| SmartPaper (India) | Open-ended | 140 | 0.686 | — | p < 1e-19 |
| DBE-KT22 (South Africa) | MCQ, 27 KCs | 168 | 0.580 | — | p < 0.001 |
| **BEA 2024 (USA)** | **MCQ, USMLE** | **595** | **0.445** | **0.310** | **p < 1e-10** |
| ~~Eedi (UK)~~ | ~~MCQ, misconception~~ | ~~105~~ | ~~0.114~~ | — | ~~ns (dropped from paper)~~ |

| Evidence | Script | Output |
|----------|--------|--------|
| SmartPaper Phase 1 | `scripts/run_prompt_screening_g3f.py` | `pilot/prompt_framing_experiment/` |
| DBE-KT22 Phase 3 | `scripts/run_dbe_kt22_validation.py` | `pilot/dbe_kt22_validation/` |
| **BEA 2024 validation** | see `pilot/bea2024/README.md` | `pilot/bea2024/` |
| ~~Eedi 105-item confirmation~~ | ~~`scripts/run_confirmation.py`~~ | ~~dropped from paper~~ |

**Pattern across datasets:** The item analysis advantage (structured prompts over teacher baseline) is strongest on open-ended items (+0.13 on SmartPaper), moderate on broad MCQs (+0.06 on DBE-KT22), and negligible on domain-specific MCQs (all prompts ρ≈0.45 on BEA). Simple teacher judgment is equally effective for MCQs.

**What explains the attenuation?** Open-ended items vary difficulty across knowledge domains; MCQs often vary difficulty through distractor design within a domain. LLMs can estimate structural prerequisites visible in item text but cannot predict which distractors will trap students.

---

## Claim 3: Small probe sets are unreliable (methodological warning)

**Strength: STRONG — NOW IN PAPER §6.4 "Methodological Recommendations"**

| Evidence | Script | Output | Key numbers |
|----------|--------|--------|-------------|
| Probe (n=20) vs full (n=105) | `scripts/run_confirmation.py` | `pilot/confirmation_experiment/` | rho=0.46 → 0.11 |
| Probe inflation analysis | `scripts/analyze_probe_inflation.py` | In-script output | 2 outliers drove 20-item signal |
| Bootstrap: only 2.5% of 20-item subsets reach rho≥0.50 | same | same | Paper §6.4 |
| SmartPaper probe→full holds | `scripts/run_smartpaper_expansion.py` | `pilot/smartpaper_expansion/` | rho=0.77 → 0.55 (attenuates but holds) |

This is a genuine contribution to the field — many published results use <50 items. **Recommendation in paper:** Use at least 100 items for validation; treat smaller samples as exploratory only.

---

## Claim 4: Counting prompts require capable models (interaction)

**Strength: MODERATE**

Phase 2 shows prerequisite_chain helps Gemini Flash (+0.083) but hurts Llama-8B (-0.256).

| Evidence | Script | Output |
|----------|--------|--------|
| Phase 2 model survey | `scripts/run_model_survey.py` | `pilot/model_survey/` |
| 7 models × 3 prompts × 3 reps | same | Paper §3.2 |

**Caveat:** Only tested on SmartPaper. The interaction may not generalize.

---

## Claim 5: Theory-grounded prompts work — but only the right kind

**Strength: STRONG (upgraded — this is a core theoretical contribution)**

### The prediction

Five learning science traditions were operationalized as prompts (see `handoffs/2026-02-02-theory-prompts.md`). The pre-registered prediction was: prompts that operationalize theory as *counting tasks* (compatible with System 1 / pattern-matching) should outperform prompts that require *cognitive simulation* (System 2 / deliberation).

### The scorecard

| Prompt | Theory | Best ρ | vs Baseline (+.555) | Prediction |
|--------|--------|--------|---------------------|------------|
| prerequisite_chain | KC Theory (Koedinger) | 0.686 | +0.131 | Confirmed: "most likely to help" |
| cognitive_load | CLT (Sweller) | 0.673 | +0.118 | Exceeded: was "moderate bet" |
| buggy_rules | Buggy Rules (Brown & Burton) | 0.655 | +0.100 | Confirmed |
| misconception_holistic | Conceptual Change (Chi) | 0.636 | +0.081 | Confirmed |
| error_analysis | Buggy Rules | 0.596 | +0.041 | Partial |
| cognitive_profile | Conceptual Change | 0.586 | +0.031 | Partial |
| classroom_sim | Student Simulation (ESS) | 0.562 | +0.007 | Confirmed null |
| familiarity_gradient | Frequency/Familiarity | 0.541 | −0.014 | Failed: was "interesting" |
| imagine_classroom | Student Simulation | 0.514 | −0.041 | Confirmed null |
| verbalized_sampling | Student Simulation | 0.496 | −0.059 | Confirmed null |
| error_affordance | Buggy Rules | 0.493 | −0.062 | **Failed: was "most likely to help"** |

### The theoretical insight (revised after examining actual prompt texts)

The initial hypothesis was that **item-structural prompts** work while **student-simulation prompts** fail. But `buggy_rules` (ρ=0.655) explicitly simulates student cognition ("identify known buggy rules — systematic procedural errors students commonly make") and is the #3 prompt. The item-structure vs. student-simulation boundary doesn't hold.

Reading the actual prompt texts reveals the real pattern: **structured enumeration as cognitive scaffolding**.

**Winners (ρ=0.65–0.69) all share the same prompt architecture:**
1. Decompose the item into parts (steps, prerequisites, elements)
2. Enumerate specific things about each part
3. Make a holistic final judgment grounded in the enumeration

| Prompt | What it enumerates | Structure |
|--------|-------------------|-----------|
| prerequisite_chain | Prerequisites per item | "List prerequisites → Count → Estimate" |
| cognitive_load | Elements in working memory | "List elements → Count → Estimate" |
| buggy_rules | Buggy rules *per procedural step* | "Step 1: List steps → Step 2: Bugs per step → Step 3: Population → Step 4: Holistic estimate" |

**Losers either enumerate without structure or skip enumeration entirely:**

| Prompt | ρ | Problem |
|--------|---|---------|
| error_affordance | 0.493 | Enumerates errors *globally* — no per-step decomposition, just a flat list |
| classroom_sim | 0.562 | No enumeration — "mentally simulate each student," a single holistic act |
| verbalized_sampling | 0.496 | Adds noise — 5 role-played perspectives that dilute rather than sharpen |
| familiarity_gradient | 0.541 | No enumeration — a single typicality judgment on a 4-point scale |

### The critical comparison: buggy_rules vs. error_affordance

Both are grounded in Brown & Burton's theory of systematic student errors. Both involve counting errors. But:
- **buggy_rules** decomposes the item into procedural steps *first*, then identifies bugs *per step*
- **error_affordance** lists errors globally ("count plausible ways to get this wrong")

The structured decomposition is what makes the difference — it forces the model to attend to item structure before judging difficulty. The *content* of what's enumerated (item features vs. student errors) matters less than whether the enumeration is **anchored to the item's structure**.

### Why this matters theoretically

1. **It's not item-structure vs. student-simulation.** Both can work. buggy_rules simulates students and works great. What matters is whether the prompt creates a structured, step-by-step decomposition of the item.

2. **Enumeration creates an intermediate representation that anchors the estimate.** The list of features/steps/bugs serves as a scaffold — without it, the model defaults to a poorly-calibrated holistic guess. This is consistent with research on structured prediction in LLMs: decomposition improves calibration even when the sub-judgments aren't individually accurate.

3. **The "System 1" finding is refined.** Deliberation hurts when it takes the form of *open-ended reasoning* (thinking tokens, CoT) or *unstructured simulation* (imagine a classroom). Deliberation *helps* when it takes the form of *constrained enumeration* (list prerequisites, count elements, identify bugs per step). The effective prompts are "System 1.5" — guided decomposition, not free reasoning.

4. **The theory-practice bridge is about *operationalization*, not *theory choice*.** KC Theory, Cognitive Load Theory, and Buggy Rules all produced good prompts — but only when operationalized as structured enumeration. The same theory (Brown & Burton) produced both the #3 prompt (buggy_rules) and the worst prompt (error_affordance). The theoretical tradition doesn't predict prompt quality; the prompt architecture does.

### Connection to System 1 / System 2

| Evidence | Key numbers |
|----------|-------------|
| Gemini Flash no-thinking: ρ=0.604 | H11 in hypothesis registry |
| Gemini Flash thinking_budget=1024: ρ=0.467 | |
| Gemini Pro (mandatory thinking): ρ=0.052 | |
| DeepSeek-Reasoner: constant output (NaN) | |
| All structured reasoning variants ≤ direct: H5 | |
| Structured enumeration prompts > holistic prompts | Phase 1 screening |

The System 1/2 finding and the enumeration finding are complementary: LLMs estimate difficulty best when given *just enough structure* to decompose the item without being forced into open-ended reasoning. Too little structure (direct prompt, familiarity_gradient) → uncalibrated holistic guess. Too much freedom (thinking tokens, verbalized_sampling) → the model wanders into unreliable simulation. The sweet spot is constrained enumeration: "list these specific things, then judge."

---

## Claim 6 (secondary): Temperature is a minor factor + Simpson's paradox warning

**Strength: STRONG — NOW IN PAPER §6.4 "Methodological Recommendations"**

| Evidence | Script | Output | Key numbers |
|----------|--------|--------|-------------|
| Balanced ANOVA (T=1.0 vs T=2.0, all 15 prompts) | `scripts/design_space_analysis.py` | `pilot/prompt_framing_experiment/figures/` | F=0.04, p=.84, η²=.001 |
| Paired within-prompt comparison | same | same | Mean Δρ=+0.004, t=0.52, p=.61 |
| Simpson's paradox diagnosis | same | same | Unbalanced ANOVA had η²=.74 (spurious) |

### What the original analysis got wrong

The unbalanced ANOVA (all temps, 3-rep configs) reported temperature as the dominant factor (η²=.74, p<.001). This was a **Simpson's paradox**: only the best-performing prompts (prerequisite_chain, cognitive_load, devil_advocate) were tested at low temperatures (0.25, 0.5), inflating the apparent low-temp advantage. When restricted to T=1.0 vs T=2.0 (the only balanced comparison), temperature explains 0.1% of variance.

### What holds up

- **Prompt family is the dominant factor**: η²=.46, p=.03 in balanced ANOVA
- **No family×temperature interaction**: F=0.17, p=.97 — temperature doesn't differentially affect any prompt family, so family-specific temperature tuning is unnecessary
- **6/15 prompts improve at T=2.0**, including some of the best (cognitive_load +.056, prerequisite_chain +.021, error_analysis +.048)

### Methodological lesson — NOW IN PAPER

**This is now explicitly stated in §6.4.** Recommendation: When hyperparameters are swept selectively for promising configurations, do not report marginal effects. Report within-configuration contrasts or restrict analysis to balanced subsets.

---

## Design Space Analysis: What Each Model Taught Us

**Script:** `scripts/design_space_analysis.py --figures`
**Data:** `pilot/prompt_framing_experiment/rsm_data.json` (34 three-rep design points)

### Model 1: Two-way ANOVA (ρ ~ family + temperature)

**What it showed:** Prompt family matters (η²=.46); temperature doesn't (η²=.001); no interaction (p=.97).

**What we learned:** Temperature is a free parameter — set it for other reasons (diversity, cost) without sacrificing accuracy. Prompt design is the only lever that moves the needle. This simplifies practical recommendations: pick a good prompt family, don't worry about temperature tuning.

**Limitation:** Families are researcher-assigned groupings. The ANOVA can't tell us *why* enumeration and cognitive_modeling families outperform others.

### Model 2: Quadratic response surface (ρ ~ family + temp + temp²)

**What it showed:** Quadratic term is not significant (F=0.23, p=.63). No evidence for an optimal interior temperature. Family×temp² interaction also not significant (p=.40).

**What we learned:** The temperature–performance relationship is flat, not U-shaped or inverted-U. There's no "sweet spot" to find. This is a *negative result about the design space* — it's flat in the temperature dimension, meaning response surface methodology is overkill for temperature optimization in this setting.

**Limitation:** Only 4 temperature levels (0.25, 0.5, 1.0, 2.0) and severely unbalanced across prompts. The quadratic has little power to detect curvature.

### Model 3: Prompt-feature regression (ρ ~ structural features)

**What it showed:** R²=.39 but F=0.84, p=.57. No individual feature (word count, enumeration steps, theory grounding, reasoning steps, population context, holistic judgment) significantly predicts ρ.

**What we learned:** We cannot currently reverse-engineer what makes a prompt work from its surface structural features. The "good prompt recipe" remains empirical. This has two implications:
1. **For practitioners:** There's no shortcut — you must test prompts empirically, not assemble them from features.
2. **For theory:** The mechanism of prompt effectiveness likely operates at a deeper level than our feature coding captures (e.g., how the prompt interacts with the model's internal representations of item content).

**Limitation:** N=15 prompts with 6 predictors is severely underpowered. The features are also hand-coded and may miss the relevant dimensions.

### Cross-level analysis: Do prompt features predict temperature sensitivity?

**What it showed:** The correlation between n_reasoning_steps and temperature slope is r=-.08, p=.78. No feature predicts which prompts benefit from high vs low temperature.

**What we learned:** The theoretically motivated hypothesis — "structured prompts degrade at high temperature because temperature corrupts enumerated output" — is specifically falsified. cognitive_load (3 steps) *improves* at T=2.0 while buggy_rules (3 steps) degrades. Temperature sensitivity is prompt-specific in ways our features can't capture.

### Summary table

| Model | Key finding | For the paper |
|-------|-------------|---------------|
| Balanced ANOVA | Family matters, temperature doesn't | Main result; report η² for both |
| Quadratic RSM | No curvature; design space is flat in temp | Brief mention; justifies not sweeping temp |
| Feature regression | Can't predict ρ from prompt structure | Important null; motivates empirical screening |
| Cross-level (steps × temp) | Can't predict temp sensitivity either | Footnote; falsifies "structure needs low temp" |
| Simpson's paradox diagnosis | Unbalanced designs create artifacts | Methodological warning for the field |

---

## BEA 2024 Shared Task — NOW IN PAPER

### Overview

We ran our difficulty estimation prompts on the BEA 2024 shared task dataset (667 USMLE medical MCQs, 595 text-only after excluding 72 image-based items) to validate cross-domain transfer and compare against published baselines.

**Location in paper:** Section 5.4 "BEA 2024 Benchmark Comparison" + Abstract

### Results

| Metric | Our Result | UnibucLLM (best published) | BEA Winner (EduTec) |
|--------|-----------|---------------------------|---------------------|
| RMSE (calibrated) | **0.280** | 0.281 | 0.299 |
| Kendall τ | **0.310** | ~0.28 | (not reported) |
| Spearman ρ | 0.445 | (not reported) | (not reported) |

### Key Finding

Zero-shot prompting with linear calibration matches the best supervised system (SVR+BERT features) despite using prompts designed for K-8 education rather than medical licensing exams. This suggests the rank-order signal from directed item analysis transfers across domains.

### Calibration Details

Raw predictions have systematic bias: models overestimate student performance.

**Calibration learned from train set (n=466):**
```
b_calibrated = 0.258 × b_predicted + 0.768
```

This linear scaling corrects the bias while preserving rank order.

### Data Files

- `pilot/bea2024/submission_gemini_ensemble.csv` — Raw predictions (uncalibrated, IRT b scale)
- `pilot/bea2024/submission_gemini_calibrated.csv` — Calibrated predictions
- `pilot/bea2024/README.md` — Full methodology documentation
- `pilot/bea2024/teacher_t1.0/`, `prerequisite_chain_t1.0/`, `buggy_rules_t1.0/` — Raw response files

### Experimental Conditions

| Condition | Model | Prompt | Reps | Items with predictions |
|-----------|-------|--------|------|------------------------|
| teacher_t1.0 | Gemini 3 Flash | teacher | 3 | 595 |
| prerequisite_chain_t1.0 | Gemini 3 Flash | prerequisite_chain | 3 | 595 |
| buggy_rules_t1.0 | Gemini 3 Flash | buggy_rules | 2 | 595 |
| scout_teacher_t1.0 | Llama-4-Scout | teacher | 3 | 595 |
| scout_prerequisite_chain_t1.0 | Llama-4-Scout | prerequisite_chain | 3 | 580 |
| scout_buggy_rules_t1.0 | Llama-4-Scout | buggy_rules | 3 | 399 |

### Comparison Notes

1. **Our approach**: Zero-shot prompting with prompts designed for K-8 Indian education
2. **BEA winners**: Supervised ML (BERT embeddings + SVR, gradient boosting, etc.)
3. **Fair comparison?**: Both use train set — they for feature extraction/training, we for calibration only
4. **Key finding**: Zero-shot + calibration matches supervised approaches

### Why Kendall τ matters

BEA teams reported different metrics. UnibucLLM reported τ ≈ 0.28; we achieve τ = 0.31. The relationship between τ and ρ is roughly τ ≈ 0.67ρ for bivariate normal data (our ratio: 0.70).

---

## What's NOT in the paper but should be considered

### Indian State Assessment
- Script exists: `scripts/run_indian_state_assessment.py`
- Status: Unknown
- Value: Another open-ended dataset would confirm the item-type finding

---

## Narrative recommendation — IMPLEMENTED

**Current title:** "It's Hard to Know How Hard It Is: Mapping the Design Space of LLM Item Difficulty Estimation"

The paper now uses the "design space mapping" framing, which positions:
- The flat-temperature finding as a mapping result (we charted the terrain)
- The Simpson's paradox finding as a methodological contribution about design-space exploration itself
- The probe-set reliability finding as a generalizable warning

**Structure implemented:**
1. Results section leads with empirical findings (item analysis prompts work, the advantage attenuates on MCQs)
2. Discussion explains mechanisms (why item analysis works, directed vs undirected analysis)
3. **New §6.4 "Methodological Recommendations"** surfaces three DOE lessons:
   - Simpson's paradox warning for unbalanced designs
   - Two-stage sequential DOE recommendation
   - ≥100 items validation requirement
4. Conclusion now lists **five findings**, with #5 being "Small samples mislead"
5. Abstract updated to mention methodological contributions + "design of experiments" keyword

---

## Practical Takeaways

### For practitioners wanting to estimate item difficulty with LLMs

1. **Pick the right prompt family, then stop tuning.** Enumeration-style prompts (prerequisite_chain, cognitive_load) and cognitive modeling prompts (buggy_rules, misconception_holistic) outperform baselines by ρ≈0.10. Temperature, number of repetitions beyond 3, and model choice (among capable models) are second-order effects.

2. **Temperature is a free parameter.** Set it based on other considerations (cost, output diversity, downstream pipeline). Don't sweep it — the expected improvement is Δρ<0.01.

3. **Use at least 100 items for validation.** Our probe-set analysis (Claim 3) shows that correlations from 20 items are unreliable. If you can't validate on 100+ items, don't trust the result.

4. **Check your item type before investing.** LLM difficulty estimation works for items where difficulty varies *across topics/skills* (open-ended items, broad-coverage MCQs) but fails for items where difficulty depends on *distractor-specific misconception prevalence* (Eedi-style diagnostic MCQs). If your items are the latter type, no amount of prompt engineering will help.

5. **Don't trust marginal means from unbalanced experiments.** If you only sweep hyperparameters for promising configurations, your marginal analyses will show Simpson's paradox artifacts. Always report within-configuration contrasts.

### For researchers studying LLM evaluation methods

1. **The design space is low-dimensional.** Of the factors we tested (15 prompt framings × 4 temperatures × 3 repetitions × 7 models × 3 datasets), only two dimensions matter: prompt family and dataset/item type. Everything else is noise.

2. **Surface features don't predict prompt quality.** Word count, enumeration steps, theory grounding, reasoning steps, and population context collectively explain 39% of variance but are not significant (F=0.84, p=.57, N=15). The mechanism by which prompts work is not accessible through structural analysis at our current granularity.

3. **Temperature doesn't interact with anything.** Not with prompt family (p=.97), not with prompt features (r=-.08), not with curvature (p=.40). This is unusual — most NLP papers find temperature effects — and may be specific to difficulty estimation tasks where the model is producing a single scalar judgment rather than generating text.

---

## Limitations

### Statistical power

- **N=15 prompts** is the effective sample size for feature regression and cross-level analyses. With 6 predictors, we have ~2.5 observations per parameter. Most null results in Analysis 3 could be Type II errors.
- **N=34 design points** for the ANOVA, with unbalanced cells. Only 3 prompts have >2 temperature levels.
- **N=3 datasets** for the item-type finding. The item-type moderator (Claim 2) rests on a qualitative comparison across 3 datasets, not a parametric test. A 4th dataset (BEA 2024) would help.

### Confounds we can't rule out

- **Prompt family vs. prompt identity:** Families contain 1–4 prompts each. Decomposition has only teacher_decomposed; baseline has only teacher. Family effects may be driven by individual prompts, not the family category.
- **Dataset vs. item type vs. population vs. domain:** SmartPaper (India, open-ended, multi-subject), DBE-KT22 (South Africa, MCQ, multi-KC), and Eedi (UK, diagnostic MCQ, misconception-targeted) differ on multiple dimensions simultaneously. We attribute the Eedi null to item type, but it could be domain, population, or an interaction.
- **Model vintage:** All experiments used Gemini 2.0 Flash (Phase 1/3) or a specific set of 7 models (Phase 2). Results may not generalize to future model generations.

### Coverage gaps

- **No temperature levels between 0 and 0.25.** The response surface below T=0.25 is unexplored.
- **No prompt families tested across all temperatures.** The Simpson's paradox arose because we didn't run a full factorial. The balanced analysis (T=1.0 vs T=2.0) is valid but loses the low-temp data entirely.
- **Feature coding is subjective.** "Has theory grounding" and "n_reasoning_steps" are hand-coded by the researchers. Different coders might assign different values, especially for borderline cases.

---

## Could We Have Done This More Efficiently?

### What we actually did

A roughly fixed-allocation design: 15 prompts × 2–4 temps × 3 reps, with the temperature range expanded for promising prompts. Total: ~102 API configurations × 140 items ≈ 14,280 item-level API calls for Phase 1.

### Multi-armed bandits (Thompson sampling, UCB)

A bandit approach would have adaptively allocated API budget toward promising prompts and away from losers.

**What it would have saved:** teacher_decomposed (ρ≈0.46) and familiarity_gradient (ρ≈0.48) were clearly poor by rep 1. Those ~1,680 item-calls (~12% of budget) could have been reallocated to more temperature levels for top prompts.

**What it would have cost us:**
- With 15 arms and noisy rewards (ρ estimated from 140 items per pull), the ρ difference between the best prompt (prerequisite_chain, 0.686) and the 5th-best (misconception_holistic, 0.636) is only 0.05 — within noise for a single run. A bandit would need substantial exploration before reliable exploitation.
- The full factorial gave us the data to *diagnose* the Simpson's paradox. A bandit optimizing for best-arm identification would have converged on prerequisite_chain at low temp and never revealed that temperature doesn't actually matter.
- **Bandits optimize for finding the best arm. Our contribution is mapping the design space.** We need coverage, not convergence. The null results (temperature doesn't matter, features don't predict ρ) are as important as identifying the best prompt.

### What would have been most efficient: two-stage sequential DOE

**Stage 1 — Cheap screening (1 rep, T=1.0, all prompts):** 15 prompts × 1 rep × 140 items = 2,100 calls. Eliminates the bottom half. Cost: ~15% of total budget.

**Stage 2 — Full factorial on survivors (3 reps, all temps, top 7–8 prompts):** 8 prompts × 4 temps × 3 reps × 140 items = 13,440 calls. Balanced design avoids Simpson's paradox entirely. Cost: ~85% of total budget.

**What this would have gained:**
- Balanced data from the start — no Simpson's paradox to diagnose post hoc
- More temperature levels for the prompts we care about
- Same total budget, better statistical power where it matters

**What this would have lost:**
- Full coverage of the design space (we wouldn't know the exact ρ for teacher_decomposed at T=2.0 — but we don't really need to)
- The ability to report all 15 prompts × all temps (less complete but more informative tables)

### Bayesian optimization

For the continuous temperature dimension, Bayesian optimization (Gaussian process surrogate + expected improvement acquisition) could have found the optimal temperature per prompt with fewer evaluations. But since the temperature surface turned out to be flat, this would have been wasted sophistication — BO is most valuable when there's a sharp optimum to find.

### Verdict — NOW IN PAPER

The two-stage sequential DOE would have been the right approach. It's simpler than bandits, doesn't require a surrogate model, and directly addresses the balance problem that created our main analytical headache. **This is now stated in §6.4 "Methodological Recommendations":** screen broadly with minimal replication first, then invest in balanced factorial designs for survivors.

---

## Claim 7: Prompt rankings partially generalize across models — but the best prompt is model-dependent

**Strength: MODERATE**

Phase 2 model survey tested 5 prompts × 7 models. Filtering to configs with n_items ≥ 100:

| Model | teacher | cognitive_load | prerequisite_chain | simulation |
|-------|---------|---------------|-------------------|------------|
| **Gemini Flash** | 0.550 | 0.615 | **0.658** | 0.163 |
| GPT-4o | **0.494** | — | — | — |
| Llama-3.3-70B | **0.480** | 0.403 | 0.396 | 0.123 |
| Llama-4-Scout | **0.474** | 0.400 | 0.345 | -0.225 |
| Gemma-3-27B | **0.501** | 0.500 | — | — |
| Llama-3.1-8B | **0.300** | 0.214 | 0.044 | 0.128 |

*Note: Gemini values updated 2026-02-03 after completing all per-item runs (140 items × 3 reps × 5 prompts).*

| Evidence | Script | Output | Key numbers |
|----------|--------|--------|-------------|
| Phase 2 model survey: 7 models × 5 prompts | `scripts/run_model_survey.py` | `pilot/model_survey/survey_results.json` | Table above |

**What generalizes across models:**
- Simulation prompts are consistently worst (ρ ≤ 0.16 on every model, often negative)
- All models achieve ρ > 0 with the teacher baseline on SmartPaper
- The gap between best and worst prompt is large (0.3–0.5) on every model
- Structured prompts (prerequisite_chain, cognitive_load) are best on Gemini Flash

**What does NOT generalize:**
- prerequisite_chain is the best prompt on Gemini (0.658) but drops below teacher on most other models
- Structured enumeration prompts (prerequisite_chain, cognitive_load) help frontier models specifically; smaller models do better with the simpler teacher prompt
- The 8B model (Llama-3.1-8B) cannot follow complex prompt instructions at all (prerequisite_chain ρ=0.044)

**Implication for cheap-model screening:** You can use a cheap model to identify prompts that *definitely don't work* (simulation consistently fails). But you **cannot** use a cheap model to identify the *best* prompt — the ranking of the top 3 prompts reverses between model families. A screening workflow should: (1) use any model to eliminate bad prompt families, then (2) test the top candidates on the target model.

---

## Remaining work before submission

1. ~~**Convert to LaTeX** for AIED submission format.~~ ✅ DONE — `paper/main.tex`
2. ~~**Add methodological recommendations section.**~~ ✅ DONE — §6.4 added
3. **Generate figures:** Need PDF figures for:
   - `figures/fig1_screening.pdf` — prompt screening results
   - `figures/fig2_temperature.pdf` — temperature × prompt interaction
   - `figures/fig3_model_heatmap.pdf` — model × prompt heatmap
   - `figures/fig4_cross_dataset.pdf` — cross-dataset comparison
   - `figures/fig5_by_subject.pdf` — SmartPaper by subject
4. **Verify all numbers in paper match evidence.** Cross-check tables against `pilot/` output files.
5. **Check AIED 2026 deadline and format requirements.**
6. ~~**Optional: BEA 2024 dataset.**~~ ✅ DONE — Full analysis complete, results in paper §5.4 and abstract. RMSE=0.280 matches best published (0.281), τ=0.31 exceeds UnibucLLM (0.28). See `pilot/bea2024/README.md` for full documentation.
