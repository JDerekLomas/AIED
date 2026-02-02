# Research Journey — 2026-02-02

## Session Goals
Continue RSM difficulty estimation experiments. Three new approaches plus Groq paid-tier cross-model retest.

## Timeline

### Prior State (from 2026-02-01 handoff)
- Teacher perspective experiment complete: **ρ=0.095** — essentially no correlation with classical difficulty. Disappointing.
- Student simulation (200 sims × 100 items) was running on Gemini Flash.
- RSM experiment had tested 11 configs. Best: teacher_prediction + t=1.5 → **ρ=0.673** (single run).

### Today's Starting Point
- Metaprompt sweep (`test_metaprompt_sweep.py`) running in terminal — 7 prompt variants × 3 temperatures × 3 reps
- Best results so far: v3_contrastive t=1.5 **ρ=0.577±0.075**, v5_error_analysis t=1.8 **ρ=0.650** (single rep)
- Cross-model test showed: Gemini Flash >> Llama-70B (0.27) >> Qwen/4o-mini (≈0)

### Work Done Today

**1. Strategy Synthesis**
Reviewed all accumulated results and identified the optimization principles:
- ~~High temperature (1.5–2.0) is critical~~ → Later shown to be flat on Eedi with good prompt; model × prompt × temp interaction (see Phase 13)
- Error-focused prompt framing beats generic teacher prediction
- Model capability is the binding constraint (Gemini Flash >> everything else)
- Aggregation across reps always improves ρ
- Documented in `.claude/handoffs/2026-02-02-optimization-strategy.md`

**2. Metaprompt Sweep Analysis (from cached data)**
Computed full results matrix from 15 completed configs:

| Rank | Config | Mean ρ |
|------|--------|--------|
| 1 | v5_error_analysis t=2.0 | 0.604 ± 0.062 |
| 2 | v3_contrastive t=1.5 | 0.577 ± 0.075 |
| 3 | v5_error_analysis t=1.8 | 0.566 ± 0.108 |
| 4 | v3_contrastive t=2.0 | 0.562 ± 0.084 |
| 5 | v8_imagine_classroom t=2.0 | 0.547 (1 rep) |

Broken: v6_distractor_first (parse failures), v7_comparative_difficulty (constant outputs → NaN)

**3. Updated Cross-Model Script**
- Reduced Groq sleep from 2.1s to 0.5s (paid account)
- Script already had groq_llama4scout defined
- Ready to run: `python3 scripts/test_cross_model.py groq_llama4scout groq_llama70b`

**4. Two-Stage Backstory Experiment** (RUNNING)
- Created `scripts/test_two_stage.py`
- Stage 1: 5 vivid student backstories generated at t=2.0 (e.g., "Leo Thorne, competitive football fan who repeats incorrect calculations out of pride")
- Stage 2: Each backstory used as context for teacher prediction at t=1.5
- 5 backstories × 20 items × 3 reps = 300 calls
- **Hypothesis:** Explicit student diversity via backstories > implicit diversity via temperature

**5. Cognitive Modeling Experiment** (RUNNING)
- Created `scripts/test_cognitive_modeling.py`
- Generates 10 incomplete chains-of-thought per item (students stopping or going wrong at different points)
- Parses answer counts from simulated student reasoning
- Also runs contrastive baseline for direct comparison
- **Hypothesis:** LLMs have better *process* knowledge of student errors than *distributional* knowledge

## Experiment Results

### Final Scoreboard (all Gemini Flash, 20 probe items, 3 reps)

| Method | Mean ρ | Std | Reps | Notes |
|--------|--------|-----|------|-------|
| v5_error_analysis t=2.0 | **0.604** | 0.062 | 0.603, 0.679, 0.528 | Best overall |
| Contrastive baseline (cog script) | **0.583** | 0.033 | 0.629, 0.562, 0.557 | Most stable |
| v3_contrastive t=1.5 | **0.577** | 0.075 | 0.590, 0.662, 0.480 | Original best |
| Two-stage backstory | **0.408** | 0.096 | 0.411, 0.289, 0.525 | Underperformed |
| Cognitive modeling (10 CoT) | **0.165** | 0.046 | 0.230, 0.135, 0.131 | Poor |

### Interpretation

**Cognitive modeling failed.** Generating 10 student chains-of-thought and counting answers (ρ=0.165) drastically underperforms simply asking a teacher to predict distributions (ρ=0.583). The model's *process-level* student simulation is much worse than its *distributional* prediction. This is the opposite of our hypothesis.

Why: The model generates plausible-looking reasoning chains, but the errors are too uniform — it doesn't vary the errors enough across students. Most simulated students either all get it right or all make the same mistake, collapsing the distribution. The model is better at *describing* what percentage would err than *enacting* individual error trajectories.

**Two-stage backstory underperformed.** ρ=0.408 is below even the contrastive prompt at t=1.5. The backstories add specificity but also add noise — the model anchors on the backstory character rather than analyzing the item's difficulty. The implicit diversity from temperature is more effective than explicit diversity from personas.

**The ceiling holds at ρ≈0.60.** Direct distribution prediction with error-focused framing and high temperature remains the best approach. Neither structured elicitation nor process simulation broke through.

## Pending
- Groq paid-tier cross-model test (script ready, not yet launched)
- Metaprompt sweep: v9, v10 may still be running in user's terminal
- Potential next experiment: **buggy reasoning** — use known misconceptions to generate misconception-specific error chains (see below)

## Next Experiment: Buggy Reasoning with Known Misconceptions (RUNNING)

The cognitive modeling approach failed because the model generates *generic* reasoning errors. The buggy reasoning approach is different — it seeds the model with the *known* misconception for one distractor (from Eedi's labeled data) and asks it to:

1. Use the known misconception to analyze the target distractor
2. **Infer** what misconceptions drive the other distractors (these are unknown/unlabeled)
3. Judge how "attractive" each buggy reasoning chain is — how naturally would a student fall into it?
4. Predict response distributions based on this analysis

This is a hybrid: part knowledge retrieval (the labeled misconception), part inference (the unlabeled ones). It tests whether grounding the model in one real misconception helps it reason more accurately about the whole item.

**Two conditions:**
- `buggy_with_hint` — gets the known misconception
- `error_analysis_no_hint` — same analytical framing but no misconception provided (control)

**Three models:** Gemini Flash (t=2.0), Llama-70B (t=1.5), Llama 4 Scout (t=1.5)

Script: `scripts/test_buggy_reasoning.py`

## Research Arc So Far

### The Question
Can LLMs estimate the difficulty of maths MCQs without seeing any student response data?

### Phase 1: Direct Approaches (Jan 26–30)
- **Structured IRT estimation** — ask the model to directly output IRT parameters. Mixed results.
- **Teacher prediction at low temp** — ρ=0.095. Essentially failed. The model's "confident" predictions don't discriminate difficulty.

### Phase 2: RSM Experiment (Jan 31–Feb 1)
- **Response Surface Methodology** — systematically varied temperature, prompt style, students-per-call, misconception hints across 11 configs.
- **Discovery:** teacher_prediction at t=1.5 hit ρ=0.673. High temperature is the key variable, not prompt engineering.
- **Cross-model test** — Gemini Flash >> Llama-70B >> Qwen/4o-mini. Model capability is the binding constraint.

### Phase 3: Metaprompt Sweep (Feb 1–2)
- **7 prompt variants × 3 temperatures × 3 reps** — systematic search for the best prompt × temp combination.
- **Winner:** v5_error_analysis at t=2.0 → ρ=0.604±0.062
- **Runner-up:** v3_contrastive at t=1.5 → ρ=0.577±0.075
- **Temperature gradient:** analytically constrained prompts peak at t=1.5; experiential prompts keep improving at t=2.0
- **Ensemble test:** averaging two top prompts doesn't help — they capture similar signal

### Phase 4: Structured Elicitation (Feb 2)
- **Two-stage backstory** (ρ=0.408) — generate student personas, then predict per-persona. Underperformed. Personas added noise; all 5 backstories were "proficient" level despite t=2.0.
- **Cognitive modeling** (ρ=0.165) — simulate 10 student reasoning chains, count answers. Failed badly. Model generates too-uniform errors.
- **Key finding:** The model is better at *describing* error distributions than *enacting* individual error trajectories. Direct prediction > simulation.

### Phase 5: Buggy Reasoning (Feb 2, running)
- Seed with known misconceptions, infer unknown ones, judge attractiveness.
- Tests whether misconception knowledge improves predictions.
- Multi-model comparison.

### Phase 6: Buggy Reasoning with Known Misconceptions (Feb 2)
- Gave the model the known misconception for the target distractor, asked it to infer misconceptions for other distractors, then predict distributions.
- **buggy_with_hint (Gemini Flash): ρ=0.193** — the misconception hint *actively hurts*. The model over-anchors on the labeled misconception and under-analyzes overall item difficulty.
- **error_analysis_no_hint (Gemini Flash): ρ=0.488** — solid but below the pure error_analysis sweep result (0.604), likely because the buggy prompt's analytical framing biases even the control.
- Groq models (Llama-70B, Llama 4 Scout) produced mostly unparseable output for the complex buggy format.
- **Key finding:** The model already has implicit knowledge of common misconceptions. Telling it explicitly narrows its reasoning rather than enriching it. Less scaffolding = better predictions.

### Emerging Principles
1. **Temperature as diversity engine** — high temp (1.5–2.0) creates a wisdom-of-crowds effect within a single model
2. **Error-focused framing** — prompts that ask about *specific calculation errors* outperform generic prediction
3. **Model capability threshold** — models below a certain capability level (Llama-70B and below) cannot do this task
4. **Direct prediction > simulation** — asking "what percentage" beats simulating individual students
5. **Aggregation across reps** — averaging 3+ repetitions consistently improves correlation
6. **Less scaffolding = better predictions** — giving the model explicit misconception hints *hurts* performance. The model already has implicit pedagogical knowledge; explicit hints narrow its reasoning.

### Phase 7: Rigor & Validation (Feb 2)

Three parallel experiments to address the fishing/multiple-comparisons concern.

**7a. High-rep stability (10 reps) — COMPLETE**
- `scripts/test_high_reps.py` — top 2 configs for 10 reps each (reused reps 0–2 from sweep)
- **Results:**

| Config | 3-rep estimate | 10-rep mean | 10-rep std | Individual reps |
|--------|---------------|-------------|------------|-----------------|
| error_analysis t=2.0 | 0.604 ± 0.062 | **0.500 ± 0.111** | | 0.60, 0.68, 0.53, 0.46, 0.55, 0.24, 0.49, 0.53, 0.41, 0.50 |
| contrastive t=1.5 | 0.577 ± 0.075 | **0.513 ± 0.097** | | 0.59, 0.66, 0.48, 0.59, 0.60, 0.47, 0.46, 0.53, 0.33, 0.41 |

- **Key finding: 3-rep estimates were optimistically biased by ~0.08–0.10.** True per-rep mean is ~0.50, not ~0.60.
- The two "best" configs are **indistinguishable** — the difference was noise.
- Averaged-prediction rho (cumulative):

| Reps averaged | error_analysis | contrastive |
|---|---|---|
| 3 | 0.666 | 0.660 |
| 5 | 0.614 | 0.636 |
| 7 | 0.573 | 0.626 |
| 10 | 0.573 | 0.571 |

- Both converge to **ρ≈0.57** with 10-rep averaging. Aggregation helps but plateaus. The 3-rep averaged number (0.66) is inflated because reps 0–2 were the cached "lucky" ones from the sweep.

**7b. Frontier model comparison — PARTIAL (GPT-4o still running)**
- `scripts/test_frontier_models.py` — error_analysis prompt across frontier models

| Model | Temp | Mean ρ | Std | Notes |
|-------|------|--------|-----|-------|
| Gemini Flash | 2.0 | **0.500** | 0.111 | Baseline (from 10-rep) |
| Claude Sonnet 4 | 1.0 | **0.263** | 0.166 | Max temp = 1.0 |
| Claude Haiku 3.5 | 1.0 | **0.077** | 0.186 | Essentially no signal |
| GPT-4o | 1.5 | 0.338 (1 rep) | — | Still running |

- **Claude models are hampered by max temperature of 1.0.** This is likely a temperature ceiling problem, not a capability problem. Claude Sonnet 4 is a frontier model but produces ρ=0.26 at t=1.0 — comparable to Gemini Flash at t=0.7.
- This suggests **temperature is the primary variable**, not model capability per se. The "model capability threshold" finding from earlier may actually be a temperature ceiling effect.

**7c. Calibration anchors (cross-validated) — COMPLETE**
- `scripts/test_calibration_anchors.py` — 4-fold CV, 5 anchors / 15 test items per fold

| Method | CV mean ρ | Std | Pooled avg ρ |
|--------|----------|-----|-------------|
| Calibrated (with anchors) | **0.445** | 0.197 | 0.397 (p=0.08) |
| Uncalibrated (control) | **0.525** | 0.172 | 0.701 (p=0.0006) |

- **Calibration anchors do NOT help on Eedi** — they actually *hurt*. The opposite of SmartPaper (+0.16).
- Likely explanation: with only 5 items per fold, the variance is extreme (individual fold rhos range from -0.82 to +0.90). The anchors may also confuse the model by providing population-level distributions that don't match the ability-stratified format we ask for.
- The uncalibrated pooled ρ=0.701 is our highest number but reflects averaging across 12 predictions per item (3 reps × 4 folds).

### Phase 7 Synthesis: The Honest Numbers

After rigorous validation:
- **True per-rep ρ ≈ 0.50** (not 0.60 as 3-rep estimates suggested)
- **10-rep averaged ρ ≈ 0.57** (plateau — more reps won't help much)
- **Temperature is the binding constraint**, not model capability — frontier models (Claude Sonnet 4) perform poorly at t=1.0
- **Calibration anchors don't transfer** from SmartPaper to Eedi
- **The two best prompt configs are indistinguishable** — prompt wording matters less than temperature

### The Temperature Ceiling Problem

A key constraint: different APIs have different max temperatures.
- Gemini: max t=2.0 (best results)
- OpenAI: max t=2.0
- Anthropic: max t=1.0 (worst results despite strong models)
- Groq/open models: max t=2.0

This means our "cross-model" comparison is confounded by temperature ceiling differences. Claude Sonnet 4 at t=1.0 is not a fair comparison to Gemini Flash at t=2.0.

**Potential workarounds to "get hotter" on low-temp APIs:**
1. **Min-p sampling** — dynamic truncation that maintains coherence at extreme temperatures. Available in vLLM/HuggingFace for local models. ([ICLR 2025 paper](https://openreview.net/forum?id=FBkpCyujtS))
2. **Run models locally** — no temperature cap via vLLM. Could test Llama-70B at t=2.0+.
3. **Prompt-level perturbation** — inject diversity through input variation rather than sampling: randomize answer order, vary role descriptions, paraphrase questions.
4. **High-temp prefill + low-temp prediction** — generate diverse student descriptions or reasoning chains at high temperature (on Gemini), then feed those as context to Claude at t=1.0. Combines Gemini's temperature range with Claude's reasoning quality.
5. **Multiple samples + aggregation** — run 50 reps at t=1.0 instead of 10 at t=2.0. Compensate for low temperature with volume.

Option 4 is particularly interesting — it's a hybrid that leverages each model's strength. The two-stage backstory approach (Phase 4) tried something similar but failed because the backstories weren't diverse enough. A better version: use Gemini at t=2.0 to generate diverse *error analyses* for each item, then feed those as context to Claude at t=1.0 for the final distribution prediction.

### Phase 8: Gemini Model Comparison (Feb 2)

Tested all 5 available Gemini models with error_analysis @ t=2.0, 3 reps:

| Model | Mean ρ | SD | Avg-pred ρ | Parse fails/20 |
|-------|--------|-----|-----------|----------------|
| **Gemini 3 Flash** | **0.604** | 0.062 | **0.666** | 0-1 |
| Gemini 2.5 Pro | 0.240 | 0.242 | 0.398 | 1-3 |
| Gemini 2.5 Flash | 0.088 | 0.131 | 0.148 | 6-9 |
| Gemini 3 Pro | 0.052 | 0.365 | -0.001 | 16-17 |
| Gemini 2.0 Flash | -0.019 | 0.050 | 0.015 | 0-1 |

**Key findings:**
- Gemini 3 Flash is uniquely good — no other model comes close
- Pro models (3 Pro, 2.5 Pro) have mandatory thinking mode that breaks structured output at t=2.0
- Gemini 2.5 Flash also struggles with parsing — high temp + thinking = format failures
- Gemini 2.0 Flash has zero signal despite clean parsing — it lacks pedagogical knowledge
- The task requires BOTH instruction-following fidelity at high temperature AND implicit pedagogical knowledge

Script: `scripts/test_gemini_models.py`

### Phase 9: Cross-Provider Model Survey (Feb 2)

15+ models tested across 5 providers. Full results in `.claude/handoffs/2026-02-02-model-selection-experiment.md`.

**Tier structure (contrastive @ t=1.5):**
- **Tier 1 (ρ > 0.35):** Gemini 3 Flash (0.470), Llama-4-Scout 17B (0.371), DeepSeek-V3 (0.338)
- **Tier 2 (ρ 0.15–0.35):** GPT-OSS-120B (0.336), Kimi-K2 (0.336), Llama-70B (0.257), GPT-4o (0.172)
- **Tier 3 (ρ ≈ 0):** Qwen-32B, Llama-8B, GPT-4o-mini, Gemini 2.0 Flash
- **Failed:** DeepSeek-Reasoner (no temp control → constant output)

**Standout: Llama-4-Scout (17B, free on Groq)**
- With error_analysis @ t=2.0 (10 reps): per-rep **ρ=0.355 ± 0.167** (initial 3-rep estimate of 0.505±0.048 was misleadingly stable)
- Avg-pred ρ improves monotonically: 3→0.609, 5→0.639, 7→0.648, **10→0.668** — matches Gemini Flash (0.666)
- Temperature sweep confirmed t=2.0 optimal (Groq max): t=1.0→0.423, t=1.5→0.445, t=2.0→0.490 avg-pred
- 17B parameters, free inference on Groq

**Frontier models (error_analysis prompt):**
- Claude Sonnet 4 (t=1.0): ρ=0.263 — hampered by max temp
- GPT-4o (t=1.5): ρ=0.198 — surprisingly weak
- Claude Haiku 3.5 (t=1.0): ρ=0.077 — no signal

**The "System 1" hypothesis:** The task requires fast pattern matching on pedagogical knowledge, not deliberative reasoning. Thinking/deliberation actively hurts (Flash with thinking_budget=1024: ρ drops from 0.604 to 0.467). This explains why Pro models fail and why "smarter" models (GPT-4o, Claude Sonnet) don't outperform.

### Phase 10: Two-Stage Diversity Injection (Feb 2, running)

`scripts/test_two_stage_diversity.py` — tests whether high-temp diversity *as context* improves prediction.

**Design:** Stage 1 generates 5 diverse seeds at t=2.0 (Gemini Flash), Stage 2 uses seeds as context for prediction at t=1.5.

The key innovation vs Phase 4 (which failed): Phase 4 used high-temp content *as the prediction mechanism* (counting chain answers). Phase 10 uses it *as context* — the model reads diverse perspectives, then makes its own judgment.

**Conditions and Stage 1 Prompts:**

**A. cognitive_chains** — 5 simulated student workings → used as context for prediction
```
You are simulating a real UK Year 9 student attempting this maths question. You are NOT a strong student — you have gaps in your knowledge and sometimes make careless errors.

{item_text}

Show your working step by step, exactly as a real student would write it. Include hesitations, crossing-out, and mistakes. You may or may not get the final answer right. Choose your answer from A/B/C/D at the end.
```

**B. buggy_analyses** — 5 misconception analyses → used as context
```
You are an expert in mathematical misconceptions. For this question, identify one specific misconception or buggy procedure that a UK Year 9 student might apply.

{item_text}

Describe:
1. The specific misconception or procedural error (be precise — name the bug)
2. How a student with this misconception would work through this question step by step
3. Which answer option (A/B/C/D) this misconception leads to
4. How natural/attractive this error is — would the student even realise they went wrong?
```

**C. error_perspectives** — 5 "why this is hard" analyses → used as context
```
You are a maths education researcher studying this question. Identify one specific reason why students might find this question difficult or easy.

{item_text}

Focus on ONE specific aspect:
- A particular step where students commonly go wrong
- A feature of the numbers/context that makes the error more or less likely
- A visual or notational trap
- A common shortcut that happens to work or fail here

Be specific and concrete, not generic.
```

**D. direct_baseline** — standard error_analysis at same temperature (no two-stage)

**Stage 2 prompt** (same for all two-stage conditions):
```
You are an experienced UK maths teacher marking a set of Year 9 mock exams.

Here are {n_seeds} different perspectives on how students might approach this question:

{seed_content}

---

Now, considering ALL of the above perspectives, predict what percentage of students at each ability level would choose each option.

{item_text}

Respond in this exact format:
below_basic: A=XX% B=XX% C=XX% D=XX%
basic: A=XX% B=XX% C=XX% D=XX%
proficient: A=XX% B=XX% C=XX% D=XX%
advanced: A=XX% B=XX% C=XX% D=XX%
```

**Cross-pipeline results:**

| Pipeline (stage1→stage2) | cognitive_chains | buggy_analyses |
|--------------------------|-----------------|----------------|
| **Gemini→Gemini** (t=2.0→t=1.5) | **0.508 ± 0.067** | 0.268 ± 0.061 |
| **Gemini→Scout** (t=2.0→t=2.0) | 0.356 ± 0.239 | -0.077 ± 0.192 |
| **Scout→Scout** (t=2.0→t=2.0) | 0.401 ± 0.082 | 0.275 ± 0.093 |

**Complete Eedi results (Gemini→Gemini, all conditions):**

| Condition | Mean ρ | SD |
|-----------|--------|-----|
| cognitive_chains | 0.508 | 0.067 |
| **direct_baseline** | **0.502** | 0.125 |
| buggy_analyses | 0.268 | 0.061 |
| error_perspectives | 0.241 | 0.061 |

Scout experiments degraded by Groq 503 capacity errors.

**Cross-model coherence effect:** Scout→Scout consistently outperforms Gemini→Scout (both cognitive_chains and buggy_analyses). A model interprets its own seeds better than another model's seeds. But Gemini→Gemini still wins overall because Gemini produces higher-quality seeds AND predictions.

**Key findings:**
1. **Cognitive chains as context works; buggy analyses doesn't.** Consistent across all three pipelines. Student reasoning chains provide useful diverse perspectives; misconception analyses are too narrow/prescriptive.
2. **Gemini→Gemini is the best two-stage pipeline** (0.508 ± 0.067). Gemini's pedagogical knowledge produces higher-quality seeds.
3. **Scout→Scout beats Gemini→Scout** for cognitive_chains (0.401 vs 0.356). Same-model coherence advantage — Scout interprets its own seeds better than Gemini's.
4. **Scout→Scout has the lowest variance** (SD=0.082) — though Scout's single-stage SD is actually 0.167 at 10 reps (the 3-rep 0.048 was misleading).
5. **Two-stage cognitive chains doesn't beat direct prediction** — Gemini→Gemini cognitive_chains (0.508) ≈ direct error_analysis (0.500 10-rep mean), but with lower variance (0.067 vs 0.111). The two-stage approach *stabilizes* rather than *boosts*.
6. **Cognitive modeling reborn** — Phase 4 cognitive chains as *mechanism* (counting answers) gave ρ=0.165. Phase 10 cognitive chains as *context* gives ρ=0.508. Same content, different role. The model is better at integrating diverse perspectives than enacting them.

**Groq capacity issues:** Scout experiments partially blocked by 503 errors. Results may be re-run when capacity recovers. Cached responses are preserved.

### The Fishing Problem
With ~20 configs tested on n=20 items, multiple comparisons were a real concern. The 10-rep validation confirmed our estimates were inflated by ~0.08–0.10. The honest claim is **ρ≈0.50 per-rep, ρ≈0.57 averaged across 10 reps**, significantly above zero (p<0.01) but more modest than initial reports.

Mitigating factors: temperature effect is monotonic across prompts (unlikely under chance), scaffolding findings are consistently negative (harder to fish for), cross-model gradient is sensible. A held-out test on the remaining ~70 Eedi items is the definitive next step.

### The Ceiling Question
With n=20 items and ρ≈0.57, the 95% CI is roughly [0.20, 0.80]. We cannot distinguish this from ρ=0.40 or ρ=0.70 with current data. Scaling to 50+ items is necessary to make precise claims.

### Emerging Principles (Revised — see Phase 13 for final version)
1. ~~**Temperature is the primary variable**~~ → See Phase 13: temperature interacts with model/prompt/dataset; flat on Eedi with good prompt, helps Gemini on SmartPaper, hurts DeepSeek
2. **Prompt design is the primary variable on Eedi** — contrastive at any temp (ρ≈0.45) vs plain (ρ≈0.12)
3. **Gemini 3 Flash is uniquely good on Eedi** — tested 5 Gemini models: 3 Flash (ρ=0.604) >> 2.5 Pro (0.240) >> rest. Pro models with thinking can't produce structured output at t=2.0
4. **Direct prediction > simulation** — asking "what percentage" (ρ≈0.50) beats simulating individual students (ρ=0.17)
5. **Aggregation helps but plateaus** — averaging 10 reps gives ρ≈0.57; diminishing returns after ~7 reps
6. **Less scaffolding = better** — misconception hints (ρ=0.19), student personas (ρ=0.41), and reasoning chains (ρ=0.17) all hurt vs. direct prediction (ρ=0.50)
7. **Calibration anchors don't transfer** — helped on SmartPaper but hurt on Eedi; possibly format-dependent
8. **3-rep estimates are unreliable at n=20** — our initial "best" of ρ=0.60 was biased; true value is ρ≈0.50
9. **Context > mechanism for structured elicitation** — high-temp content used as *context* for prediction (ρ=0.508) dramatically outperforms same content used as *prediction mechanism* (ρ=0.165). Two-stage stabilizes variance but doesn't boost mean vs direct prediction.
10. **The signal does not generalize beyond hand-picked items on Eedi** — see Phase 11. But it DOES generalize on SmartPaper — see Phase 12.

### Phase 11: The Expansion Test — Signal Collapse (Feb 2)

**This is the most important experiment of the day.**

Expanded the probe set from 20 → 50 items by adding 30 randomly sampled Eedi items (stratified by classical difficulty: 10 easy, 10 medium, 10 hard). Computed proper 2PL IRT parameters for all 50 items using the full 15.8M-response dataset (validated at ρ=1.000 against original params).

**Results:**

| Model | All 50 items | Original 20 | New 30 |
|---|---|---|---|
| Gemini 3 Flash (3 reps) | **ρ=0.039** (p=0.79) | ρ=0.462 (p=0.04) | ρ=-0.176 (p=0.35) |
| Llama-4-Scout (3 reps) | **ρ=0.001** (p=0.99) | ρ=0.089 (p=0.71) | ρ=-0.123 (p=0.52) |

**The signal collapsed completely on new items.**

- On the original 20: Gemini still shows signal (ρ=0.462, consistent with earlier results)
- On the new 30 randomly sampled items: **zero signal** for both models (ρ≈0, not significant)
- Combined 50: washed out to ρ≈0

**Why?**

The original 20 items were hand-selected to test 5 specific, well-documented misconceptions:
- Inverse operations (using same op instead of inverse)
- Order of operations (left-to-right instead of PEMDAS)
- Fraction addition (adding numerators AND denominators)
- Negative multiplication (neg × neg = neg)
- (one more category)

These are among the most studied misconceptions in math education literature — extensively discussed in teacher forums, textbooks, and research papers. This is exactly the content that ends up in LLM training data.

The randomly sampled 30 items test a much wider variety of knowledge gaps, many less well-documented.

**Sensitivity check:** Even removing the 2 most extreme items (b_2pl < -3), Gemini still shows ρ=0.585 on the remaining 18 original items. The signal is REAL for those specific items — it just doesn't transfer to arbitrary items.

**Scout 10-rep stability (original 20 items, error_analysis @ t=2.0):**
- Mean per-rep ρ: 0.355 ± 0.167 (3-rep estimate of 0.505±0.048 was misleadingly stable)
- Avg-pred ρ improves monotonically: 3→0.609, 5→0.639, 7→0.648, **10→0.668** (p=0.001)
- Individual reps: 0.543, 0.535, 0.437, 0.403, 0.530, 0.116, 0.304, 0.422, 0.070, 0.194
- Temperature sweep: t=2.0 confirmed optimal (Groq max); t=1.0→0.312, t=1.5→0.301, t=2.0→0.424 mean per-rep

### Option Shuffle Analysis (Feb 2)

Computed correlations on 6 reps (3 original order, 3 shuffled A→B/C/D permutations):

| Condition | Per-rep ρ range | Avg-pred ρ |
|---|---|---|
| Original 3 reps | 0.528–0.679 | **0.666** |
| Shuffled 3 reps | 0.316–0.527 | 0.617 |
| All 6 reps | — | 0.665 |

Shuffling option order doesn't help — original order is slightly better. More reps past 3 don't improve averaged predictions on these items.

### The Honest Verdict

**The ρ≈0.50-0.67 results were an artifact of item selection, not a generalizable capability.**

What we thought: LLMs (especially Gemini 3 Flash) can predict item difficulty at ρ≈0.50-0.67 with the right prompt and temperature.

What's actually true: LLMs can predict difficulty **only for items testing well-documented misconceptions** that appear extensively in training data. For random items, the prediction is ρ≈0 — no better than chance.

This is the same overfitting pattern seen in the replication track: feature extraction v1 showed r=0.770 on 10 test items, collapsed to r=0.063 on 1,869 items.

### What This Means for the Paper

1. **The negative result IS the story.** LLMs cannot predict item difficulty in general. This is important and publishable — it contradicts the optimistic claims from papers working with small, curated test sets.
2. **The partial success is informative.** The model succeeds specifically on items testing well-known misconceptions. This maps to pedagogical content knowledge (PCK) — the model has learned SOME PCK from training data, but only for the most commonly discussed misconceptions.
3. **The "System 1" finding remains a secondary result.** Deliberation hurts even on the items where prediction works.
4. **The methodology tells a cautionary tale.** Small probe sets (n=20) can produce impressive-looking correlations that don't replicate. This is relevant to the broader psychometrics + AI literature.

### Files from Phase 11
- `pilot/rsm_experiment/probe_items_expanded50.csv` — 50-item probe with 2PL IRT params
- `pilot/rsm_experiment/expanded50/gemini_flash/` — 3 reps, 50 items
- `pilot/rsm_experiment/expanded50/llama4_scout/` — 3 reps, 50 items
- `pilot/rsm_experiment/scout_10rep/` — 10 reps, 20 items
- `scripts/run_expanded_probe.py` — experiment runner

### Phase 12: SmartPaper Expansion — Signal Holds (Feb 2)

Ran the same expansion test on SmartPaper: all 140 items (open-ended, rubric-scored) across 4 subjects and 3 grades.

**Results:**

| Model | All 140 | Original 20 | Remaining 120 |
|---|---|---|---|
| **Gemini 3 Flash** | **ρ=0.547** (p<0.0001) | ρ=0.768 | ρ=0.518 |
| Llama-4-Scout | ρ=0.250 (p=0.003) | ρ=0.663 | ρ=0.184 |

**By subject (Gemini, avg-pred):**

| Subject | ρ | n | p |
|---|---|---|---|
| Science | **0.734** | 32 | <0.0001 |
| Social Science | **0.702** | 34 | <0.0001 |
| Mathematics | **0.600** | 33 | 0.0002 |
| English | 0.432 | 41 | 0.005 |

**The signal GENERALIZES on SmartPaper.** Even on the 120 non-probe items, Gemini shows ρ=0.518 (p<0.0001). This is completely different from Eedi, where non-probe items showed ρ≈0.

### The Contrast: SmartPaper vs Eedi

| | SmartPaper | Eedi |
|---|---|---|
| Item type | Open-ended + rubric | MCQ with distractors |
| All items | **ρ=0.547** (n=140) | **ρ=0.039** (n=50) |
| Probe items | ρ=0.768 (n=20) | ρ=0.462 (n=20) |
| Non-probe items | **ρ=0.518** (n=120) | **ρ=-0.176** (n=30) |
| Generalizes? | **YES** | **NO** |

**Why?** SmartPaper items are open-ended with rubrics — the model judges "how hard is this for this grade level." This is general pedagogical knowledge about curriculum difficulty, prerequisite chains, and age-appropriate expectations.

Eedi items require predicting which *specific wrong answer* students choose and how often. MCQ difficulty depends on distractor quality — how well each wrong answer maps to a specific misconception. This requires knowledge of misconception-to-distractor mappings that only exists in training data for the most well-studied misconceptions.

### Revised Paper Story

**LLMs can estimate difficulty for open-ended items (ρ≈0.55) but not for MCQ items (ρ≈0).** MCQ difficulty depends on distractor quality — a form of specific pedagogical content knowledge that LLMs lack except for well-documented misconceptions. Open-ended difficulty depends on general curriculum knowledge that LLMs have absorbed from training data.

Secondary findings:
1. Gemini 3 Flash >> all other models (including GPT-4o, Claude Sonnet 4)
2. Deliberation hurts (System 1 > System 2)
3. High temperature + averaging creates wisdom-of-crowds effect
4. Small evaluation sets (n=20) produce inflated correlations — generalization testing is essential

### Parsing & Calibration Check

**Parsing:** 100% parse rate for both models (420/420 Gemini, 419/419 Scout). Simple numeric output format works cleanly even at t=2.0.

**Calibration problem:**

| | Estimated mean p | Actual mean p | Offset |
|---|---|---|---|
| Gemini | 0.727 | 0.293 | +0.434 |
| Scout | 0.685 | 0.293 | +0.392 |

Both models massively overestimate how easy items are. Spearman ρ=0.547 (rank order preserved) but Pearson r=0.454 (poor linear fit). The models know *which* items are harder but not *how hard* they are.

**Likely explanation:** SmartPaper students are Indian Grades 6-8. The models' implicit calibration is trained on populations with higher baseline performance (US/UK test-takers, online tutoring users). The relative ordering transfers cross-culturally but the absolute scale doesn't.

**Implication for the paper:** LLMs can *rank* open-ended item difficulty but cannot *calibrate* it without population-specific anchoring. This is a useful practical finding — ranking is sufficient for item selection and test assembly, but not for setting pass marks or reporting scores.

### Files from Phase 12
- `pilot/smartpaper_expansion/gemini_flash/` — 3 reps, 140 items
- `pilot/smartpaper_expansion/llama4_scout/` — 3 reps, 140 items
- `scripts/run_smartpaper_expansion.py` — experiment runner

### Phase 13: Temperature Reframing — Controlled Sweeps (Feb 2)

Two temperature sweep experiments reveal that the "temperature cliff" narrative was wrong, and the true picture is a 3-way interaction of model × prompt × temperature.

**13a. Eedi sweep (contrastive prompt, Gemini Flash, 3 reps each, deterministic parsing)**

Data: `pilot/rsm_experiment/transition_zone/gemini_sweep.json`

| Temp | Mean ρ | SD | Individual reps |
|------|--------|-----|-----------------|
| 0.3 | 0.449 | 0.118 | 0.283, 0.543, 0.520 |
| 0.6 | 0.458 | 0.046 | 0.395, 0.472, 0.506 |
| 0.9 | 0.358 | 0.070 | 0.278, 0.448, 0.347 |
| 1.2 | 0.354 | 0.127 | 0.529, 0.232, 0.302 |
| 1.5 | 0.462 | 0.031 | 0.424, 0.499, 0.463 |
| 2.0 | 0.580 | 0.110 | 0.703, 0.436, 0.602 |

**On Eedi: temperature is flat (ρ≈0.35–0.58).** No cliff, no clear trend. The original RSM confounded prompt × temperature — it compared a weak prompt at t=0.3 (ρ=0.12) vs a strong prompt at t=1.5 (ρ=0.67, later shown to be an outlier). With contrastive prompt held constant, temperature barely matters.

Scout sweep (fixed max_tokens=512→1024 truncation bug): ρ=0.398 (t=0.3), 0.466 (t=0.6), 0.552 (t=0.9). Incomplete due to Groq rate limits.

**13b. SmartPaper sweep (20 probe items, 5 reps averaged, 2 models × 3 strategies × 4 temps)**

Data: `pilot/smartpaper_rsm_v2/temp_sweep/results.json`

**Gemini 2.5 Flash on SmartPaper:**

| Strategy | t=0.5 | t=1.0 | t=1.5 | t=2.0 |
|----------|-------|-------|-------|-------|
| baseline | 0.571 | **0.743** | 0.311 | 0.398 |
| errors | 0.774 | **0.803** | 0.783 | **0.841** |
| anchors | 0.811 | **0.821** | 0.783 | **0.877** |

**DeepSeek Chat on SmartPaper:**

| Strategy | t=0.5 | t=1.0 | t=1.5 | t=2.0 |
|----------|-------|-------|-------|-------|
| baseline | **0.800** | **0.809** | 0.728 | 0.675 |
| errors | **0.799** | 0.787 | 0.783 | 0.766 |
| anchors | 0.743 | 0.729 | 0.726 | 0.691 |

**Key findings from combined sweeps:**

1. **SmartPaper ρ values are much higher** (0.69–0.88) than Eedi (0.35–0.58). Open-ended items are fundamentally easier to predict.
2. **Temperature helps Gemini on SmartPaper** — monotonic increase for errors (0.774→0.841) and anchors (0.811→0.877). Best single result: **anchors t=2.0 ρ=0.877**.
3. **Temperature HURTS DeepSeek** — monotonic decrease across ALL strategies (baseline: 0.800→0.675; anchors: 0.743→0.691).
4. **Baseline is fragile for Gemini** — collapses at t=1.5 (0.311) while errors/anchors stay robust. Structured prompts provide temperature-robustness.
5. **DeepSeek is strong across the board** — even its worst (anchors t=2.0: 0.691) beats Gemini's best baseline. Better intrinsic calibration.
6. **The interaction is model × prompt × temperature.** No single variable dominates. "Temperature is the primary variable" was wrong — that was a Gemini-specific effect on Eedi with a weak prompt.

### Revised Emerging Principles (Final)

1. ~~**Temperature as diversity engine**~~ → **Temperature interacts with model and prompt.** Helps Gemini on SmartPaper (ρ increases 0.81→0.88), hurts DeepSeek (ρ decreases 0.80→0.68), flat on Eedi with good prompt (ρ≈0.35–0.58). The original "temperature cliff" was an artifact of confounding prompt × temperature.
2. **Prompt design is the primary variable** on Eedi — contrastive/error-focused at any temp (ρ≈0.45) vs plain teacher_prediction (ρ≈0.12). A 3–4× effect.
3. **Structured prompts provide temperature-robustness** — Gemini baseline swings wildly (0.31–0.74) while errors/anchors stay in a tight band (0.77–0.88) on SmartPaper.
4. **Item type determines generalizability** — SmartPaper ρ=0.55 on 140 items; Eedi collapses beyond curated 20. Open-ended difficulty is general curriculum knowledge; MCQ difficulty requires specific distractor-misconception mappings.
5. **Direct prediction > simulation** — asking "what percentage" (ρ≈0.50) beats simulating individual students (ρ=0.17)
6. **Less scaffolding = better** on Eedi — misconception hints, personas, and reasoning chains all hurt
7. **Deliberation hurts** — System 1 > System 2 across all tests
8. **Aggregation helps but plateaus** — 10 reps → ρ≈0.57 on Eedi; diminishing returns after ~7
9. **Small probe sets produce inflated estimates** — n=20 items, 3-rep: ρ=0.60; validated at 10-rep: ρ=0.50; expanded to n=50: ρ=0.04
10. **Models overestimate easiness** — +0.40 calibration offset for Indian students; rank order transfers but absolute scale doesn't
11. **Two-stage diversity injection fails at scale** — simulated student attempts as context actively destroy signal on full item sets (SmartPaper 134 items: ρ=0.06 two-stage vs ρ=0.55 direct)

### Phase 14: Two-Stage Diversity on SmartPaper — Context Destroys Signal (Feb 2)

Tested the two-stage cognitive chains approach on SmartPaper's full 134 non-visual items. This is the generalization test for the Eedi two-stage findings.

**Script:** `scripts/test_two_stage_smartpaper.py`

**Design:** Stage 1 generates 5 diverse student attempts at t=2.0 (Gemini Flash), Stage 2 predicts difficulty using attempts as context at t=1.5.

**Eedi results (20 probe items, for comparison):**

| Condition | Gemini→Gemini | Notes |
|-----------|--------------|-------|
| cognitive_chains | 0.508 ± 0.067 | Matched direct |
| direct_baseline | 0.502 ± 0.125 | Higher variance |
| buggy_analyses | 0.268 ± 0.061 | Weak |
| error_perspectives | 0.241 ± 0.061 | Weak |

**SmartPaper results (134 items):**

| Pipeline | cognitive_chains rep0 | direct (Phase 12) |
|----------|-----------------------|-------------------|
| Gemini→Gemini | **ρ=0.059** (p=0.50) | **ρ=0.547** (p<0.0001) |
| Gemini→Scout | ρ=0.182 (p=0.04) | ρ=0.250 |

**Two-stage destroys SmartPaper signal.** From ρ=0.547 (direct) to ρ=0.059 (two-stage cognitive chains). The effect is catastrophic.

**By subject (two-stage cognitive chains vs Phase 12 direct):**

| Subject | Two-stage ρ | Direct ρ | Damage |
|---------|------------|----------|--------|
| English | 0.318 | 0.432 | -0.114 |
| Science | 0.173 | **0.734** | **-0.561** |
| Social Science | -0.011 | **0.702** | **-0.713** |
| Mathematics | 0.086 | **0.600** | **-0.514** |

Worst damage in subjects where direct prediction was strongest (Science, Social Science). English, where the model was weakest, suffers least — but still no improvement.

**By difficulty tercile:** Flat — hard (0.124), medium (0.237), easy (0.130). Two-stage doesn't selectively help hard items.

**By grade:** Grade 6 (0.263) > Grade 7 (0.097) > Grade 8 (-0.159). Model simulates younger students better?

**Seed quality audit:**
Seeds are actually realistic — proper Hindi-medium student English with spelling errors, grade-appropriate content, and genuine misconceptions. Example (Grade 7 Science, testing indicator knowledge, difficulty=0.078):
```
In class teacher told that haldi is indicator. I take liquid in bowl. I put
turmeric powder. If color changes to red then it is soap or base. Then I use
gurhal (hibiscus) flower...
```

The seeds are good quality but *they don't help prediction*. The model's implicit knowledge — accessed via direct prompting — is more informative than 5 explicitly generated student attempts.

**Why context hurts:**
1. **Noise injection.** Even realistic simulated attempts are a lossy representation of the model's knowledge. Direct prediction accesses implicit knowledge without this lossy intermediary.
2. **Context window pollution.** ~2000 tokens of student attempts dilute the model's attention. For items where it "knows" the difficulty, this is pure distraction.
3. **Anchoring on seed content.** The model anchors on the specific errors/successes in the seeds rather than using its broader knowledge base. 5 samples is too few to represent the true population distribution.
4. **The model is better at *judging* than *integrating*.** Direct prediction asks "how hard is this?" — a judgment. Two-stage asks "given these 5 attempts, how hard is this?" — an integration task. The model's pedagogical knowledge is better accessed as a judgment.

**Cross-pipeline coherence results (Eedi 20 items):**

| Pipeline | cognitive_chains | buggy_analyses |
|----------|-----------------|----------------|
| Gemini→Gemini | **0.508 ± 0.067** | 0.268 ± 0.061 |
| Gemini→Scout | 0.356 ± 0.239 | -0.077 ± 0.192 |
| Scout→Scout | 0.401 ± 0.082 | 0.275 ± 0.093 |

Same-model pipelines (Gemini→Gemini, Scout→Scout) outperform cross-model (Gemini→Scout). Models interpret their own generated context better.

**Files:**
- `scripts/test_two_stage_smartpaper.py` — SmartPaper two-stage experiment
- `scripts/test_two_stage_diversity.py` — Eedi two-stage experiment (updated with Scout/Groq support)
- `pilot/smartpaper_two_stage/` — SmartPaper cached seeds and predictions
- `pilot/rsm_experiment/two_stage_diversity/` — Eedi cached seeds and predictions
