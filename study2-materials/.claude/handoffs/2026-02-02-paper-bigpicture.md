# Handoff: Paper Big Picture & DBE-KT22 Experiments (2026-02-02)

## What this paper is

An **optimization paper with potential theory contributions**. The core question is practical: *Can LLMs replace empirical calibration of test item difficulty?* We use response surface methodology (RSM) — borrowed from industrial process optimization — to map the parameter landscape (model × prompt × temperature × item type) across ~200 experimental conditions. The result is not a single correlation but a map of when LLM difficulty estimation works and when it doesn't.

### The optimization framing
- We're optimizing a pipeline (LLM difficulty estimation) across a high-dimensional parameter space
- **DOE is the method** — Phase 1 uses RSM (full factorial screening: 7 models × 4 prompts), subsequent phases use sequential experimentation (vary one factor, hold others fixed, use results to inform next experiment)
- The human-in-the-loop sequential design is deliberate: the parameter space is too large for full factorial (~thousands of conditions), so we prune dead branches early and direct resources to informative regions. This is standard practice in industrial process optimization (Box & Draper, 2007) and is closer to **adaptive DOE** than classical pre-specified designs.
- The field currently evaluates with single-configuration studies; we argue this is inadequate
- The practical deliverable: a decision procedure for practitioners ("characterize your difficulty source before choosing your method")
- **Methodological contribution**: the screening → anomaly → hypothesis → targeted test sequence is the hypothetico-deductive method applied to LLM evaluation. Most papers report one configuration; we demonstrate what systematic exploration looks like.

### The theory contributions (potential, not proven)
- **Complexity-driven vs. selectivity-driven difficulty**: Items whose difficulty comes from cognitive complexity (recall vs. reasoning, single-step vs. multi-step) are predictable from text. Items whose difficulty depends on empirical student-item interactions (which distractors trigger which misconceptions in which students) are not. This is a *hypothesis* about why the literature disagrees — not a proven mechanism.
- **"System 1" finding**: Fast pattern matching on absorbed pedagogical content knowledge outperforms deliberative cognitive simulation. Four learning science traditions (KC theory, BUGGY, conceptual change, ESS) all predict more scaffolding should help. It doesn't. This says something about what LLMs encode: pedagogical content knowledge (what errors are common), not cognitive process models (how errors arise).
- **Small-set overfitting**: ρ=0.50 on 20 curated items → ρ=0.11 on 105 random items. A methodological cautionary tale about LLM evaluation that generalizes beyond this task.

### What it draws on (theoretical and methodological foundations)

1. **Psychometrics / IRT** — defining and measuring "difficulty"; when classical p-correct suffices vs. when IRT is needed; the distinction between item parameters and person parameters

2. **Learning sciences theories of student error** — KC theory (Koedinger et al.), BUGGY (Brown & Burton), conceptual change (Vosniadou, Chi), error-specific simulations (ESS). These aren't just prompting strategies — they serve as a **theoretical sampling frame**: four distinct accounts of *why* students make errors, each operationalized as a prompt. The collective negative result is a substantive claim that LLMs lack access to the cognitive mechanisms these theories describe.

3. **LLM-as-cognitive-model debate** — can LLMs simulate students? (Binz & Schulz 2023, Argyle et al. 2023, Shanahan 2024). Our contribution: LLMs can retrieve frequency information about common errors (pedagogical content knowledge) but cannot simulate cognitive processes that generate those errors. This connects to the broader question of what LLMs encode vs. what they can do.

4. **Design of experiments / sequential experimentation** — Phase 1 is RSM (Box & Wilson 1951); subsequent phases are adaptive sequential DOE (Box & Draper 2007). The human-in-the-loop design is justified by the parameter space size and the need to prune uninformative regions. This is also a **methodological contribution** to LLM evaluation: most studies report a single configuration, we argue capability claims require systematic multi-factor exploration.

5. **Replication / meta-science** — transparent reporting of all conditions including failures; the small-set inflation problem (ρ=0.50 on 20 items → ρ=0.11 on 105). Connects to broader concerns about evaluation methodology in AI (Raji et al. 2021, Liao et al. 2021).

6. **Item difficulty theory** — what makes a question hard? Two traditions: (a) cognitive complexity / depth of knowledge (Webb 1997, Bloom's taxonomy) — difficulty from the task structure itself; (b) empirical calibration / selectivity — difficulty from student-item interactions that can't be predicted from text alone. The complexity/selectivity distinction is our post-hoc framework that synthesizes why LLMs succeed on some items and fail on others.

7. **Pedagogical content knowledge (PCK)** — Shulman (1986). The "System 1" finding maps onto PCK: LLMs appear to have absorbed *knowledge about common student errors* from training data (a form of PCK) but lack the *cognitive process models* that would let them predict errors from first principles. This reframes what "LLM-as-teacher" means.

### Prior work to engage with

- **Ha et al. (2023), Attali et al. (2023)** — direct precursors doing LLM difficulty estimation on small curated sets. Our small-set inflation finding is a direct replication critique.
- **Tack & Piech (2023)** — NeurIPS competition on predicting student performance; different framing (knowledge tracing vs. item-level difficulty) but overlapping territory.
- **Yaneva et al. (2019, 2020)** — pre-LLM difficulty estimation using linguistic features. Useful baseline comparison: what did feature engineering achieve vs. what LLMs achieve?
- **Benedetto et al. (2023)** — BERT-based difficulty estimation. Bridge between feature engineering and LLM approaches.
- **Byrd & Srivastava (2022)** — LLM benchmark difficulty prediction. Different domain (AI benchmarks vs. educational items) but same core question.
- **Raji et al. (2021)** — AI evaluation methodology critique. Supports our argument for multi-configuration evaluation.
- **Embretson (1998)** — cognitive complexity model. Theoretical foundation for why some difficulty is predictable from item text.

## Current state of experiments

### Completed results

| Dataset | Model | Method | ρ | p | n | Notes |
|---|---|---|---|---|---|---|
| Eedi | Gemini 3 Flash | Contrastive, t=1.5 | 0.114 | 0.25 (ns) | 105 | IRT b₂PL ground truth |
| Eedi | 7 models × 4 prompts | Direct estimation | ≈0 | ns | 1,869 | Classical p-correct |
| SmartPaper | Gemini 3 Flash | Teacher prediction, t=2.0 | 0.547 | <0.0001 | 140 | Classical p-correct |
| DBE-KT22 | Gemini 2.5 Flash | Direct estimation, t=0 | 0.342 | <0.001 | 168 | Classical p-correct |

### Running experiments (background jobs, PID 6182 and 9707)

1. **Gemini 3 Flash direct on DBE-KT22** — needed for clean cross-dataset comparison (all three datasets on same model). Cache file: `predictions_gemini3flash.json` (not yet written at checkpoint)
2. **Contrastive prompt on DBE-KT22** (3 reps, Gemini 2.5 Flash, t=1.5) — tests whether Eedi's best prompt helps on clean data. Rep0 complete (168 items), rep1 at 40/168.
3. **GPT-4o direct on DBE-KT22** — tests cross-model generalization. Queued after contrastive reps finish.
4. **Classroom simulation on DBE-KT22** (Gemini 2.5 Flash, t=1.0) — tests whether simulation failure was Eedi-specific. Queued.

Script: `scripts/run_dbe_kt22_hypotheses.py`
Output dir: `pilot/dbe_kt22_validation/`
Cache files: `predictions_{experiment_name}.json`

### What these experiments test

The DBE-KT22 experiments distinguish two explanations for the Eedi null:
- **Explanation A**: Eedi's difficulty is selectivity-driven → LLMs can't predict it (the paper's claim)
- **Explanation B**: Eedi's data is bad (answer ordering issues, adaptive assignment, Kaggle quality)

If methods that failed on Eedi work on DBE-KT22, the Eedi failures are at least partly dataset-specific. If they also fail on DBE-KT22, the selectivity framework is stronger.

## Key editorial decisions made this session

1. **H14 needs revision**: The claim should be "difficulty source determines predictability" not "MCQ vs. open-ended determines predictability." DBE-KT22 (MCQ, ρ=0.34) breaks the format confound.

2. **The paper is an optimization paper first, theory paper second.** The RSM/DOE framing gives it methodological identity. The complexity/selectivity distinction is a hypothesis that emerges from the optimization, not a pre-specified theory being tested.

3. **IRT not needed for SmartPaper or DBE-KT22.** When all students take all items, classical p-correct and IRT b are nearly perfectly correlated (DBE-KT22: ρ=0.999). IRT was necessary for Eedi because of adaptive assignment.

4. **Author-assigned difficulty on DBE-KT22 is useless** (ρ=0.069 vs empirical, ns). Only empirical p-correct is valid ground truth. (Documented in `pilot/dbe_kt22_validation/RESULTS.md`)

5. **Model consistency needed for final table**: All three datasets should use Gemini 3 Flash for the main comparison. The Gemini 3 Flash DBE-KT22 run is in progress.

## Key files

- `results-site/paper.html` — current paper draft (needs sections 1-2 replaced with draft.md content, needs DBE-KT22 results added)
- `paper_sections_draft.md` — preferred intro + method text (RSM/DOE framing, 5-phase pipeline)
- `scripts/run_dbe_kt22_validation.py` — baseline DBE-KT22 experiment (Gemini 2.5 Flash, completed)
- `scripts/run_dbe_kt22_hypotheses.py` — 3 hypothesis tests (contrastive, GPT-4o, classroom sim, running)
- `pilot/dbe_kt22_validation/` — all DBE-KT22 results and caches
- `pilot/dbe_kt22_validation/RESULTS.md` — documents IRT analysis, author difficulty finding
- `.claude/handoffs/2026-02-02-paper-rewrite.md` — previous session handoff (still relevant for Eedi/SmartPaper context)
- `.claude/handoffs/2026-02-02-hypothesis-registry.md` — full hypothesis registry

## Three-Phase Sequential DOE

### Design overview

The study uses **sequential experimentation** — each phase's results inform the next phase's design, pruning dead branches and directing resources to informative regions. This is standard adaptive DOE practice (Box & Draper, 2007).

| Phase | Purpose | Dataset | Design | Status |
|-------|---------|---------|--------|--------|
| 1. Screening | Identify best prompt × calibration × temperature | SmartPaper 20 probes | 5 prompts × 3 calibrations × 2 temps × 5 reps = 30 configs | RUNNING (15/30 done) |
| 2. Model survey | Test 8 models on full item set with top configs | SmartPaper 140 items | 8 models × 2–3 prompt configs × 3 reps | NEW — ready to launch |
| 3. Confirmation | Pre-specified test on held-out dataset | DBE-KT22 168 items | Top 3–5 configs × 5 reps | After Phase 2 |

### What we already have (not re-running)

| Dataset | Model | Method | ρ | n | Notes |
|---------|-------|--------|---|---|-------|
| SmartPaper 140 | Gemini 3 Flash | Teacher prediction, t=2.0, 3 reps | 0.547 | 140 | Averaged-pred |
| SmartPaper 140 | Scout | Teacher prediction, t=2.0, 3 reps | ~0.45 | 140 | From expansion |
| Indian State 210 | Scout | Direct, t=2.0, 3 reps | 0.319 | 210 | ~47% items have incomplete text |
| DBE-KT22 | Gemini 2.5 Flash | Direct, t=0 | 0.342 | 168 | Baseline |
| DBE-KT22 | Gemini 3 Flash | Direct, t=0 | Running | 168 | In progress (PID 9707) |
| DBE-KT22 | Gemini 2.5 Flash | Contrastive, t=1.5, 3 reps | Running | 168 | In progress (PID 6182) |

These serve as **internal replication checks** — Phase 2 should reproduce the Gemini 3 Flash SmartPaper result.

### Phase 1: Prompt Framing Experiment (RE-RUNNING — v2, clean)

**Critical fix:** The original Phase 1 screening (v1) had **data leakage** — every prompt contained "Average pass rate across items on this exam is about 29%", which is the mean of the ground truth distribution. Theory-based prompts (#6-10) also contained count-to-percentage lookup tables. All v1 results are invalidated.

**v1 results (INVALIDATED — archived in `pilot/smartpaper_rsm_v2/g3f_screening/`):**
- contrastive_errors_t1.0: ρ=0.811 (inflated by base-rate anchoring)
- contrastive_none_t2.0: ρ=0.694 (inflated)

**v2 design changes:**
- Removed "29% average pass rate" from all prompts
- Removed percentage lookup tables from theory prompts
- Removed percentage annotations from comparative difficulty scale
- Dropped 4 weak/redundant framings (contrastive, imagine_classroom, comparative_difficulty, population_aware)
- Runs on **full 140 items** (not 20 probes — 20-probe screening doesn't transfer, as contrastive ρ=0.694→0.077 proved)
- 3 reps (not 5 — Gemini is consistent enough)
- Output format: decimal 0-1 (not XX% — better parse rate)

**Design:** 7 framings × 2 temps × 3 reps × 140 items = 5,880 calls on Gemini 3 Flash Preview.

**Prompt factor (7 levels):**
1. **teacher** — direct teacher assessment (baseline, matches Phase 2)
2. **error_analysis** — "what errors would lead students to get this wrong?"
3. **devil_advocate** — "teachers overestimate — challenge your assumptions"
4. **prerequisite_chain** — KC Theory: count prerequisite failure points
5. **error_affordance** — BUGGY: count plausible error paths
6. **cognitive_load** — Sweller CLT: count elements held in working memory
7. **familiarity_gradient** — distance from textbook drills

**Temperature factor (2 levels):** 1.0 and 2.0

**No calibration factor** — pure zero-calibration only (the realistic use case).

**Script:** `scripts/run_prompt_screening_g3f.py`
**Output:** `pilot/prompt_framing_experiment/<framing>_t<temp>/rep<N>/<item>.txt`

**Status:** RUNNING — all 7 framings launched in parallel. Results pending.

**Portability note:** These prompts are SmartPaper-specific — they reference Indian government schools, EWS population, Hindi-medium backgrounds, open-ended format, and rubric scoring. The prompt *structures* (teacher, devil_advocate, prerequisite_chain, etc.) are generic, but the surface text must be adapted for Phase 3 (DBE-KT22: South African university, MCQ format, "answer correctly" not "score full marks"). This is by design: Phase 1 optimizes for the screening dataset, Phase 3 tests whether the *framing type* transfers with population-appropriate wording.

### Phase 2: Model Survey (RUNNING)

**Purpose:** Test whether the best prompt configurations generalize across models, and whether model size/capability predicts performance.

**Script:** `scripts/run_model_survey.py`
**Output:** `pilot/model_survey/<model>/<prompt>/rep<N>/<item>.txt`

**Design:** 6 models × 3 prompts × 3 reps on all 140 SmartPaper items.

**Model factor (6 levels):**

| Model | Parameters | Provider | Literature basis |
|-------|-----------|----------|-----------------|
| Llama-3.1-8B | 8B | Groq | Smallest — lower bound |
| Gemma-3-27B | 27B | Google GenAI | Closest to Kröger et al.'s Gemma-2-27B |
| Llama-4-Scout | 17B active (109B MoE) | Groq | Already have SmartPaper baseline |
| Llama-3.3-70B | 70B | Groq | Dense large model |
| Gemini 3 Flash | ~medium (unknown) | Google | Our anchor model |
| GPT-4o | ~large (unknown) | OpenAI | **Razavi & Powers' model** |

**Dropped:** Qwen3-32B (ρ≈0.05, 36% parse failures at t=2.0), Llama-4-Maverick (redundant with Scout), DeepSeek-V3 (unavailable on Groq), Gemma-2-27B (unavailable on Groq — substituted Gemma-3-27B via Google API).

**Prompt factor (3 levels):**
- **teacher** — direct teacher assessment ("estimate what proportion would score full marks")
- **contrastive** — error-focused framing ("what makes this easy/hard compared to similar content?")
- **simulation** — classroom simulation ("imagine 30 students with varying abilities")

**Temperature:** 2.0 (fixed).

**Existing data that overlaps (internal replication checks):**
- `smartpaper_expansion/gemini_flash/` — Gemini 3 Flash × teacher × 3 reps × 140 items (ρ=0.547)
- `smartpaper_expansion/llama4_scout/` — Scout × teacher × 3 reps × 140 items (ρ≈0.45)

**Early results (teacher prompt):**

| Model | Params | ρ (teacher) | 95% CI |
|-------|--------|-------------|--------|
| **Gemini 3 Flash** | ? | **0.550** | [0.411, 0.664] |
| Gemma-3-27B | 27B | 0.501 | [0.332, 0.648] |
| GPT-4o | ? | 0.360 | [0.205, 0.502] |
| Llama-3.1-8B | 8B | pending analysis | |
| Llama-4-Scout | 109B MoE | pending analysis | |
| Llama-3.3-70B | 70B | pending analysis | |

**Status:** All 6 models have teacher data collected. Contrastive + simulation running for all. Will run `--analyze` when complete.

**Hypotheses tested in Phase 2:**
| Hypothesis | Test |
|-----------|------|
| Model capability predicts success | Rank models by benchmark → rank by ρ |
| Parameter count predicts success | Plot ρ vs log(params), 8B → GPT-4o range |
| Razavi & Powers replication | GPT-4o with teacher prompt — they reported r=0.83 |
| System 1 > System 2 | Teacher vs simulation prompt, all models |
| Multi-rep averaging helps | Per-rep vs averaged ρ across all conditions |
| Prompt × model interaction | Does the best prompt depend on the model? |

### Phase 3: Confirmation on DBE-KT22 (after Phase 2)

**Purpose:** Pre-specified test of top configurations on a held-out dataset (different domain, MCQ format, university-level).

**Design:** Top 3-5 model×prompt configurations from Phase 2, run on all 168 DBE-KT22 items, 5 reps each.

**Pre-specified hypotheses** (written before seeing Phase 3 results):
1. The top SmartPaper configuration produces ρ > 0.30 on DBE-KT22 (above chance)
2. SmartPaper ρ > DBE-KT22 ρ for every configuration tested (dataset difficulty moderates)
3. Prompt ranking is preserved across datasets (no prompt × dataset crossover interaction)
4. Model ranking is preserved across datasets (no model × dataset crossover interaction)

**Prompt adaptation required:** Phase 1/2 prompts are SmartPaper-specific (Indian government schools, open-ended, rubric scoring). For Phase 3, prompts must be rewritten with DBE-KT22 population context (South African university students, MCQ format, "answer correctly" not "score full marks"). The *framing type* (e.g., devil_advocate, prerequisite_chain) is preserved; only the population/format surface text changes.

**Total calls:** ~5 configs × 5 reps × 168 items = **~4,200 calls**
**Cost:** ~$3-5
**Time:** ~1-2 hrs

### Phase 4 (optional): Indian State Assessment cross-validation

If time permits, run top configs on the 210-item Indian State Assessment. Tests generalization to a third population (Indian government schools) and to items with incomplete text (~47%). Already have Scout baseline (ρ=0.319).

### Total new experiment budget

| Phase | Calls | Cost | Time (parallel) |
|-------|-------|------|-----------------|
| Phase 1 (remaining 15 configs) | ~1,500 | ~$1 | Running |
| **Phase 2 (model survey)** | **~10,000** | **~$12** | **~2 hrs** |
| **Phase 3 (confirmation)** | **~4,200** | **~$4** | **~1 hr** |
| Phase 4 (optional) | ~3,000 | ~$3 | ~1 hr |
| **Total new** | **~18,700** | **~$20** | **~4-5 hrs** |

### Literature hypotheses coverage

| # | Hypothesis | Source | Where tested | Phase |
|---|-----------|--------|-------------|-------|
| 1 | Direct estimation works | Razavi & Powers 2025 | GPT-4o + direct prompt | 2 |
| 2 | Classroom sim beats direct | Kröger et al. 2025 | Simulation vs teacher prompt, all models | 1 + 2 |
| 3 | Cognitive scaffolding helps | AIED 2025, KC/BUGGY | 5 framings vs teacher baseline | 1 |
| 4 | Calibration anchors help | Our prior finding | 3 calibration levels | 1 |
| 5 | Error patterns help | Novel | errors vs none calibration | 1 |
| 6 | Temperature helps | Wisdom-of-crowds | t=1.0 vs t=2.0 | 1 |
| 7 | Model size predicts success | General expectation | ρ vs log(params), 8B-671B | 2 |
| 8 | Model capability predicts success | General expectation | ρ vs benchmark rank | 2 |
| 9 | Kröger's Gemma-2-27B works | Kröger et al. 2025 | Gemma-2-27B, simulation prompt | 2 |
| 10 | Razavi's GPT-4o works | Razavi & Powers 2025 | GPT-4o, direct prompt | 2 |
| 11 | System 1 > System 2 | Our Eedi finding | Teacher vs simulation, all models | 2 |
| 12 | Multi-rep averaging helps | Ensemble literature | Per-rep vs averaged, all conditions | 1 + 2 |
| 13 | Prompt is primary lever | Our Eedi finding | Prompt effect size vs model/temp | 1 + 2 |
| 14 | Dataset moderates everything | Our framework | SmartPaper vs DBE-KT22 | 3 |

**14 hypotheses, 3 phases, ~$20, ~5 hours.**

### Paper structure (revised)

1. **Intro** — Literature claims LLMs can predict item difficulty, but studies use small sets and single configurations. We apply sequential DOE to map the landscape.
2. **Datasets & Ground Truth** — SmartPaper (140 items, 728K students, open-ended), DBE-KT22 (168 items, 1.3K students, MCQ). Both complete assignment, classical p-correct. Brief note on Eedi exclusion.
3. **Phase 1: Prompt Screening** — 30-config DOE on SmartPaper probes. Identifies best framings. Tests scaffolding, calibration, temperature.
4. **Phase 2: Model Survey** — 6 models spanning 8B to GPT-4o on SmartPaper full set. Tests model size, replicates Razavi. Confirms System 1 finding.
5. **Phase 3: Cross-Dataset Confirmation** — Top configs on DBE-KT22. Pre-specified hypotheses. Tests generalization.
6. **Discussion** — Complexity/selectivity framework (post-hoc), what LLMs encode (PCK not cognitive models), practical recommendations, limitations.

## What still needs doing

1. **Wait for Phase 1 screening to finish** (15/30 configs done)
2. **Wait for running DBE-KT22 experiments** (Gemini 3 Flash direct, contrastive reps, GPT-4o, classroom sim)
3. **Launch Phase 2 model survey** — script: `scripts/run_model_survey.py` (WRITING NOW)
4. **After Phase 2**: analyze results, select top configs, design Phase 3
5. **Write paper** with sequential DOE narrative
6. **LaTeX conversion** for AIED 2026 submission
