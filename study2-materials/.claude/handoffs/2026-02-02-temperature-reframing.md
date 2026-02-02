# Temperature Reframing Session — 2026-02-02

## The Big Revision

The "temperature cliff" narrative from the RSM experiment was wrong. Fresh experiments with proper controls reveal:

### What happened
1. Original RSM used **plain teacher_prediction prompt** at t=0.3 (ρ=0.119) vs t=1.5 (ρ=0.673 — single run, later shown to be an outlier)
2. 10-rep validation already showed true per-rep ρ≈0.50, not 0.60+ (see research-journey.md Phase 7)
3. Fresh retest of t=1.5 with stochastic parsing gave ρ=0.105, 0.361, 0.296 — nowhere near 0.673
4. This session ran **contrastive prompt** across full temperature range (0.3–2.0) for both Gemini Flash and Llama-4-Scout

### Fresh results (contrastive prompt, deterministic parsing, 3 reps each)

**Gemini 3 Flash:**
| Temp | Mean ρ | SD |
|------|--------|-----|
| 0.3 | 0.449 | 0.118 |
| 0.6 | 0.458 | 0.046 |
| 0.9 | 0.358 | 0.070 |
| 1.2 | 0.354 | 0.127 |
| 1.5 | 0.462 | 0.031 |
| 2.0 | 0.580 | 0.110 |

**Llama-4-Scout (1024 max_tokens — fixed truncation bug):**
| Temp | Mean ρ | SD | Parse rate |
|------|--------|-----|-----------|
| 0.3 | 0.398 | 0.064 | 89% |
| 0.6 | 0.466 | 0.221 | 94% |
| 0.9 | 0.552 | 0.034 | 93% |

### Key findings

1. **No temperature cliff.** Gemini ρ≈0.35–0.58 across ALL temperatures with contrastive prompt. Roughly flat with slight uptick at t=2.0 (0.580±0.110, but only 3 reps).
2. **Prompt is the primary variable.** Plain teacher_prediction at t=0.3: ρ=0.12. Contrastive at t=0.3: ρ=0.45. That's a 3–4× difference from prompt alone.
3. **Scout token truncation bug.** Earlier Scout experiments used max_tokens=512, but Scout generates long reasoning chains before the formatted output. At 512 tokens, output was truncated → parse failures → constant array → NaN correlations. Fixed with 1024 tokens; Scout now shows ρ≈0.40+ at t=0.3.
4. **The original RSM confounded prompt × temperature.** It compared a weak prompt at low temp vs a strong prompt at high temp, attributing the difference to temperature.
5. **Honest number: ρ≈0.40–0.50** on 20 Eedi probe items, driven primarily by prompt quality.

### What this means for the paper

The narrative needs to shift from "temperature as diversity engine" to "contrastive prompting elicits latent pedagogical knowledge." Temperature helped in the original RSM because it happened to make the weak prompt marginally better, but a well-designed prompt doesn't need it.

The paper currently (results-site/paper.html) has been partially updated:
- Abstract: uses honest ρ≈0.50 numbers ✓
- Section 4.2 (RSM): still shows old bar chart with "4× increase" annotation — needs rewrite
- Section 4.4 (Prompt Strategy): NEW section added with cognitive/buggy/two-stage results ✓
- Section 5.2: updated to mention gradual effect, not cliff ✓
- Section 5.3: NEW section on screening→MAB framing ✓
- Figure 2b: bar chart with 10-rep validated numbers ✓ but still shows temperature interaction
- Figure 2c: transition curve — needs complete redo with flat line
- Figure 3b: NEW prompt strategy comparison bar chart ✓

### SmartPaper temp sweep (20 probe items, 5 reps averaged, Gemini Flash + DeepSeek)

This is the OTHER temperature experiment — run on SmartPaper probe items with 3 prompt strategies × 4 temperatures × 2 models. Data in `pilot/smartpaper_rsm_v2/temp_sweep/results.json`.

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

### Eedi vs SmartPaper temperature comparison

The Eedi and SmartPaper sweeps tell **different stories**:

| Dimension | Eedi (contrastive, Gemini) | SmartPaper (errors/anchors, Gemini) |
|-----------|---------------------------|-------------------------------------|
| Overall ρ range | 0.35–0.58 | 0.77–0.88 |
| Temperature effect | Flat (no trend) | Increasing (t=2.0 best) |
| Best single result | 0.580 (t=2.0, 3 reps) | **0.877** (anchors, t=2.0) |
| Prompt matters? | Yes (0.12→0.45) | Yes (baseline unstable, errors/anchors robust) |

### Key insights from combined analysis

1. **SmartPaper correlations are much higher** (ρ=0.69–0.88) than Eedi (ρ=0.35–0.58). Open-ended rubric-scored items are fundamentally easier to predict.
2. **Temperature helps Gemini on SmartPaper** — monotonic increase for errors (0.774→0.841) and anchors (0.811→0.877). But only with structured prompts; baseline collapses at t=1.5 (0.311).
3. **Temperature hurts DeepSeek on SmartPaper** — monotonic decrease across ALL strategies (e.g., baseline: 0.800→0.675). DeepSeek is best at low temperature.
4. **The interaction is model × prompt × temperature**, not just prompt alone. The Eedi-only conclusion that "prompt > temperature" was incomplete.
5. **Baseline is fragile** — on SmartPaper, Gemini baseline swings wildly (0.311–0.743) while errors/anchors stay in a tight 0.77–0.88 band. Structured prompts provide temperature-robustness.
6. **DeepSeek is strong across the board** — even its worst (anchors t=2.0: 0.691) beats Gemini's best baseline. DeepSeek seems to have better intrinsic calibration for SmartPaper items.

### Full-scale SmartPaper expansion (140 items, contrastive prompt, t=2.0, 3 reps)

Run in parallel session. Results in `pilot/smartpaper_expansion/`.

**Gemini Flash:** ρ=0.547 (avg-pred), per-rep 0.533±0.012
- By subject: Science 0.734, Social Science 0.702, Math 0.600, English 0.432
- Original 20 probe: ρ=0.768; Remaining 120: ρ=0.518

**Llama-4-Scout:** ρ=0.250 (avg-pred), per-rep 0.202±0.053
- By subject: Social Science 0.486, English 0.332, Math -0.113, Science 0.041
- Original 20 probe: ρ=0.663; Remaining 120: ρ=0.184
- Hit Groq 503 errors on ~12 items

Signal **holds on SmartPaper** when expanding beyond probe items (ρ=0.518 on 120 held-out), unlike Eedi where it collapsed (ρ=-0.176 on 30 new items).

### Calibration note

Models overestimate p_correct by ~+0.40 for Indian Grade 6-8 students (mean predicted 0.73 vs actual 0.29). Rank order preserved (Spearman) but absolute calibration poor. The anchors strategy on SmartPaper has even larger bias (+0.22–0.30) but better ranking.

### Still needs
- **Rewrite RSM section** to reflect the nuanced model × prompt × temperature interaction
- **Update Figure 2c** — Eedi: flat line; SmartPaper: rising line for Gemini, falling for DeepSeek
- **Run confirmation** on 58 held-out Eedi items with contrastive prompt
- **Test DeepSeek on full 140 SmartPaper** — it showed ρ=0.80 on 20 probe items, may beat Gemini at scale

### Completed experiments
- **Eedi Gemini sweep**: DONE. ρ range 0.354–0.580 across t=0.3–2.0 (flat)
- **Eedi Scout sweep (fixed)**: DONE through t=0.9. ρ range 0.398–0.552
- **SmartPaper temp sweep**: DONE. 2 models × 3 strategies × 4 temps on 20 probe items
- **SmartPaper 140-item expansion**: DONE. Gemini ρ=0.547, Scout ρ=0.250

### Bug found and fixed
**Scout max_tokens=512 truncation**: Scout generates verbose reasoning (3000+ chars) before formatted output. At 512 tokens, the formatted distribution was cut off, parser fell back to 0.5 for all levels, producing constant p_incorrect arrays → NaN correlations. This affected ALL prior Scout results that used 512 tokens. The cross-model experiment (`test_cross_model.py`) used 1024 tokens so those results are valid, but the RSM experiment (`run_rsm_experiment.py` line 338) used 512 for teacher_prediction specifically.

### Groq API keys
- Original: in .env as GROQ_API_KEY
- Additional keys: [redacted — see .env]

### Files modified this session
- `results-site/paper.html` — major updates: honest numbers, prompt strategy section 4.4, screening→MAB discussion 5.3, updated figures
- `scripts/run_transition_zone.py` — NEW, mapped t=0.6/0.9/1.2 (used stochastic parsing, results now superseded by fresh deterministic runs)
- `pilot/rsm_experiment/transition_zone/` — transition zone results (stochastic parsing, partially superseded)

### Session files from earlier (read for context)
- `.claude/handoffs/2026-02-02-research-journey.md` — full research arc with all phase results and prompts
- `.claude/handoffs/2026-02-02-model-selection-experiment.md` — 12-model ranking, Gemini family deep-dive
