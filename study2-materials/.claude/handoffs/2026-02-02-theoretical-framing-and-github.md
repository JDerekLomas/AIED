# Theoretical Framing & GitHub Push — 2026-02-02

## What happened this session

### 1. Theoretical framing added to paper

User asked what theory underpins the prompt improvements. I synthesized across `prompt-strategies-v2.md`, `docs/difficulty_estimation_prompts.md`, and the paper itself, then updated `results-site/paper.html` with:

**Section 1 — "Theoretical Traditions Under Test"** (new subsection):
- KC theory (Koedinger et al., 2012) → teacher prediction framing, Knowledge State (S2) prompts
- Buggy Procedures (Brown & Burton, 1978) → Production Rules (S4) prompts
- Conceptual Change (Chi et al., 1994, 2008) → Mental Model (S3) prompts
- ESS framework (arXiv:2601.05473) → first empirical test of E3-level prompting
- Key setup: all four traditions predict more specification = better estimation. Our finding: it doesn't.

**Section 4.4 — retitled "Testing Theoretical Predictions"**:
- Added "Theory" column to strategy comparison table
- New paragraph explaining WHY BUGGY and ESS fail: they model *which* errors, not their *prevalence*
- Strengthened "context ≠ mechanism" finding: LLMs have implicit PCK but can't *be* students

**Section 5.1 — strengthened complexity/selectivity framework**:
- Connected to KC theory (complexity-driven) and BUGGY (selectivity-driven)
- Three numbered explanatory patterns with citations

**Section 5.4 — "What We Replicated and What We Didn't"** (new):
- Replication scorecard table: 8 prior claims, sources, and results
- Pattern: claims from cognitively diverse item pools fail on misconception-targeted items

**Conclusion — rewritten**:
- Theoretical contribution framed as the complexity/selectivity boundary
- All four traditions predict richer specification should help; central finding is it doesn't

**References added**: Brown & Burton (1978), Chi et al. (1994), Chi (2008), Koedinger et al. (2012), Matsuda et al. (2015), Razavi & Powers (2025)

### 2. GitHub push

Committed and pushed everything to https://github.com/JDerekLomas/AIED (commit 3fb75f2):
- ~25k raw model response files in `pilot/`
- 40+ experiment scripts in `scripts/`
- SmartPaper dataset (140 items), Eedi curated items, IRT parameters
- Full paper (`results-site/paper.html`)
- Research handoffs, literature review, design docs
- **API keys stripped** from all files before commit:
  - Groq keys removed from handoffs and `run_enhanced_classroom_sim.py`
  - DeepSeek keys removed from handoffs, `test_cross_model.py`, `run_3rep_fullset.py`
  - All replaced with `os.environ.get()` or `[REDACTED]`
- `node_modules/` added to `.gitignore`

### Key theoretical insight (the paper's contribution)

LLM difficulty estimation is bounded by the **source** of difficulty:
- **Complexity-driven** (KC theory): how many/which knowledge components → LLMs can estimate from text → SmartPaper ρ=0.55, generalizes
- **Selectivity-driven** (BUGGY): population prevalence of specific buggy procedures triggered by specific item features → LLMs fail → Eedi ρ≈0

No amount of cognitive scaffolding (misconception hints, confusion tuples, production rules, student personas) helps because the binding constraint is not what the model knows about misconceptions but what it cannot know about their prevalence in a specific population.

### Files modified
- `results-site/paper.html` — theoretical framing throughout (intro, 4.4, 5.1, 5.4, conclusion, references)
- `.gitignore` — added `node_modules/`
- `.claude/handoffs/2026-02-02-temperature-reframing.md` — API keys redacted
- `.claude/handoffs/2026-02-02-dashboard-and-3rep.md` — API keys redacted
- `.claude/handoffs/2026-02-02-two-stage-diversity.md` — API keys redacted
- `.claude/handoffs/2026-02-02-model-selection-experiment.md` — API keys redacted
- `.claude/handoffs/2026-02-02-cross-model-expansion.md` — API keys redacted
- `scripts/run_enhanced_classroom_sim.py` — hardcoded Groq key → env var
- `scripts/test_cross_model.py` — hardcoded DeepSeek key → env var
- `scripts/run_3rep_fullset.py` — hardcoded DeepSeek key → env var

### Still needs
- Section numbering in 5.x shifted (5.4→5.5 for data bug, 5.5→5.6 for limitations, 5.6→5.7 for implications) — verify rendering
- The 3-rep fullset experiment is still partially complete (Scout ~40%, DeepSeek/Gemini not started)
- Phase 14 (two-stage destroys SmartPaper signal) is in research-journey.md but only briefly in the conclusion, not yet a full paper section
- Could add a "replication attempts" subsection specifically for the reasoning-augmented approach (arXiv:2503.08551) which we tested and failed to replicate
