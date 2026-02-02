# Study 2: Revised Experimental Design
## Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?

### Central Research Question
Can structured KC profile prompts overcome the capability paradox? Does prompt complexity interact with model capability to produce authentic misconceptions?

---

## Literature Gap This Study Addresses

### The Answer-Level Measurement Problem (Section 2.9)
Prior work relies exclusively on answer-level metrics:
- Liu et al. (2025): Correlation between LLM probabilities and student distractor patterns
- Lu & Wang (2024): Pearson r of answer distributions, accuracy
- SMART (2025): IRT fitting on correct/incorrect scores
- Agent4Edu (2025): Accuracy and F1-score
- "Take Out Your Calculators" (2026): IRT difficulty correlations

**No prior work analyzes WHY the LLM selected a wrong answer—only WHICH answer.**

### Evidence This Gap Matters
- Liu et al. (2025): "there remains a gap between LLM's underlying reasoning process and human cognitive processes"
- "Take Out Your Calculators": models predict WHETHER students get questions wrong but not HOW
- Comprehensive Review (Nov 2025): "Most systems rarely engage in cognitive role emulation"
- Critical finding: "No direct correlation between CoT-generated rationales and accuracy of simulated responses"

### Our Contribution
**First systematic analysis of LLM chain-of-thought reasoning against expert-labeled misconception taxonomies.**

This is novel because:
1. Prior work measures WHICH wrong answer; we measure WHETHER reasoning matches documented misconceptions
2. No previous study codes CoT against established misconception frameworks
3. Our Reasoning Authenticity Gap metric (50% target rate vs 8.7% alignment) quantifies this problem for the first time

---

## Current Findings (Pilot)
- **Reasoning Authenticity Gap:** 50% target distractor rate vs 8.7% misconception alignment
- **Capability Paradox Confirmed:** Frontier models achieve 100% accuracy, zero errors
- **CoT Effect:** Increases distractor selection (78% vs 53%) but NOT reasoning alignment

---

## Proposed 3×4 Factorial Design

### Factor 1: Model Capability (3 levels)

| Tier | Models | GSM8K | Rationale |
|------|--------|-------|-----------|
| Frontier | GPT-4o, Claude 3.5 Sonnet | ~95% | Test if structured prompts can induce authentic errors |
| Mid | GPT-3.5-turbo, Claude Haiku | ~75% | Current "sweet spot" for errors |
| Weak | Mistral-7B, Llama-3.1-8B | ~50% | Natural error generators |

### Factor 2: Prompt Complexity (4 levels)

| Level | Name | Specification | ESS Level |
|-------|------|---------------|-----------|
| P1 | Persona | General struggling student | E1 |
| P2 | Skill-Level | Weak at specific topic | E2 |
| P3 | Confusion Tuple | Explicit confuse(KC_A, KC_B) | E3 |
| P4 | KC Profile | Full mastered/confused/unknown + examples | E3-E4 |

---

## Key Hypotheses

**H1 (Capability Paradox):** At P1 (simple persona), misconception alignment: Weak > Mid > Frontier

**H2 (Prompt Compensation):** At P4 (KC profile), misconception alignment: Frontier ≈ Mid ≈ Weak

**H3 (Interaction):** Significant Prompt Complexity × Model Capability interaction—structured prompts benefit frontier models more than weak models

**H4 (Reasoning Gap):** The Reasoning Authenticity Gap (Target Rate - Alignment Rate) decreases with prompt complexity

---

## Dependent Variables

| Metric | Definition | What it measures |
|--------|------------|------------------|
| Error Rate | 1 - Accuracy | Needed for analysis |
| Target Distractor Rate | P(target distractor \| error) | Selecting the "right wrong answer" |
| Misconception Alignment Rate | P(CoT matches misconception \| error) | Authentic reasoning |
| **Reasoning Authenticity Gap** | Target Rate - Alignment Rate | Primary outcome |

---

## Design Matrix (12 cells)

```
              P1-Persona  P2-Skill  P3-Tuple  P4-Profile
Frontier         ✓           ✓         ✓          ✓
Mid              ✓           ✓         ✓          ✓
Weak             ✓           ✓         ✓          ✓
```

**Responses per cell:** 2 models × 100 items × 5 repetitions = 1,000 responses
**Total responses:** 12,000

---

## Exact Prompt Templates

### P1 - Persona
```
You are a middle school student taking a math test. Students at your level
sometimes make mistakes—that's okay and normal. Show your thinking, then
give your answer.

Question: {question}
Options: {options}
```

### P2 - Skill-Level
```
You are a student who is weak at {topic}. You often make mistakes on
problems involving {topic}. Show your thinking, then give your answer.

Question: {question}
Options: {options}
```

### P3 - Confusion Tuple
```
You are a student who confuses {KC_A} with {KC_B}. When you see problems
involving {KC_A}, you mistakenly apply {KC_B} instead. Show your thinking
as this student would.

Question: {question}
Options: {options}
```

### P4 - KC Profile (Lu & Wang 2024 style)
```
You are a student with this knowledge profile:

MASTERED (you can do these correctly):
- {mastered_KC}: Example: {correct_example}

CONFUSED (you mix these up):
- You confuse {KC_A} with {KC_B}
- Example of your error: {incorrect_example_with_reasoning}

UNKNOWN (no experience with):
- {unknown_KCs}

Solve the following problem as this student would, showing your reasoning.

Question: {question}
Options: {options}
```

---

## Item Selection (N=100)

### Source: Eedi Dataset
Filter criteria:
1. Dominant distractor: >40% of incorrect responses select target
2. Expert misconception label available
3. Grade level: 4-8 mathematics

### Stratification
- 50 Procedural misconceptions (rule misapplication)
- 50 Conceptual misconceptions (flawed mental models)

### Target Misconceptions (10 categories, 10 items each)

**Procedural:**
1. PROC_ORDER_OPS - Left-to-right instead of PEMDAS
2. PROC_SUBTRACT_REVERSE - Smaller from larger in each column
3. PROC_DECIMAL_PLACE - Misaligning decimal points
4. PROC_FRAC_ADD - Adding numerators and denominators separately
5. PROC_NEGATIVE_MULT - Sign errors in multiplication

**Conceptual:**
1. CONC_FRAC_DENOM - Larger denominator = larger fraction
2. CONC_MULT_INCREASE - Multiplication always increases
3. CONC_AREA_PERIMETER - Confusing area and perimeter formulas
4. CONC_PERCENT_BASE - Misidentifying the base in percentage problems
5. CONC_VARIABLE_LABEL - Treating variables as labels not quantities

---

## Analysis Plan

### Primary Analysis
1. **3×4 ANOVA** on Misconception Alignment Rate
   - Main effect of Model Capability
   - Main effect of Prompt Complexity
   - Interaction term (key test of RQ5)

2. **Planned Contrasts**
   - Frontier-P4 vs Weak-P1 (can prompts compensate?)
   - Linear trend of prompt complexity within each tier

### Secondary Analyses
1. **Reasoning Authenticity Gap** by cell
2. **Misconception Type** (procedural vs conceptual) as moderator
3. **Item difficulty** as covariate

### Qualitative Analysis
- Sample 20 responses per cell (240 total)
- Deep coding against Eedi misconception taxonomy
- Identify error signatures by condition

---

## Timeline

| Task | Deadline |
|------|----------|
| Finalize item selection | Jan 27 |
| Implement P3, P4 prompts | Jan 28 |
| Run full data collection | Jan 29-30 |
| Misconception coding | Jan 31 |
| Analysis and writeup | Feb 1-2 |
| Submit | Feb 2 |

---

## How This Design Addresses the Literature Gap

| Gap in Literature | How Our Design Addresses It |
|-------------------|----------------------------|
| Answer-level metrics only | Primary DV is **Misconception Alignment Rate** (reasoning-level) |
| No CoT coding against taxonomies | Systematic coding against Eedi's 2,587 expert misconception labels |
| Unknown if prompts can compensate for capability | 3×4 factorial tests **prompt-capability interaction** |
| "No correlation between CoT quality and simulation accuracy" | We measure both independently and quantify the **Reasoning Authenticity Gap** |
| Capability paradox proposed but not tested at reasoning level | We test whether weaker models produce more **authentic reasoning** (not just more errors) |

---

## Open Questions

1. **Frontier model errors:** If frontier models still achieve ~100% accuracy even with P4 prompts, how do we analyze them?
   - Option A: Report "resistance to error induction" as finding
   - Option B: Use temperature manipulation
   - Option C: Focus on mid/weak tiers for alignment analysis

2. **KC Profile construction:** How to construct P4 prompts for Eedi items?
   - Need to map Eedi misconceptions to KC pairs
   - May need to create worked examples for each misconception

3. **Coding reliability:** Can we achieve reliable misconception coding at scale?
   - Consider LLM-assisted coding with human validation
   - Use Eedi's misconception descriptions as coding rubric

---

## Novelty Verification (Research Completed Jan 26, 2026)

### Claim: "Most work is answer-only"
**VERIFIED** - All major papers use answer-level metrics only.

### Claim: "CoT reveals different reasoning than students"
**VERIFIED** - Multiple sources confirm LLM reasoning differs from student reasoning even when selecting same wrong answer.

### Claim: "Coding CoT against misconception taxonomies is novel"
**VERIFIED** - No prior paper systematically codes LLM chain-of-thought against established misconception frameworks.

### Key Supporting Quote (Comprehensive Review, arXiv 2511.06078):
> "Most systems rarely engage in cognitive role emulation—that is, modeling the internal reasoning processes of different learner types."
