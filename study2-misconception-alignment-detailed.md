# Study 2: Misconception Alignment (Detailed Design)

## Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?

*Draft v3: January 26, 2026*
*Updated: Added capability gradient hypothesis (RQ5, H4, H6)*

---

## Executive Summary

This study tests whether LLM-simulated students exhibit the same *underlying misconceptions* as real students—not just whether they get the same items wrong, but whether their reasoning reflects the same flawed mental models. Using the Eedi dataset (1,868 items with expert-labeled misconceptions per distractor), ASSISTments (computed distractor rates), and researcher-generated probe items, we compare LLM chain-of-thought reasoning against established misconception taxonomies. Primary outcome: misconception category match rate between LLM reasoning and the misconception label associated with the distractor the LLM selected.

---

## Research Questions

| RQ | Question | Analysis |
|----|----------|----------|
| **RQ1** | When an LLM selects a distractor, does its reasoning reflect the misconception that distractor was designed to capture? | Misconception match rate (LLM reasoning coded against Eedi taxonomy) |
| **RQ2** | Does misconception alignment vary by misconception type (procedural vs. conceptual vs. interpretive)? | Stratified analysis by misconception category |
| **RQ3** | Does prompting strategy affect misconception alignment? | Comparison: answer-only vs. explain-then-answer vs. student-persona |
| **RQ4** | Can we generate synthetic probe items that reliably elicit specific misconceptions from LLMs? | Validation of generated items against LLM responses |
| **RQ5** | Does model capability predict misconception alignment—and does this relationship differ when simulating low-performing vs. high-performing students? | Capability × simulated-ability interaction analysis |

---

## Theoretical Framework

### Why Misconception Match > Distractor Match

Prior work focuses on: *"Did the LLM pick distractor B?"*

We focus on: *"Did the LLM's reasoning show the misconception that distractor B tests?"*

This matters because:
1. **Diagnostic validity**: If LLMs pick the right wrong answer for wrong reasons, they can't help us understand student thinking
2. **Item design**: Synthetic students are only useful for distractor evaluation if they engage with distractors as intended
3. **Generalization**: Misconception-level alignment may transfer across items testing the same misconception

### Misconception Taxonomy

We adopt Eedi's taxonomy (derived from mathematics cognition literature) with three top-level categories:

| Category | Definition | Example |
|----------|------------|---------|
| **Procedural** | Correct concept, flawed execution | Forgetting to carry; wrong order of operations |
| **Conceptual** | Fundamental misunderstanding | Believing subtraction is commutative; thinking multiplying by fraction increases value |
| **Interpretive** | Misreading problem or representation | Misidentifying axes; confusing units; misreading word problem |

Eedi provides fine-grained misconception labels (e.g., "Believes multiplication is repeated addition only," "Confuses area and perimeter"). We use both levels.

### The Capability Paradox Hypothesis

A counterintuitive possibility: **weaker models may be better at simulating struggling students than frontier models.**

**Rationale:**
- Frontier models must *perform* incompetence—their errors are roleplay, not authentic failure
- When GPT-4o "pretends" to be a struggling student, its errors may be qualitatively different from real student errors
- Weaker models might *naturally* produce errors that resemble student misconceptions
- The optimal simulator for a low-performing student may be a model whose actual capability matches the target student level

**Prior evidence:**
- "Models that are adept at solving mathematical problems may struggle to authentically simulate the response patterns of struggling students" (Take Out Your Calculators, 2026)
- "Smaller models can be efficiently utilized for distractor generation" (Liu et al., AIED 2025)
- Pedagogically fine-tuned models showed no alignment advantage over base models (NAEP study, 2025)

**Implication:** If true, the right tool for simulating struggling students isn't the smartest model with a persona prompt, but a model whose actual capability matches the target student's ability level.

---

## Datasets

### 1. Eedi (Primary)

**Source:** [Kaggle Competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

| Attribute | Value |
|-----------|-------|
| Items | 1,868 diagnostic questions |
| Format | MCQ (1 correct + 3 distractors) |
| Labels | Each distractor mapped to specific misconception from taxonomy |
| Student data | Response frequencies per option; optional written explanations |
| Topics | Grades 4-8 mathematics |

**Why it's ideal:**
- Ground truth misconception labels per distractor (not just "wrong")
- Human-annotated by 15 trained math tutors
- Student explanations available for validation

**Sampling strategy:**
- Select items where one distractor is clearly dominant (>40% of incorrect responses)
- Balance across misconception types (procedural/conceptual/interpretive)
- Balance across difficulty levels
- Target: N=400 items

### 2. ASSISTments (Secondary)

**Source:** [ASSISTments Data](https://sites.google.com/site/assistmentsdata/)

| Attribute | Value |
|-----------|-------|
| Items | ~26,000 questions |
| Format | Mixed (MCQ + open response) |
| Labels | Knowledge components; no misconception labels |
| Student data | Individual attempt records |

**Use case:**
- Compute distractor selection rates from raw data
- Cross-validate findings from Eedi
- Test generalization to different item style

**Sampling strategy:**
- Filter to MCQ items only
- Compute distractor frequencies from attempt data
- Select items with clear dominant distractor
- Researcher codes misconception type post-hoc
- Target: N=200 items

### 3. Generated Probe Items (Validation)

**Purpose:** Test whether we can *deliberately* elicit specific misconceptions from LLMs

**Method:**
- Select 10 high-frequency misconceptions from Eedi taxonomy
- For each, generate 5 novel items designed to trigger that misconception
- Validate: Does the LLM exhibit the target misconception?

**Example:**

| Target Misconception | Generated Item |
|---------------------|----------------|
| "Believes subtraction is commutative" | "Calculate: 8 - 15. Options: A) -7, B) 7, C) 23, D) -23" |
| "Confuses perimeter and area" | "A rectangle is 4m by 6m. What is its perimeter? A) 24m, B) 20m, C) 10m, D) 48m" |

**Target:** 50 generated items (10 misconceptions × 5 items)

---

## Independent Variables

### Factor 1: Model Capability (3 tiers × 2 models = 6 models)

We deliberately select models across a capability gradient, indexed by GSM8K performance (grade-school math benchmark). This enables testing whether model capability predicts misconception alignment differently when simulating low- vs. high-performing students.

| Tier | Model | GSM8K (approx) | Rationale |
|------|-------|----------------|-----------|
| **Frontier** | GPT-4o | ~92% | State-of-the-art reasoning |
| **Frontier** | Claude 3.5 Sonnet | ~90% | Different training approach |
| **Mid** | Llama-3.1-70B | ~83% | Strong open-source |
| **Mid** | GPT-4o-mini | ~75% | Cost-efficient frontier derivative |
| **Weaker** | Llama-3.1-8B | ~56% | Small open-source |
| **Weaker** | Mistral-7B | ~52% | Popular small model |

**Capability as continuous predictor:** In addition to tier-based analysis, we treat GSM8K score as a continuous predictor to test linear/nonlinear relationships with misconception alignment.

**Cost note:** Weaker models are substantially cheaper per token, so expanding from 3 to 6 models has minimal budget impact (~$100-150 additional).

### Factor 2: Prompting Strategy (3 levels)

**Level 1: Answer-Only**
```
Solve this math problem. Select the best answer.

[Question and options]

Answer:
```

**Level 2: Explain-Then-Answer**
```
Solve this math problem. Show your reasoning step by step, then select the best answer.

[Question and options]

Reasoning:
Answer:
```

**Level 3: Student Persona**
```
You are a middle school student who sometimes makes mistakes in math.
Solve this problem as that student would—it's okay to make errors.
Show your thinking, then give your answer.

[Question and options]

Thinking:
Answer:
```

### Factor 3: Simulated Ability Level (3 levels, for Persona condition only)

When using the Student Persona prompt, we vary the described ability level:

| Level | Prompt Modification | Target |
|-------|--------------------| -------|
| **Low** | "You are a middle school student who struggles with math and often makes mistakes..." | Below Basic / Basic students |
| **Medium** | "You are an average middle school student who sometimes makes mistakes in math..." | Basic / Proficient students |
| **High** | "You are a middle school student who is good at math but occasionally makes careless errors..." | Proficient / Advanced students |

**Key test:** Does model capability × simulated ability interact? Specifically:
- Do weaker models show *better* misconception alignment when simulating low-ability students?
- Do frontier models show *better* alignment when simulating high-ability students?

### Factor 4: Misconception Type (3 levels)

Coded from Eedi taxonomy:
- Procedural
- Conceptual
- Interpretive

---

## Dependent Variables

### Primary DV: Misconception Category Match

**Definition:** When the LLM selects a distractor, does its reasoning reflect the misconception that distractor was designed to capture?

**Measurement:**
1. LLM selects distractor (e.g., option B)
2. Look up Eedi's misconception label for option B (e.g., "Confuses multiplication and addition")
3. Code LLM's reasoning for expressed misconception
4. Score: Match (1) if LLM reasoning aligns with label; No Match (0) if different error or no clear error

**Coding scheme:**

| Code | Definition | Example |
|------|------------|---------|
| **Full Match** | LLM reasoning clearly expresses the labeled misconception | Label: "Forgets to carry"; LLM: "8+7=15, write 5..." (no carry shown) |
| **Partial Match** | Related error in same category | Label: "Forgets to carry"; LLM: "Made arithmetic error" (procedural, but vague) |
| **Category Match Only** | Same top-level category, different specific misconception | Label: "Forgets to carry"; LLM: "Wrong order of operations" (both procedural) |
| **No Match** | Different category or unrelated error | Label: "Forgets to carry"; LLM: "Misread the problem" (interpretive, not procedural) |
| **Correct/No Error** | LLM answered correctly | — |
| **Uncodeable** | Reasoning unclear or missing | — |

### Secondary DVs

| DV | Definition |
|----|------------|
| **Distractor Match Rate** | % of LLM errors selecting the most common human distractor |
| **Error Rate Calibration** | Correlation between LLM error rate and human error rate per item |
| **Misconception-Level Accuracy** | For items testing same misconception, consistency of LLM behavior |

---

## Procedure

### Phase 1: Item Selection & Preparation (Week 1-2)

1. **Eedi sampling:**
   - Download full dataset from Kaggle
   - Filter to items with dominant distractor (>40% of incorrect)
   - Stratify by misconception type and difficulty
   - Select N=400

2. **ASSISTments preparation:**
   - Download 2012 dataset
   - Filter to MCQ items
   - Compute distractor frequencies
   - Code misconception types (researcher + RA)
   - Select N=200

3. **Probe item generation:**
   - Identify top 10 misconceptions by frequency in Eedi
   - Generate 5 items per misconception using GPT-4 (with human review)
   - Validate items with small pilot (N=3 students or cognitive interview)

### Phase 2: Response Generation (Week 3-4)

**Design structure:**
- 6 models × 2 baseline prompts (Answer-Only, Explain) × N=10 responses = 120 responses per item
- 6 models × 3 ability levels (Persona condition) × N=10 responses = 180 responses per item
- Total per item: 300 responses
- Total: 650 items × 300 = 195,000 responses

**Note on response count:** Reduced from N=20 to N=10 per condition to accommodate expanded model set while maintaining statistical power for between-condition comparisons.

**Data captured:**
- Selected answer (A/B/C/D)
- Full reasoning text (for Explain and Persona conditions)
- Token probabilities for answer options (if available)
- Model capability index (GSM8K score)

**Infrastructure:**
- Batch API calls (OpenAI, Anthropic, Together.ai for Llama/Mistral)
- Estimated cost: ~$400-600 (weaker models offset frontier costs)

### Phase 3: Coding & Analysis (Week 5-7)

**Quantitative analysis:**
1. Compute distractor match rates per condition
2. Compute error rate correlations with human data
3. 3×3×3 factorial ANOVA on misconception match rate

**Qualitative coding (light touch):**
1. Sample N=50 items stratified by:
   - High vs. low distractor match
   - Misconception type
   - Model
2. Two coders independently code LLM reasoning
3. Compute inter-rater reliability (Cohen's κ)
4. Resolve disagreements through discussion
5. Identify themes: When does misconception match succeed/fail?

### Phase 4: Probe Item Validation (Week 6)

1. Run all 50 probe items through 3 models × 3 prompts
2. Code: Did LLM exhibit target misconception?
3. Compute "elicitation rate" per misconception
4. Identify which misconceptions are reliably elicitable

---

## Analysis Plan

### Primary Analysis 1: Misconception Match (Overall)

**Model:**
```
Match ~ ModelCapability * Prompt * MisconceptionType + (1|Item) + (1|MisconceptionID)
```

Mixed-effects logistic regression:
- Fixed effects: ModelCapability (continuous GSM8K or 3-level tier), Prompt, MisconceptionType, interactions
- Random effects: Item (account for item difficulty), MisconceptionID (account for misconception-level clustering)

### Primary Analysis 2: Capability × Simulated Ability Interaction

**Model (Persona condition only):**
```
Match ~ ModelCapability * SimulatedAbility * MisconceptionType + (1|Item) + (1|MisconceptionID)
```

**Key test:** Is the ModelCapability × SimulatedAbility interaction significant?
- If positive: higher capability models are better across all simulated ability levels
- If negative: lower capability models are better for simulating low-ability students
- If null: model capability doesn't matter for simulation quality

**Visualization:** Heatmap of misconception match rates with ModelCapability (rows) × SimulatedAbility (columns)

**Planned contrasts:**
- Procedural vs. Conceptual misconceptions
- Student Persona vs. other prompts
- Frontier tier vs. Weaker tier
- Linear effect of GSM8K on alignment (is it monotonic?)

### Secondary Analyses

1. **By misconception frequency:** Do LLMs better match common vs. rare misconceptions?
2. **By item difficulty:** Does alignment vary with item difficulty?
3. **Cross-dataset:** Do findings replicate from Eedi to ASSISTments?
4. **Capability-difficulty matching:** Do weaker models show better alignment on harder items (where students struggle)?
5. **Error authenticity:** Qualitative comparison of error types across capability tiers

### Qualitative Analysis

**Thematic coding of N=100 items (stratified by model tier):**

| Theme | Description |
|-------|-------------|
| **Shallow match** | LLM picks right distractor but reasoning is generic/vague |
| **Deep match** | LLM reasoning clearly parallels documented student thinking |
| **Sophisticated error** | LLM makes error a student wouldn't (too complex) |
| **Random/lucky** | No clear reasoning but happened to match |
| **Overthinking** | LLM reasons itself into wrong answer via excess complexity |
| **Authentic struggle** | Reasoning shows genuine confusion similar to student explanations |
| **Performed incompetence** | Reasoning feels artificial, like "acting" at being wrong |

**Capability-specific coding:**
- Sample 50 items from frontier models, 50 from weaker models
- Code for "authenticity of error" — does this feel like a real student or a smart system pretending to be wrong?
- Compare qualitative patterns across tiers

---

## Hypotheses & Expected Findings

| Hypothesis | Prediction | Rationale |
|------------|------------|-----------|
| **H1** | Overall misconception match rate: 30-50% | Prior work shows moderate alignment; full match is harder than distractor match |
| **H2** | Procedural > Conceptual match | Procedural errors are surface-level; conceptual require deeper (mis)understanding |
| **H3** | Student Persona > Explain > Answer-only | Persona prompting licenses errors; explanation reveals reasoning |
| **H4** | **Capability × SimulatedAbility interaction: negative** | Weaker models will show *better* misconception alignment when simulating low-performing students; frontier models' "performed incompetence" will be less authentic |
| **H4a** | Weaker models (Llama-8B, Mistral-7B) will outperform frontier models on low-ability simulation | Their natural errors may resemble student errors more than roleplay |
| **H4b** | Frontier models will show more "sophisticated errors" (errors students wouldn't make) | Overthinking, unusual reasoning paths |
| **H5** | Generated probes: 60-80% elicitation rate | Targeted items should reliably trigger target misconceptions |
| **H6** | Optimal model-student matching: alignment peaks when model capability ≈ target student ability | "Capability matching" as a design principle |

---

## Limitations & Mitigations

| Limitation | Mitigation |
|------------|------------|
| Eedi taxonomy may not cover all LLM errors | Include "other" category; report uncategorized rate |
| Coding is subjective | Two coders + IRR; coding training; clear rubric |
| Only math domain | Acknowledge; discuss generalization cautiously |
| LLM reasoning may be post-hoc rationalization | Compare answer-only vs. explain conditions |
| Temperature introduces noise | N=20 per condition; report variance |

---

## Contributions

1. **First study of misconception-level alignment** (not just distractor-level)
2. **Leverages expert-labeled misconception dataset** (Eedi) as ground truth
3. **Tests prompting effects on misconception expression**
4. **Develops coding scheme** for LLM-misconception mapping
5. **Validates probe item generation** for targeted misconception elicitation
6. **Tests the "capability paradox" hypothesis:** Are weaker models better simulators of struggling students?
7. **Practical implications:** Which model capability level is appropriate for simulating which student population?
8. **Design principle:** "Capability matching" as a framework for selecting synthetic student models

---

## Timeline

| Week | Phase | Activities |
|------|-------|------------|
| 1-2 | Preparation | Item selection, probe generation, pilot coding |
| 3-4 | Generation | API calls, data collection |
| 5-6 | Coding | Qualitative coding, probe validation |
| 7 | Analysis | Statistical modeling, visualization |
| 8-10 | Writing | Draft, revisions, submission |

---

## Budget

| Item | Cost |
|------|------|
| API calls - Frontier (GPT-4o, Claude 3.5 Sonnet) | $300-400 |
| API calls - Mid (Llama-70B, GPT-4o-mini) | $100-150 |
| API calls - Weaker (Llama-8B, Mistral-7B) | $20-40 |
| Coding assistance (RA, 25 hrs) | $500 |
| Kaggle/data access | Free |
| **Total** | ~$920-1,090 |

**Note:** Expanding to 6 models is roughly cost-neutral because weaker models are 10-50x cheaper per token than frontier models.

---

## Submission Target

**Primary:** AIED 2026 (deadline passed for main track; check late-breaking/workshop)

**Alternatives:**
- EDM 2026 (Feb 9 deadline) — fits "data mining" angle
- LAK 2026 — fits "learning analytics" angle
- L@S 2026 (Feb 16 deadline) — fits "scale" angle via synthetic data
- JEDM or IJAIED journal — longer format for full study

---

## Open Questions for Discussion

1. **Coding granularity:** Code to Eedi's fine-grained labels (100+ categories) or collapse to top-level (3 categories)?
   - Recommendation: Code fine-grained, analyze both levels

2. **What counts as "match"?** Full match only, or include partial/category match?
   - Recommendation: Report all levels; primary analysis on full+partial

3. **How to handle correct answers?** Exclude from analysis, or include as "no misconception"?
   - Recommendation: Report error rates separately; misconception analysis on errors only

4. **Inter-rater reliability threshold?** κ > 0.6? > 0.8?
   - Recommendation: Target κ > 0.7; if lower, simplify scheme

5. **GSM8K as capability index:** Is this the right benchmark? Alternatives: MATH, specific grade-level tests
   - Recommendation: Use GSM8K as primary (most comparable to Eedi content level); report MATH as secondary

6. **How many models per tier?** Current design uses 2 per tier (6 total). Could reduce to 1 per tier for pilot.
   - Recommendation: Start with 1 per tier (3 models) in pilot; expand if effect detected

7. **Simulated ability prompts:** Are the three levels (Low/Medium/High) distinct enough? Need pilot testing.
   - Recommendation: Pilot with N=20 items to check error rate calibration across levels

---

## Next Steps

1. [ ] Download Eedi dataset and explore misconception labels
2. [ ] Draft coding rubric with examples
3. [ ] **Capability pilot:** Run 20 items through 3 models (1 per tier: GPT-4o, Llama-70B, Mistral-7B)
4. [ ] Test simulated ability prompts — do Low/Medium/High produce different error rates?
5. [ ] Look for early signal on capability × alignment relationship
6. [ ] Refine prompts based on pilot
7. [ ] Generate probe items
8. [ ] Full data collection (expand to 6 models if pilot supports hypothesis)

---

*Ready for review and iteration*
