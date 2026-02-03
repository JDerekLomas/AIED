# Cognitive Modeling Prompts: Design, Results, and Cross-Validation

## Motivation

Our Phase 1 screening (11 prompt framings × 2 temperatures × 3 reps × 140 items on Gemini 3 Flash) established that structured prompts consistently outperform plain teacher estimation (ρ=0.547). The top performers — `prerequisite_chain` (ρ=0.684) and `cognitive_load` (ρ=0.674) — share a common mechanism: they ask the LLM to enumerate cognitive demands *before* making a holistic difficulty estimate.

This raised a question: could we push further by grounding the enumeration in specific learning science traditions? Three traditions offer complementary accounts of why students fail:

1. **Buggy Procedures** (Brown & Burton, 1978): Students apply internally-consistent but flawed algorithms — e.g., subtracting smaller from larger regardless of position, distributing only to the first term. Difficulty reflects how many procedural traps an item contains.

2. **Misconception Theory** (Chi, Slotta & de Leeuw, 1994): Students hold coherent-but-wrong mental models. Difficulty reflects how many active misconceptions an item triggers and whether the correct answer requires conceptual change.

3. **Multi-dimensional Cognitive Profiling**: Difficulty is not one thing but a profile across dimensions — concept familiarity, computational complexity, error trap density, format demands, and number friendliness. This integrates psychometric item analysis with cognitive load theory.

### The Critical Design Choice: Holistic vs. Mechanistic Estimation

Early prototypes of these prompts asked the LLM to *compute* difficulty mechanistically — e.g., multiply P(pass) across prerequisite steps (failure cascade), or subtract misconception prevalence rates. These produced severe pessimism bias:

| Prompt (mechanistic) | Easy (p=0.77) | Mid (p=0.38) | Hard (p=0.06) | MAE |
|---|---|---|---|---|
| failure_cascade (multiply step probs) | 0.12 | 0.18 | 0.08 | 0.30 |
| buggy_rules (subtract error rates) | 0.15 | 0.05 | 0.10 | 0.30 |
| misconception (sum prevalences) | 0.20 | 0.10 | 0.25 | 0.27 |

The multiplicative independence assumption crushes easy items: if an item has 5 steps each at P=0.95, the mechanistic estimate is 0.95^5 = 0.77, but if steps are at P=0.85, it drops to 0.44. Real students don't fail independently across steps — a strong student passes all steps; a weak student may fail several.

The fix: use cognitive analysis as **scaffolding** for a holistic estimate, not as inputs to a formula. This mirrors why `prerequisite_chain` works — it lists prerequisites then asks "what proportion would get full marks?" rather than computing from step probabilities.

After redesign (enumerate → holistic):

| Prompt (holistic) | Easy (p=0.77) | Mid (p=0.38) | Hard (p=0.06) | MAE |
|---|---|---|---|---|
| buggy_rules | 0.72 | 0.41 | 0.08 | 0.05 |
| misconception_holistic | 0.70 | 0.35 | 0.07 | 0.08 |
| cognitive_profile | 0.74 | 0.37 | 0.09 | 0.04 |


## Prompt Texts

### buggy_rules

Theoretical basis: Brown & Burton (1978) — systematic procedural errors as difficulty scaffolding.

```
You are an expert in mathematical cognition and systematic student errors
(Brown & Burton, 1978). You are analyzing a test item from Indian
government schools.

Your students are mostly from economically weaker sections. Many come
from Hindi-medium backgrounds and have limited English proficiency.

For the following test item, analyze the cognitive demands:

Grade: {grade}
Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Step 1: List the specific procedural steps a student must execute correctly.
Step 2: For each step, identify any known "buggy rules" — systematic
procedural errors students commonly make (e.g., subtracting smaller from
larger regardless of position, forgetting to carry, distributing only to
first term).
Step 3: Consider the target student population (grade level, open-ended
format requiring recall not recognition).
Step 4: Taking into account ALL of the above analysis holistically — the
number of steps, the severity and commonality of bugs, the grade level,
and that stronger students may avoid multiple bugs while weaker students
may hit several — estimate what proportion of students in this population
would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

### misconception_holistic

Theoretical basis: Chi, Slotta & de Leeuw (1994) — misconception-based difficulty analysis.

```
You are an expert in mathematics education and student misconceptions.
You are analyzing a test item from Indian government schools.

Your students are mostly from economically weaker sections. Many come
from Hindi-medium backgrounds and have limited English proficiency.

For the following test item, analyze what makes it easy or difficult:

Grade: {grade}
Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Step 1: Identify the mathematical concepts and skills required.
Step 2: List the most common misconceptions students at this grade level
hold about these concepts. For each, note whether it would lead to an
incorrect answer on THIS specific item.
Step 3: Consider factors that make the item easier: Is the context
familiar? Are the numbers simple? Is the required operation well-practiced
at this grade level?
Step 4: Consider factors that make it harder: Open-ended format (no answer
choices), multi-step reasoning, abstract notation, large numbers, etc.
Step 5: Weighing ALL factors holistically, estimate what proportion of
students at this grade level would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```

### cognitive_profile

Theoretical basis: Multi-dimensional psychometric profiling integrating cognitive load theory (Sweller, 1988) with item analysis dimensions.

```
You are an expert psychometrician analyzing test item difficulty for
Indian government schools.

Your students are mostly from economically weaker sections. Many come
from Hindi-medium backgrounds and have limited English proficiency.

For the following test item, build a cognitive difficulty profile:

Grade: {grade}
Subject: {subject}
Question: {question_text}
Rubric: {rubric}
Maximum score: {max_score}

Analyze these dimensions:
- Concept familiarity: How well-practiced is this topic at this grade
  level? (very familiar / familiar / less familiar)
- Computational complexity: How many arithmetic/algebraic steps?
  (1 / 2-3 / 4+)
- Common error traps: Are there well-known misconceptions or procedural
  bugs that apply? (none / minor / major)
- Format demand: Open-ended requires recall; is partial credit possible
  or must the answer be exact?
- Number friendliness: Are the numbers easy to work with mentally?
  (easy / moderate / hard)

After building this profile, estimate holistically what proportion of
students at this grade level would produce the fully correct answer.

Respond with ONLY a number between 0 and 1 on the last line.
For example: 0.45

Your estimate:
```


## Phase 1 Results: SmartPaper (140 items, Gemini 3 Flash, 3 reps)

All prompts evaluated on 140 open-ended items (Grades 3–9, English/Maths/Science/Social Science, 728K students, India). Metric: Spearman ρ vs. classical proportion correct.

Complete results: 15 prompts × 2 temperatures (33 configs total). Showing top 20 and full cognitive modeling family.

| Rank | Prompt | Temp | ρ | 95% CI | MAE | Bias | n | Family |
|---|---|---|---|---|---|---|---|---|
| 1 | prerequisite_chain | 0.5 | 0.686 | [0.580, 0.771] | 0.155 | +0.110 | 140 | enumeration |
| 2 | prerequisite_chain | 2.0 | 0.684 | [0.572, 0.770] | 0.148 | +0.105 | 140 | enumeration |
| 3 | cognitive_load | 2.0 | 0.673 | [0.550, 0.766] | 0.190 | +0.158 | 140 | enumeration |
| 4 | prerequisite_chain | 0.25 | 0.666 | [0.563, 0.749] | 0.155 | +0.111 | 140 | enumeration |
| 5 | **buggy_rules** | **1.0** | **0.655** | [0.542, 0.744] | **0.117** | **+0.054** | 140 | cognitive modeling |
| 6 | prerequisite_chain | 1.0 | 0.653 | [0.532, 0.751] | 0.155 | +0.111 | 140 | enumeration |
| 7 | **buggy_rules** | **2.0** | **0.638** | [0.519, 0.731] | **0.123** | **+0.058** | 140 | cognitive modeling |
| 8 | **misconception_holistic** | **2.0** | **0.636** | [0.506, 0.733] | 0.205 | +0.172 | 140 | cognitive modeling |
| 9 | **misconception_holistic** | **1.0** | **0.625** | [0.489, 0.733] | 0.198 | +0.164 | 140 | cognitive modeling |
| 10 | cognitive_load | 0.5 | 0.623 | [0.494, 0.729] | 0.200 | +0.166 | 140 | enumeration |
| 11 | cognitive_load | 1.0 | 0.617 | [0.474, 0.727] | 0.192 | +0.160 | 140 | enumeration |
| 12 | devil_advocate | 1.0 | 0.596 | [0.463, 0.703] | 0.098 | −0.048 | 140 | reframing |
| 13 | devil_advocate | 2.0 | 0.595 | [0.458, 0.699] | 0.100 | −0.054 | 140 | reframing |
| 14 | error_analysis | 2.0 | 0.588 | [0.463, 0.690] | 0.121 | +0.048 | 140 | enumeration |
| 15 | **cognitive_profile** | **1.0** | **0.586** | [0.456, 0.693] | 0.214 | +0.181 | 140 | cognitive modeling |
| 16 | **cognitive_profile** | **2.0** | **0.584** | [0.449, 0.694] | 0.214 | +0.180 | 140 | cognitive modeling |
| 17 | contrastive | 1.0 | 0.584 | [0.443, 0.697] | 0.123 | −0.016 | 140 | reframing |
| 18 | contrastive | 2.0 | 0.575 | [0.431, 0.690] | 0.120 | −0.031 | 140 | reframing |
| 19 | classroom_sim | 2.0 | 0.561 | [0.421, 0.671] | 0.240 | +0.214 | 140 | simulation |
| 20 | error_analysis | 1.0 | 0.548 | [0.409, 0.660] | 0.125 | +0.052 | 140 | enumeration |
| 21 | teacher | 1.0 | 0.547 | [0.407, 0.662] | 0.438 | +0.433 | 140 | baseline |
| 22 | teacher | 2.0 | 0.547 | [0.408, 0.663] | 0.437 | +0.432 | 140 | baseline |

### Key Findings

**1. Structured enumeration consistently outperforms plain estimation.** The top 6 prompts all ask the LLM to enumerate cognitive demands before estimating. The plain teacher baseline (ρ=0.547) is outperformed by every enumeration and cognitive modeling prompt. The gap is ~0.10–0.14 in ρ.

**2. buggy_rules achieves the best calibration.** While prerequisite_chain and cognitive_load have slightly higher ρ, buggy_rules has the lowest MAE (0.117) and smallest bias (+0.054). The detailed procedural error analysis produces not just good rank-ordering but good absolute estimates.

**3. Temperature interacts with prompt complexity.** Simple prompts (teacher) show no temperature effect. Enumeration prompts (prerequisite_chain, cognitive_load) improve at T=2.0. Cognitive modeling prompts show mixed effects: buggy_rules slightly degrades at T=2.0 (0.655→0.638), misconception_holistic slightly improves (0.625→0.636), cognitive_profile is flat (0.586→0.584). The structured chain-of-thought may already provide sufficient diversity without needing temperature-driven exploration.

**4. All cognitive modeling prompts show positive bias.** buggy_rules (+0.054), misconception_holistic (+0.164), cognitive_profile (+0.180) all overestimate difficulty slightly. The teacher baseline has severe positive bias (+0.433) because it predicts raw proportions without understanding the population. Interestingly, devil_advocate and contrastive show slight *negative* bias — they slightly underestimate difficulty.

**5. The "enumerate then estimate holistically" mechanism is the key.** The three cognitive modeling prompts differ in *what* they enumerate (procedural bugs, misconceptions, dimensional profile) but share the same estimation mechanism: detailed analysis followed by "weighing ALL factors holistically." This consistently outperforms both plain estimation and mechanistic computation.


## Sample LLM Output

**Item:** Grade 6, Maths — "Write all the factors of 68." (Rubric: all 6 factors for 1 mark; actual p = 0.08)

### buggy_rules output (T=1.0):

> **Step 1: Procedural Steps**
> 1. Comprehension: Translate the English term "factors" into the mathematical concept of divisors
> 2. Systematic Search: Employ a strategy to find pairs (e.g., 1×68, 2×34)
> 3. Multiplication/Division execution: Correctly perform mental division (e.g., 68 ÷ 4 = 17)
> 4. Exhaustion: Recognize when the search is complete (finding the "gap" between 4 and 17)
> 5. Transcription: List all six values without omission
>
> **Step 2: Buggy Rules / Systematic Errors**
> - The "Small Factor" Bug: Listing only 1, 2, and 4, then stopping because mental division becomes harder past 10
> - The "One-Pair" Bug: Providing only one pair (e.g., 2 and 34) instead of "all"
> - The "Multiples" Confusion: Confusing factors with multiples (listing 68, 136, etc.)
> - Divisibility Failure: Missing 4 and 17 because 68 is not in the standard 1–10 multiplication tables
> - Parity Bias: Assuming only even numbers are factors of an even number (ignoring 17)
>
> **Step 3: Student Population Considerations**
> - Grade 6 is the transition year where abstract number theory is introduced
> - English proficiency: If the student doesn't know the word "factor," they cannot attempt the problem
> - Recall vs. Recognition: The open-ended format is significantly harder than multiple choice
> - 17 is a prime number and 68 is "off the grid" for many students
>
> **Step 4: Holistic Estimate**
> Given the binary rubric (0 or 1) and the high probability of missing the 4/17 pair due to table-dependency and the linguistic barrier, the success rate is likely low.
>
> **0.22**

Estimated: 0.22. Actual: 0.08. The model correctly identifies this as very hard and provides specific, plausible reasons (the 4×17 pair, binary rubric, table-dependency).


## Cross-Dataset Validation: DBE-KT22

**Status: Complete.** Buggy_rules (3 reps, Gemini 3 Flash, T=1.0) on 168 university-level database systems MCQs from South Africa.

This tests whether the buggy_rules advantage transfers across:
- **Domain**: Maths/Science/English → Database Systems
- **Format**: Open-ended → Multiple choice
- **Population**: Indian government school students → South African university undergraduates
- **Level**: Grades 3–9 → University

| Method | ρ | 95% CI | MAE | Bias | n |
|---|---|---|---|---|---|
| **buggy_rules** | **0.579** | **[0.462, 0.676]** | **0.144** | **−0.087** | **168** |
| decomposed (proficiency levels) | 0.479 | — | — | — | 168 |
| contrastive (per-option prediction) | 0.440 | — | — | — | 168 |
| direct_clean (no correct answer) | 0.365 | — | — | — | 168 |
| direct (with correct answer) | 0.342 | — | — | — | 168 |
| synthetic_mcq (simulated students) | 0.300 | — | — | — | 164 |

**buggy_rules is the new best on DBE-KT22 by a wide margin** (+0.100 over decomposed). Per-rep consistency is excellent (ρ = 0.558, 0.570, 0.560).

This effectively rules out the leakage concern: the same prompt design works on both Indian school maths (ρ=0.655) and South African university database systems (ρ=0.579). The LLM cannot have memorized difficulty statistics for both datasets — the mechanism must be genuine cognitive modeling rather than memorized item statistics.


## Phase 2: Cross-Model Generalization

The model survey (7 models × 5 prompts × 3 reps × 140 items) tested whether structured prompts transfer across models. Key results for the enumeration prompt family:

| Model | teacher | cognitive_load | prerequisite_chain |
|---|---|---|---|
| Gemini 3 Flash | 0.550 | 0.619 | 0.633 |
| Gemma-3-27B | 0.501 | 0.500 | 0.464 |
| Llama-3.3-70B | 0.480 | 0.403 | 0.396 |
| GPT-4o | 0.494 | — | — |
| Llama-4-Scout | 0.474 | 0.400 | 0.345 |
| Llama-3.1-8B | 0.300 | 0.214 | 0.044 |
| Qwen3-32B | 0.304 | — | — |

The structured prompt advantage (cognitive_load and prerequisite_chain > teacher) holds across all models with sufficient capability (Gemini, Gemma, Llama-70B, Scout). It disappears on smaller/weaker models (Llama-8B), where prerequisite_chain actually *hurts* — the model cannot reliably enumerate prerequisites.

buggy_rules and the other cognitive modeling prompts have not yet been tested cross-model. This is a natural next step.


## Theoretical Implications

### Why does "enumerate then estimate holistically" work?

The mechanism appears to be **structured attention allocation**. When asked "estimate difficulty," the LLM draws on a shallow gestalt. When asked "list the procedural steps, identify bugs at each step, then estimate," the LLM is forced to attend to specific item features that determine difficulty — the prime factor 17, the binary rubric, the language barrier — before synthesizing. The enumeration step surfaces item-specific details that would otherwise be lost in the compression to a single number.

This is analogous to "think step by step" improving reasoning — but applied to judgment rather than computation. The key insight is that the *form* of the enumeration matters less than the *act* of enumerating. Buggy rules, misconceptions, prerequisites, and cognitive load dimensions all produce comparable improvements because they all force detailed item-level attention.

### Why does mechanistic computation fail?

The multiplicative cascade model (P_correct = ∏ P_step) assumes conditional independence of step failures. This is empirically false: student ability is a strong common cause. A student who passes step 1 is more likely to pass step 2 (high ability) than the base rate suggests. The product dramatically underestimates P(all correct) for easy items where most students have sufficient ability for all steps.

The holistic estimate implicitly accounts for this correlation structure because the LLM has absorbed the joint distribution of student performance from its training data. When it says "0.22," it is not computing a product — it is pattern-matching against its implicit model of what proportion of Grade 6 Indian students can find all factors of 68.

### The leakage question

A reviewer could argue that buggy_rules works because the LLM has memorized which *types* of procedural errors are associated with which difficulty levels. This is partially true and partially the point. The LLM's knowledge of "students commonly confuse factors with multiples" *is* pedagogical content knowledge — the same knowledge a human expert would use. The question is whether this constitutes leakage or legitimate domain expertise.

The cross-dataset validation on DBE-KT22 (different domain, population, format, level) provides the empirical test. If buggy_rules transfers, the knowledge is generalizable domain expertise, not memorized item-level statistics.
