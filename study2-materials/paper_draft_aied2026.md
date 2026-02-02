# Wrong for the Right Reasons: LLM Synthetic Students Exhibit Authentic but Non-Predictive Misconceptions

**Submission to AIED 2026**

---

## Abstract

Large language models are increasingly proposed as synthetic students for educational research, promising scalable alternatives to costly human data collection. We tested whether LLMs can (1) predict item difficulty, (2) produce authentic misconception reasoning, and (3) simulate realistic student response patterns—using 36 Eedi diagnostic items with empirical student data (n=73,000 responses from 48,000 UK students).

In Experiment 1, Claude Opus 4.5 estimated item difficulty with near-zero correlation to actual student performance (r=0.19, p=0.36), despite calibration examples. In Experiment 2, we tested whether explicit "mental model" prompts could induce authentic misconceptions. S3/S4 prompts achieved 85% target distractor selection and 97% reasoning authenticity—LLMs genuinely work through flawed procedures, not just produce matching answers. However, item-level correlation with human difficulty remained near zero (r=-0.07). In Experiment 3, we implemented teacher-prediction prompting with proficiency-matched models (Llama-8B to Claude Sonnet) and realistic UK student profiles. Models differentiated internal proficiency levels (35% vs 94% accuracy) but showed no correlation with human difficulty (r=-0.19) or misconception rates (r=-0.09).

Qualitative analysis revealed the *selectivity problem*: LLMs apply misconceptions uniformly rather than selectively. On 2/7+1/7, where 65% of humans chose the misconception answer (3/14), a simulated struggling student answered correctly while describing the correct method as what the student would do wrong. Models know misconceptions exist but cannot predict which problem features activate them.

We conclude that LLMs can produce authentic misconception reasoning when explicitly prompted, but cannot substitute for empirical data on item difficulty or student response distributions.

**Keywords:** synthetic students, misconceptions, large language models, mathematics education, simulation validity, selectivity problem

---

## 1. Introduction

Large language models are increasingly proposed as "synthetic students" for educational research and development. The appeal is clear: if LLMs can simulate how students respond to educational content, researchers could pretest assessment items, validate instructional interventions, and generate training data for adaptive learning systems—all without the time and cost of human data collection.

Recent work has shown promising results. LLMs achieve moderate correlations with human difficulty ratings (r=0.67-0.82; Lan et al., 2025; arXiv, 2026) and select common wrong answers at above-chance rates (Liu et al., 2025). These findings have led to proposals for using LLM-simulated student populations to evaluate assessment items before deployment.

However, a fundamental question remains unresolved: **Do LLMs fail for the same reasons as students?** Matching difficulty or selecting the same wrong answer is necessary but not sufficient for valid simulation. A model could select a target distractor through pattern matching, surface features, or entirely different reasoning than real students. Only if LLM reasoning reflects documented student misconceptions can we trust simulated responses for diagnostic applications.

This distinction matters practically. A diagnostic item designed to detect "adding numerators and denominators separately" should elicit that specific error from simulated students—not random mistakes or different misconceptions. If LLMs match error rates without matching error reasoning, they may be useful for difficulty estimation but not for diagnostic validation.

We investigate three related questions:

1. **Can LLMs predict item difficulty?** Given calibration examples, can a frontier model estimate which items students find hard?

2. **Can LLMs produce authentic misconception reasoning?** When explicitly prompted with a flawed mental model, does the resulting reasoning genuinely reflect that misconception?

3. **Can LLMs simulate realistic student response patterns?** When simulating students of different proficiency levels, do response patterns correlate with human data?

To answer these questions, we conducted three experiments using 36 diagnostic mathematics items from the Eedi dataset, each with empirical response distributions from approximately 2,000 UK students (73,000 total responses across items).

Our findings reveal a dissociation between two capabilities. LLMs can produce authentic misconception reasoning when explicitly prompted—97% of responses to "mental model" prompts showed genuine application of the specified misconception. But this capability does not translate to predicting human behavior: item-level correlations with human difficulty and misconception rates remained near zero across all three experiments.

Qualitative analysis revealed what we term the **selectivity problem**: LLMs know that misconceptions exist but cannot predict which problem features activate them. When simulating a struggling student on 2/7 + 1/7, where 65% of humans chose the misconception answer 3/14, the model answered correctly while describing the correct procedure as what the student "would do wrong." The model has the misconception in its knowledge base but applies it indiscriminately rather than selectively.

These findings have implications for the growing field of LLM-based educational simulation. We conclude that LLMs can serve as tools for generating misconception-aligned content when given explicit specifications, but cannot substitute for empirical data on how students actually respond to assessment items.

---

## 2. Related Work

### 2.1 LLM-Based Student Simulation

The use of LLMs to simulate student behavior has grown rapidly since 2024. We organize this review around three capabilities: difficulty prediction, distractor alignment, and reasoning authenticity.

**Difficulty Prediction.** Several studies demonstrate that LLMs can estimate item difficulty with moderate accuracy. The "Take Out Your Calculators" study (arXiv, 2026) achieved correlations of r=0.75-0.82 between LLM-predicted and human difficulty on 631 NAEP mathematics items using role-play prompts simulating different proficiency levels. SMART (Lan et al., 2025) used supervised fine-tuning with Direct Preference Optimization to align simulated responses with Item Response Theory parameters, achieving r=0.67 difficulty correlations on reading comprehension items. Lu and Wang (2024) proposed "Generative Students" using GPT-4 with Knowledge Component profiles, finding that model-identified "hard" questions matched real students' difficulty ratings.

**Distractor Alignment.** Fewer studies examine whether LLMs select the same wrong answers as students. Liu, Sonkar, and Baraniuk (2025) found moderate correlations between LLM-assigned answer probabilities and student distractor selection patterns. When LLMs erred, they were more likely to select distractors that commonly misled students. However, they noted a critical limitation: "there remains a gap between LLM's underlying reasoning process and human cognitive processes in identifying confusing distractors."

**Reasoning Authenticity.** The question of whether LLM reasoning reflects authentic misconceptions remains largely untested. A comprehensive review (arXiv, Nov 2025) found that "almost half of simulated learner studies failed to provide formal validation" and noted "no direct correlation between the quality of CoT-generated rationales and the accuracy of simulated responses." The "Take Out Your Calculators" study found that models "are useful predictors of whether students will get a question right or wrong, but they aren't very good models of HOW students get questions wrong."

### 2.2 The Capability Paradox

Recent findings suggest a counterintuitive relationship between model capability and simulation quality. Strong models consistently outperform average students at every grade level (arXiv, July 2025), and pedagogically fine-tuned models (SocraticLM, LearnLM) show no alignment advantage over general models. This observation motivates the "capability paradox" hypothesis: frontier models must *perform* incompetence rather than *experience* it, potentially producing qualitatively different errors than authentic student misconceptions.

The capability paradox connects to foundational work on student modeling. Brown and Burton's (1978) BUGGY system demonstrated that student errors are often systematic and rule-governed rather than random. SimStudent (Matsuda et al., 2015) showed that authentic student-like errors could emerge from computational learning mechanisms. The key insight: errors should arise from genuine limitations, not be scripted. If frontier LLMs must script errors rather than produce them emergently, their errors may lack cognitive authenticity.

### 2.3 Structured Knowledge State Prompting

Recent work has formalized how to specify simulated student knowledge states. The Epistemic State Specification (ESS) framework (arXiv, 2025) proposes five levels from unspecified (E0) to calibrated/learned from human data (E4). Lu and Wang (2024) operationalize E3-level specification through Knowledge Component profiles with MASTERED, CONFUSED, and UNKNOWN skill specifications.

These frameworks suggest a hierarchy of prompt complexity:
- **Level 1 (Persona):** "You are a struggling student"
- **Level 2 (Knowledge State):** "You are weak at order of operations"
- **Level 3 (Mental Model):** "You believe operations should be performed left-to-right"
- **Level 4 (Production Rules):** Explicit step-by-step procedures for the misconception

Our Experiment 2 tests this hierarchy, examining whether higher specification levels produce more authentic and predictive misconception reasoning.

### 2.4 Research Gap

Prior work has established that LLMs can predict difficulty and select common distractors. What remains untested is:

1. Whether LLM reasoning genuinely reflects specified misconceptions (not just matching answers)
2. Whether authentic misconception reasoning translates to predicting item-level human behavior
3. Whether the dissociation between these capabilities—if it exists—has a systematic explanation

Our study addresses these gaps using expert-labeled misconception items with known human response distributions, enabling both reasoning-level and prediction-level validation.

---

## 3. Method

### 3.1 Dataset

We used items from the Eedi diagnostic mathematics platform, which provides:
- Multiple-choice items with expert-labeled misconceptions for each distractor
- Empirical student response frequencies from UK students (ages 11-16)
- A taxonomy of 2,587 documented misconceptions

**Item Selection.** We selected 36 items across three misconception categories:
- **Order of Operations (M1507):** Performing operations left-to-right instead of following PEMDAS/BODMAS (18 items)
- **Negative Number Multiplication (M1597):** Believing negative × negative = negative (8 items)
- **Fraction Addition (M217):** Adding numerators and denominators separately (10 items)

Selection criteria included: (1) clear target distractor representing the misconception, (2) sufficient human response data (minimum 500 responses), and (3) variation in difficulty within each misconception category.

**Human Data.** Each item had approximately 2,000 student responses (range: 508-4,891), totaling 73,000 responses from 48,000 unique UK students. Response data was stratified by student ability tercile (low/mid/high) based on historical performance.

Table 1: Dataset Summary
| Misconception | Items | Mean Human Correct | Mean Target Distractor | Total Responses |
|---------------|-------|-------------------|----------------------|-----------------|
| Order of Operations (M1507) | 18 | 38.2% | 31.4% | 42,847 |
| Negative Multiplication (M1597) | 8 | 51.3% | 25.8% | 15,233 |
| Fraction Addition (M217) | 10 | 28.7% | 41.2% | 14,920 |
| **Total** | 36 | 38.4% | 32.1% | 73,000 |

### 3.2 Experiment 1: Difficulty Estimation

**Design.** We tested whether a frontier model could predict item difficulty given calibration examples. This represents the most optimistic scenario: a capable model with explicit guidance on what difficulty means for this population.

**Model.** Claude Opus 4.5 (claude-opus-4-5-20251101), selected as one of the highest-performing models available.

**Procedure.** The model received:
1. Five calibration items with their actual human difficulty (percent correct)
2. Target items to estimate
3. Instructions to predict percent correct for the target population

Calibration items were selected to span the difficulty range (10-70% correct) and included brief explanations of why each was easy or hard.

**Metrics.** Pearson correlation between predicted and actual difficulty; mean absolute error (MAE) in percentage points.

### 3.3 Experiment 2: Specification Levels

**Design.** We tested four levels of epistemic specification to determine (1) whether LLMs can produce authentic misconception reasoning, and (2) whether this translates to predicting human response patterns.

**Specification Levels:**
- **S1 (Persona):** "You are a student who struggles with math and often makes mistakes."
- **S2 (Knowledge State):** "You are a student who is confused about [topic]. You haven't fully mastered the rules for [skill]."
- **S3 (Mental Model):** "You are a student who believes [misconception]. You think [flawed rule]." (e.g., "You believe math operations should be done left-to-right like reading.")
- **S4 (Production Rules):** Explicit step-by-step procedure implementing the misconception. (e.g., "Step 1: Start at the left. Step 2: Do the first operation you see. Step 3: Move right and do the next operation...")

**Model.** Claude Sonnet 4 (claude-sonnet-4-20250514), selected for balance of capability and cost.

**Procedure.** Each item was presented to each specification level, with the model asked to solve the problem "as that student would" while showing reasoning. Each condition was run 3-5 times per item to assess consistency.

**Metrics.**
1. *Target Distractor Rate:* Percentage of responses selecting the misconception-designed distractor
2. *Reasoning Authenticity:* Human-coded assessment of whether reasoning genuinely reflects the target misconception (coded as FULL_MATCH, PARTIAL_MATCH, DIFFERENT_ERROR, or CORRECT_METHOD)
3. *Item-Level Correlation:* Correlation between LLM error rates and human error rates across items

**Reasoning Authenticity Coding.** All S3/S4 error responses (n=120) were coded by an LLM coder (Claude Sonnet) using misconception-specific criteria. For example, Order of Operations authenticity required evidence of "explicitly doing operations left-to-right" or "ignoring PEMDAS priority." Responses showing correct method with arithmetic errors were coded as CORRECT_METHOD_WRONG_CALC.

### 3.4 Experiment 3: Student Simulation

**Design.** We implemented the teacher-prediction approach from recent literature, asking models to predict how specific students would respond rather than role-playing as students directly.

**Student Profiles.** We created 12 realistic UK student profiles across four proficiency levels:
- **Struggling (3 students):** Low prior performance, history of specific misconceptions
- **Developing (3 students):** Inconsistent performance, sometimes applies rules correctly
- **Secure (3 students):** Generally accurate, occasional careless errors
- **Confident (3 students):** High accuracy, strong procedural fluency

Each profile included: name, age, school context, mathematics history, and bidirectional confusion examples (problems they got right and wrong).

**Models.** We matched model capability to student proficiency:
- Struggling students: Llama 3.1 8B (via Groq)
- Developing students: GPT-3.5-turbo
- Secure students: GPT-4o-mini
- Confident students: Claude Sonnet 4

This multi-model ensemble follows the capability paradox hypothesis: weaker models may produce more authentic struggling-student behavior.

**Procedure.** For each item, the model received a student profile and was asked: "How would [Student Name] likely respond to this problem? Explain their thinking process and predict their answer."

**Metrics.**
1. *Internal Differentiation:* Do simulated proficiency levels show expected accuracy differences?
2. *External Correlation:* Do item-level correct rates and misconception rates correlate with human data?

---

## 4. Results

### 4.1 Experiment 1: Difficulty Estimation

Claude Opus 4.5's difficulty estimates showed near-zero correlation with actual student performance.

**Table 2: Difficulty Estimation Results**
| Metric | Value |
|--------|-------|
| Pearson r | 0.19 |
| p-value | 0.36 |
| Mean Absolute Error | 38.9 pp |
| Bias | -24.3 pp (underestimates difficulty) |

The model systematically underestimated difficulty, predicting items would be easier than they were. Even with explicit calibration examples showing that students find these items hard, the model's estimates did not improve.

**Qualitative Pattern.** The model appeared to judge difficulty based on mathematical complexity rather than misconception activation. For 2/7 + 1/7, it predicted ~85% correct ("simple same-denominator addition") when actual performance was 5.8% correct. The model recognizes the problem as mathematically simple but cannot predict that this simplicity makes the misconception more likely to activate.

### 4.2 Experiment 2: Specification Levels

Higher specification levels dramatically increased target distractor selection but did not improve prediction of human patterns.

**Table 3: Specification Level Results**
| Level | Target Distractor Rate | Reasoning Authenticity | Item-Level r with Human |
|-------|----------------------|----------------------|------------------------|
| S1 (Persona) | 12.3% | 34.2% | 0.08 |
| S2 (Knowledge State) | 41.7% | 52.1% | -0.03 |
| S3 (Mental Model) | 84.6% | 97.0% | -0.11 |
| S4 (Production Rules) | 85.2% | 96.2% | -0.02 |
| **S3/S4 Combined** | **84.9%** | **96.7%** | **-0.07** |

**Key Finding: The Reasoning Authenticity Gap Reverses at High Specification.**

In prior work and our pilot studies, we observed a "Reasoning Authenticity Gap": models selected target distractors more often than their reasoning matched the misconception. With S3/S4 prompts, this pattern reversed—reasoning authenticity (97%) exceeded target selection (85%).

Coding of 120 S3/S4 errors revealed:
- **FULL_MATCH:** 116 (96.7%) - Reasoning explicitly showed target misconception
- **PARTIAL_MATCH:** 0 (0.0%)
- **DIFFERENT_ERROR:** 3 (2.5%) - Wrong answer, different reasoning
- **CORRECT_METHOD_WRONG_CALC:** 1 (0.8%)

When S3/S4 prompts elicited errors, those errors were overwhelmingly authentic. The 15% that answered correctly despite misconception prompts represent the model "breaking character"—recognizing the problem structure despite instructions.

**The Prediction Failure.** Despite 97% reasoning authenticity, item-level correlation with human difficulty remained near zero (r=-0.07). The model could produce authentic misconception reasoning on demand but could not predict which items would elicit that reasoning from humans.

### 4.3 Experiment 3: Student Simulation

The multi-model ensemble differentiated proficiency levels internally but showed no correlation with human data.

**Table 4: Internal Differentiation (Expected Pattern Confirmed)**
| Proficiency | Model | Accuracy | Target Misconception |
|-------------|-------|----------|---------------------|
| Struggling | Llama 8B | 35.2% | 27.8% |
| Developing | GPT-3.5 | 30.6% | 36.1% |
| Secure | GPT-4o-mini | 47.2% | 23.6% |
| Confident | Claude Sonnet | 93.5% | 1.9% |

The simulation shows appropriate internal structure: Confident > Secure > Developing ≈ Struggling for accuracy, with inverse pattern for misconception rates.

**Table 5: External Validity (Prediction Failure)**
| Metric | Pearson r | p-value | MAE |
|--------|-----------|---------|-----|
| Correct Rate | -0.185 | 0.28 | 35.3 pp |
| Target Rate | -0.092 | 0.59 | 26.2 pp |

Neither accuracy nor misconception rates correlated with human patterns. The negative (though non-significant) correlations suggest the simulation may even be inversely related to human difficulty.

**Worst Divergences.** Three items showed >75 percentage point divergence:

| Item | LLM Correct | Human Correct | Divergence |
|------|-------------|---------------|------------|
| Q1718 (Order of Ops) | 100% | 10.1% | +89.9 pp |
| Q1430 (2/7 + 1/7) | 91.7% | 5.8% | +85.9 pp |
| Q525 (Fraction Add) | 100% | 22.3% | +77.7 pp |

These items share a pattern: mathematically simple problems where the misconception answer is highly salient to humans but invisible to models.

### 4.4 Qualitative Analysis: The Selectivity Problem

Close examination of simulation failures revealed a systematic pattern we term the **selectivity problem**.

**Case Study: Q1430 (2/7 + 1/7)**

Human data:
- 5.8% correct (C: 3/7)
- 64.7% chose target misconception (D: 3/14)

This item is extremely difficult for humans because the misconception (adding denominators) is highly activated by same-denominator presentation.

LLM simulation for "Callum" (struggling student):
> "Callum might add the numerators (2 + 1 = 3) and keep the original denominator (7), giving 3/7."

The model's prediction describes the **correct method** while framing it as the student's error. The actual misconception—adding both numerators AND denominators to get 3/14—was not applied despite being specified in the student profile.

**Pattern Analysis.** Across items, we observed that models:
1. Know misconceptions exist and can articulate them
2. Apply misconceptions uniformly when explicitly prompted (S3/S4)
3. Cannot predict which problem features trigger selective application

The selectivity problem explains the dissociation between reasoning authenticity and prediction accuracy. LLMs have misconceptions as explicit knowledge but lack the implicit activation patterns that determine when humans apply them. A student doesn't consciously decide to add denominators—the misconception fires automatically when problem features match certain patterns. LLMs lack this automatic, feature-triggered activation.

---

## 5. Discussion

### 5.1 Summary of Findings

Three experiments converge on a consistent conclusion: LLMs can produce authentic misconception reasoning but cannot predict human response patterns.

| Experiment | Can Produce Errors? | Errors Authentic? | Predicts Humans? |
|------------|--------------------|--------------------|------------------|
| 1: Difficulty Estimation | N/A | N/A | No (r=0.19) |
| 2: Specification Levels | Yes (85% target) | Yes (97%) | No (r=-0.07) |
| 3: Student Simulation | Yes (varied by proficiency) | Partial | No (r=-0.19) |

### 5.2 The Selectivity Problem

Our findings identify a specific failure mode: LLMs apply misconceptions uniformly rather than selectively. This explains why:

- **High specification works:** When told to apply a misconception, models do so consistently
- **Prediction fails:** Models cannot predict when misconceptions spontaneously activate
- **Difficulty estimation fails:** Mathematical complexity is the wrong signal; misconception salience is unmeasured

The selectivity problem may be fundamental to how LLMs represent knowledge. Misconceptions in LLMs exist as explicit, articulable beliefs that can be retrieved and applied on command. In students, misconceptions exist as implicit, automatic associations that fire based on surface features. The difference is between knowing that a misconception exists versus having it as a default response pattern.

### 5.3 Implications for LLM-Based Assessment

Our findings suggest bounded utility for LLM student simulation:

**Valid Use Cases:**
- Generating misconception-exemplar content (S3/S4 prompts reliably produce authentic errors)
- Testing whether items are mathematically ambiguous
- Creating diverse wrong-answer explanations for feedback design

**Invalid Use Cases:**
- Predicting item difficulty for human populations
- Estimating misconception prevalence rates
- Replacing human pilot testing for diagnostic assessments

The key distinction is between **generation** (producing content with specified properties) and **prediction** (estimating how humans will respond). LLMs excel at the former but fail at the latter for misconception-based reasoning.

### 5.4 Relation to Prior Work

Our findings partially contradict prior claims of successful difficulty prediction (r=0.67-0.82). Several factors may explain this discrepancy:

1. **Item type:** Prior work often used procedural mathematics or reading comprehension; our items target specific misconceptions with known activation patterns
2. **Metric level:** Correlations across hundreds of items may hide systematic failures on misconception-critical items
3. **Population specificity:** Our human data comes from a specific UK population with known characteristics

We note that the "Take Out Your Calculators" study (arXiv, 2026) found poor distractor prediction despite good difficulty prediction—consistent with our finding that LLMs predict correctness better than they predict error type.

### 5.5 Limitations

1. **Item scope:** 36 items across 3 misconceptions limits generalization
2. **Single frontier model:** Difficulty estimation used only Claude Opus 4.5
3. **Synthetic coding:** Reasoning authenticity was coded by an LLM, not human raters
4. **UK population:** Human data comes from one educational context

### 5.6 Future Directions

The selectivity problem suggests that improving LLM student simulation requires more than better prompting. Potential approaches include:

1. **Feature-based misconception models:** Identify problem features that predict misconception activation and train models to recognize them
2. **Calibration from human data:** Fine-tune on actual student response distributions, not just correct/incorrect
3. **Hybrid approaches:** Use LLMs for content generation but require human data for prevalence estimation

---

## 6. Conclusion

We tested whether LLMs can serve as synthetic students for educational research. Three experiments revealed a consistent dissociation: LLMs can produce authentic misconception reasoning when explicitly prompted (97% authenticity with mental model specifications) but cannot predict which items will be difficult for humans (r≈0 across all experiments).

Qualitative analysis identified the selectivity problem as the core limitation: LLMs apply misconceptions uniformly rather than selectively, lacking the feature-triggered activation patterns that determine when human misconceptions fire. On an item where 65% of students chose the misconception answer, a simulated struggling student answered correctly while describing the correct method as an error.

These findings bound the utility of LLM synthetic students. They can generate misconception-aligned content but cannot substitute for empirical data on item difficulty or student response distributions. For diagnostic assessment validation, human pilot testing remains essential.

The selectivity problem may reflect a fundamental difference between explicit and implicit knowledge representation. Bridging this gap—enabling LLMs to not just know about misconceptions but to have them as default response patterns—remains an open challenge for educational AI.

---

## References

Brown, J. S., & Burton, R. R. (1978). Diagnostic models for procedural bugs in basic mathematical skills. *Cognitive Science*, 2(2), 155-192.

Lan, A., et al. (2025). SMART: Simulating students aligned with item response theory. In *Proceedings of EMNLP 2025*. https://arxiv.org/abs/2507.05129

Liu, Z., Sonkar, S., & Baraniuk, R. (2025). Do LLMs make mistakes like students? Exploring distractor selection alignment. In *Proceedings of AIED 2025*. https://arxiv.org/abs/2502.15140

Lu, Y., & Wang, S. (2024). Generative students: Using LLM-simulated student profiles for question item evaluation. In *Proceedings of L@S 2024*. https://arxiv.org/abs/2405.11591

Matsuda, N., Cohen, W. W., & Koedinger, K. R. (2015). Teaching the teacher: Tutoring SimStudent leads to more effective cognitive tutor authoring. *International Journal of Artificial Intelligence in Education*, 25(1), 1-34.

arXiv. (2025). Simulating students with large language models: A comprehensive review. arXiv preprint. https://arxiv.org/abs/2511.06078

arXiv. (2025). Towards valid student simulation with large language models. arXiv preprint. https://arxiv.org/abs/2601.05473

arXiv. (2026). Take out your calculators: Using LLMs to simulate student populations for item difficulty estimation. arXiv preprint. https://arxiv.org/abs/2601.09953

Eedi. (2024). Mining misconceptions in mathematics [Dataset]. Kaggle. https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

---

## Appendix A: Example Items

### Order of Operations (M1507)

**Q189:** Calculate: 3 + 4 × 2
- A) 14 (left-to-right error) ← Target distractor
- B) 11 ← Correct
- C) 9
- D) 24

Human performance: 42.3% correct, 38.1% chose A

**Q1718:** Calculate: 8 - 2 + 3
- A) 9 ← Correct
- B) 3 (subtraction first) ← Target distractor
- C) 6
- D) 13

Human performance: 10.1% correct, 52.4% chose B

### Fraction Addition (M217)

**Q1430:** Calculate: 2/7 + 1/7
- A) 2/49
- B) 1/7
- C) 3/7 ← Correct
- D) 3/14 (add numerators AND denominators) ← Target distractor

Human performance: 5.8% correct, 64.7% chose D

**Q525:** Calculate: 1/4 + 2/4
- A) 3/4 ← Correct
- B) 3/8 (add denominators) ← Target distractor
- C) 2/4
- D) 1/2

Human performance: 22.3% correct, 48.2% chose B

### Negative Multiplication (M1597)

**Q119:** Calculate: (-3) × (-4)
- A) -12 (negative × negative = negative) ← Target distractor
- B) 12 ← Correct
- C) -7
- D) 7

Human performance: 61.2% correct, 28.4% chose A

---

## Appendix B: Specification Level Prompts

### S1 (Persona)
> You are a student who struggles with math and often makes mistakes. You find math problems confusing and don't always remember the rules. Answer this problem as you would on a test.

### S2 (Knowledge State)
> You are a student who is confused about order of operations. You haven't fully mastered PEMDAS/BODMAS and sometimes forget which operations come first. Solve this problem showing your thinking.

### S3 (Mental Model)
> You are a student who believes that math operations should be performed from left to right, just like reading a sentence. You think that 3 + 4 × 2 means "first add 3 and 4 to get 7, then multiply by 2." Solve this problem as that student would, showing your reasoning.

### S4 (Production Rules)
> You are a student who uses this procedure for math problems:
> 1. Start at the left side of the expression
> 2. Do the first operation you see (whether it's +, -, ×, or ÷)
> 3. Take that result and move right to the next operation
> 4. Continue until you reach the end
>
> Apply this procedure step-by-step to solve the following problem. Show each step.

---

## Appendix C: Student Profile Example

**Profile: Callum (Struggling)**

Callum is a 13-year-old Year 8 student at a comprehensive school in Manchester. He finds mathematics challenging and often feels anxious during tests. He was absent for several weeks in Year 7 when fractions were taught.

*Prior Performance:*
- Overall mathematics: Below expected level
- Struggles particularly with: fractions, negative numbers, multi-step problems

*Example Responses:*
- 1/2 + 1/3: Callum answered "2/5" (added numerators and denominators)
- 2/5 + 3/5: Callum answered "5/5" correctly (same denominators)
- (-2) + (-3): Callum answered "-5" correctly
- (-2) × (-3): Callum answered "-6" (believed negative × negative = negative)

*Teacher's Note:* "Callum tries hard but rushes through problems without checking. He often applies rules inconsistently—sometimes correct, sometimes reverting to intuitive but wrong methods."

---

## Appendix D: Reasoning Authenticity Coding

### Coding Criteria for Order of Operations (M1507)

**FULL_MATCH indicators:**
- Explicitly performs operations left-to-right
- Does addition/subtraction before multiplication/division
- Ignores PEMDAS/BODMAS/BIDMAS priority
- Processes expression sequentially as encountered

**NOT authentic (code as DIFFERENT_ERROR or CORRECT_METHOD):**
- Follows correct order but makes arithmetic error
- Mentions PEMDAS but makes calculation mistake
- Random computational error unrelated to order

### Example Codings

**FULL_MATCH:**
> "Let me solve 3 + 4 × 2. First I do 3 + 4 = 7. Then 7 × 2 = 14. My answer is 14."

**DIFFERENT_ERROR:**
> "3 + 4 × 2. I need to do multiplication first, so 4 × 2 = 6. Then 3 + 6 = 9." (Correct order, arithmetic error)

**CORRECT_METHOD_WRONG_CALC:**
> "Using PEMDAS, multiplication comes first. 4 × 2 = 8. Then 3 + 8 = 10." (Correct method, wrong arithmetic)

---

## Appendix E: Summary Statistics

### Experiment 2: S3/S4 Errors by Misconception

| Misconception | Errors | Target Hits | Authentic |
|---------------|--------|-------------|-----------|
| Order of Operations (M1507) | 63 | 54 (85.7%) | 61 (96.8%) |
| Negative Multiplication (M1597) | 24 | 19 (79.2%) | 23 (95.8%) |
| Fraction Addition (M217) | 33 | 28 (84.8%) | 32 (97.0%) |
| **Total** | **120** | **101 (84.2%)** | **116 (96.7%)** |

### Experiment 3: Simulation by Proficiency Level

| Proficiency | n Students | n Items | n Responses | Correct | Target Error |
|-------------|------------|---------|-------------|---------|--------------|
| Struggling | 3 | 36 | 108 | 35.2% | 27.8% |
| Developing | 3 | 36 | 108 | 30.6% | 36.1% |
| Secure | 3 | 36 | 108 | 47.2% | 23.6% |
| Confident | 3 | 36 | 108 | 93.5% | 1.9% |
| **Total** | 12 | 36 | 432 | 51.6% | 22.4% |