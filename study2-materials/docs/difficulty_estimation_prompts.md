# LLM Item Difficulty Estimation: Literature Review

*Compiled: 2026-01-26*

---

## Summary Table: What Works?

| Approach | Best Result | Notes |
|----------|-------------|-------|
| Direct "estimate %" prompting | r ≈ 0 | Fails on standardized tests |
| Direct estimation (GPT-4o, K-5) | r = 0.83 | Works for simple items |
| Proficiency role-play simulation | <1% change | Inconsistent |
| Classroom simulation + IRT | r = 0.75-0.82 | Requires aggregation |
| Feature extraction + ML | r = 0.87 | Best overall |
| Model uncertainty + ML | SOTA RMSE | Promising new approach |
| LLM embeddings | r = 0.70-0.77 | Moderate |
| **Reasoning-augmented** | **10-28% MSE reduction** | **Analyze each option** |
| **Confusion tuples** | High correlation | Explicit misconception specification |
| **LLM error alignment** | 0.73-0.80 | LLMs naturally make student-like errors |

---

## Papers Attempting Direct Difficulty Estimation

### 1. "Take Out Your Calculators" (Kröger et al., 2025)
**Source:** https://arxiv.org/abs/2601.09953

**Direct Estimation Prompt:**
```
Estimate the percentage of students at this grade level who will answer
the question correctly. Your prediction should be based on factors such
as problem difficulty and cognitive load...
```

**Results - Direct Estimation (FAILS):**
| Model | Correlation Range |
|-------|-------------------|
| All tested models | r = -0.14 to +0.14 |
| Grade 12 | Predominantly negative |

**Results - Classroom Simulation (WORKS):**
| Model | Grade 4 | Grade 8 | Grade 12 |
|-------|---------|---------|----------|
| Gemma-2-27b | r = 0.75 | r = 0.76 | r = 0.82 |

**Key Finding:** "Correlations remain consistently near zero... with no model showing moderate or strong correlations" for direct estimation.

---

### 2. Razavi & Powers (2025)
**Source:** https://arxiv.org/abs/2504.08804

**Dataset:** 5,170 K-5 mathematics and reading items

**Model:** GPT-4o (2024-11-20)

**Direct Estimation Results:**
| Subject | RMSE | MAE | Correlation |
|---------|------|-----|-------------|
| Math | 0.91 | 0.72 | r = 0.83 |
| Reading | 0.86 | 0.69 | r = 0.83 |

**Feature Extraction + GBM Results:**
| Subject | RMSE | Correlation |
|---------|------|-------------|
| Reading | 0.73 | r = 0.87 |
| Math | 0.53-0.89 | r = 0.62-0.82 |

**Features Extracted:**
- Vocabulary complexity
- Syntax complexity
- Conceptual complexity
- Cognitive load
- Depth of Knowledge (DOK) level
- Skill difficulty
- Distractor quality

**Rating Scales Tested:** -3 to +3, -5 to +5, 1 to 100

**Key Finding:** Direct estimation works moderately well (r=0.83) for K-5 items with GPT-4o, but feature extraction + ML is better (r=0.87).

---

### 3. "Human-AI Difficulty Alignment" (2024)
**Source:** https://arxiv.org/abs/2512.18880

**Proficiency Prompt Structure:**
```
System: Suppose you are a student taking the [EXAM]. You are a
[PROFICIENCY LEVEL] student with [LEVEL]-level [SUBJECT] proficiency.

User: Analyze the difficulty [values/levels] of the question...provide
the final [value/category] in \boxed{...}:
```

**Proficiency Levels:**
- Low-proficiency ("weak student with low-level")
- Medium-proficiency ("average student with medium-level")
- High-proficiency ("good student with high-level")

**Results:**
| Domain | Spearman ρ |
|--------|------------|
| Overall | 0.28 |
| USMLE (medical) | 0.13 |
| SAT Math | 0.41 |
| SAT Reading | 0.29 |
| Cambridge English | 0.30 |

**Scaling Paradox:**
- GPT-5: ρ = 0.34
- GPT-4.1: ρ = 0.44
- "Scaling does not reliably translate into alignment"

**Key Findings:**
- 75.6% saturation rate in USMLE (models solve uniformly)
- 70.4% "Savant Rate" (hard for humans, trivial for models)
- "Models struggle to authentically mimic different proficiency levels"
- Accuracy changes <1% across personas

---

### 4. Reading Comprehension Difficulty (HCI 2025)
**Source:** https://arxiv.org/abs/2502.17785

**Models:** GPT-4o, o1

**Prompt Approach:**
```
[Full question text]
[Context with multiple-choice options]
[Ground truth answer]
[Aggregated student performance data]
[Explicit IRT methodological guidelines]

Estimate the IRT parameters:
- Discrimination parameter (a)
- Difficulty parameter (b)
```

**Results:**
- GPT-4o: 98.84% answering accuracy
- o1: 99.14% answering accuracy
- Difficulty estimates "meaningfully align" with IRT parameters
- But estimates cluster in narrower ranges than true IRT values

---

### 5. "Do LLMs Give Psychometrically Plausible Responses?" (2025)
**Source:** https://arxiv.org/abs/2506.09796

**Models:** 18 instruction-tuned LLMs (Llama 3, OLMo 2, Phi 3/4, Qwen 2.5), 0.5B-72B parameters

**Results:**
| Subject | Best Correlation |
|---------|------------------|
| Reading | r = 0.32-0.56 |
| History | Poor/negative |
| Economics | Poor/negative |

**Key Finding:** "LLMs should not be used for piloting educational assessments in a zero-shot setting."

**Problems Identified:**
- Larger models excel at correct answers but fail at predicting distractor attractiveness
- Temperature scaling offers minimal benefit
- "Fundamentally non-human response patterns"

---

## Papers Using Reasoning-Augmented Approaches

### 6. "Reasoning and Sampling-Augmented MCQ Difficulty Prediction" (AIED 2025)
**Source:** https://arxiv.org/abs/2503.08551

**Method:** Generate reasoning for each option before predicting difficulty

**Prompt Strategy:** GPT-4o generates:
- Reasoning steps to reach the correct answer
- "Feedback messages explaining the potential error a student might make in selecting each distractor"

**Results:**
| Dataset | Without Reasoning (MSE) | With Reasoning (MSE) | Improvement |
|---------|-------------------------|----------------------|-------------|
| APT | 0.622 | 0.566 | 9% |
| EEDI | 0.522 | 0.471 | 10% |

**Best overall:** 28.3% MSE reduction, 34.6% R² improvement with full method

#### Our Replication Attempt (Jan 2026)

We tested reasoning-augmented estimation on our Eedi curated items using the prompt:
```
## Step 1: Analyze Each Option
For each option, explain:
- What misconception or error would lead a student to select this?
- How plausible is this error for a typical middle schooler?

## Step 2: Estimate Distribution
A: [percentage]%  B: [percentage]%  C: [percentage]%  D: [percentage]%
```

**Our Results (N=8 items):**
| Model | MAE (pp) | Correlation |
|-------|----------|-------------|
| gpt-4o-mini (direct) | 22.9 | 0.19 |
| gpt-4o (direct) | 23.9 | 0.05 |
| gpt-4o-mini (reasoning) | 23.9 | 0.03 |
| gpt-4o (reasoning) | 24.2 | 0.12 |

**Key Finding:** Reasoning did NOT improve estimation on our data. Unlike the AIED paper (which used fine-tuned models on training data), zero-shot reasoning augmentation shows no benefit.

**Failure Case Analysis (Q644):**
- Question: Evaluate -1 × -4
- Actual distribution: 1% correct, **99% selected -5** (confused × with +)
- gpt-4o predicted: 70% correct, 15% to -5
- The model recognized the misconception exists but couldn't predict its overwhelming dominance

**Implication:** LLMs cannot predict the *magnitude* of specific cognitive errors. They know what mistakes are possible but not which dominate in practice.

---

### 7. "Generating Plausible Distractors via Student Choice Prediction" (2025)
**Source:** https://arxiv.org/abs/2501.13125

**Method:** Pairwise ranker with explicit reasoning about misconceptions

**Prompt asks model to reason about:**
1. What knowledge the question tests
2. Why each distractor might confuse students

**Results:**
| Model | Ranking Accuracy |
|-------|------------------|
| GPT-3.5-turbo | 58.7% |
| GPT-4o | 64.0% |
| Fine-tuned with reasoning | 67.5% |
| Human experts | 71.7% |

---

### 8. "Do LLMs Make Mistakes Like Students?" (Feb 2025)
**Source:** https://arxiv.org/abs/2502.15140

**Key Finding:** When LLMs make mistakes, they tend to select the same wrong answers that students select.

**Index-Based Approach:** Present options as A/B/C/D, compute probability of selecting each letter.

**Results:**
| Metric | Correlation |
|--------|-------------|
| Probability alignment (Pearson) | 0.28-0.37 |
| Error selection alignment | 0.73-0.80 |
| Smallest model (0.5B) selects student's top wrong answer | 51.6% |
| Largest model (72B) selects student's top wrong answer | 59.3% |

**Key insight:** "Students compare options simultaneously rather than evaluating each independently" - index-based beats text-based.

---

### 9. "Towards Valid Student Simulation with LLMs" (Jan 2026)
**Source:** https://arxiv.org/abs/2601.05473

**Method:** Confusion Tuples - explicitly instruct model what concepts a student cannot distinguish

**Prompt Strategy:**
```
The student cannot distinguish between [concept A] and [concept B].
```

**Knowledge Component States:**
- Mastered
- Unknown
- Confusion (between two specific concepts)

**Result:** High correlation with real student difficulty rankings. Errors become "causally attributable to defined knowledge gaps rather than stochastic model behavior."

---

## Papers Using Other Indirect Approaches

### 10. SMART Framework (Scarlatos et al., 2025)
**Source:** https://arxiv.org/abs/2507.05129

**Method:** Simulate students at IRT ability levels, aggregate responses

**Ability Bucket Labels (10 levels):**
1. Minimal (θ < -2.5)
2. Emerging
3. Developing
4. Approaching Proficiency
5. Proficient
6. Competent
7. Skilled
8. Advanced
9. Exceptional
10. Mastery (θ ≥ 2.5)

**Result:** "Outperforms all direct prediction methods"

---

### 7. Model Uncertainty Approach (EDM 2025)
**Source:** https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.104/

**Method:** Use LLM uncertainty (not predictions) as difficulty signal

**Uncertainty Metrics:**
1. **1st Token Probability:** Softmax probability of answer token
2. **Choice-Order Sensitivity:** Consistency across randomized option orderings

**Models Tested:** Phi-3.5, Llama-3.2-3B, Qwen2.5-3B/14B/32B/72B, Yi-34B, Llama-3.1-8B/70B

**Results (Uncertainty + Text Features):**
| Dataset | RMSE | Prior Best |
|---------|------|------------|
| USMLE | 0.2864 | 0.291 |
| CMCQRD | 8.2325 | 8.5 |
| Biopsychology | 0.1359 | — |

**Key Finding:** Model uncertainty features substantially outperform text-only baselines.

---

### 8. EDM 2024 - Loss-Based Features
**Source:** https://educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.90/

**Method:** Infer difficulty from answer generation confidence, not explicit estimation

**Prompt (for answering):**
```
Answer the following question based on the context. Keep the answer
short, maximum 1 sentence, without any additional explanations.
Context: [context]. Question: [question]
```

**Features Extracted from Response:**
- MinLossIdx - context size needed for optimal answering
- LossRange - performance variance across context sizes
- FullLoss - loss with complete context
- MinLossAUC - uncertainty reduction slope
- ExpertiseGap - performance difference between model sizes

---

### 9. UnibucLLM (BEA 2024)
**Source:** https://aclanthology.org/2024.bea-1.41/

**Method:** Use LLM-generated answers as features for traditional ML

**Models for Feature Generation:** Falcon-7B, Meditron-7B, Mistral-7B

**Best Result:**
- MSE: 0.0534 (SVR+BERT with LLM answer features)
- Kendall τ: 0.1592
- Ranked 9th/43 in shared task

**Key Finding:** LLM answers as features + traditional ML beats direct LLM estimation.

---

## Student Simulation Prompts

### "Take Out Your Calculators" - Main Method
```
You are a {skill level} student in the {grade}th grade, given the task
to answer a math word problem question on {content area of problem},
taking into account the difficulty of this question. {Definition of
skill level continues}. In all your responses, you have to completely
forget that you are an AI model, but rather this {skill level} student,
and completely simulate yourself as one.
```

**Skill Levels (NAEP mapping):**
- Below Basic (25% of simulated classroom)
- Basic (35%)
- Proficient (25%)
- Advanced (15%)

**Variants:**
- With Names: "You are [NAME], a student in the {grade}th grade..."
- With Student IDs: "You are [STDID], a student in the {grade}th grade..."

**Key Finding:** Diverse demographic names improve predictions.

---

## Key Insights

### Why Direct Estimation Often Fails

1. **Capability Gap:** Models solve problems too easily - can't distinguish what's hard for humans
2. **No Variance:** Models achieve uniform success rates, losing difficulty signal
3. **Savant Items:** 70%+ of human-hard items are trivial for models
4. **Distractor Blindness:** Models can't predict which wrong answers attract humans

### When Direct Estimation Works

1. **Simple items:** K-5 level (Razavi & Powers)
2. **Strong models:** GPT-4o outperforms open-source
3. **With student data:** Including performance statistics in prompt helps

### Best Practices

1. **Feature extraction + ML** (r = 0.87) beats direct estimation
2. **Classroom simulation + IRT** works when direct fails
3. **Model uncertainty** is a promising new signal
4. **Weaker models** sometimes predict difficulty better than stronger ones

---

## Implications for Study 2: Reasoning Authenticity Gap

Our study investigates whether LLMs can simulate students with specific math misconceptions. The pilot found:
- S3 (Mental Model) produces 72% target distractor rate
- Human baseline: 38% target distractor rate
- S4 (Production Rules): 46% - closest to human baseline

### How This Literature Informs Our Work

**1. The "Overshoot" Problem is Well-Documented**

Our finding that S3 overshoots (72% vs 38%) aligns with literature:
- "Savant Rate" of 70%+ means models find human-hard items trivial
- Models achieve "uniform success rates" - they either get it right or wrong, no middle ground
- When prompted to simulate misconceptions, models may over-comply rather than calibrate

**2. "Distractor Blindness" is Our Core Challenge**

The literature explicitly identifies this:
> "Larger models excel at correct answers but fail at predicting which wrong answers humans are likely to be distracted by"

This is exactly our research question - can LLMs select the *specific* distractor that reflects a misconception, not just any wrong answer?

**3. Simulation vs. Estimation: We're on the Right Track**

Literature shows:
- Direct estimation: r ≈ 0 (fails)
- Classroom simulation + IRT: r = 0.75-0.82 (works)

Our S1-S4 specification levels are a form of simulation. The literature supports this approach over direct labeling.

**4. Specification Level Matters**

Our finding that S3 > S4 for procedural misconceptions contradicts our initial hypothesis but aligns with:
- Proficiency simulation is "highly inconsistent" (<1% change across personas)
- Mental model framing (S3) may be more effective than procedural rules (S4)
- The "capability gap" means explicit rules don't help models "dumb down"

**5. Model Selection May Matter More Than Prompting**

Key insight from Kröger et al.:
> "Weaker models (Gemma) actually yield better real-world difficulty predictions than mathematically stronger models"

**Implication:** We should test smaller/weaker models for misconception simulation. GPT-4/Claude may be too capable to authentically simulate confusion.

**6. Alternative Approaches to Consider**

Based on literature, we could explore:

| Approach | How It Would Work | Rationale |
|----------|-------------------|-----------|
| Model uncertainty | Measure LLM confidence per option, use as difficulty proxy | EDM 2025 showed this beats direct estimation |
| Feature extraction | Extract misconception-relevant features, train classifier | r = 0.87 vs r ≈ 0 for direct |
| Weaker models | Test Gemma, Phi, smaller Llama | May simulate confusion better |
| Classroom aggregation | Simulate N students, aggregate to distribution | Kröger's successful approach |

### Revised Research Questions

Based on literature, we should reframe:

| Original Question | Revised Question |
|-------------------|------------------|
| Can LLMs simulate misconceptions? | Under what conditions do LLM misconception simulations align with student populations? |
| Which specification level is best? | Does specification level interact with model capability? |
| Do LLMs show authentic reasoning? | Is authentic reasoning necessary for population-aligned behavior? |

### Key Risk: Population Alignment vs. Misconception Induction

The literature reveals a tension:
- **High misconception rate** (S3 = 72%) ≠ **Population alignment** (human = 38%)
- Maximizing misconception expression may *reduce* ecological validity
- The goal should be matching student distributions, not maximizing target selection

**Recommendation:** Reframe success metric from "target rate" to "distance from human baseline" (what S4 optimizes for).

---

## References

1. Kröger et al. (2025). "Take Out Your Calculators: Estimating the Real Difficulty of Question Items with LLM Student Simulations." arXiv:2601.09953

2. Razavi & Powers (2025). "Estimating Item Difficulty Using Large Language Models and Tree-Based Machine Learning Algorithms." arXiv:2504.08804

3. "Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction." arXiv:2512.18880

4. Scarlatos et al. (2025). "SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction." arXiv:2507.05129

5. "Exploring the Potential of Large Language Models for Estimating the Reading Comprehension Question Difficulty." arXiv:2502.17785 (HCI 2025)

6. "Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?" arXiv:2506.09796

7. EDM 2024. "How Hard can this Question be? An Exploratory Analysis of Features Assessing Question Difficulty using LLMs."

8. EDM 2025. "Are You Doubtful? Oh, It Might Be Difficult Then! Exploring the Use of Model Uncertainty for Question Difficulty Estimation."

9. UnibucLLM (BEA 2024). "Harnessing LLMs for Automated Prediction of Item Difficulty and Response Time."

10. "Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs." AIED 2025. arXiv:2503.08551

11. "Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction." arXiv:2501.13125

12. "Do LLMs Make Mistakes Like Students? Exploring Natural Alignment between Language Models and Human Error Patterns." arXiv:2502.15140

13. "Towards Valid Student Simulation with Large Language Models." arXiv:2601.05473
