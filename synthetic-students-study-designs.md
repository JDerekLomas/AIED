# LLM Synthetic Students: Two Study Designs

*Draft: January 26, 2026*

---

## Study 1: Model Capability × Prompting Interaction

### Title

**Do Smarter Models Make Better Synthetic Students? A Factorial Study of Model Capability, Prompting Strategy, and Simulated Ability on Item Difficulty Prediction**

### Abstract

Large Language Models show promise as synthetic students for predicting item difficulty, potentially reducing costly human pretesting. However, prior work uses ad-hoc model selection and inconsistent prompting strategies, leaving unclear whether model capability or prompt engineering drives simulation quality. We present a 3×2×5 factorial study examining the interaction between model capability (small, mid-tier, frontier—indexed by GSM8K performance), prompting strategy (minimal persona vs. detailed cognitive profile), and simulated student ability (five levels from novice to advanced). Using released items from NAEP (N=631), ASSISTments (N=500), and LSAT (N=300) with known human performance data, we generate synthetic student responses and fit IRT models to estimate item difficulty. We evaluate against human ground truth using Pearson correlation, RMSE, and ability distribution divergence (KL). Our results reveal a significant capability × prompting interaction: [hypothesized finding—e.g., complex prompting compensates for weaker models but provides diminishing returns for frontier models]. Across datasets, synthetic student difficulty predictions correlate r=X with human data, with frontier models achieving r=X under minimal prompting alone. We discuss implications for cost-effective item pretesting and identify boundary conditions where synthetic students fail to match human performance patterns.

**Keywords:** synthetic students, item difficulty prediction, large language models, item response theory, educational assessment

---

### Experimental Design

#### Research Questions

- **RQ1:** Does model capability (as indexed by GSM8K) predict synthetic student simulation quality?
- **RQ2:** Does prompting complexity improve simulation quality, and does this interact with model capability?
- **RQ3:** Can synthetic students accurately simulate different ability levels, and does this vary by model/prompt?
- **RQ4:** Do findings generalize across item types (math, reading, logical reasoning)?

#### Independent Variables

| Factor | Levels | Operationalization |
|--------|--------|-------------------|
| **Model Capability** | 3 | Small (~60% GSM8K): Llama-3.1-8B, Mistral-7B; Mid (~80% GSM8K): GPT-3.5, Llama-3.1-70B; Frontier (~95% GSM8K): GPT-4o, Claude Sonnet |
| **Prompting Strategy** | 2 | *Minimal:* "You are a [ability] student. Answer this question." / *Detailed:* Full cognitive profile with KC mastery, reasoning constraints, demographic context, worked example of target performance |
| **Simulated Ability** | 5 | Novice (θ=-2), Below Average (θ=-1), Average (θ=0), Above Average (θ=+1), Advanced (θ=+2) — mapped to percentile descriptors |

#### Dependent Variables

| Measure | Description |
|---------|-------------|
| **Difficulty Correlation (PCC)** | Pearson r between IRT difficulty from synthetic responses and human ground truth |
| **Difficulty RMSE** | Root mean squared error of difficulty estimates |
| **Ability Alignment (θ-Align)** | Spearman correlation between instructed ability and actual response correctness |
| **Distribution Divergence (KL)** | KL divergence between synthetic and human ability distributions |
| **Distractor Alignment** | % of incorrect responses matching most common human distractor (MCQ only) |

#### Datasets

| Dataset | Domain | Items | Ground Truth | Access |
|---------|--------|-------|--------------|--------|
| **NAEP** | Math (Gr 4,8,12) | 631 MCQ | Population % correct by skill level | Public/restricted |
| **ASSISTments** | Math (middle school) | 500 items | Individual attempt data, ~100K responses | Open |
| **LSAT** | Logical reasoning | 300 items | Published % correct | Public (prep materials) |

#### Procedure

1. **Item Selection:** Stratified sample ensuring difficulty distribution coverage (easy/medium/hard)

2. **Prompt Construction:**
   - *Minimal:* `You are a [Below Basic/Basic/Proficient/Advanced] student in grade [X]. Answer the following question. Just give your answer choice.`
   - *Detailed:*
     ```
     You are simulating a [grade] student with the following profile:
     - Overall ability: [percentile description]
     - Mastered concepts: [list relevant KCs]
     - Struggling concepts: [list relevant KCs]
     - Common error patterns: [describe typical mistakes at this level]
     - Reading level: [grade equivalent]

     Instructions: Answer as this student would, including making mistakes
     they would make. Show brief reasoning, then give your final answer.
     ```

3. **Response Generation:**
   - N=100 synthetic students per ability level per item
   - Temperature=0.7 for response variability
   - Total: 3 models × 2 prompts × 5 abilities × 100 students × ~1400 items = ~4.2M responses

4. **Scoring:**
   - MCQ: Exact match
   - Open response (ASSISTments): LLM-based scoring with rubric

5. **IRT Fitting:**
   - Fit 2PL model to synthetic response matrix
   - Extract difficulty (b) and discrimination (a) parameters
   - Compare to human-derived parameters

6. **Analysis:**
   - 3×2×5 factorial ANOVA on difficulty correlation
   - Interaction plots for capability × prompting
   - Separate analyses by dataset/domain
   - Ablation: effect of N synthetic students on stability

#### Hypotheses

- **H1:** Frontier models outperform smaller models on difficulty prediction (main effect of capability)
- **H2:** Detailed prompting improves performance for small/mid models more than frontier (interaction)
- **H3:** All models struggle to simulate low-ability students accurately (floor effect on θ-Align at low ability)
- **H4:** Distractor alignment is weaker than difficulty prediction across all conditions

#### Power Analysis

- Target: detect medium effect (f=0.25) for interaction
- With 6 cells (3×2), α=.05, power=.80 → need ~180 items minimum per dataset
- Our design (500+ items per dataset) is well-powered

#### Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Setup | 2 weeks | API access, prompt engineering, pilot |
| Generation | 3 weeks | Response generation across conditions |
| Analysis | 2 weeks | IRT fitting, statistical analysis |
| Writing | 3 weeks | Paper drafting, revisions |

---

## Study 2: Cross-Domain Generalization & Misconception Modeling

### Title

**Beyond Difficulty: Can LLM Synthetic Students Predict What Students Get Wrong? A Cross-Domain Study of Misconception Alignment**

### Abstract

While recent work shows LLMs can predict item difficulty, a harder test of simulation fidelity is whether synthetic students make the same *mistakes* as real students. We investigate misconception alignment: when an LLM-simulated student answers incorrectly, does it choose the same wrong answer that human students most frequently select? Using multiple-choice items from three domains—mathematics (NAEP, N=400), science (TIMSS, N=300), and reading comprehension (NAEP, N=250)—we compare synthetic student distractor selections against human response distributions. We test three frontier models (GPT-4o, Claude Sonnet, Gemini Pro) under two prompting conditions: answer-focused (select best answer) and reasoning-focused (explain thinking, then answer). Our analysis reveals that LLMs achieve moderate distractor alignment (X% match with top human distractor) but this varies substantially by misconception type: procedural errors are better predicted than conceptual misconceptions. Reasoning-focused prompting improves alignment for conceptual items but degrades it for procedural items, suggesting different error-generation mechanisms. Chain-of-thought analysis reveals that LLM "mistakes" often reflect computational shortcuts rather than the knowledge gaps that drive student errors. We propose a taxonomy of LLM-student error alignment and discuss implications for using synthetic students in formative assessment design.

**Keywords:** misconceptions, distractor analysis, synthetic students, error patterns, formative assessment

---

### Experimental Design

#### Research Questions

- **RQ1:** To what extent do LLM distractor selections match human student distractor selections?
- **RQ2:** Does misconception alignment vary by error type (procedural vs. conceptual)?
- **RQ3:** Does reasoning-focused prompting improve misconception alignment?
- **RQ4:** What do chain-of-thought explanations reveal about LLM vs. human error mechanisms?

#### Independent Variables

| Factor | Levels | Operationalization |
|--------|--------|-------------------|
| **Model** | 3 | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro |
| **Prompting** | 2 | *Answer-focused:* Direct answer selection / *Reasoning-focused:* Explain thinking step-by-step before answering |
| **Domain** | 3 | Mathematics, Science, Reading Comprehension |
| **Misconception Type** | 2 | Procedural (calculation, process errors) vs. Conceptual (fundamental misunderstanding) |

#### Dependent Variables

| Measure | Description |
|---------|-------------|
| **Top-1 Distractor Match** | % of LLM errors matching most common human distractor |
| **Distractor Rank Correlation** | Spearman ρ between LLM distractor probabilities and human selection frequencies |
| **Misconception Category Match** | % agreement on error taxonomy (coded by experts) |
| **CoT-Error Alignment** | Qualitative: does LLM reasoning reflect same misconception as human errors? |

#### Datasets

| Dataset | Domain | Items | Distractor Data | Access |
|---------|--------|-------|-----------------|--------|
| **NAEP (Math)** | Mathematics | 400 MCQ | % selecting each option by demographic | Restricted |
| **TIMSS** | Science | 300 MCQ | International % by option | Public |
| **NAEP (Reading)** | Reading Comp | 250 MCQ | % selecting each option | Restricted |
| **Eedi** | Math misconceptions | 500 MCQ | Student selection rates | Kaggle (open) |

#### Item Classification

Each item pre-coded for misconception type:

| Type | Definition | Example |
|------|------------|---------|
| **Procedural** | Error in executing known procedure | Sign error, order of operations mistake |
| **Conceptual** | Fundamental misunderstanding of concept | Believing multiplication always increases |
| **Reading/Interpretation** | Misreading question or data | Misidentifying axis on graph |
| **Guessing** | Random selection, no systematic error | — |

#### Procedure

1. **Item Selection & Coding:**
   - Select items with clear "most common" distractor (>30% of incorrect responses)
   - Expert coding of misconception type (2 raters, resolve disagreements)
   - Balance across difficulty levels

2. **Prompt Construction:**
   - *Answer-focused:*
     ```
     You are a student who sometimes makes mistakes.
     Select the best answer: [options A-D]
     ```
   - *Reasoning-focused:*
     ```
     You are a student who sometimes makes mistakes.
     Think through this problem step by step, showing your reasoning.
     Then select your answer.
     Note: It's okay to make errors—answer as a real student would.
     ```

3. **Response Generation:**
   - N=200 responses per item per condition
   - Temperature=1.0 for maximum variability
   - Record: selected answer + full reasoning (for CoT analysis)

4. **Quantitative Analysis:**
   - Compute distractor selection frequencies per item
   - Compare to human ground truth
   - 3×2×2 factorial ANOVA (model × prompt × misconception type)

5. **Qualitative Analysis:**
   - Sample N=100 incorrect LLM responses with CoT
   - Code reasoning for: correct procedure/wrong execution, flawed concept, misreading, random
   - Compare to known human misconception literature

#### Hypotheses

- **H1:** Distractor alignment will be moderate overall (40-60% top-1 match)
- **H2:** Procedural errors show higher alignment than conceptual errors
- **H3:** Reasoning-focused prompting improves conceptual error alignment but may hurt procedural (overthinking simple procedures)
- **H4:** CoT analysis will reveal LLMs make "different mistakes for different reasons"

#### Analysis Plan

**Quantitative:**
```
DV: Distractor_Match ~ Model * Prompt * Misconception_Type + (1|Item) + (1|Domain)
```
Mixed-effects logistic regression with item and domain as random effects.

**Qualitative:**
- Thematic analysis of CoT explanations
- Develop taxonomy: LLM error types vs. documented student misconceptions
- Case studies of high-alignment vs. low-alignment items

#### Expected Contributions

1. **First systematic study of distractor alignment** across multiple domains
2. **Misconception type moderator** — identifies where synthetic students work/fail
3. **Reasoning analysis** — explains *why* alignment succeeds or fails
4. **Practical guidance** — which items are safe to pretest with synthetic students?

#### Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Item selection & coding | 3 weeks | Dataset curation, expert coding |
| Generation | 2 weeks | Response generation |
| Quantitative analysis | 2 weeks | Statistical modeling |
| Qualitative analysis | 3 weeks | CoT coding, thematic analysis |
| Writing | 3 weeks | Paper drafting |

---

## Comparison of Study Designs

| Dimension | Study 1 | Study 2 |
|-----------|---------|---------|
| **Primary focus** | Difficulty prediction | Misconception/distractor prediction |
| **Key manipulation** | Model capability × prompting | Prompting × misconception type |
| **Novel contribution** | GSM8K as capability index; interaction effects | First systematic distractor alignment study |
| **Methodological emphasis** | Quantitative (IRT) | Mixed methods (quant + CoT analysis) |
| **Practical implication** | When can we skip human pretesting? | Which item types need human validation? |
| **Risk** | Findings may replicate SMART | CoT coding is labor-intensive |
| **Venue fit** | EDM, LAK (psychometric focus) | AIED (misconception focus) |

---

## Hybrid Option: Combined Design

Could merge into single paper with two studies:

**Study 1a:** Factorial design establishing difficulty prediction across model/prompt conditions (quantitative foundation)

**Study 1b:** Deep dive on distractor alignment for subset of items, with CoT analysis (mechanistic explanation)

This would be comprehensive but may exceed page limits for short paper formats. Better suited for journal (IJAIED, JEDM) or full paper track.

---

*Draft ready for review*
