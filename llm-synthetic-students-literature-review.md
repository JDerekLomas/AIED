# LLM Synthetic Students: Literature Review (2024-2025)

*Compiled: January 26, 2026*

## Research Context

This review covers recent work on using Large Language Models to simulate student performance on educational assessments. The goal is to understand how well LLM "synthetic students" can predict human student performance, with applications to item difficulty estimation, pretesting, and adaptive learning.

---

## Review Papers

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| [Simulating Students with LLMs: A Review](https://arxiv.org/abs/2511.06078) | arXiv Nov 2025 | Comprehensive taxonomy: direct prompting vs. knowledge tracing vs. knowledge graphs; identifies validation crisis (~50% of studies lack formal validation) |
| [Large Language Models in Student Simulation: A Survey](https://www.researchgate.net/publication/397754575) | ResearchGate 2024 | Earlier survey covering emergence of LLM-based approaches |

### Key Themes from Reviews

**Cognitive Architecture & Memory Management**
- Three dominant approaches: direct prompt-based simulation, knowledge tracing models, and knowledge graphs with heuristics
- Memory systems distinguish between short-term (dialogue history) and long-term (reflection databases) components
- "Combining factual response records (specific) with learning memory summaries (general) enables more accurate modeling of student practice"

**Personality and Individual Differences**
- Researchers increasingly integrate psychological frameworks (Big Five, MBTI) into simulated agents
- "Big Five for Tutoring Conversation (BF-TC)" reformulates traits for educational contexts

**Identified Gaps**
1. **Validation Crisis**: "Almost half of simulated learner studies failed to provide formal validation"
2. **Diversity Representation**: Automatic evaluation methods "frequently fail to authentically capture learner diversity"
3. **Pedagogical Efficacy vs. Stylistic Realism**: Few studies distinguish whether agents sound convincing vs. facilitate learning
4. **Temporal Consistency**: Maintaining coherent learning trajectories across extended interactions needs development

---

## Core Empirical Work

### 1. SMART: Simulated Students Aligned with IRT

**Citation:** Lan et al., EMNLP 2025
**Paper:** https://arxiv.org/abs/2507.05129

**Method:**
- Three-stage pipeline: (1) SFT to train student simulator, (2) DPO to align with IRT ability parameters, (3) automated scoring + IRT fitting
- Novel use of Direct Preference Optimization where preference pairs are created by comparing responses under IRT model
- Ability values converted to strings (4 decimal places) rather than categorical bins

**Datasets:**
- Smarter Balanced: 85K responses to 49 reading comprehension items from 63K students
- CodeWorkout: 10,834 Java programming solutions to 50 problems

**Results:**
- Difficulty prediction: PCC 0.67, RMSE 0.62 (Smarter Balanced)
- θ-alignment: 0.77 (vs 0.52 for SFT without DPO)
- "Student simulation is key when working with few training items"

**Limitations:**
- Only ~50 items per dataset
- Struggles with higher-order thinking items (summarization, thesis writing)
- Generated responses occasionally have "fidelity issues" (passage repetition, irrelevant content)
- "Simulated students may not sufficiently represent minority or non-English speaking students"

---

### 2. Can LLMs Reliably Simulate Real Students?

**Citation:** arXiv July 2025
**Paper:** https://arxiv.org/abs/2507.08232

**Method:**
- IRT (Rasch model) to estimate ability on shared scale
- Two conditions: unenforced prompting vs. grade-enforced prompting
- Grade-enforcement: three variants from minimal ("you are a 4th grader") to full chain-of-thought

**Dataset:**
- 489 NAEP multiple-choice questions
- Subjects: Mathematics and reading comprehension
- Grades: 4, 8, and 12

**Results:**
- **RQ1 (Unenforced):** "Strong general-purpose models consistently outperform average student at every grade"
- **RQ2 (Grade-Enforcement):** Changed performance but inconsistently; no model achieved alignment "across subjects and grades"
- Some models improved; others worsened dramatically (Qwen2.5-7B: 98.2 → 5.2 percentile)
- Pedagogically fine-tuned models (SocraticLM, LearnLM) showed NO alignment advantage

**Key Finding:**
"Reliable grade-level emulation will require tailored prompting that does not ensure generalization to other grades or subjects"

**Limitations:**
- Text-only (excludes visual reasoning)
- MCQ format only
- No cross-grade data
- Prompting methods only (no fine-tuning explored)

---

### 3. Take Out Your Calculators

**Citation:** arXiv January 2026
**Paper:** https://arxiv.org/html/2601.09953v1

**Method:**
- Role-play prompts with NAEP skill categories: Below Basic (25%), Basic (35%), Proficient (25%), Advanced (15%)
- Simulate classrooms → aggregate to success rates → fit IRT → extract difficulty
- Tested effect of student identifiers (none, alphanumeric, homogeneous names, diverse names)

**Dataset:**
- 631 NAEP mathematics MCQs
- Grades 4, 8, and 12

**Results:**
- Difficulty prediction: r = 0.75-0.82 across grades
- AUC: 0.78-0.90 for discriminating easy vs. hard items
- Ensemble approaches yield r = 0.82 at grade 12
- Diverse demographic names improve correlations vs. homogeneous identifiers

**Key Finding:**
"Models that are adept at solving mathematical problems may struggle to authentically simulate the response patterns of struggling students"

**Limitations:**
- Poor distractor prediction—models predict *correctness* better than *which wrong answer*
- Content area variation: excels at Measurement, struggles with Algebra
- Limited to 631 MCQs from single source

---

### 4. Generative Students

**Citation:** Lu & Wang, L@S 2024
**Paper:** https://dl.acm.org/doi/10.1145/3657604.3662031
**arXiv:** https://arxiv.org/html/2405.11591v1

**Method:**
- GPT-4 prompted with Knowledge Component (KC) based student profiles
- Student profile defined by: mastered KCs, confused KCs, unknown KCs
- Based on KLI (Knowledge-Learning-Instruction) framework

**Results:**
- LLM responses align with intended profiles
- Set of "hard" questions aligns with real students' difficulty ratings
- Supports question item evaluation without full data collection

**Framework:**
Uses knowledge components to define what students are expected to learn, enabling targeted profile construction.

---

## Distractor Analysis

### Do LLMs Make Mistakes Like Students?

**Citation:** Liu, Sonkar, & Baraniuk, AIED 2025
**Paper:** https://arxiv.org/abs/2502.15140

**Research Questions:**
1. Do distractors students frequently select correspond to those LLMs assign higher generation likelihood?
2. When LLMs select incorrect choice, do they choose the same distractor most students pick?

**Results:**
- Moderate correlations between LLM-assigned probabilities and student selection patterns
- When LLMs make mistakes, they ARE more likely to select the same incorrect answers that commonly mislead students
- Pattern consistent across both small and large language models
- **Prompting matters:** Index-based (A/B/C/D) shows higher alignment than text-based (Qwen-72B: 59.3% vs 42.6%)

**Implications:**
- Smaller models can be efficiently utilized for distractor generation
- LLM-generated distractors could "probe different aspects of student understanding, complementing rather than replacing human expertise"

---

### Generating Plausible Distractors

**Paper:** https://arxiv.org/html/2501.13125v2

**Method:**
- Overgenerate-and-rank approach
- DPO-trained ranker to select distractors most likely to be chosen by students

**Finding:**
LLMs generate mathematically valid distractors but are "less adept at anticipating common student misconceptions"

---

### Systematic Control of Item Difficulty Through Distractor Generation

**Citation:** AIED 2025
**Paper:** https://link.springer.com/chapter/10.1007/978-3-031-99261-2_14

Uses LLM-based distractor generation to systematically control MCQ difficulty levels.

---

## Knowledge Tracing Integration

| Paper | Venue | Finding |
|-------|-------|---------|
| [LLMKT for Dialogue-based KT](https://learninganalytics.upenn.edu/ryanbaker/Dialogue_KT_LAK_25-2.pdf) | LAK 2025 | LLM-based KT significantly outperforms existing methods in dialogue tutoring (Scarlatos, Baker, Lan) |
| [Towards Modeling Learner Performance](https://educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.84/) | EDM 2024 | Fine-tuned GPT-3 on ASSISTments achieves comparable AUC to BKT |
| [2T-KT Model](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006499) | Expert Systems 2025 | LLM-enriched knowledge graphs for cold-start KT |
| [Next Token Knowledge Tracing](https://arxiv.org/html/2511.02599) | arXiv Nov 2025 | Pretrained LLM representations adapted for structured KT tasks |

### ASSISTments + LLM Findings

- Fine-tuned LLMs achieved higher (Statics, ASSISTments 2017) or similar (ASSISTments 2009) AUC to standard BKT
- However, DKT, Best-LR, and SAKT consistently outperformed fine-tuned LLMs
- Performance gap narrows as training data increases

---

## Agent-Based Systems

| System | Year | Approach |
|--------|------|----------|
| **Agent4Edu** | 2025 | LLM agents with predefined profiles + memory modules + reflection mechanisms for personalized learning simulation |
| **Student Development Agent** | 2025 | Risk-free simulation for evaluating AIED interventions; validated against MAIC outcomes for non-cognitive development |
| **Classroom Simulacra** | 2025 | Multi-agent classroom simulation for learning behavioral modeling |
| **TeachTune** | CHI 2025 | Evaluating pedagogical agents against diverse simulated student profiles |

---

## Psychometric Applications

### LLMs as Synthetic Examinees

- **Liu et al. (2025):** GPT-3.5/4 answer sets yield 3PL item statistics matching 50-student baseline, reducing pretest costs
- **Grohs et al. (2024):** ChatGPT can pre-flag weak or biased items
- **Dueñas et al. (2024):** GPT-3.5 simulates medical candidate responses; extracts features like justification length, consistency under temperature variation
- **Liu et al. (2024):** Multiple LLMs (GPT-3.5/4, Llama 2/3, Gemini-Pro) to pretest College Algebra items

### IRT Framework for LLM Evaluation

**Paper:** [Leveraging LLM-Respondents for Item Evaluation](https://arxiv.org/html/2407.10899v1)

- IRT maps both persons and items onto same scales
- Enables equating without assuming population score distributions
- Tested GPT-3.5, GPT-4, Llama 2, Llama 3, Gemini-Pro, Cohere Command R Plus
- 150 responses per model on 20 College Algebra items

---

## Prompting Techniques

### Persona Prompting Findings

- "A one-sentence student-level prompt lets GPT-4 imitate weak, average, and strong test-takers, although adherence to target level is uneven"
- Lu and Wang (2024) and Hayakawa and Saggion (2024) applied prompting to simulate multiple test takers with single LLM
- Zotos et al. (2025) used LLM uncertainty as proxy for predicting student response distributions

### Expert Persona Study

**Paper:** [Playing Pretend: Expert Personas Don't Improve Factual Accuracy](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5879722)

- Tested domain-specific expert personas and low-knowledge personas
- Six models on graduate-level questions (science, engineering, law)
- **Finding:** Assigning expert persona had no significant impact on performance (except Gemini 2.0 Flash)

---

## Datasets for Synthetic Student Research

| Dataset | Granularity | Items | Access |
|---------|-------------|-------|--------|
| **NAEP** | Aggregate % correct by demographic | 600+ math/reading | Public (restricted use) |
| **ASSISTments 2009** | Individual attempts | 26K questions, 340K records | [Open](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data) |
| **ASSISTments 2012** | Individual attempts | 45K questions, 2.54M records | Open |
| **ASSISTments 2015** | Individual attempts | 100 KCs, 700K records | Open |
| **Eedi** | Individual + distractor selection | Math misconceptions | [Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) |
| **Smarter Balanced** | Individual responses | State assessments | Via partnership (restricted) |
| **CodeWorkout** | Individual programming attempts | Java exercises | Research access |

---

## Research Gaps & Opportunities

### 1. Systematic Model Capability Comparison
Most studies use 1-2 models. A GSM8K-indexed capability factor enabling systematic comparison across small/old/frontier models is novel.

### 2. Prompting Strategy Rigor
"Simple vs. complex" prompting is underexplored. Most use ad-hoc personas without systematic comparison.

### 3. Multi-Dataset Validation
Studies typically use one dataset. Cross-dataset validation (NAEP + LSAT + ASSISTments) enables generalization claims.

### 4. Distractor Prediction
Current work shows models predict difficulty better than distractor choice. Room to improve misconception modeling.

### 5. Ability Distribution Matching
Review notes most studies "rarely test whether ability distribution of synthetic learners matches real students."

### 6. Representational Bias
"Simulated students may not sufficiently represent minority or non-English speaking students" — validation needed.

### 7. Domain Transfer
Most studies focus on math; reading comprehension and other domains less explored.

---

## Key Citations

### Must-Cite Papers

1. **SMART** (Lan et al., EMNLP 2025) — IRT-aligned simulation via DPO
2. **Generative Students** (Lu & Wang, L@S 2024) — KC-based profiles
3. **Do LLMs Make Mistakes Like Students** (Liu et al., AIED 2025) — Distractor alignment
4. **Can LLMs Reliably Simulate** (arXiv 2025) — NAEP grade-level prompting
5. **Simulating Students Review** (arXiv Nov 2025) — Comprehensive taxonomy

### Additional References

- Agent4Edu (2025) — Agent-based simulation
- Student Development Agent (2025) — AIED intervention evaluation
- LLMKT (Scarlatos et al., LAK 2025) — Dialogue-based knowledge tracing
- Leveraging LLM-Respondents (2024) — Psychometric framework

---

## Implications for Proposed Study Design

### Strengths of Proposed Design

| Factor | Prior Work Gap | Your Contribution |
|--------|----------------|-------------------|
| Model capability (3 levels) | Ad-hoc model selection | Systematic GSM8K-indexed comparison |
| Prompting strategy (2 levels) | Inconsistent approaches | Controlled simple vs. complex comparison |
| Student profiles (5 levels) | Often 3-4 levels | Finer-grained ability distribution |
| Multiple datasets | Single-dataset studies | Cross-dataset generalization |
| Distractor analysis | Underexplored | Qualitative misconception analysis |

### Recommended Additions

1. **Include IRT fitting** to enable direct comparison with SMART and related work
2. **Report both difficulty correlation AND distractor alignment** (current gap)
3. **Test representational bias** across demographic subgroups if data available
4. **Compare prompting methods** (index-based vs. text-based for MCQ)

---

*Last updated: January 26, 2026*
