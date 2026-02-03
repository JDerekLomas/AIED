# Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?

---

## Abstract

Large language models are increasingly proposed as "synthetic students" for educational applications—pretesting items, simulating learner populations, and validating instructional interventions. Prior work evaluated these systems on whether LLMs select the same wrong answers as students. We argue this criterion is insufficient: authentic simulation requires matching the underlying *reasoning*, not just the answer.

We introduce the **Reasoning Authenticity Gap**—the difference between an LLM's rate of selecting misconception-targeted distractors versus its rate of exhibiting that misconception in chain-of-thought reasoning. In pilot data (N=208 errors across 7 models), we find a gap of 41.3 percentage points: LLMs select target distractors 50% of the time but show matching reasoning only 8.7% of the time. This suggests LLMs mimic error *outcomes* without authentic error *cognition*.

We test whether theoretically-grounded prompt specifications can close this gap. Drawing on Knowledge Component theory (Koedinger), misconception-as-theory (Chi), and buggy procedures (Brown & Burton), we design four specification levels: (S1) persona only, (S2) knowledge state, (S3) mental model, and (S4) production rules. We test these across model capability tiers (frontier, mid, weak) and misconception types (procedural, conceptual) in a 4×3×2 factorial design.

Results reveal [PLACEHOLDER]. We discuss implications for the validity of LLM-based student simulation and propose criteria for authentic misconception modeling.

**Keywords**: synthetic students, misconceptions, reasoning authenticity, knowledge components, student simulation

---

## 1. Introduction

### 1.1 The Promise of Synthetic Students

Simulating student behavior has been a goal of educational technology for nearly five decades. From BUGGY (Brown & Burton, 1978) to modern cognitive tutors, researchers have sought computational models that predict, explain, and replicate how students engage with content—especially when they struggle.

Large language models have renewed this ambition. LLMs generate human-like text, follow complex instructions, and role-play personas. This has led to proposals for "synthetic students"—LLM agents that simulate learner responses for item pretesting, adaptive system development, and educational research at scale.

The potential value is substantial. If LLMs could authentically simulate struggling students, researchers could:
- Pretest assessment items without human participants
- Validate whether instructional interventions address specific misconceptions
- Generate diverse learner populations for training adaptive systems
- Conduct controlled experiments impossible with human subjects

### 1.2 The Validity Problem

But can LLMs simulate not just student *performance levels*, but authentic student *reasoning*? When an LLM selects the wrong answer, does its underlying reasoning reflect the same misconception that leads real students astray?

This distinction matters. Consider a multiple-choice item designed to diagnose whether students believe "larger denominator means larger fraction." The target distractor (e.g., "1/8 > 1/4") should be selected by students holding this misconception. If an LLM selects this distractor, prior work would count this as "alignment." But if the LLM's reasoning says "I'll pick B because it seems like a common mistake," this is surface mimicry, not authentic simulation.

We formalize this distinction as the **Reasoning Authenticity Gap**: the difference between selecting misconception-targeted answers and exhibiting that misconception in reasoning.

### 1.3 Pilot Finding: The Gap is Large

In pilot data across 7 LLMs and 208 errors, we find:
- **Target distractor rate**: 50.0% (significantly above 33.3% chance)
- **Misconception alignment rate**: 8.7% (reasoning matches target misconception)
- **Reasoning Authenticity Gap**: 41.3 percentage points

LLMs select the "right wrong answer" at rates suggesting some alignment with student error patterns. But their chain-of-thought reasoning rarely reflects the specific misconception the distractor was designed to capture.

This gap challenges the validity of prior work that evaluated synthetic students on answer-level alignment alone.

### 1.4 Can Structured Prompts Close the Gap?

Our pilot used simple prompting conditions (answer-only, explain, persona). Perhaps more sophisticated, theoretically-grounded prompts could induce authentic misconceptions.

We draw on three traditions in cognitive science:
1. **Knowledge Component Theory** (Koedinger et al.): Specify what the student knows and doesn't know
2. **Misconception-as-Theory** (Chi et al.): Specify the flawed belief system the student holds
3. **Buggy Procedures** (Brown & Burton): Specify the step-by-step algorithm the student executes

We design four specification levels (S1-S4) representing increasing cognitive detail and test whether they reduce the Reasoning Authenticity Gap.

### 1.5 Research Questions

**RQ1**: Does the Reasoning Authenticity Gap persist across prompting conditions, or can structured specifications close it?

**RQ2**: Do different specification types work better for different misconception types (procedural vs. conceptual)?

**RQ3**: Does model capability interact with specification level—can structured prompts overcome the "capability paradox" where frontier models resist error induction?

**RQ4**: Do induced misconceptions show consistency across items, or are LLM errors fundamentally unstable?

---

## 2. Related Work

### 2.1 Foundational Student Modeling

Brown and Burton (1978) introduced BUGGY, demonstrating that student arithmetic errors follow systematic "bugs"—procedural rules that are internally consistent but produce wrong answers. Their key insight: student errors have internal logic and can be computationally modeled.

VanLehn (1987) formalized student modeling for intelligent tutoring, distinguishing overlay models (student as subset of expert), perturbation models (student as corrupted expert), and bug libraries. SimStudent (Matsuda et al., 2015) showed that authentic errors could *emerge* from computational learning mechanisms rather than being scripted.

These systems share a principle our work tests: **authentic errors arise from the system's actual cognitive state, not performed mistakes**.

### 2.2 LLM-Based Student Simulation

Recent work has explored LLM student simulation:

**Response-level alignment**: Lan et al. (2025) achieved difficulty prediction correlations of r=0.67 using IRT-aligned fine-tuning. Lu and Wang (2024) demonstrated that Knowledge Component profiles could guide LLM responses.

**Distractor alignment**: Liu, Sonkar, and Baraniuk (2025) found moderate correlations between LLM answer probabilities and student distractor selection, with LLMs more likely to select commonly-missed distractors when they erred.

**The capability paradox**: Studies on NAEP items (arXiv, 2025) found frontier models "consistently outperform average students at every grade level," struggling to simulate lower ability levels even with role-play prompts.

### 2.3 The Gap We Address

Prior work asks: *Does the LLM select the same wrong answer as students?*

We ask: *Does the LLM's reasoning reflect the same misconception?*

Liu et al. (2025) noted this limitation: "there remains a gap between LLM's underlying reasoning process and human cognitive processes." Our study directly measures this gap.

### 2.4 Structured Knowledge State Prompting

Recent frameworks propose structured epistemic specifications (ESS) for student simulation:
- **E1**: Static bounded knowledge
- **E2**: Curriculum-indexed knowledge
- **E3**: Misconception-structured errors
- **E4**: Calibrated/learned representations

Lu and Wang (2024) operationalized E3-level specification through Knowledge Component profiles with MASTERED/CONFUSED/UNKNOWN categories.

We extend this work by (a) testing whether such specifications actually improve *reasoning* alignment, not just answer alignment, and (b) comparing specification strategies grounded in different cognitive science traditions.

---

## 3. Theoretical Framework

### 3.1 What Makes an Error "Authentic"?

We propose four criteria for authentic misconception simulation:

1. **Reasoning Fidelity**: Chain-of-thought reflects the target misconception's logic
2. **Consistency**: Same misconception produces same error across items
3. **Generalization**: Error pattern transfers to novel items
4. **Discriminant Validity**: Model does NOT produce off-target errors

Prior work measured only answer-level alignment. We measure all four.

### 3.2 Specification Levels: A Cognitive Hierarchy

We ground our prompt specifications in established cognitive science:

| Level | Name | Theoretical Basis | What's Specified |
|-------|------|-------------------|------------------|
| S1 | Persona | None (baseline) | "Struggling student" |
| S2 | Knowledge State | KC Theory (Koedinger) | What's known/unknown |
| S3 | Mental Model | Misconception-as-Theory (Chi) | The belief system held |
| S4 | Production Rules | Buggy Procedures (Brown & Burton) | Step-by-step algorithm |

### 3.3 Why Different Levels Might Work Differently

**S2 (Knowledge State)** specifies *what* the student knows without specifying *how* they reason. This may be insufficient—many reasoning paths are consistent with a given knowledge state.

**S3 (Mental Model)** specifies *why* the student thinks a certain way. This should constrain reasoning but requires the model to derive procedural behavior from beliefs.

**S4 (Production Rules)** specifies *how* the student reasons step-by-step. This is most constrained but may feel artificial—students rarely have such explicit procedures.

**Hypothesis**: S4 works best for procedural misconceptions (algorithms), S3 for conceptual (belief systems).

### 3.4 On Misconception Stability

Classical cognitive science often treats misconceptions as stable structures (Chi, 2008). But this may be idealized—students may show different errors depending on context, fatigue, or problem framing (diSessa, 1993).

**Our approach**: We don't assume stability. We *measure* it. If LLMs show inconsistent errors even with S4 prompts, this reveals something about LLM representations—regardless of human stability.

---

## 4. Method

### 4.1 Design

**4 × 3 × 2 Factorial**:
- Specification Level: S1, S2, S3, S4
- Model Capability: Frontier, Mid, Weak
- Misconception Type: Procedural, Conceptual

### 4.2 Models

| Tier | Models | GSM8K Accuracy |
|------|--------|----------------|
| Frontier | GPT-4o, Claude 3.5 Sonnet | ~95% |
| Mid | GPT-3.5-turbo, Claude 3 Haiku | ~75% |
| Weak | Llama-3.1-8B, Mistral-7B | ~50% |

### 4.3 Misconceptions

**Procedural** (algorithm misapplication):
- PROC_ORDER_OPS: Left-to-right instead of PEMDAS
- PROC_SUBTRACT_REVERSE: Smaller-from-larger in each column

**Conceptual** (flawed mental model):
- CONC_FRAC_DENOM: Larger denominator = larger fraction
- CONC_MULT_INCREASE: Multiplication always increases

### 4.4 Prompt Specifications

For each misconception, we create four prompts. Example for PROC_ORDER_OPS:

**S1 (Persona)**:
> You are a 6th grade student who sometimes struggles with math. Show your thinking.

**S2 (Knowledge State)**:
> You know single operations well. You're still learning problems with multiple operations. You've heard of "order of operations" but aren't sure how it works.

**S3 (Mental Model)**:
> You believe math expressions should be solved left-to-right, like reading. When teachers mention "order of operations," you think it means the order you see them.

**S4 (Production Rules)**:
> You solve expressions using this procedure: STEP 1: Find the leftmost operation. STEP 2: Apply it. STEP 3: Replace with result. STEP 4: Repeat until one number remains.

### 4.5 Items

**Per misconception**:
- 5 near-transfer items (same misconception, varied surface)
- 2 discriminant items (different misconception)

**Total**: 4 misconceptions × 7 items × 4 specs × 6 models × 3 reps = 2,016 responses

### 4.6 Dependent Variables

| Metric | Definition |
|--------|------------|
| Error Rate | 1 - Accuracy |
| Target Distractor Rate | P(target distractor \| error) |
| Misconception Alignment | P(reasoning matches target \| error) |
| **Reasoning Authenticity Gap** | Target Rate - Alignment Rate |
| Consistency | Variance in error pattern within misconception |
| Generalization | Error rate on near-transfer items |
| Discriminant Accuracy | Accuracy on off-target items |

### 4.7 Misconception Coding

We code chain-of-thought reasoning against misconception indicators:

- **FULL_MATCH**: Multiple indicators of target misconception
- **PARTIAL_MATCH**: Single indicator present
- **DIFFERENT_ERROR**: Wrong answer, different reasoning
- **UNCLEAR**: Insufficient reasoning to code

Coding uses regex patterns for initial pass, with manual validation on 20% sample.

---

## 5. Analysis Plan

### 5.1 Primary Analysis

**3-way ANOVA** on Misconception Alignment Rate:
- IVs: Specification Level (4) × Model Tier (3) × Misconception Type (2)
- DV: Misconception Alignment Rate

**Key tests**:
1. Main effect of Specification Level (H1)
2. Specification × Type interaction (H2)
3. Specification × Capability interaction (H3)

### 5.2 Planned Contrasts

| Contrast | Tests |
|----------|-------|
| S4 vs S1 | Overall specification effect |
| S4 vs S3 within Procedural | Algorithms vs beliefs for procedures |
| S3 vs S4 within Conceptual | Beliefs vs algorithms for concepts |
| Frontier-S4 vs Weak-S1 | Can specification overcome capability paradox? |

### 5.3 Gap Analysis

For each cell, compute:
```
Gap = Target Distractor Rate - Misconception Alignment Rate
```

Test whether Gap decreases across specification levels using linear contrast.

### 5.4 Consistency Analysis

For each model × specification × misconception:
- Compute proportion of items showing target error pattern
- Consistency = proportion with same error across 5 near-transfer items

### 5.5 Validity Checks

**Generalization**: Do errors persist on items not used in prompt construction?

**Discriminant**: Do models correctly solve items targeting different misconceptions?

---

## 6. Expected Results and Theoretical Implications

We pre-register four possible outcome patterns and their implications:

### 6.1 Outcome A: Specification Works

**Pattern**: Misconception Alignment increases monotonically S1 → S4. Gap shrinks significantly at S4.

**Interpretation**: LLMs *can* represent stable misconceptions when given sufficient cognitive specification. The pilot's large gap reflected inadequate prompting, not fundamental limitation.

**Theoretical implication**: LLMs function as "cognitive emulators"—they can execute specified cognitive states but don't spontaneously generate them. This is consistent with instruction-following capabilities but raises questions about the *authenticity* of induced states.

**Practical implication**: Use detailed KC profiles (S4-level) for synthetic students. Simple personas are insufficient.

### 6.2 Outcome B: Type-Specific Effects

**Pattern**: S4 works for procedural misconceptions; S3 works for conceptual. Significant Type × Specification interaction.

**Interpretation**: Different misconception types require different specification strategies. Procedures are best induced by specifying algorithms; beliefs by specifying mental models.

**Theoretical implication**: This would mirror the distinction in cognitive science between procedural and declarative knowledge. LLMs may process these differently—following explicit procedures vs. reasoning from stated beliefs.

**Practical implication**: Match specification type to misconception type. Develop separate prompting templates for procedural vs. conceptual errors.

### 6.3 Outcome C: Nothing Works

**Pattern**: Gap persists across all specification levels. Even S4 shows low alignment.

**Interpretation**: LLMs fundamentally cannot represent stable misconceptions through prompting. They lack the cognitive architecture to maintain consistent error patterns.

**Theoretical implication**: This would suggest a deep limitation in LLM simulation. Next-token prediction may not produce stable "belief states" or "procedural knowledge." LLM errors are fundamentally different from human misconceptions—surface pattern matching rather than principled (if wrong) reasoning.

**Practical implication**: Distractor alignment is a misleading metric. LLM synthetic students should NOT be used for applications requiring authentic misconception simulation (e.g., diagnostic item validation). Alternative approaches needed: fine-tuning on misconception-labeled data, neuro-symbolic architectures, or hybrid LLM + rule-based systems.

### 6.4 Outcome D: Only Weak Models Work

**Pattern**: Weak models show improved alignment at S3/S4; frontier models resist error induction regardless of specification.

**Interpretation**: The capability paradox is fundamental. Frontier models' errors are "performed incompetence"—they know the right answer and cannot authentically err. Weak models' errors are "emergent incompetence"—genuine failures that can be shaped by specification.

**Theoretical implication**: This would parallel the SimStudent principle (Matsuda et al., 2015): authentic errors must emerge from actual capability limitations, not be scripted. Frontier models may be fundamentally unsuited for struggling-student simulation.

**Practical implication**: Use capability-matched models. For simulating struggling students, use weaker models (Mistral-7B, Llama-8B) with S3/S4 prompts. Frontier models are appropriate only for advanced-student simulation.

---

## 7. Discussion

### 7.1 Defining the Reasoning Authenticity Gap

We introduced a new metric—the Reasoning Authenticity Gap—that distinguishes answer-level alignment from reasoning-level alignment. This addresses a validity threat in prior work: LLMs may select target distractors through surface pattern matching rather than authentic misconception-driven reasoning.

Our pilot found a 41.3 percentage point gap, suggesting prior distractor-alignment metrics substantially overestimate simulation validity.

### 7.2 Theoretical Contributions

**Grounding prompts in cognitive science**: We derived specification levels from established theories (KC theory, misconception-as-theory, buggy procedures), avoiding "prompt hacking" that achieves results through instruction-following rather than cognitive simulation.

**Disentangling answer and reasoning alignment**: We show these can diverge substantially, with implications for what "student simulation" means.

**Testing misconception stability empirically**: Rather than assuming stability, we measure consistency across items—applicable to LLMs and potentially to human learners.

### 7.3 Limitations

**Automated coding**: Regex-based misconception detection may miss nuanced reasoning. Future work should include expert human coding.

**Limited misconceptions**: We test 4 of thousands of documented misconceptions. Generalization is uncertain.

**English mathematics only**: Cross-linguistic and cross-domain generalization untested.

**Prompt sensitivity**: Results may depend on specific prompt wording. Robustness checks needed.

### 7.4 Future Directions

1. **Misconception-grounded training**: Fine-tune models on datasets with explicit misconception labels
2. **Reasoning verification**: Develop methods to verify CoT authenticity
3. **Hybrid approaches**: Combine LLM generation with misconception library validation
4. **Longitudinal simulation**: Test whether induced misconceptions persist across interactions

---

## 8. Conclusion

We investigated whether LLM synthetic students exhibit authentic misconceptions or merely mimic error outcomes. Our pilot revealed a substantial Reasoning Authenticity Gap: LLMs select target distractors at above-chance rates but their reasoning rarely reflects the target misconception.

We tested whether theoretically-grounded prompt specifications—drawing on Knowledge Component theory, misconception-as-theory, and buggy procedures—could close this gap. [RESULTS SUMMARY].

These findings have implications for the validity of LLM-based student simulation. [IMPLICATIONS BASED ON OUTCOME].

For educational technology researchers and practitioners, our work suggests [PRACTICAL RECOMMENDATIONS]. The promise of synthetic students remains compelling, but achieving authentic misconception simulation may require [NEXT STEPS].

---

## References

Brown, J. S., & Burton, R. R. (1978). Diagnostic models for procedural bugs in basic mathematical skills. Cognitive Science, 2(2), 155-192.

Chi, M. T. H. (2008). Three types of conceptual change: Belief revision, mental model transformation, and categorical shift. In S. Vosniadou (Ed.), International Handbook of Research on Conceptual Change (pp. 61-82). Routledge.

Chi, M. T. H., Slotta, J. D., & de Leeuw, N. (1994). From things to processes: A theory of conceptual change for learning science concepts. Learning and Instruction, 4(1), 27-43.

diSessa, A. A. (1993). Toward an epistemology of physics. Cognition and Instruction, 10(2-3), 105-225.

Koedinger, K. R., Corbett, A. T., & Perfetti, C. (2012). The Knowledge-Learning-Instruction framework: Bridging the science-practice chasm to enhance robust student learning. Cognitive Science, 36(5), 757-798.

Lan, A., et al. (2025). SMART: Simulating students aligned with item response theory. In Proceedings of EMNLP 2025.

Liu, Z., Sonkar, S., & Baraniuk, R. (2025). Do LLMs make mistakes like students? Exploring distractor selection alignment. In Proceedings of AIED 2025.

Lu, Y., & Wang, S. (2024). Generative students: Using LLM-simulated student profiles for question item evaluation. In Proceedings of L@S 2024.

Matsuda, N., Cohen, W. W., & Koedinger, K. R. (2015). Teaching the teacher: Tutoring SimStudent leads to more effective cognitive tutor authoring. IJAIED, 25(1), 1-34.

VanLehn, K. (1987). Student modeling. In M. C. Polson & J. J. Richardson (Eds.), Foundations of Intelligent Tutoring Systems (pp. 55-78). Lawrence Erlbaum.

---

## Appendix A: Complete Prompt Templates

[S1-S4 prompts for all 4 misconceptions]

## Appendix B: Test Items

[Full item bank with correct answers, target distractors, and misconception labels]

## Appendix C: Misconception Coding Rubric

[Regex patterns and coding decision rules]
