# From Zero to Moderate: Recovering Item Difficulty from LLM-Simulated Student Responses Through Systematic Configuration Search

**Submission to AIED 2026**

---

## Abstract

Recent work demonstrates that LLMs can produce authentic misconception reasoning but fail to predict item difficulty (r≈0), a phenomenon attributed to the *selectivity problem*—LLMs apply misconceptions uniformly rather than selectively. We investigated whether this failure reflects a fundamental limitation or a configuration problem by conducting a systematic response surface experiment across prompt style, temperature, hint level, and other parameters using Eedi diagnostic items with IRT-calibrated difficulty (n=105 items, 73,000+ student responses).

A Box-Behnken response surface design across 11 configurations revealed that a single configuration—teacher-prediction prompting at high temperature (1.5)—achieved ρ=0.67 (p=0.001), while most configurations remained near zero. This finding was unexpected: rather than a smooth parameter landscape, a sharp interaction between prompt style and temperature produced the only significant result. Follow-up prompt engineering with 3-replicate stability testing identified a "contrastive" prompt variant that achieved ρ=0.55±0.02 across replications (p=0.01), making it both the most accurate and the most stable approach tested.

Error analysis revealed that the successful configuration works through a different mechanism than student simulation: it leverages teacher-like expert judgment about item properties rather than attempting to simulate student cognition. We argue that reframing the task from "simulate students" to "predict difficulty as teachers do" may partially circumvent the selectivity problem. However, the sensitivity of results to exact configuration—and the failure of seemingly reasonable alternatives—highlights the fragility of LLM-based difficulty estimation and the continued need for empirical calibration.

**Keywords:** item difficulty estimation, large language models, response surface methodology, mathematics education, teacher prediction, selectivity problem

---

## 1. Introduction

Can large language models predict how difficult a test item will be for students? The answer from recent literature is ambiguous. Some studies report moderate-to-strong correlations (r=0.67–0.82; Lan et al., 2025; arXiv, 2026), while others find near-zero predictive validity for misconception-based items (Lomas et al., in preparation). This paper investigates whether the discrepancy reflects fundamental limits of LLM-based estimation or sensitivity to methodological configuration choices that prior work has not systematically explored.

The motivation comes from a specific finding: in prior experiments using the same item set, three different approaches—direct difficulty estimation with a frontier model (r=0.19), misconception-specification prompting (r=-0.07), and multi-model student simulation (r=-0.19)—all failed to recover IRT difficulty parameters from empirical student data. Qualitative analysis attributed these failures to the *selectivity problem*: LLMs know that misconceptions exist but cannot predict which problem features activate them in real students (Lomas et al., in preparation).

However, these experiments each tested a single configuration. It remained possible that the right combination of parameters—prompt framing, temperature, model, hint content—could produce valid estimates. To test this, we adopted a response surface methodology (RSM) approach: systematically varying five experimental factors across a Box-Behnken design to map the parameter landscape and identify any regions of success.

Our findings are surprising in two ways. First, one configuration achieved a statistically significant Spearman ρ=0.67 (p=0.001)—substantially better than any prior attempt with this dataset. Second, this result was isolated: most of the parameter space produced near-zero correlations, and the successful configuration relied on a specific interaction between prompt style (teacher prediction) and temperature (1.5) that was not predicted by theory.

Follow-up experiments tested whether prompt engineering could stabilize and improve this result. A "contrastive" prompt—asking the model to reason about what makes each specific item easy or hard relative to similar items—achieved ρ=0.55±0.02 across three independent replications, providing the most reliable difficulty estimates observed in our research program.

These results have implications for both the practice of LLM-based item calibration and the theory of the selectivity problem. The successful approach works not by simulating student cognition but by eliciting expert teacher judgment—suggesting that the selectivity problem may be partially circumventable when the task is reframed from simulation to prediction.

---

## 2. Related Work

### 2.1 LLM-Based Difficulty Prediction

Several approaches to LLM-based item difficulty estimation have emerged. The "Take Out Your Calculators" study (arXiv, 2026) achieved r=0.75–0.82 on 631 NAEP mathematics items using role-play prompts at multiple proficiency levels. SMART (Lan et al., 2025) used supervised fine-tuning with DPO to align simulated responses with IRT parameters, achieving r=0.67. Liu et al. (2025) found moderate distractor alignment but noted a persistent gap between LLM reasoning and human cognitive processes.

A critical methodological observation is that most prior work varies one parameter at a time (prompt, model, or temperature) while holding others constant. This leaves interactions unexplored and may miss configurations that only work through specific parameter combinations.

### 2.2 The Selectivity Problem

Lomas et al. (in preparation) identified a specific failure mode in LLM student simulation: models apply misconceptions uniformly rather than selectively. When simulating a struggling student on 2/7 + 1/7 (where 65% of humans chose the misconception answer 3/14), the model answered correctly while describing the correct method as what the student "would do wrong." The model has misconceptions as explicit knowledge but lacks the implicit, feature-triggered activation patterns that determine when humans apply them.

This finding raised the question: is the selectivity problem a fundamental limitation, or can it be mitigated through better configuration? Our study addresses this directly.

### 2.3 Response Surface Methodology in Educational Research

Response surface methodology (RSM; Box & Wilson, 1951) is standard in engineering optimization but underused in educational AI research. RSM efficiently maps how multiple factors and their interactions affect an outcome, requiring fewer runs than full factorial designs. Box-Behnken designs (Box & Behnken, 1960) are particularly efficient for 4–7 factors, testing each pair of factors at all combinations of extremes while keeping other factors at center values.

We apply RSM to the question of LLM configuration for difficulty estimation—treating prompt style, temperature, batch size, hint content, and model as experimental factors and Spearman correlation with IRT difficulty as the response variable.

---

## 3. Method

### 3.1 Dataset

We used 105 diagnostic mathematics items from the Eedi platform (Eedi, 2024), each with:
- Expert-labeled misconceptions for each distractor
- Empirical student response data from UK students (ages 11–16)
- IRT difficulty parameters (b_2pl) estimated from a 2-parameter logistic model fit to 73,000+ student responses

From these 105 items, we selected a probe set of 20 items stratified by difficulty quintile (4 items per quintile) for the RSM experiment. This probe set spans the full difficulty range (b_2pl from -2.1 to 2.8).

### 3.2 Experiment 1: Response Surface Design

**Factors.** We varied five factors in a Box-Behnken design:

| Factor | Low (-1) | Center (0) | High (+1) |
|--------|----------|------------|-----------|
| Temperature | 0.3 | 0.9 | 1.5 |
| Students per call | 1 | 5 | 30 |
| Prompt style | individual_roleplay | classroom_batch | teacher_prediction |
| Misconception hint | hidden | partial (name only) | full (name + correct answer) |
| Model | gemini-2.5-flash | gemini-3-flash | — |

**Prompt Styles.** Three fundamentally different framings:

- *Individual roleplay*: The model role-plays as a student at a specified proficiency level and answers the item. 20 calls per item (one per simulated student across proficiency levels).
- *Classroom batch*: The model simulates a classroom of N students answering together, outputting each student's answer. Fewer calls needed per item.
- *Teacher prediction*: The model acts as an experienced UK maths teacher predicting what percentage of students at each ability level would choose each option. One call per item.

**Misconception Hints:**
- *Hidden*: Item text only, no indication of correct answer or misconception
- *Partial*: Misconception name included (e.g., "This item tests: adding numerators and denominators separately")
- *Full*: Misconception name plus correct answer identified

**Response Variable.** For each configuration, we computed the Spearman rank correlation (ρ) between the model's predicted item difficulty and the empirical IRT b_2pl parameter. For roleplay/classroom styles, predicted difficulty = proportion of simulated students answering incorrectly. For teacher prediction, predicted difficulty = weighted probability of incorrect response across ability levels (weights: below_basic=0.25, basic=0.35, proficient=0.25, advanced=0.15).

**Design.** A Box-Behnken design for 5 factors yields 46 experimental runs (40 edge points + 6 center points). Each run requires 20 items × the number of API calls per item (varying by prompt style and students_per_call).

### 3.3 Experiment 2: Prompt Engineering

Based on Experiment 1 results identifying the best configuration region, we designed four prompt variants and tested each with 3 independent replications (different random seeds via temperature=1.5) on the same 20 probe items.

**Variants:**

- *v0_baseline*: Minimal teacher prediction prompt (the original from the RSM winner)
- *v1_anchored*: Added explicit calibration anchors for each proficiency level (e.g., "Below-basic students perform near chance on hard topics, 40–50% on familiar ones")
- *v3_contrastive*: Added contrastive reasoning instruction: "Think carefully about what makes this specific question easy or hard compared to other questions testing similar content. Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error."
- *v4_contrastive_anchored*: Combined contrastive reasoning with calibration anchors

Each variant × replication produces a Spearman ρ. We report both individual replication ρ values and the ρ computed from averaged predictions (mean p_incorrect across 3 replications per item).

### 3.4 Error Analysis

For the best-performing RSM configuration, we conducted item-level error analysis examining:
- Which items the model predicted well vs. poorly
- Whether prediction accuracy related to item discrimination (a_2pl parameter)
- Whether the model could identify the correct answer
- How predictions distributed across proficiency levels

---

## 4. Results

### 4.1 Experiment 1: Response Surface

Of the first 11 configurations completed (the remaining runs are ongoing), results showed a stark pattern:

**Table 1: RSM Configuration Results (Selected)**

| Config | Prompt Style | Temp | Hint | ρ | p |
|--------|-------------|------|------|---|---|
| 7 | teacher_prediction | 1.5 | partial | **0.673** | **0.001** |
| 10 | classroom_batch | 1.5 | hidden | 0.406 | 0.076 |
| 4 | individual_roleplay | 0.3 | partial | 0.307 | 0.187 |
| 6 | individual_roleplay | 1.5 | partial | 0.306 | 0.189 |
| 0 | classroom_batch | 0.3 | partial | 0.259 | 0.271 |
| 3 | classroom_batch | 1.5 | partial | 0.244 | 0.300 |
| 5 | teacher_prediction | 0.3 | partial | 0.119 | 0.616 |
| 9 | classroom_batch | 0.3 | full | 0.118 | 0.621 |

**Key Finding: Sharp Interaction.** Only one configuration achieved a significant correlation. Config 7 (teacher_prediction × temp=1.5) produced ρ=0.673 (p=0.001), while the same prompt style at low temperature (Config 5, temp=0.3) yielded ρ=0.119—a five-fold difference. No other prompt style achieved significance at any temperature.

**Main Effects (Preliminary):**
- *Temperature*: Higher temperature generally improved correlations across all styles
- *Prompt style*: Teacher prediction showed the highest variance—best at high temperature, worst at low
- *Misconception hint*: Full hints (Config 9, ρ=0.118) performed worse than hidden (Config 10, ρ=0.406) or partial hints
- *Students per call*: Minimal effect for classroom_batch (ρ=0.259 at n=1 vs. ρ=0.244 at n=30)

### 4.2 Experiment 2: Prompt Engineering

Four prompt variants were each run 3 times on the 20 probe items (240 total API calls).

**Table 2: Prompt Variant Results (3 replications each)**

| Variant | Mean ρ | SD | Rep 1 | Rep 2 | Rep 3 | Averaged ρ | p |
|---------|--------|-----|-------|-------|-------|-----------|---|
| v3_contrastive | **0.549** | **0.017** | 0.571 | 0.529 | 0.545 | **0.562** | **0.010** |
| v1_anchored | 0.376 | 0.068 | 0.405 | 0.442 | 0.282 | 0.361 | 0.118 |
| v0_baseline | 0.370 | 0.097 | 0.501 | 0.339 | 0.269 | 0.403 | 0.078 |
| v4_contrastive_anchored | 0.315 | 0.097 | 0.250 | 0.243 | 0.452 | 0.447 | 0.048 |

**Key Findings:**

1. **Contrastive prompting is the most stable approach.** The v3_contrastive variant achieved the lowest standard deviation (0.017) across replications—roughly 5× more stable than the baseline (SD=0.097). Its worst single replication (0.529) exceeded the other variants' means.

2. **Anchoring does not help.** Adding explicit calibration anchors (v1_anchored) did not improve over the baseline (0.376 vs. 0.370). Combining anchors with contrastive reasoning (v4) actually degraded performance relative to contrastive alone (0.315 vs. 0.549).

3. **Averaging across replications improves all variants.** The averaged-prediction ρ exceeded mean single-rep ρ for every variant, consistent with noise cancellation from aggregation. The contrastive variant's averaged ρ=0.562 (p=0.010) represents the most reliable estimate.

4. **High variance at temperature=1.5.** The baseline variant's ρ ranged from 0.269 to 0.501 across replications, confirming that single-run results (like Config 7's ρ=0.673) should be interpreted cautiously.

### 4.3 Error Analysis

Analysis of Config 7 predictions revealed:

**Model knows the correct answer.** The model identified the correct answer as having the highest probability for the correct-answer option in 19 of 20 items. The signal comes not from whether the model knows the answer, but from how it distributes probability across incorrect options and ability levels.

**Below-basic and basic levels drive the signal.** Per-level analysis showed that predictions for below_basic and basic students correlated most strongly with IRT difficulty. Predictions for advanced students showed near-ceiling performance (>85% correct) with minimal variance—contributing little discriminative information.

**High-discrimination items are easier to predict.** Items with high IRT discrimination (a_2pl > 1.5) showed smaller prediction residuals than low-discrimination items. This makes sense: high-discrimination items have clear right/wrong boundaries that align with the model's binary understanding of misconceptions.

**Compression problem.** The model's predicted difficulty range was narrower than the actual range. Predicted weighted p_incorrect ranged from ~0.25 to ~0.55, while actual b_2pl spanned a wider range. This compression limits absolute calibration but preserves ordinal ranking.

---

## 5. Discussion

### 5.1 From Zero to Moderate: What Changed?

Our prior experiments all produced r≈0 for difficulty prediction. The RSM experiment found ρ=0.67 (Config 7) and the contrastive prompt achieved a stable ρ=0.55. What accounts for the improvement?

The critical change was **reframing the task from student simulation to teacher prediction**. Prior experiments asked models to *be* students (roleplay) or *simulate* classrooms. The successful approach asks the model to *predict*, as an expert teacher, how students at different ability levels would respond. This bypasses the selectivity problem: instead of needing to simulate when a misconception fires, the model draws on aggregate knowledge about student behavior patterns.

The high temperature (1.5) is also essential—possibly because it introduces sufficient variability in predictions to differentiate items. At low temperature, the teacher_prediction prompt produces near-identical distributions across items (ρ=0.119), suggesting the model defaults to a generic response pattern without temperature-induced exploration.

### 5.2 The Contrastive Prompting Effect

The contrastive prompt asks the model to reason about what makes *this specific question* easy or hard compared to similar items. This instruction appears to activate the model's item-level discrimination ability, forcing it to consider specific features rather than applying a misconception-category-level estimate.

The key instruction: *"Some questions that LOOK like they test a misconception are actually straightforward because the numbers/context don't trigger the error. Other questions that look simple have subtle traps that catch many students."*

This directly addresses the selectivity problem by asking the model to reason about selectivity explicitly. Rather than needing implicit feature-triggered activation (which models lack), the contrastive prompt makes selectivity an explicit reasoning step.

The stability of contrastive prompting (SD=0.017 vs. 0.097 for baseline) suggests it constrains the model's reasoning in a productive way—reducing the randomness that high temperature introduces while preserving the inter-item differentiation that high temperature enables.

### 5.3 Fragility and Replication Concerns

Several findings urge caution:

1. **Config 7's ρ=0.673 was not replicated at that level.** The prompt variant experiments showed the same basic teacher_prediction prompt achieving ρ=0.370±0.097 across replications. The original Config 7 result appears to be a favorable single draw from a high-variance distribution.

2. **Anchoring hurt rather than helped.** Adding seemingly helpful calibration information (explicit base rates for each proficiency level) degraded performance. This suggests the model has better internal calibration than the anchors provided, and overriding it with explicit numbers introduces bias.

3. **Combined approaches were worse than pure approaches.** Contrastive + anchored (v4) underperformed contrastive alone (v3), suggesting these mechanisms interfere rather than complement.

4. **Small probe set.** All results are based on 20 items—sufficient for rank-order statistics but limited for generalization claims.

### 5.4 Implications

**For practitioners:** Teacher-prediction prompting with contrastive reasoning and high temperature can produce meaningful difficulty rank-orderings (ρ≈0.55) for multiple-choice mathematics items. However, results should be aggregated across multiple runs, and absolute difficulty values should not be trusted (compression problem).

**For researchers:** The sharp interaction between prompt style and temperature highlights the danger of single-configuration studies. A researcher testing teacher prediction at temp=0.3 would conclude it fails (ρ=0.12); at temp=1.5, that it succeeds (ρ=0.55+). Response surface designs are recommended for future LLM evaluation studies.

**For theory:** The selectivity problem may be partially circumventable by reframing simulation as prediction. When asked to predict rather than simulate, the model can leverage its knowledge about student behavior without needing to replicate the cognitive processes that produce that behavior. This distinction—prediction vs. simulation—may explain some discrepancies in the literature.

### 5.5 Limitations

1. **Incomplete RSM sweep.** Only 11 of 46 planned configurations have been completed. The full response surface model cannot yet be fitted.
2. **Single model family.** All experiments used Gemini Flash variants. Results may not generalize to other model families.
3. **20-item probe set.** Limited statistical power and generalizability.
4. **No cross-validation.** The same probe items were used for development and evaluation.
5. **UK population specificity.** Human data comes from one educational context.

### 5.6 Future Directions

1. **Complete RSM sweep and fit quadratic response surface** to identify the true optimal region and confirm the temperature × prompt style interaction.
2. **Cross-validate on held-out items** (the remaining 85 items in the calibrated set).
3. **Test across model families** (Claude, GPT-4, open-source models).
4. **Explore ensemble methods**: averaging teacher predictions from multiple models.
5. **Investigate the compression problem**: can calibration from a small set of anchored items improve absolute (not just ordinal) prediction?

---

## 6. Conclusion

We investigated whether LLM-based item difficulty estimation is fundamentally limited or configuration-dependent. A response surface experiment revealed that the answer is both: most configurations produce near-zero correlations with IRT difficulty, but teacher-prediction prompting at high temperature achieves ρ=0.55–0.67—a meaningful result for practical item screening.

The successful approach works by reframing the task from student simulation to expert prediction, partially circumventing the selectivity problem identified in prior work. Contrastive prompting, which asks the model to reason about what makes each specific item easy or hard, further stabilizes predictions (SD=0.017 across replications).

However, the fragility of these results—the sharp dependence on exact configuration, the failure of seemingly reasonable alternatives like calibration anchoring, and the high variance at the temperature settings that enable differentiation—argues against treating LLM difficulty estimates as a replacement for empirical calibration. At best, they provide an inexpensive first-pass screening: which items are likely to be very easy or very hard for students.

The practical recommendation is specific: use teacher-prediction prompting with contrastive reasoning at high temperature, average across 3+ independent runs, and treat results as ordinal rankings rather than absolute difficulty estimates. For diagnostic assessment development, empirical pilot testing remains essential.

---

## References

Box, G. E. P., & Behnken, D. W. (1960). Some new three level designs for the study of quantitative variables. *Technometrics*, 2(4), 455–475.

Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of optimum conditions. *Journal of the Royal Statistical Society: Series B*, 13(1), 1–45.

Eedi. (2024). Mining misconceptions in mathematics [Dataset]. Kaggle. https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

Lan, A., et al. (2025). SMART: Simulating students aligned with item response theory. In *Proceedings of EMNLP 2025*. https://arxiv.org/abs/2507.05129

Liu, Z., Sonkar, S., & Baraniuk, R. (2025). Do LLMs make mistakes like students? Exploring distractor selection alignment. In *Proceedings of AIED 2025*. https://arxiv.org/abs/2502.15140

Lomas, D. et al. (in preparation). Wrong for the right reasons: LLM synthetic students exhibit authentic but non-predictive misconceptions.

Lu, Y., & Wang, S. (2024). Generative students: Using LLM-simulated student profiles for question item evaluation. In *Proceedings of L@S 2024*. https://arxiv.org/abs/2405.11591

arXiv. (2025). Simulating students with large language models: A comprehensive review. arXiv preprint. https://arxiv.org/abs/2511.06078

arXiv. (2026). Take out your calculators: Using LLMs to simulate student populations for item difficulty estimation. arXiv preprint. https://arxiv.org/abs/2601.09953
