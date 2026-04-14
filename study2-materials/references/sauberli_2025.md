# Sauberli et al. 2025

## Citation
Sauberli, A., Frassinelli, D., & Plank, B. (2025). Do LLMs Give Psychometrically Plausible Responses in Educational Assessments? In Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), co-located with ACL 2025.

## Summary
Evaluates the psychometric plausibility of responses from 18 instruction-tuned LLMs (from the Llama 3, OLMo 2, Phi 3/4, and Qwen 2.5 families, ranging from 0.5B to 72B parameters) on two datasets of multiple-choice items: NAEP (549 items across reading, U.S. history, economics) and CMCQRD (504 English proficiency items, B1-C2 levels). Analyzes alignment using both classical test theory (CTT) and IRT frameworks.

## Key Numbers
- 18 LLMs tested across 4 model families
- NAEP: 549 items (252 reading, 204 U.S. history, 93 economics)
- CMCQRD: 504 items (B1-C2 English proficiency)
- CTT correlations with human difficulty: 0.32-0.56 range for CMCQRD B1 reading (best case)
- Most correlations described as "not very strong" and "fluctuate substantially"
- IRT analysis showed weaker results than CTT, with some significant negative correlations (e.g., 4th grade history)
- Temperature scaling improved alignment for large models but only marginally
- Core finding: "LLMs should not be used for piloting educational assessments in a zero-shot setting"

## Relevance to Our Work
Highly relevant -- they also conduct a large-scale evaluation of multiple LLMs on item difficulty estimation, finding weak and inconsistent correlations in zero-shot settings. Their pessimistic conclusion about zero-shot LLM use for assessment piloting contrasts with results from studies using more sophisticated prompting. Our work directly addresses this gap by showing that prompt strategy matters: systematic prompt comparison across models can identify configurations that substantially outperform zero-shot baselines. Their finding of subject-dependent performance (reading > history/economics) also relates to our cross-domain analysis.
