# Srivatsa et al. 2025

## Citation
Srivatsa, K. V. A., Maurya, K. K., & Kochmar, E. (2025). Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension? In Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), co-located with ACL 2025.

## Summary
Evaluates 11 LLMs (LLaMA2-13B, LLaMA2-70B, LLaMA3.1-8B, LLaMA3.1-70B, Mistral-7B, Qwen2.5-7B, Qwen2.5-Math, GPT-3.5-Turbo, o3-Mini, SocraticLM, LearnLM-1.5-Pro) on 489 items from the National Assessment of Educational Progress (NAEP) covering math and reading comprehension across grades 4, 8, and 12. Uses IRT to place models on the same ability scale as real students. Tests both default and grade-enforcement prompting.

## Key Numbers
- 489 NAEP items (249 math, 240 reading) across grades 4, 8, 12
- Without prompting: models deviate from target 50th percentile by 25-41 percentile points
- With grade-enforcement prompting: no model-prompt pair aligns across all subjects and grades
- Qwen2.5-7B showed extreme instability (delta = -93.1 percentile points in grade 4 reading with enforcement)
- GPT-3.5-Turbo closest to target in multiple scenarios
- Core finding: "no evaluated model-prompt pair fits the bill across subjects and grades"

## Relevance to Our Work
Directly relevant -- they also find that LLMs struggle to reliably simulate student ability, especially across domains and grade levels. Their finding that grade-enforcement prompting produces inconsistent results across models supports our systematic comparison of prompt strategies. Our work complements theirs by focusing on item difficulty estimation (ranking items) rather than ability simulation (matching student distributions), and by testing a much larger prompt space (16 prompts vs. their grade-enforcement approach).
