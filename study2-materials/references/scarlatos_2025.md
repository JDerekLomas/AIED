# Scarlatos et al. (2025)

## Citation
Scarlatos, A., Fernandez, N., Ormerod, C., Lottridge, S., & Lan, A. (2025). SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), pages 25071-25094.

## Summary
Proposes SMART, a method that fine-tunes LLMs (Llama-3.2-Instruct 1B and 3B) using direct preference optimization (DPO) to create simulated students aligned with specific IRT ability levels. Generates thousands of simulated responses, scores them with an LLM-based scorer, then fits IRT to estimate difficulty. Evaluated on two datasets: Smarter Balanced (~858K responses to 49 reading comprehension items) and CodeWorkout (50 Java coding problems, 10,834 responses from 246 students).

## Key Numbers
- Smarter Balanced (3B model): Pearson r = 0.674, RMSE = 0.620
- CodeWorkout (3B model): Pearson r = 0.393, RMSE = 0.589
- Outperforms direct prediction baselines (e.g., ModernBERT achieved r = -0.04 on CodeWorkout)
- Outperforms Response Gen SFT baseline (r = 0.607 on Smarter Balanced)
- Uses Llama-3.2-Instruct as base model
- Designed specifically for open-ended items, not MCQs

## Relevance to Our Work
Represents a training-based approach to difficulty estimation, complementary to our zero-shot prompt engineering approach. Their moderate correlations (r = 0.39-0.67) on open-ended items provide a useful comparison point. Our work tests whether careful prompt design can achieve comparable performance without fine-tuning, across a broader range of item types and domains. Their finding that simulation-based approaches outperform direct prediction aligns with our inclusion of simulation-style prompts.
