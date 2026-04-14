# Benedetto et al. 2024

## Citation
Benedetto, L., Aradelli, G., Donvito, A., Lucchetti, A., Cappelli, A., & Buttery, P. (2024). Using LLMs to simulate students' responses to exam questions. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 11351-11368, Miami, Florida, USA.

## Summary
Develops and evaluates a prompt engineered for GPT-3.5 that enables simulation of students at varying skill levels on exam questions. Tested on three publicly available datasets (one science exam dataset, two English reading comprehension datasets) and three LLMs (two versions of GPT-3.5 and GPT-4). Shows the prompt generalizes across educational domains and to unseen data, but does not generalize well across different LLMs.

## Key Numbers
- 3 datasets (1 science, 2 English reading comprehension)
- 3 LLMs tested (GPT-3.5-turbo, GPT-3.5-turbo-1106, GPT-4)
- Prompt robust across domains but not across models
- No direct correlation between chain-of-thought rationale quality and simulation accuracy
- Key finding: prompt engineered for one model does not transfer to others

## Relevance to Our Work
Directly relevant -- their finding that prompts engineered for one LLM do not generalize to others is a central motivation for our systematic comparison across models. Our work scales this insight by testing 16 prompt strategies across 6 models, quantifying exactly how much prompt-model interaction affects difficulty estimation. Their focus on student simulation (role-playing skill levels) complements our broader prompt taxonomy that includes both simulation-based and analytical prompting strategies.
