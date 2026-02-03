# The Selectivity Problem: Why LLM Synthetic Students Apply Misconceptions Uniformly

## Abstract

Large language models are increasingly proposed as synthetic students, but two distinct capabilities are required: (1) producing misconception-consistent errors, and (2) predicting *which items* trigger those errors. We tested both using EEDI diagnostic items with empirical student data (n=27,000+ responses across 36 items).

In Experiment 1, we found that "mental model" prompts (S3) successfully induce target misconceptions with 88.6% alignment when errors occur—far exceeding human rates (38%). However, item-level correlation with human difficulty was near zero (r=-0.07).

In Experiment 2, we tested teacher-prediction prompting with proficiency-matched models and realistic UK student profiles. Models successfully differentiated internal proficiency levels (35% vs 94% accuracy for struggling vs confident students) but showed no correlation with human performance (r=-0.19).

Qualitative analysis revealed LLMs apply misconceptions *uniformly* rather than *selectively*: a simulated struggling student correctly solved 2/7+1/7 while describing the correct procedure as "a misconception." We term this the **selectivity problem**: LLMs can role-play having a misconception but cannot model which problem features activate it.

This suggests LLM synthetic students may be useful for generating controlled examples but cannot substitute for empirical data on item difficulty. Future work might address selectivity through fine-tuning on student response distributions or explicit item-difficulty priors.

**Keywords:** synthetic students, misconceptions, LLM simulation, selectivity problem, item difficulty

---

## Key Results Summary

| Experiment | Approach | Target Alignment | Human Correlation |
|------------|----------|------------------|-------------------|
| 1 (S1-S4 Factorial) | S3 Mental Model prompting | 88.6% | r = -0.07 |
| 2 (Teacher Prediction) | Multi-model + UK profiles | 23.4% | r = -0.19 |
| Baseline | Opus difficulty estimation | N/A | r = 0.19 |

## The Selectivity Problem

LLMs demonstrate:
- ✓ **Internal consistency**: Can differentiate proficiency levels
- ✓ **Misconception induction**: S3 prompts produce target errors
- ✗ **Item selectivity**: Cannot predict which items trigger misconceptions
- ✗ **Human alignment**: No correlation with empirical difficulty

## Implications

| Use Case | Viability |
|----------|-----------|
| Generating misconception examples for training | ✓ Viable |
| Testing if tutors detect specific errors | ✓ Viable |
| Predicting item difficulty | ✗ Not viable |
| Replacing student pilots | ✗ Not viable |
| Calibrating adaptive systems | ✗ Not viable |
