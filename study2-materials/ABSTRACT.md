# Wrong for the Right Reasons: LLM Synthetic Students Exhibit Authentic but Non-Predictive Misconceptions

**Submission to AIED 2026**

---

## Abstract

Large language models are increasingly proposed as synthetic students for educational research, promising scalable alternatives to costly human data collection. We tested whether LLMs can (1) predict item difficulty, (2) produce authentic misconception reasoning, and (3) simulate realistic student response patterns—using 36 Eedi diagnostic items with empirical student data (n=73,000 responses from 48,000 UK students).

In Experiment 1, Claude Opus 4.5 estimated item difficulty with near-zero correlation to actual student performance (r=0.19, p=0.36), despite calibration examples. In Experiment 2, we tested whether explicit "mental model" prompts could induce authentic misconceptions. S3/S4 prompts achieved 85% target distractor selection and 97% reasoning authenticity—LLMs genuinely work through flawed procedures, not just produce matching answers. However, item-level correlation with human difficulty remained near zero (r=-0.07). In Experiment 3, we implemented teacher-prediction prompting with proficiency-matched models (Llama-8B to Claude Sonnet) and realistic UK student profiles. Models differentiated internal proficiency levels (35% vs 94% accuracy) but showed no correlation with human difficulty (r=-0.19) or misconception rates (r=-0.09).

Qualitative analysis revealed the *selectivity problem*: LLMs apply misconceptions uniformly rather than selectively. On 2/7+1/7, where 65% of humans chose the misconception answer (3/14), a simulated struggling student answered correctly while describing the correct method as what the student would do wrong. Models know misconceptions exist but cannot predict which problem features activate them.

We conclude that LLMs can produce authentic misconception reasoning when explicitly prompted, but cannot substitute for empirical data on item difficulty or student response distributions.

---

**Keywords:** synthetic students, misconceptions, large language models, mathematics education, simulation validity, selectivity problem

**Word count:** 280
