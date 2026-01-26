# AIED 2026 Abstract Draft

## Title

**Beyond Difficulty: Can LLM Synthetic Students Exhibit Human-Like Misconceptions?**

---

## Abstract (250 words)

Large Language Models show promise as synthetic students for predicting item difficulty, potentially reducing costly human pretesting. However, predicting *which items are hard* is easier than predicting *why students fail*—the underlying misconceptions that drive incorrect responses. We investigate whether LLM-simulated students exhibit the same misconceptions as real students when answering mathematics items incorrectly.

Using the Eedi dataset of diagnostic mathematics questions, where each distractor is expert-labeled with a specific misconception, we test whether LLM reasoning reflects the intended misconception when selecting that distractor. We employ a 3×3 factorial design: three frontier models (GPT-4o, Claude 3.5 Sonnet, Llama-3.1-70B) crossed with three prompting strategies (answer-only, explain-then-answer, student persona). For each incorrect LLM response, we code whether its chain-of-thought reasoning matches the misconception associated with the selected distractor (full match, partial match, category match, or no match).

Preliminary results from 50 researcher-generated probe items suggest moderate misconception alignment: LLMs select distractors at above-chance rates, but their reasoning often reflects different error patterns than the intended misconception—particularly for conceptual versus procedural errors. Student persona prompting increases error rates but does not consistently improve misconception alignment.

Our findings have implications for using synthetic students in educational assessment. While LLMs may effectively predict item difficulty, their utility for distractor evaluation and misconception-targeted feedback may be limited by fundamental differences in how LLMs and humans arrive at incorrect answers.

---

## Keywords

synthetic students, misconceptions, large language models, diagnostic assessment, distractor analysis

---

## Track

**Human Aspects of AIED** (learner modeling, cognition)

or

**Technical Aspects of AIED** (evaluation methods)

---

## Notes for Submission

- Abstract is ~247 words (limit typically 250)
- Can adjust framing toward "technical" or "human" depending on track
- "Preliminary results" language allows flexibility since full data not yet collected
- Emphasizes the gap we're addressing (misconception-level vs difficulty-level)
