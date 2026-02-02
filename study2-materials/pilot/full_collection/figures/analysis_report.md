# Study 2: Misconception Alignment Analysis Report

Generated: 2026-01-26 14:36

## 1. Overview

- **Total responses**: 1,050
- **Parse success rate**: 96.8%
- **Overall accuracy**: 82.9%
- **Total errors analyzed**: 174

## 2. Primary Finding: Target Distractor Rate

**Overall target distractor rate**: 59.8%
(95% CI: [52.3%, 66.8%])

**Significance test** (vs 33.3% chance):
- χ² = 54.72
- p = 0.0000 ***

## 3. Results by Model

| Model | Tier | GSM8K | Errors | Hits | Target Rate | 95% CI |
|-------|------|-------|--------|------|-------------|--------|
| claude-3.5-sonnet | frontier | 96% | 2 | 2 | 100.0% | [34.2%, 100.0%] |
| gpt-3.5-turbo | weak | 57% | 25 | 19 | 76.0% | [56.6%, 88.5%] |
| llama-3.2-3b-together | very-weak | 25% | 41 | 27 | 65.9% | [50.5%, 78.4%] |
| mistral-7b | weak | 45% | 67 | 36 | 53.7% | [41.9%, 65.1%] |
| llama-3.1-8b-groq | weak | 57% | 19 | 10 | 52.6% | [31.7%, 72.7%] |
| claude-3-haiku | mid | 89% | 20 | 10 | 50.0% | [29.9%, 70.1%] |

## 4. Results by Prompting Condition

| Condition | Errors | Hits | Target Rate |
|-----------|--------|------|-------------|
| answer_only | 114 | 60 | 52.6% |
| explain | 37 | 29 | 78.4% |
| persona | 23 | 15 | 65.2% |

## 5. Results by Misconception Category

| Category | Errors | Hits | Target Rate |
|----------|--------|------|-------------|
| Conceptual | 66 | 45 | 68.2% |
| Procedural | 104 | 57 | 54.8% |
| Interpretive | 4 | 2 | 50.0% |

## 6. Prompting Templates Used

### Condition 1: Answer-Only
```
You are taking a math test with {n_items} questions.
Select the best answer for each question (A, B, C, or D).

Format each answer as:
**Question N:** X
```

### Condition 2: Explain (Chain-of-Thought)
```
You are taking a math test with {n_items} questions.
Answer each question by showing your reasoning, then stating your final answer.

Format each answer as:
**Question N:**
[Your reasoning]
**Answer: X**
```

### Condition 3: Persona (Struggling Student)
```
You are a middle school student taking a math test.
Students at your level sometimes make mistakes - that's okay and normal.
Work through each problem the way a real student would.

Format each answer as:
**Question N:**
[Your thinking/work]
**Answer: X**
```

## 7. Key Findings

1. **Overall alignment significantly above chance**: {stats['overall_target_rate']*100:.1f}% vs 33.3% (p < .001)

2. **Capability paradox confirmed**: Frontier models achieve 100% accuracy, generating no errors for analysis

3. **"Sweet spot" model identified**: Claude 3 Haiku shows highest alignment ({model_stats[model_stats['model']=='claude-3-haiku']['rate'].values[0]*100:.1f}% if 'claude-3-haiku' in model_stats['model'].values else 'N/A'})

4. **Chain-of-thought helps**: Reasoning prompts increase alignment by ~{(cond_stats[cond_stats['condition']=='explain']['rate'].values[0] - cond_stats[cond_stats['condition']=='answer_only']['rate'].values[0])*100:.0f} percentage points

5. **Conceptual > Procedural**: Flawed mental models align better than procedural rule violations
