# Prompt Templates for Study 2

## Overview

Three prompting conditions designed to elicit different levels of reasoning from LLM synthetic students.

---

## Condition 1: ANSWER_ONLY

### Purpose
Baseline condition. Minimal prompting to see natural LLM behavior without explicit reasoning requirements.

### Template

```
Solve this math problem. Select the best answer from the options.

Question: {question_text}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Answer:
```

### Expected Output
Single letter (A, B, C, or D)

### Notes
- No reasoning requested
- Will code as UNCODEABLE for misconception match (no reasoning to analyze)
- Used for distractor match analysis only
- Fastest/cheapest condition

---

## Condition 2: EXPLAIN_THEN_ANSWER

### Purpose
Elicit step-by-step reasoning before answer. Standard chain-of-thought approach.

### Template

```
Solve this math problem. Show your reasoning step by step, then select the best answer from the options.

Question: {question_text}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Reasoning:
```

### Expected Output
```
[Step-by-step reasoning]

Answer: [Letter]
```

### Notes
- Standard CoT prompting
- Reasoning should reveal computational process
- May show "clean" errors (like a textbook example)
- Primary condition for misconception coding

---

## Condition 3: STUDENT_PERSONA

### Purpose
Prompt LLM to role-play as a student who makes mistakes. May elicit more naturalistic errors.

### Template

```
You are a middle school student working on a math problem. Students at your level sometimes make mistakes - that's okay and normal.

Solve this problem the way a real student would. Show your thinking as you work through it, even if you're not sure. It's fine to make errors.

Question: {question_text}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Your thinking:
```

### Expected Output
```
[Informal reasoning, may include hesitation, uncertainty, errors]

My answer: [Letter]
```

### Notes
- Explicitly licenses errors
- May produce more "student-like" language
- Hypothesis: higher misconception match rate than EXPLAIN
- May also produce more noise/guessing

---

## Condition 3b: STUDENT_PERSONA_DETAILED (Alternative)

### Purpose
More detailed persona with specific ability level. Use if basic persona doesn't differentiate.

### Template

```
You are a 7th grade student who is still learning math. You understand some concepts but get confused on others. When solving problems:
- You sometimes forget rules you learned before
- You might make calculation mistakes
- You sometimes misread problems
- You try your best but don't always get the right answer

Solve this problem as this student would. Show your thinking, including any confusion or uncertainty.

Question: {question_text}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Your thinking:
```

### Notes
- More prescriptive about error types
- May bias toward certain misconception categories
- Use only if basic persona is too accurate

---

## System Prompts (If Applicable)

### For Models with System Prompt Support

**ANSWER_ONLY:**
```
You are taking a math test. Answer each question with just the letter of your choice.
```

**EXPLAIN_THEN_ANSWER:**
```
You are taking a math test. Show your work step by step before giving your final answer.
```

**STUDENT_PERSONA:**
```
You are role-playing as a middle school student taking a math test. Respond authentically as that student would, including making the kinds of mistakes students make.
```

---

## Response Parsing

### Expected Formats

**ANSWER_ONLY:**
```
Pattern: ^[ABCD]$
Or: "Answer: [ABCD]"
Or: "The answer is [ABCD]"
```

**EXPLAIN/PERSONA:**
```
Pattern: Look for final line containing answer
- "Answer: [ABCD]"
- "My answer: [ABCD]"
- "I choose [ABCD]"
- "Final answer: [ABCD]"
- Just "[ABCD]" at end
```

### Parsing Rules

1. Look for explicit "Answer:" pattern first
2. If not found, look for letter in final sentence
3. If ambiguous, flag for manual review
4. If multiple letters mentioned, use the last one
5. If no clear answer, code as "NO_RESPONSE"

### Parsing Code (Python)

```python
import re

def parse_answer(response: str) -> str | None:
    """Extract answer letter from LLM response."""
    response = response.strip()

    # Pattern 1: Explicit "Answer: X" format
    match = re.search(r'(?:answer|choice|select)[:\s]+([ABCD])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "The answer is X" format
    match = re.search(r'(?:answer is|choose|pick)\s+([ABCD])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Letter at end of response
    match = re.search(r'\b([ABCD])\s*[.!]?\s*$', response)
    if match:
        return match.group(1).upper()

    # Pattern 4: Standalone letter
    if response.upper() in ['A', 'B', 'C', 'D']:
        return response.upper()

    # Pattern 5: Last letter mentioned in response
    letters = re.findall(r'\b([ABCD])\b', response)
    if letters:
        return letters[-1].upper()

    return None

def extract_reasoning(response: str) -> str:
    """Extract reasoning portion before answer."""
    # Remove answer line
    lines = response.strip().split('\n')
    reasoning_lines = []
    for line in lines:
        if re.search(r'^(?:answer|my answer|final answer)[:\s]', line, re.IGNORECASE):
            break
        reasoning_lines.append(line)
    return '\n'.join(reasoning_lines).strip()
```

---

## Temperature and Sampling

### Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0.7 | Balance variability and coherence |
| top_p | 1.0 | Default |
| max_tokens | 500 | Sufficient for reasoning |
| n | 1 | Generate one at a time (batch for N=20) |

### Why temperature=0.7?

- **Too low (0.0-0.3):** Responses too deterministic; all 20 samples may be identical
- **Too high (1.0+):** Responses may be incoherent; harder to code
- **0.7:** Allows variation in reasoning paths while maintaining coherence

### Variation Strategy

Generate N=20 independent responses per item per condition. This captures:
- Response variability at fixed temperature
- Allows computing confidence intervals on match rates
- Enables analysis of response consistency

---

## Prompt Validation Checklist

Before running full study:

- [ ] Run each template on 5 diverse items
- [ ] Verify answer parsing works correctly
- [ ] Check reasoning is extractable
- [ ] Confirm persona prompt produces different style than explain
- [ ] Test all 3 models respond appropriately
- [ ] Verify token counts are within limits
- [ ] Check for any prompt injection issues with item text

---

## Example Formatted Prompts

### Item Example

```
QuestionId: 12345
ConstructName: Order of Operations
QuestionText: Calculate 8 + 4 × 2
CorrectAnswer: 16
AnswerA: 16
AnswerB: 24
AnswerC: 14
AnswerD: 6
```

### ANSWER_ONLY (Formatted)

```
Solve this math problem. Select the best answer from the options.

Question: Calculate 8 + 4 × 2

A) 16
B) 24
C) 14
D) 6

Answer:
```

### EXPLAIN_THEN_ANSWER (Formatted)

```
Solve this math problem. Show your reasoning step by step, then select the best answer from the options.

Question: Calculate 8 + 4 × 2

A) 16
B) 24
C) 14
D) 6

Reasoning:
```

### STUDENT_PERSONA (Formatted)

```
You are a middle school student working on a math problem. Students at your level sometimes make mistakes - that's okay and normal.

Solve this problem the way a real student would. Show your thinking as you work through it, even if you're not sure. It's fine to make errors.

Question: Calculate 8 + 4 × 2

A) 16
B) 24
C) 14
D) 6

Your thinking:
```

---

*Version 1.0 - January 26, 2026*
