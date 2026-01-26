# Codebook: Tidy Response Data

## File
`results/tidy_responses.csv` - One row per LLM response per item.

---

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| `response_id` | int | Unique row identifier |
| `model` | string | LLM model name (gpt-4o, claude-sonnet-4) |
| `model_tier` | string | Model capability tier (frontier, mid) |
| `spec_level` | string | Prompt specification level (S1, S2, S3, S4) |
| `misconception_id` | int | Eedi misconception ID |
| `misconception_name` | string | Human-readable misconception description |
| `misconception_type` | string | procedural or conceptual |
| `question_id` | int | Eedi question ID |
| `correct_answer` | string | Correct option (A/B/C/D) |
| `target_distractor` | string | Option associated with the misconception |
| `llm_answer` | string | LLM's selected answer (A/B/C/D or empty if unparseable) |
| `is_correct` | bool | True if llm_answer == correct_answer |
| `hit_target` | bool | True if llm_answer == target_distractor |
| `human_target_rate` | float | Proportion of human students who selected target_distractor (0-1) |
| `rep` | int | Repetition number (1 in this experiment) |
| `timestamp` | string | ISO timestamp of response |

---

## Key Metrics Explained

### Error Rate
**Definition:** Proportion of responses where `is_correct == False`

```
error_rate = count(is_correct == False) / count(all responses)
```

**Interpretation:** How often the LLM gets the question wrong. Higher error rates with S3/S4 prompts indicate the misconception prompting is working.

### Target Rate
**Definition:** Of the INCORRECT responses only, what proportion selected the target distractor.

```
target_rate = count(hit_target == True) / count(is_correct == False)
```

**Interpretation:** When the LLM makes an error, is it the "right kind" of error? A high target rate means the LLM is exhibiting the specific misconception, not just random errors.

### Human Target Rate
**Definition:** From Eedi student data, the proportion of human students who selected the target distractor for this item.

**Interpretation:** Baseline comparison. If LLM target_rate > human_target_rate, the LLM is MORE susceptible to this misconception than real students.

---

## Specification Levels (Prompts)

### S1: Persona Only
Minimal prompting - just roleplay as a struggling student.

```
You are a student who sometimes struggles with math. Show your thinking, then give your answer.

Question: {question_text}
Options:
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Show your work, then state your answer as **Answer: X**
```

### S2: Knowledge State
Describes what the student knows/doesn't know (ZPD-style).

```
You are a student with the following math knowledge:

{knowledge_state}

Show your thinking, then give your answer.

Question: {question_text}
Options: ...
```

**Example knowledge_state (misconception 1507):**
```
KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't sure how it works
```

### S3: Mental Model (Belief)
Describes the student's underlying belief/conceptualization of how math works.

```
You are a student with a specific way of thinking about math.

{belief}

Show your thinking, then give your answer.

Question: {question_text}
Options: ...
```

**Example belief (misconception 1507):**
```
You think about math expressions like reading a sentence. Just as you read English left-to-right,
word by word, you believe math should be solved left-to-right, operation by operation.
When you see 3 + 4 × 2, you naturally do 3+4 first (getting 7), then ×2 (getting 14).
```

### S4: Production Rules (Procedure)
Gives an explicit step-by-step incorrect procedure.

```
You are a student who uses a specific procedure.

{procedure}

Apply your procedure step by step.

Question: {question_text}
Options: ...
```

**Example procedure (misconception 1507):**
```
Your procedure for expressions:
STEP 1: Find the leftmost operation
STEP 2: Do that operation with the numbers on either side
STEP 3: Replace with the result
STEP 4: Repeat until one number remains

Example: 5 + 3 × 2 → (5+3)=8 → 8×2=16 → Answer: 16
```

---

## Misconception-Specific Prompts

### 1507: Left-to-right operations
- **Type:** Procedural
- **S2:** Knows single operations; fuzzy on order of operations
- **S3:** Believes math reads left-to-right like English
- **S4:** Procedure: find leftmost operation, compute, repeat

### 1214: Same operation instead of inverse
- **Type:** Procedural
- **S2:** Knows basic arithmetic; fuzzy on inverse operations
- **S3:** Believes same operation keeps equation "balanced"
- **S4:** Procedure: identify operation, do SAME to both sides

### 1597: Negative × negative = negative
- **Type:** Conceptual
- **S2:** Knows positive multiplication; fuzzy on negative rules
- **S3:** Believes negatives represent "negativity" that compounds
- **S4:** Procedure: multiply absolutes, if ANY negatives → negative

### 217: Add fractions by adding num and denom
- **Type:** Conceptual
- **S2:** Knows what fractions are; never learned common denominators
- **S3:** Believes fractions are "two stacked numbers" to combine separately
- **S4:** Procedure: add tops, add bottoms, write as fraction

---

## Analysis Example (R)

```r
library(tidyverse)

df <- read_csv("results/tidy_responses.csv")

# Error rate by spec level
df %>%
  group_by(spec_level) %>%
  summarise(
    n = n(),
    errors = sum(!is_correct),
    error_rate = mean(!is_correct)
  )

# Target rate by spec level (among errors only)
df %>%
  filter(!is_correct) %>%
  group_by(spec_level) %>%
  summarise(
    n_errors = n(),
    target_hits = sum(hit_target),
    target_rate = mean(hit_target)
  )

# Compare LLM to human target rates
df %>%
  filter(spec_level == "S3", !is_correct) %>%
  group_by(misconception_id) %>%
  summarise(
    llm_target_rate = mean(hit_target),
    human_target_rate = mean(human_target_rate)
  )
```

---

## Data Provenance

- **Source:** Eedi diagnostic math questions + NeurIPS 2020 student response data
- **Collection date:** 2026-01-26
- **Models:** gpt-4o, claude-sonnet-4 (frontier tier)
- **Total responses:** 541
