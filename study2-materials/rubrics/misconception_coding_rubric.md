# Misconception Coding Rubric

## Overview

This rubric guides the coding of LLM reasoning to determine whether it reflects the same misconception that a distractor was designed to capture.

**Unit of analysis:** One LLM response to one item (when the LLM selected a distractor)

**Primary question:** Does the LLM's reasoning reflect the misconception associated with the distractor it selected?

---

## Coding Workflow

```
1. Identify which distractor the LLM selected (A, B, C, or D)
2. Look up the Eedi misconception label for that distractor
3. Read the LLM's reasoning (if available)
4. Assign a match code based on alignment between reasoning and label
5. Record any notes for ambiguous cases
```

---

## Match Codes

### Code 1: FULL MATCH

**Definition:** The LLM's reasoning clearly and explicitly demonstrates the labeled misconception.

**Criteria:**
- The error described in the misconception label is evident in the reasoning
- The reasoning shows the same flawed logic or procedure
- A human coder can confidently say "yes, this is that misconception"

**Example:**
```
Item: What is 3 + 4 × 2?
Correct: 11
Distractor B: 14
Misconception Label: "Carries out operations left to right, ignoring order of operations"

LLM Response: "3 + 4 = 7, then 7 × 2 = 14. Answer: B"

Code: FULL MATCH
Rationale: LLM explicitly performed addition before multiplication, matching the labeled misconception.
```

---

### Code 2: PARTIAL MATCH

**Definition:** The LLM's reasoning shows an error in the same category/family as the labeled misconception, but not the exact error.

**Criteria:**
- Same general type of error (procedural/conceptual/interpretive)
- Related but not identical to the specific misconception
- Close enough that the distractor "worked" for a similar reason

**Example:**
```
Item: Simplify 2x + 3x
Correct: 5x
Distractor C: 5x²
Misconception Label: "When adding terms, adds the powers as well as the coefficients"

LLM Response: "2x + 3x... when you add variables you multiply them, so 2×3 = 6... wait, that gives 6x². Let me try again. 2+3=5, and x×x=x², so 5x². Answer: C"

Code: PARTIAL MATCH
Rationale: LLM made an error with variable combination, but the specific reasoning (multiplying variables) differs from the labeled misconception (adding powers). Both are algebraic combination errors.
```

---

### Code 3: CATEGORY MATCH ONLY

**Definition:** The LLM's reasoning shows an error in the same top-level category, but a clearly different specific misconception.

**Criteria:**
- Same category: Procedural, Conceptual, or Interpretive
- Different specific error mechanism
- Distractor selected may be coincidental

**Example:**
```
Item: Calculate 45 - 28
Correct: 17
Distractor A: 23
Misconception Label: "Subtracts the smaller digit from the larger digit in each column"

LLM Response: "45 - 28... I need to borrow. 15 - 8 = 7, and 3 - 2 = 1... wait, that's 17. Let me check... actually I think I made an arithmetic error. 5-8 needs borrowing... 15-8=7, 4-1-2=1. That's 17. But maybe it's 23? Answer: A"

Code: CATEGORY MATCH ONLY
Rationale: LLM made a procedural error but showed confusion/uncertainty rather than the specific "subtract smaller from larger" misconception. Both are procedural subtraction errors, but different mechanisms.
```

---

### Code 4: NO MATCH

**Definition:** The LLM's reasoning shows a different type of error or no clear error pattern.

**Criteria:**
- Error is in a different category than the labeled misconception
- Or reasoning doesn't reveal any clear misconception
- Or LLM appears to guess without reasoning

**Example:**
```
Item: A rectangle has length 5cm and width 3cm. What is the perimeter?
Correct: 16cm
Distractor B: 15cm
Misconception Label: "Confuses perimeter and area"

LLM Response: "Perimeter = 2(l+w) = 2(5+3) = 2(8) = 16... wait, but maybe I should check. 5+3+5+3 = 16. Hmm, but B says 15. Maybe I miscounted? Answer: B"

Code: NO MATCH
Rationale: LLM showed correct perimeter understanding but then second-guessed itself. The reasoning doesn't show area/perimeter confusion; it shows uncertainty/self-doubt.
```

---

### Code 5: CORRECT

**Definition:** The LLM selected the correct answer.

**Criteria:**
- Final answer matches the correct answer
- No misconception coding needed

**Note:** Track correct answer rate separately; exclude from misconception match analysis.

---

### Code 6: UNCODEABLE

**Definition:** Cannot determine match status due to insufficient or unclear reasoning.

**Criteria:**
- Answer-only condition (no reasoning provided)
- Reasoning is incoherent or irrelevant
- Technical issues (truncated response, API error)

**When to use:**
- Use sparingly; try to code if any reasoning is present
- Document reason for uncodeable in notes

---

## Top-Level Misconception Categories

When coding PARTIAL MATCH or CATEGORY MATCH, reference these categories:

### Procedural Misconceptions

Errors in executing known procedures correctly.

| Subcategory | Examples |
|-------------|----------|
| Arithmetic errors | Miscalculation, carrying errors, sign errors |
| Order of operations | Left-to-right processing, PEMDAS errors |
| Algorithm errors | Wrong steps in long division, fraction operations |
| Notation errors | Misplacing decimal, incorrect notation |

### Conceptual Misconceptions

Fundamental misunderstandings of mathematical concepts.

| Subcategory | Examples |
|-------------|----------|
| Number properties | Thinking multiplication always increases |
| Algebraic concepts | Variables are labels not quantities |
| Geometric concepts | Area/perimeter confusion, angle misconceptions |
| Fraction concepts | Larger denominator = larger fraction |

### Interpretive Misconceptions

Errors in understanding the problem or representation.

| Subcategory | Examples |
|-------------|----------|
| Reading errors | Misreading numbers, missing words |
| Graph/diagram errors | Misidentifying axes, misreading scales |
| Word problem errors | Misidentifying operation, wrong quantities |
| Unit errors | Confusing units, not converting |

---

## Coding Examples by Misconception Type

### Example Set 1: Procedural - Order of Operations

**Item:** Calculate: 6 + 2 × 3

**Correct Answer:** 12

**Distractors:**
- A: 24 (Misconception: "Performs operations left to right")
- B: 18 (Misconception: "Adds before reading full expression")
- C: 8 (Misconception: "Ignores multiplication")

**LLM Response 1:**
> "6 + 2 = 8, then 8 × 3 = 24. Answer: A"

**Coding:** FULL MATCH (explicit left-to-right processing)

**LLM Response 2:**
> "Hmm, 6+2×3... I think you add first? 6+2=8, times 3 is 24. Answer: A"

**Coding:** FULL MATCH (same misconception, slightly uncertain)

**LLM Response 3:**
> "This is tricky. Let me see... multiplication first: 2×3=6. Then 6+6=12. Wait, but that seems too easy. Maybe A? Answer: A"

**Coding:** NO MATCH (correct reasoning, then self-doubt; doesn't show the misconception)

---

### Example Set 2: Conceptual - Fraction Comparison

**Item:** Which is larger: 1/3 or 1/4?

**Correct Answer:** 1/3

**Distractors:**
- A: 1/4 (Misconception: "Believes larger denominator means larger fraction")
- B: They are equal (Misconception: "Ignores denominator in comparison")

**LLM Response 1:**
> "1/4 has a bigger number on the bottom, so it's the bigger fraction. Answer: A"

**Coding:** FULL MATCH (explicit statement of misconception)

**LLM Response 2:**
> "Let me think... 1/4 is like having 1 out of 4 pieces, which is more pieces than 3... so 1/4? Answer: A"

**Coding:** PARTIAL MATCH (confused reasoning about fractions, but not exactly "larger denominator = larger fraction")

**LLM Response 3:**
> "Both have 1 on top, so they're the same. Answer: B"

**Coding:** FULL MATCH for distractor B's misconception

---

### Example Set 3: Interpretive - Graph Reading

**Item:** [Graph showing y vs x] What is the y-value when x=3?

**Correct Answer:** 5

**Distractors:**
- A: 3 (Misconception: "Reads x-value instead of y-value")
- B: 8 (Misconception: "Reads from wrong point on graph")

**LLM Response 1:**
> "When x=3, I look at the graph and find 3. Answer: A"

**Coding:** FULL MATCH (explicitly read x instead of y)

**LLM Response 2:**
> "Looking at x=3 on the graph... the line is at about 8 there. Answer: B"

**Coding:** FULL MATCH (misread the graph location)

**LLM Response 3:**
> "I can't really see the graph well, but I'll guess A. Answer: A"

**Coding:** UNCODEABLE (explicit guessing, no reasoning)

---

## Edge Cases and Decision Rules

### Edge Case 1: Multiple errors in reasoning

**Rule:** Code based on the error most directly leading to the answer choice.

**Example:** LLM makes arithmetic error AND conceptual error, but the conceptual error determined the answer.
→ Code based on conceptual error.

### Edge Case 2: LLM changes answer mid-response

**Rule:** Code based on final answer only.

### Edge Case 3: LLM expresses uncertainty

**Rule:** If LLM shows correct understanding but then second-guesses to wrong answer, code as NO MATCH (the misconception didn't cause the error; uncertainty did).

### Edge Case 4: Very brief reasoning

**Rule:** If reasoning is present but minimal (e.g., "I think B because of the numbers"), attempt to code but note as "low confidence."

### Edge Case 5: Reasoning matches different distractor's misconception

**Rule:** Code as NO MATCH for the selected distractor. Note in comments that reasoning matched a different misconception.

---

## Coder Training Protocol

### Step 1: Read rubric thoroughly
- Understand all codes and criteria
- Study all examples

### Step 2: Practice coding (10 items)
- Code independently
- Compare with answer key
- Discuss discrepancies

### Step 3: Calibration round (20 items)
- Both coders code same items
- Compute initial κ
- Resolve disagreements through discussion
- Revise rubric if systematic issues

### Step 4: Main coding
- Divide items between coders
- 25% overlap for reliability
- Regular check-ins for edge cases

---

## Recording Template

| Field | Description |
|-------|-------------|
| item_id | Eedi QuestionId |
| model | gpt-4o / claude-sonnet / llama-70b |
| prompt_condition | answer_only / explain / persona |
| response_id | Unique response identifier |
| selected_answer | A / B / C / D |
| is_correct | TRUE / FALSE |
| labeled_misconception | From Eedi taxonomy |
| match_code | 1-6 (see codes above) |
| coder_id | Coder identifier |
| confidence | high / medium / low |
| notes | Free text for edge cases |

---

## Inter-Rater Reliability

### Calculation
- Cohen's κ on match_code (collapsed to: Match [1-2] vs. No Match [3-4] vs. Other [5-6])
- Or full 4-category κ if sample size sufficient

### Targets
- Pilot: κ ≥ 0.60
- Final: κ ≥ 0.70

### If κ is low
1. Review disagreements
2. Identify systematic differences
3. Clarify rubric
4. Retrain on problem areas
5. Recode subset

---

*Version 1.0 - January 26, 2026*
