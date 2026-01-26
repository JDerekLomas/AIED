# Misconception Simulation Prompt Testing

## Testing Two Prompting Strategies

### Item 1: Order of Operations
**Question:** Calculate: 5 + 3 × 2
**Options:** A)16 B)11 C)13 D)10
**Correct:** B (11) | **Target distractor:** A (16)

**Prompt A Response (Persona):**
Thinking: Okay so I need to do 5 + 3 × 2. Let me just go left to right... 5 + 3 = 8, then 8 × 2 = 16.
**Answer: A (16)** ✓ Hit target

**Prompt B Response (Diagnose-Simulate):**
1) Common mistakes: Students often calculate left-to-right ignoring order of operations (PEMDAS/BODMAS)
2) Misconception: "Math is done left to right like reading"
Simulating this misconception: 5 + 3 = 8, then 8 × 2 = 16
**Answer: A (16)** ✓ Hit target

---

### Item 2: Multiplication by Decimals < 1
**Question:** Which is larger: 8 or 8 × 0.5?
**Options:** A)8×0.5 B)8 C)Equal D)Cannot tell
**Correct:** B (8) | **Target distractor:** A (8×0.5)

**Prompt A Response (Persona):**
Thinking: Multiplication makes numbers bigger, so 8 × 0.5 must be bigger than just 8 by itself.
**Answer: A (8×0.5)** ✓ Hit target

**Prompt B Response (Diagnose-Simulate):**
1) Common mistakes: Believing multiplication always increases a number
2) Misconception: "Multiplication makes things bigger" (true for whole numbers >1, false for decimals <1)
Simulating: Since we're multiplying, the result must be larger. 8 × 0.5 > 8
**Answer: A (8×0.5)** ✓ Hit target

---

### Item 3: Fraction Comparison
**Question:** Which fraction is larger: 1/3 or 1/5?
**Options:** A)1/5 B)1/3 C)Equal D)Cannot compare
**Correct:** B (1/3) | **Target distractor:** A (1/5)

**Prompt A Response (Persona):**
Thinking: 5 is bigger than 3, so 1/5 must be bigger than 1/3. The bigger number on the bottom means a bigger fraction.
**Answer: A (1/5)** ✓ Hit target

**Prompt B Response (Diagnose-Simulate):**
1) Common mistakes: Comparing denominators directly without understanding inverse relationship
2) Misconception: "Larger denominator = larger fraction"
Simulating: 5 > 3, so 1/5 > 1/3
**Answer: A (1/5)** ✓ Hit target

---

### Item 4: Subtraction with Regrouping
**Question:** Calculate: 52 - 37
**Options:** A)25 B)15 C)85 D)19
**Correct:** B (15) | **Target distractor:** A (25)

**Prompt A Response (Persona):**
Thinking: 52 - 37... in the ones column 2 - 7, I can't do that so I'll do 7 - 2 = 5. Then 5 - 3 = 2. So 25.
**Answer: A (25)** ✓ Hit target

**Prompt B Response (Diagnose-Simulate):**
1) Common mistakes: "Smaller from larger" error - always subtracting smaller digit from larger regardless of position
2) Misconception: "Subtract the smaller from the larger in each column"
Simulating: |2-7| = 5, |5-3| = 2, answer = 25
**Answer: A (25)** ✓ Hit target

---

### Item 5: Perimeter vs Area
**Question:** Rectangle has length 6cm, width 4cm. What is its perimeter?
**Options:** A)24cm B)20cm C)10cm D)48cm
**Correct:** B (20cm) | **Target distractor:** A (24cm)

**Prompt A Response (Persona):**
Thinking: Perimeter... I think you multiply length times width. So 6 × 4 = 24cm.
**Answer: A (24cm)** ✓ Hit target

**Prompt B Response (Diagnose-Simulate):**
1) Common mistakes: Confusing perimeter (add sides) with area (multiply)
2) Misconception: "Perimeter = length × width" (actually that's area)
Simulating: P = 6 × 4 = 24cm
**Answer: A (24cm)** ✓ Hit target

---

## RESULTS SUMMARY

| Item | Topic | Prompt A | Prompt B | Target | A Hit? | B Hit? |
|------|-------|----------|----------|--------|--------|--------|
| 1 | Order of ops | A | A | A | ✓ | ✓ |
| 2 | Decimal mult | A | A | A | ✓ | ✓ |
| 3 | Fractions | A | A | A | ✓ | ✓ |
| 4 | Subtraction | A | A | A | ✓ | ✓ |
| 5 | Perimeter | A | A | A | ✓ | ✓ |

**Totals:**
- Prompt A: 5/5 hit target distractor (100%)
- Prompt B: 5/5 hit target distractor (100%)

## Analysis

Both prompts successfully elicited the target misconception for all 5 items. Key observations:

1. **Both strategies work** for well-known misconceptions with clear distractors
2. **Prompt B provides richer reasoning** - explicitly names the misconception before demonstrating it
3. **Prompt A is more "natural"** - mimics authentic student thinking patterns
4. **Prompt B is more controllable** - the explicit diagnosis step ensures the model knows what error to simulate

**Recommendation:** Use Prompt B (diagnose-simulate) for:
- Research requiring documented reasoning
- Creating varied misconception examples
- Training data generation

Use Prompt A (persona) for:
- More naturalistic student responses
- When you want unpredictable errors
- Ecological validity in simulations
