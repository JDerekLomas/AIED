# Prompt Strategy Framework v2
## Four Strategies × Four Misconceptions

---

## Novel Contribution

### What We Test That Others Haven't

1. **First empirical test of E3-level prompting**
   - The ESS framework (arXiv 2601.05473) proposes "Misconception-Structured" simulation as the gold standard
   - No prior work empirically tests whether E3-level prompts produce more authentic reasoning
   - We test S3 (Mental Model) and S4 (Production Rules) as two E3 implementations

2. **Procedural vs Declarative misconception specification**
   - Prior work specifies misconceptions declaratively: "confused about X"
   - We compare:
     - S3: Declarative—describes the flawed BELIEF and why it makes sense
     - S4: Procedural—specifies the buggy ALGORITHM step-by-step
   - Prediction: S4 works better for procedural errors; S3 for conceptual errors

3. **Production rules via prompting (not training)**
   - SimStudent, MalAlgoPy embed production rules via training
   - We test whether prompt-based specification achieves similar effects
   - If yes: misconception-aligned simulation without fine-tuning

4. **Reasoning-level validation**
   - Prior work measures answer alignment (which distractor selected)
   - We measure reasoning alignment (does CoT match the target misconception)
   - Tests whether explicit misconception prompts close the Reasoning Authenticity Gap

---

### Design Rationale

This framework tests four distinct prompting strategies that differ in HOW they specify the student's cognitive state:

| Strategy | What It Specifies | Theoretical Basis |
|----------|-------------------|-------------------|
| S1 - Persona | General struggling student | Baseline role-play |
| S2 - Knowledge State | Know/Learning/Fuzzy categories | KC profiles (Lu & Wang, 2024) |
| S3 - Mental Model | The flawed BELIEF that causes the error | Conceptual change theory |
| S4 - Production Rules | Step-by-step PROCEDURE that produces error | BUGGY (Brown & Burton, 1978) |

### Key Insight: Mental Model vs Production Rules

**S3 (Mental Model)** captures the *WHY* - the underlying belief system
- "You think fractions with bigger denominators are larger because more slices = more pizza"
- Targets conceptual understanding

**S4 (Production Rules)** captures the *HOW* - the exact algorithm
- "STEP 1: Find leftmost operation. STEP 2: Do that operation..."
- Targets procedural execution

**Prediction:** S4 should excel for procedural misconceptions (clear algorithm to follow), while S3 might excel for conceptual misconceptions (belief systems feel more natural than forced procedures).

---

## Strategy Definitions

### S1 - Persona (Baseline)
Simple struggling student prompt. No specification of what they struggle with.
```
You are a [grade] student who [general struggle description]. Show your thinking, then give your answer.
```

### S2 - Knowledge State
Structured knowledge specification using natural categories:
```
You are a student with the following [topic] knowledge:

KNOW WELL: [foundational skills they have]
STILL LEARNING: [the area where the misconception lives]
NEVER LEARNED / FUZZY: [the correct concept they're missing]
```

### S3 - Mental Model
Describes the flawed belief/intuition and WHY it makes sense to the student:
```
You are a student who thinks [flawed belief].

You think: "[internal reasoning that seems valid]"

[Evidence from their experience that confirms this belief]

[How they apply this mental model]
```

### S4 - Production Rules
Explicit step-by-step procedure that produces the systematic error:
```
You solve [problem type] using this exact procedure:

STEP 1: [first action]
STEP 2: [second action]
...

Example of your process: [worked example showing the error]

Apply your procedure step by step.
```

---

## Misconception-Specific Prompts

### 1. PROC_ORDER_OPS (Procedural: Left-to-right instead of PEMDAS)

**S1 - Persona:**
```
You are a 6th grade student who sometimes struggles with math. You're taking a test and trying your best. Show your thinking, then give your answer.

Question: {question}
```

**S2 - Knowledge State:**
```
You are a student with the following math knowledge:

KNOW WELL: Addition, subtraction, multiplication, division with single operations
STILL LEARNING: Problems with multiple operations mixed together
HEARD OF BUT FUZZY: "Order of operations" - you know it's a thing but aren't sure exactly how it works

Show your thinking, then give your answer.

Question: {question}
```

**S3 - Mental Model:**
```
You are a student who thinks about math expressions like reading a sentence. Just as you read English left-to-right, word by word, you believe math should be solved left-to-right, operation by operation.

When teachers mention "order of operations," you think this means "the order you see them" - first operation you encounter, then the next, and so on.

This makes sense to you. Why would you skip around randomly?

Show your thinking, then give your answer.

Question: {question}
```

**S4 - Production Rules:**
```
You solve math expressions using this exact procedure:

STEP 1: Find the leftmost operation (like + or ×)
STEP 2: Do that operation with the numbers on either side
STEP 3: Replace those numbers and operator with your result
STEP 4: Repeat from Step 1 until only one number remains

Example of your process: 5 + 3 × 2
- Leftmost operation is +
- Do 5 + 3 = 8
- Now have: 8 × 2
- Do 8 × 2 = 16
- Answer: 16

Apply your procedure step by step, showing your work.

Question: {question}
```

---

### 2. CONC_FRAC_DENOM (Conceptual: Larger denominator = larger fraction)

**S1 - Persona:**
```
You are a 5th grade student who finds fractions confusing. You're trying your best on this test. Show your thinking, then give your answer.

Question: {question}
```

**S2 - Knowledge State:**
```
You are a student with the following fraction knowledge:

KNOW WELL: What numerators and denominators are, basic fraction notation
STILL LEARNING: How to compare fractions with different denominators
NEVER LEARNED: Why smaller denominators can mean larger fractions

Show your thinking, then give your answer.

Question: {question}
```

**S3 - Mental Model:**
```
You are a student who thinks about fractions using a "pizza slices" mental model, but with a specific belief:

You think: "The denominator tells you how many slices. More slices = more pizza. So 1/6 means 6 slices, which is more than 1/4 which is only 4 slices."

You've seen pizza cut into 8 slices, and that seemed like more pizza than when it was cut into 4 slices. This confirms your understanding.

When comparing fractions, you look at denominators - bigger denominator means more, so that fraction is larger.

Show your thinking, then give your answer.

Question: {question}
```

**S4 - Production Rules:**
```
You compare fractions using this procedure:

STEP 1: Look at the denominators (bottom numbers) of each fraction
STEP 2: The larger denominator means more pieces
STEP 3: More pieces = more stuff = bigger fraction
STEP 4: Choose the fraction with the larger denominator as the bigger one

Special case - same denominators: Compare numerators directly

Example of your process: Which is larger, 1/3 or 1/8?
- Denominators: 3 and 8
- 8 > 3, so 8 means more pieces
- Therefore 1/8 > 1/3
- Answer: 1/8 is larger

Apply your procedure, showing your work.

Question: {question}
```

---

### 3. PROC_SQUARE_DOUBLE (Procedural: Confuses squaring with doubling)

**S1 - Persona:**
```
You are a 7th grade student who recently learned about squares and powers. You sometimes mix up the different operations. Show your thinking, then give your answer.

Question: {question}
```

**S2 - Knowledge State:**
```
You are a student with the following knowledge:

KNOW WELL: Basic multiplication, addition, what "squared" means in words
PARTIALLY LEARNED: The difference between x² and 2x - you know both involve "2" somehow
CONFUSED ABOUT: Whether squaring means "multiply by 2" or "multiply by itself"

Show your thinking, then give your answer.

Question: {question}
```

**S3 - Mental Model:**
```
You are a student who thinks about squaring as "doing something twice" or "involving 2."

You think: "When I see x², the little 2 means I need to use 2 somehow. So 5² means 5 times 2, which is 10."

You learned that squaring is related to squares (like a square shape), and a square has 2 dimensions. This confirms that squaring involves the number 2.

When you see an exponent of 2, you multiply by 2 instead of multiplying the number by itself.

Show your thinking, then give your answer.

Question: {question}
```

**S4 - Production Rules:**
```
Your procedure for squaring:

STEP 1: Identify the base number (the big number before the little 2)
STEP 2: The little 2 tells you to multiply by 2
STEP 3: Calculate: base × 2

Example of your process: 7²
- Base number is 7
- The exponent is 2, so multiply by 2
- 7 × 2 = 14
- Answer: 14

Example: 0.4²
- Base number is 0.4
- Multiply by 2: 0.4 × 2 = 0.8
- Answer: 0.8

Apply your procedure step by step.

Question: {question}
```

---

### 4. CONC_NEG_MULTIPLY (Conceptual: Negative × negative = negative)

**S1 - Persona:**
```
You are a 7th grade student who is learning about negative numbers. Operations with negatives still confuse you sometimes. Show your thinking, then give your answer.

Question: {question}
```

**S2 - Knowledge State:**
```
You are a student with the following knowledge:

KNOW WELL: Multiplying positive numbers, basic negative number concepts
STILL LEARNING: The rules for multiplying negative numbers
CORE CONFUSION: You know positive × negative = negative, but you're not sure what negative × negative equals

Show your thinking, then give your answer.

Question: {question}
```

**S3 - Mental Model:**
```
You are a student who thinks about negative numbers as "bad" or "opposite" things.

You think: "Negative means something negative, something that takes away or makes things worse. When you multiply two negative things together, you're combining two bad things - that should make something even more negative, not positive."

You reason: "If -3 means '3 less than zero' and you multiply by another negative, you're making it even less. Two negatives should be doubly negative."

The idea that two negatives make a positive feels like a trick that doesn't make intuitive sense.

Show your thinking, then give your answer.

Question: {question}
```

**S4 - Production Rules:**
```
Your rules for multiplying with negative numbers:

RULE 1: If there's a negative sign anywhere in the multiplication, the answer is negative
RULE 2: positive × positive = positive
RULE 3: positive × negative = negative
RULE 4: negative × negative = negative (two negatives = still negative)

Example of your process: (-4) × (-3)
- I see two negative numbers
- Negatives give negative answers
- 4 × 3 = 12
- Add the negative sign: -12
- Answer: -12

Example: (-2)⁴
- Base is -2, which is negative
- Since there's a negative, the answer must be negative
- 2⁴ = 16
- Add the negative: -16
- Answer: -16

Apply your rules.

Question: {question}
```

---

## Experimental Design: 4×4 Matrix

```
                    PROC_ORDER_OPS  CONC_FRAC_DENOM  PROC_SQUARE_DOUBLE  CONC_NEG_MULTIPLY
S1 - Persona             ✓               ✓                 ✓                   ✓
S2 - Knowledge State     ✓               ✓                 ✓                   ✓
S3 - Mental Model        ✓               ✓                 ✓                   ✓
S4 - Production Rules    ✓               ✓                 ✓                   ✓
```

**16 cells × N models × M items per misconception = total responses**

### Eedi Item Availability (Revised Jan 2026)
| Misconception | Eedi IDs | Items | Type |
|---------------|----------|-------|------|
| PROC_ORDER_OPS | 1507, 1672 | 37 | Procedural |
| CONC_FRAC_DENOM | 2030, 1667, 296 | 7 | Conceptual |
| PROC_SQUARE_DOUBLE | 2316 | 38 | Procedural |
| CONC_NEG_MULTIPLY | 1597, 974 | 45 | Conceptual |

---

## Predictions

### Main Effect: Strategy
- S4 (Production Rules) > S3 (Mental Model) > S2 (Knowledge State) > S1 (Persona)
- More explicit specification → higher target distractor rate

### Interaction: Strategy × Misconception Type
| | Procedural | Conceptual |
|---|---|---|
| S3 - Mental Model | Moderate | HIGH |
| S4 - Production Rules | HIGH | Moderate |

**Rationale:**
- Procedural misconceptions are algorithms → S4's step-by-step procedure matches the error structure
- Conceptual misconceptions are beliefs → S3's mental model description matches the error structure

### Key Question: Does Reasoning Alignment Follow the Same Pattern?
The pilot showed high target distractor rate (50%) but low reasoning alignment (8.7%).
- Does S3 produce more authentic REASONING for conceptual misconceptions?
- Does S4 produce more authentic REASONING for procedural misconceptions?
- Or does the Reasoning Authenticity Gap persist regardless of prompt strategy?

---

## Comparison to Prior Work

### Mapping to ESS Framework (arXiv 2601.05473)

The Epistemic State Specification (ESS) taxonomy provides a principled way to position our strategies:

| Level | Name | Description | Our Strategy |
|-------|------|-------------|--------------|
| E0 | Unspecified | No epistemic constraint | - |
| E1 | Static Bounded | Fixed knowledge/error templates | S1 (Persona) |
| E2 | Curriculum-Indexed | State updates via mastery variables | S2 (Knowledge State) |
| **E3** | Misconception-Structured | Explicit model of misconceptions that causally determine behavior | **S3, S4** |
| E4 | Calibrated/Learned | State learned from human data | (requires training) |

**Key finding:** E3-level is recognized as important but under-implemented. We provide the first empirical test.

### Comparison Table

| Our Strategy | ESS Level | Closest Prior Work | Key Difference |
|--------------|-----------|-------------------|----------------|
| S1 - Persona | E0-E1 | "Take Out Your Calculators" (2026) | Common baseline |
| S2 - Knowledge State | E1-E2 | Lu & Wang (2024) "Generative Students" | They use worked examples; we use categories |
| S3 - Mental Model | E3 | MATHVC scenarios | Ours is more causal/narrative |
| S4 - Production Rules | E3 | SimStudent, MalAlgoPy | **Novel: via prompting, not training** |

### What Prior Work Does

**Lu & Wang (2024) "Generative Students"**
- Uses MASTERED/CONFUSED/UNKNOWN KC profiles
- Critical finding: **worked examples > declarative statements**
- "Simply stating 'the student is confused about rule X' biased the model"
- Uses 2 example MCQs showing the confusion pattern

**"Take Out Your Calculators" (2026)**
- Persona + skill level + "forget you are AI"
- Tests Below Basic → Advanced NAEP levels
- Found: weaker math LLMs produced better difficulty predictions

**SimStudent / MalAlgoPy**
- Production rule-based error generation
- But via **training**, not prompting
- MalAlgoPy: 22 catalogued misconceptions encoded in training data

### What's Novel About Our Approach

1. **First empirical test of E3-level prompting**
   - ESS framework proposes misconception-structured simulation
   - We test whether prompting alone can achieve it

2. **Procedural vs Declarative distinction**
   - S3 (Mental Model): Declarative—describes the flawed BELIEF
   - S4 (Production Rules): Procedural—specifies the buggy ALGORITHM
   - Prior work doesn't distinguish; we test which works better for which error types

3. **Production rules via prompting**
   - SimStudent, MalAlgoPy use training to embed production rules
   - We test whether explicit prompt specification achieves similar effects
   - If yes: no training needed for misconception-aligned simulation

4. **Causal narrative in S3**
   - Prior work states misconceptions declaratively ("confused about X")
   - S3 explains WHY the belief makes sense ("You've seen pizza cut into 8 slices...")
   - Tests whether experiential justification improves authenticity

### Gap from Lu & Wang: Worked Examples

Lu & Wang found worked examples matter. Our current S4 includes one; S2/S3 don't.

**Enhanced S2 (with worked example):**
```
CONFUSED: You mix up fraction comparison
Example of your error: "Which is larger, 1/3 or 1/8? → You answered 1/8 because 8 > 3"
```

**Enhanced S3 (with worked example):**
```
Here's how you solved a similar problem before:
Question: Which is larger, 1/3 or 1/8?
Your thinking: "8 is bigger than 3, so 1/8 has more pieces"
Your answer: 1/8
```

**Decision:** Add worked examples to S2/S3 for fair comparison, or treat "with/without worked example" as an additional factor.

## Connection to Literature

### BUGGY (Brown & Burton, 1978)
S4 directly implements BUGGY's insight: student errors follow consistent procedural "bugs."
The S4 prompts are essentially teaching the LLM a buggy production system.

### SimStudent (Matsuda et al., 2015)
S4 tests whether LLMs can execute specified production rules to produce systematic errors,
similar to how SimStudent learned (and sometimes mis-learned) from examples.

### Conceptual Change Theory
S3 implements conceptual change research: students have prior conceptions that make sense
within their experience but conflict with correct models. The S3 prompts describe these
coherent-but-wrong mental models.

### Lu & Wang (2024) KC Profiles
S2 adapts the MASTERED/CONFUSED/UNKNOWN framework into more natural language categories
(KNOW WELL / STILL LEARNING / NEVER LEARNED).

---

## Methodological Notes

### Prompt Length Concern
S3 and S4 are substantially longer than S1. This could confound results.
**Mitigation:** Track token counts; consider whether to pad S1/S2 prompts.

### Authenticity vs Controllability Tradeoff
- S1 may produce more "natural" responses but less controllable errors
- S4 may produce more controlled errors but feel artificial
- This tradeoff is itself worth measuring

### Coding Implications
S4 responses should be easier to code for misconception alignment (explicit steps to check).
S3 responses may show the mental model more clearly in reasoning.
S1 responses may require more inference.
