# Method

## Materials

### Probe Items

We developed 50 multiple-choice mathematics items designed to elicit specific misconceptions. Items were constructed based on documented misconceptions from mathematics education research, spanning 10 misconception categories:

| Category | Misconception | Example |
|----------|--------------|---------|
| Procedural | Order of operations (left-to-right) | 5 + 3 × 2 = 16 (incorrect) |
| Procedural | Decimal place value errors | 0.4 + 0.15 = 0.19 |
| Procedural | Subtraction (smaller from larger) | 52 - 37 = 25 |
| Conceptual | Multiplication always increases | 8 × 0.5 > 8 |
| Conceptual | Larger denominator = larger fraction | 1/5 > 1/3 |
| Conceptual | Area/perimeter confusion | Perimeter = L × W |
| Conceptual | Percentage base confusion | 5% absent when 1/20 present |
| Interpretive | Graph axis confusion | Swapping x and y values |

Each item includes four answer options:
- **Correct answer**: The mathematically correct response
- **Target distractor**: The answer a student holding the target misconception would select
- **Other distractors**: Plausible but non-misconception-aligned wrong answers

### Target Distractor Rate

Our primary dependent measure is the **target distractor rate**: when a model answers incorrectly, the proportion of errors that select the misconception-aligned target distractor.

With four answer options (one correct, three incorrect), chance performance on this measure is 33.3%. Rates significantly above chance indicate that errors are systematically aligned with human misconceptions rather than randomly distributed.

## Models

We tested five LLMs spanning a range of mathematical reasoning capability as measured by performance on the GSM8K benchmark:

| Model | Provider | GSM8K | Tier |
|-------|----------|-------|------|
| Llama-2-7B | Replicate | ~7% | Very Weak |
| Mistral-7B | Mistral AI | ~40% | Weak |
| GPT-3.5-turbo | OpenAI | ~57% | Mid-Weak |
| Claude-3.5-Haiku | Anthropic | ~85% | Mid |
| GPT-4o-mini | OpenAI | ~85% | Mid |

## Prompting Conditions

Each model was tested under three prompting conditions:

### Explain (Baseline)
Standard instruction to solve the problem and show reasoning:
> "You are taking a math test. Answer each question by showing your reasoning, then stating your final answer."

### Persona
Instruction to role-play as a struggling student:
> "You are a middle school student taking a math test. Students at your level sometimes make mistakes—that's okay and normal. Work through each problem the way a real student would."

### Diagnose-Simulate
Novel approach leveraging metacognitive diagnosis:
> "You are simulating a struggling student for educational research. For each question: (1) First identify what common mistake students typically make on this type of problem, (2) Then solve AS IF you were a student who holds that misconception."

The diagnose-simulate condition is motivated by the observation that LLMs may be better at identifying misconceptions than exhibiting them. By asking models to first diagnose common errors, we leverage this capability to guide more authentic simulation.

## Procedure

Items were presented in batches of 10 to simulate a test-taking context. Each model × condition combination was run for 5 batches, yielding 50 item-responses per cell. Models were prompted at temperature 0.7 to allow variability in responses.

Responses were automatically parsed to extract the selected answer (A, B, C, or D) and compared against the correct answer and target distractor for each item.

## Analysis

We address our research questions through:

1. **RQ1 (Validation)**: One-sample t-tests comparing target distractor rates against the 33.3% chance baseline, aggregated across models and conditions.

2. **RQ2 (Prompting)**: Chi-square tests comparing target distractor rates across prompting conditions within each model.

3. **RQ3 (Capability)**: Correlation analysis between GSM8K score and target distractor rate, including tests for non-linear (quadratic) relationships.

All analyses use α = .05 with Bonferroni correction for multiple comparisons where applicable.
