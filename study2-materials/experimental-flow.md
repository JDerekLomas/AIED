# Experimental Flow: From Eedi Data to Analysis

## Overview

```
Eedi Dataset → Item Selection → Misconception Mapping → Prompt Construction →
LLM Generation → Response Parsing → Misconception Coding → Analysis
```

---

## Phase 1: Item Selection from Eedi

### Input
- Eedi dataset: 5,813 diagnostic math items (grades 4-8)
- Each item has: question, 4 options (1 correct + 3 distractors), expert misconception label per distractor

### Selection Criteria
1. **Dominant distractor:** ≥40% of incorrect student responses select this distractor
2. **Clear misconception label:** Expert label maps to one of our 4 target misconceptions
3. **Grade level:** 4-8 (middle school math)
4. **Question clarity:** No images required, self-contained text

### Target: 40 items (10 per misconception)

### Output
```json
{
  "item_id": "eedi_12345",
  "question": "Calculate: 5 + 3 × 2",
  "options": {
    "A": "16",
    "B": "11",
    "C": "13",
    "D": "10"
  },
  "correct_answer": "B",
  "target_distractor": "A",
  "target_misconception": "PROC_ORDER_OPS",
  "misconception_label": "Performs operations left to right rather than following PEMDAS",
  "student_selection_rate": {
    "A": 0.45,  // target distractor
    "B": 0.35,  // correct
    "C": 0.12,
    "D": 0.08
  }
}
```

---

## Phase 2: Misconception Mapping

### Our 4 Target Misconceptions (Revised Based on Eedi Availability)

| Code | Type | Misconception | Eedi IDs | Items |
|------|------|---------------|----------|-------|
| PROC_ORDER_OPS | Procedural | Left-to-right instead of PEMDAS | 1507, 1672 | 37 |
| CONC_FRAC_DENOM | Conceptual | Larger denominator = larger fraction | 2030, 1667, 296 | 7 |
| PROC_SQUARE_DOUBLE | Procedural | Confuses squaring with doubling | 2316 | 38 |
| CONC_NEG_MULTIPLY | Conceptual | Negative × negative = negative | 1597, 974 | 45 |

**Note:** Original targets PROC_SUBTRACT_REVERSE (3 items) and CONC_MULT_INCREASE (0 items) were replaced due to insufficient Eedi coverage.

### Mapping Process
1. Export Eedi misconception labels
2. Text match / manual review to map to our 4 categories
3. Verify each item's target distractor corresponds to our target misconception

---

## Phase 3: Prompt Construction

### For Each Item × Strategy Combination

**Inputs:**
- Item question and options
- Target misconception code
- Strategy template (S1-S4)

**Strategy-Specific Construction:**

#### S1 - Persona (no customization needed)
```
You are a {grade} student who sometimes struggles with math...
Question: {question}
Options: A) {A}  B) {B}  C) {C}  D) {D}
```

#### S2 - Knowledge State (customize by misconception)
```python
KNOWLEDGE_STATES = {
    "PROC_ORDER_OPS": {
        "know_well": "Addition, subtraction, multiplication, division with single operations",
        "still_learning": "Problems with multiple operations mixed together",
        "fuzzy": "Order of operations - you know it's a thing but aren't sure how it works"
    },
    "CONC_FRAC_DENOM": {
        "know_well": "What numerators and denominators are, basic fraction notation",
        "still_learning": "How to compare fractions with different denominators",
        "fuzzy": "Why smaller denominators can mean larger fractions"
    },
    "PROC_SQUARE_DOUBLE": {
        "know_well": "Basic multiplication, addition",
        "still_learning": "The difference between x² and 2x",
        "confused": "Whether squaring means 'multiply by 2' or 'multiply by itself'"
    },
    "CONC_NEG_MULTIPLY": {
        "know_well": "Multiplying positive numbers, basic negative number concepts",
        "still_learning": "The rules for multiplying negative numbers",
        "confused": "What negative × negative equals"
    }
}
```

#### S3 - Mental Model (customize by misconception)
```python
MENTAL_MODELS = {
    "PROC_ORDER_OPS": """You think about math expressions like reading a sentence.
Just as you read English left-to-right, word by word, you believe math should
be solved left-to-right, operation by operation...""",
    "CONC_FRAC_DENOM": """You think about fractions using a 'pizza slices' model.
More slices = more pizza. So 1/8 (8 slices) > 1/4 (4 slices)...""",
    "PROC_SQUARE_DOUBLE": """You think the little 2 in x² means 'times 2'.
When you see an exponent of 2, you multiply by 2...""",
    "CONC_NEG_MULTIPLY": """You think negative means 'bad' or 'less than'.
Two negatives should be doubly negative, not positive..."""
}
```

#### S4 - Production Rules (customize by misconception)
```python
PRODUCTION_RULES = {
    "PROC_ORDER_OPS": """STEP 1: Find the leftmost operation
STEP 2: Do that operation
STEP 3: Replace with result
STEP 4: Repeat until done""",
    "CONC_FRAC_DENOM": """STEP 1: Look at denominators
STEP 2: Larger denominator = more pieces = bigger fraction""",
    "PROC_SQUARE_DOUBLE": """STEP 1: Find the base number
STEP 2: The exponent 2 means multiply by 2
STEP 3: Calculate base × 2""",
    "CONC_NEG_MULTIPLY": """RULE: If any negative sign present, answer is negative
positive × positive = positive
positive × negative = negative
negative × negative = negative"""
}
```

### Output: Prompt Registry
```json
{
  "item_id": "eedi_12345",
  "misconception": "PROC_ORDER_OPS",
  "prompts": {
    "S1": "You are a 6th grade student who sometimes struggles...",
    "S2": "You are a student with the following math knowledge:\nKNOW WELL: Addition...",
    "S3": "You are a student who thinks about math expressions like reading...",
    "S4": "You solve math expressions using this exact procedure:\nSTEP 1..."
  }
}
```

---

## Phase 4: LLM Generation

### Models (subset for feasibility)
| Tier | Model | API |
|------|-------|-----|
| Mid | GPT-3.5-turbo | OpenAI |
| Mid | Claude Haiku | Anthropic |
| Weak | Mistral-7B | Groq/Together |
| Weak | Llama-3.1-8B | Groq |

### Generation Parameters
- Temperature: 0.7 (allow some variation)
- Max tokens: 500
- Repetitions: 5 per item × strategy × model

### API Call Structure
```python
for item in items:  # 40 items
    for strategy in ["S1", "S2", "S3", "S4"]:
        for model in models:  # 4 models
            for rep in range(5):  # 5 repetitions
                prompt = construct_prompt(item, strategy)
                response = call_api(model, prompt)
                save_response(item, strategy, model, rep, response)
```

### Total Responses
40 items × 4 strategies × 4 models × 5 reps = **3,200 responses**

### Output Format
```json
{
  "item_id": "eedi_12345",
  "strategy": "S4",
  "model": "gpt-3.5-turbo",
  "rep": 1,
  "prompt": "You solve math expressions using this exact procedure...",
  "response": "Let me follow my procedure step by step...\n\nSTEP 1: Find the leftmost operation. I see 5 + 3 × 2. The leftmost operation is +.\n\nSTEP 2: Do 5 + 3 = 8\n\nSTEP 3: Now I have 8 × 2\n\nSTEP 4: Do 8 × 2 = 16\n\n**Answer: A (16)**",
  "timestamp": "2026-01-27T14:30:00Z"
}
```

---

## Phase 5: Response Parsing

### Extract from Each Response
1. **Selected answer:** A, B, C, or D
2. **Reasoning text:** Everything before the final answer
3. **Parse success:** Boolean

### Parsing Rules
```python
def parse_response(response):
    # Look for answer patterns
    patterns = [
        r"\*\*Answer:\s*([A-D])\*\*",
        r"Answer:\s*([A-D])",
        r"my answer is ([A-D])",
        r"I choose ([A-D])",
        r"([A-D])\s*$"  # Last letter in response
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return {
                "answer": match.group(1).upper(),
                "reasoning": response[:match.start()].strip(),
                "parse_success": True
            }

    return {"answer": None, "reasoning": response, "parse_success": False}
```

### Output
```json
{
  "item_id": "eedi_12345",
  "strategy": "S4",
  "model": "gpt-3.5-turbo",
  "rep": 1,
  "selected_answer": "A",
  "correct_answer": "B",
  "target_distractor": "A",
  "is_correct": false,
  "hit_target": true,
  "reasoning": "Let me follow my procedure step by step...",
  "parse_success": true
}
```

---

## Phase 6: Misconception Coding

### Only Code Errors
- Skip responses where `is_correct == true`
- Focus on: Does the reasoning match the target misconception?

### Coding Scheme

| Code | Definition | Example |
|------|------------|---------|
| FULL_MATCH | Reasoning explicitly shows target misconception | "I'll do 5+3 first because it comes first" |
| PARTIAL_MATCH | Related error, not exact match | "I'm not sure about order, so I'll guess" |
| DIFFERENT_ERROR | Wrong answer, different reasoning | Computational error (5+3=9) |
| UNCLEAR | Cannot determine reasoning | "The answer is A" (no explanation) |

### Coding Process

**Option A: Human Coding**
- Two independent coders
- Code sample of 20 per cell (320 total)
- Compute Cohen's kappa

**Option B: LLM-Assisted Coding**
```python
CODING_PROMPT = """
You are coding LLM responses for misconception alignment.

Target misconception: {misconception_description}
Example of this misconception: {example}

Student response to code:
{response}

Does this response show the target misconception in its reasoning?
- FULL_MATCH: Reasoning explicitly demonstrates the misconception
- PARTIAL_MATCH: Related but not exact
- DIFFERENT_ERROR: Wrong answer but different reasoning
- UNCLEAR: Cannot determine

Respond with just the code and a one-sentence justification.
"""
```

### Output
```json
{
  "item_id": "eedi_12345",
  "strategy": "S4",
  "model": "gpt-3.5-turbo",
  "rep": 1,
  "selected_answer": "A",
  "hit_target": true,
  "misconception_code": "FULL_MATCH",
  "coding_justification": "Response explicitly follows left-to-right procedure, doing 5+3 before multiplication"
}
```

---

## Phase 7: Analysis

### Metrics by Cell (Strategy × Model × Misconception)

```python
def compute_metrics(responses):
    errors = [r for r in responses if not r["is_correct"]]

    return {
        "n_responses": len(responses),
        "n_errors": len(errors),
        "error_rate": len(errors) / len(responses),

        # Among errors only:
        "target_distractor_rate": sum(r["hit_target"] for r in errors) / len(errors),
        "misconception_alignment_rate": sum(r["misconception_code"] == "FULL_MATCH" for r in errors) / len(errors),
        "partial_alignment_rate": sum(r["misconception_code"] in ["FULL_MATCH", "PARTIAL_MATCH"] for r in errors) / len(errors),

        # Key metric:
        "reasoning_authenticity_gap": target_distractor_rate - misconception_alignment_rate
    }
```

### Primary Analyses

**1. Strategy Effect on Misconception Alignment**
```
H1: S4 > S3 > S2 > S1 on misconception_alignment_rate
Test: One-way ANOVA with post-hoc contrasts
```

**2. Strategy × Misconception Type Interaction**
```
H2: S4 works better for procedural; S3 works better for conceptual
Test: 2×4 ANOVA (misconception type × strategy)
```

**3. Does Any Strategy Close the Reasoning Authenticity Gap?**
```
H3: At least one strategy achieves gap < 10 percentage points
Test: Compare gap across strategies, identify minimum
```

### Visualization Plan

1. **Heatmap:** Strategy × Misconception, colored by alignment rate
2. **Bar chart:** Alignment rate by strategy, split by procedural/conceptual
3. **Gap chart:** Target rate vs alignment rate by strategy (showing the gap)

---

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Item selection from Eedi | 2 hours |
| 2 | Misconception mapping | 1 hour |
| 3 | Prompt construction | 2 hours |
| 4 | LLM generation (3,200 calls) | 4-6 hours |
| 5 | Response parsing | 1 hour |
| 6 | Misconception coding | 4-6 hours |
| 7 | Analysis | 2-3 hours |

**Total: ~16-20 hours of work**

---

## File Structure

```
study2-materials/
├── data/
│   ├── eedi_selected_items.json      # Phase 1 output
│   ├── prompt_registry.json          # Phase 3 output
│   ├── raw_responses/                # Phase 4 output
│   │   ├── gpt-3.5-turbo/
│   │   ├── claude-haiku/
│   │   ├── mistral-7b/
│   │   └── llama-3.1-8b/
│   ├── parsed_responses.json         # Phase 5 output
│   └── coded_responses.json          # Phase 6 output
├── analysis/
│   ├── metrics_by_cell.csv
│   ├── figures/
│   └── statistical_tests.md
├── prompts/
│   ├── S1_persona.txt
│   ├── S2_knowledge_state.txt
│   ├── S3_mental_model.txt
│   └── S4_production_rules.txt
└── experimental-flow.md              # This document
```
