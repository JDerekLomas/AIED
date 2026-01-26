#!/usr/bin/env python3
"""
Test "diagnose-then-simulate" prompting approach.

Compares:
- persona: Standard "act like a student" prompt
- diagnose_simulate: First identify common errors, then exhibit one
"""

import json
from pathlib import Path

# Load a few test items
items_path = Path("/Users/dereklomas/AIED/study2-materials/data/probe_items.json")
with open(items_path) as f:
    data = json.load(f)

# Extract 5 diverse items
test_items = []
for misc in data['misconceptions'][:5]:
    test_items.append({
        'item': misc['items'][0],
        'misconception': misc['name'],
        'category': misc['category']
    })

# Define prompts
PERSONA_PROMPT = """You are a middle school student working on a math problem. Students at your level sometimes make mistakes - that's okay and normal.

Solve this problem the way a real student would. Show your thinking as you work through it, even if you're not sure. It's fine to make errors.

Question: {question}

{options}

Your thinking:"""

DIAGNOSE_SIMULATE_PROMPT = """You are simulating a struggling middle school student for educational research.

Before answering, briefly consider:
1. What common mistakes do students typically make on this type of problem?
2. What misconception might lead a student to choose a wrong answer?

Then solve the problem AS IF you were a student who holds one of those common misconceptions. Show your (flawed) student reasoning and select your answer.

Question: {question}

{options}

Student thinking:"""

def format_options(opts: dict) -> str:
    return "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()))

# Print test cases for manual inspection / API testing
print("="*70)
print("DIAGNOSE-THEN-SIMULATE PROMPT TEST CASES")
print("="*70)

for i, item_data in enumerate(test_items):
    item = item_data['item']
    print(f"\n{'='*70}")
    print(f"TEST ITEM {i+1}: {item['item_id']}")
    print(f"Target misconception: {item_data['misconception']}")
    print(f"Correct answer: {item['correct']}")
    print(f"Target distractor: {item['target_distractor']}")
    print("="*70)

    opts_str = format_options(item['options'])

    print("\n--- PERSONA PROMPT ---")
    print(PERSONA_PROMPT.format(question=item['question'], options=opts_str))

    print("\n--- DIAGNOSE-SIMULATE PROMPT ---")
    print(DIAGNOSE_SIMULATE_PROMPT.format(question=item['question'], options=opts_str))

# Save prompts for API testing
output = {
    'persona_template': PERSONA_PROMPT,
    'diagnose_simulate_template': DIAGNOSE_SIMULATE_PROMPT,
    'test_items': test_items
}

output_path = Path("/Users/dereklomas/AIED/study2-materials/pilot/prompt_comparison_items.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nTest items saved to {output_path}")
print("\nTo test: Run both prompts through GPT-3.5 or Claude and compare target distractor rates")
