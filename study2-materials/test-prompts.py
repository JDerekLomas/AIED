#!/usr/bin/env python3
"""
Test misconception simulation prompts via Anthropic API
"""

import anthropic
import json

client = anthropic.Anthropic()

ITEMS = [
    {
        "id": 1,
        "question": "Calculate: 5 + 3 × 2",
        "options": "A)16 B)11 C)13 D)10",
        "correct": "B",
        "target": "A",
        "topic": "Order of operations"
    },
    {
        "id": 2,
        "question": "Which is larger: 8 or 8 × 0.5?",
        "options": "A)8×0.5 B)8 C)Equal D)Cannot tell",
        "correct": "B",
        "target": "A",
        "topic": "Decimal multiplication"
    },
    {
        "id": 3,
        "question": "Which fraction is larger: 1/3 or 1/5?",
        "options": "A)1/5 B)1/3 C)Equal D)Cannot compare",
        "correct": "B",
        "target": "A",
        "topic": "Fraction comparison"
    },
    {
        "id": 4,
        "question": "Calculate: 52 - 37",
        "options": "A)25 B)15 C)85 D)19",
        "correct": "B",
        "target": "A",
        "topic": "Subtraction with regrouping"
    },
    {
        "id": 5,
        "question": "Rectangle has length 6cm, width 4cm. What is its perimeter?",
        "options": "A)24cm B)20cm C)10cm D)48cm",
        "correct": "B",
        "target": "A",
        "topic": "Perimeter vs area"
    }
]

PROMPT_A = """You are a middle school student. Solve this problem the way a real student would. It's fine to make errors.

{question}
{options}

Your thinking:"""

PROMPT_B = """You are simulating a struggling student. Before answering:
1) What common mistakes do students make on this type?
2) What misconception leads to wrong answers?

Then solve AS IF you held one of those misconceptions.

{question}
{options}

Student thinking:"""

def extract_answer(response_text):
    """Extract the letter answer from response"""
    # Look for explicit answer patterns
    text = response_text.upper()
    for pattern in ["ANSWER: ", "ANSWER IS ", "I CHOOSE ", "MY ANSWER: ", "I'LL GO WITH "]:
        if pattern in text:
            idx = text.find(pattern) + len(pattern)
            if idx < len(text) and text[idx] in "ABCD":
                return text[idx]

    # Look for last mentioned letter option
    last_answer = None
    for char in text:
        if char in "ABCD":
            last_answer = char
    return last_answer

def test_prompt(item, prompt_template):
    """Run a single test"""
    prompt = prompt_template.format(
        question=item["question"],
        options=item["options"]
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text
    answer = extract_answer(response_text)
    return answer, response_text

def main():
    results = []

    print("Testing misconception simulation prompts...\n")
    print("=" * 70)

    for item in ITEMS:
        print(f"\nItem {item['id']}: {item['topic']}")
        print(f"Q: {item['question']}")
        print(f"Target distractor: {item['target']}")
        print("-" * 50)

        # Test Prompt A
        ans_a, text_a = test_prompt(item, PROMPT_A)
        hit_a = "✓" if ans_a == item["target"] else "✗"
        print(f"Prompt A answer: {ans_a} {hit_a}")

        # Test Prompt B
        ans_b, text_b = test_prompt(item, PROMPT_B)
        hit_b = "✓" if ans_b == item["target"] else "✗"
        print(f"Prompt B answer: {ans_b} {hit_b}")

        results.append({
            "item": item["id"],
            "topic": item["topic"],
            "target": item["target"],
            "prompt_a": ans_a,
            "prompt_b": ans_b,
            "a_hit": ans_a == item["target"],
            "b_hit": ans_b == item["target"]
        })

    # Summary
    print("\n" + "=" * 70)
    print("\nRESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Item':<6} {'Topic':<25} {'Prompt A':<10} {'Prompt B':<10} {'Target':<8}")
    print("-" * 70)

    a_hits = 0
    b_hits = 0

    for r in results:
        a_mark = "✓" if r["a_hit"] else "✗"
        b_mark = "✓" if r["b_hit"] else "✗"
        print(f"{r['item']:<6} {r['topic']:<25} {r['prompt_a']} {a_mark:<8} {r['prompt_b']} {b_mark:<8} {r['target']}")
        a_hits += r["a_hit"]
        b_hits += r["b_hit"]

    print("-" * 70)
    print(f"\nPrompt A hit rate: {a_hits}/5 ({a_hits*20}%)")
    print(f"Prompt B hit rate: {b_hits}/5 ({b_hits*20}%)")

    # Save detailed results
    with open("/Users/dereklomas/AIED/study2-materials/test-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to test-results.json")

if __name__ == "__main__":
    main()
