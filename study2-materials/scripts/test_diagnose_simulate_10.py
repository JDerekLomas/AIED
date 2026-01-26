#!/usr/bin/env python3
"""
Test diagnose_simulate prompting condition on 10 misconception items using Claude Haiku.
"""

import anthropic
import re

client = anthropic.Anthropic()

PROMPT_TEMPLATE = """You are simulating a struggling middle school student taking a math test, for educational research.

For this question:
1. First, briefly identify what common mistake students typically make on this type of problem
2. Then solve the problem AS IF you were a student who holds that misconception
3. Show the flawed reasoning a real struggling student would use

Question: {question}
A) {A}  B) {B}  C) {C}  D) {D}

Respond with:
Common student error: [brief description]
Student thinking: [flawed reasoning]
Answer: [letter]"""

# Test items
ITEMS = [
    {"question": "Calculate: 5 + 3 × 2", "A": "16", "B": "11", "C": "13", "D": "10", "correct": "B", "target": "A"},
    {"question": "Which is larger: 8 or 8 × 0.5?", "A": "8×0.5", "B": "8", "C": "Equal", "D": "Cannot tell", "correct": "B", "target": "A"},
    {"question": "Which fraction is larger: 1/3 or 1/5?", "A": "1/5", "B": "1/3", "C": "Equal", "D": "Cannot compare", "correct": "B", "target": "A"},
    {"question": "Calculate: 52 - 37", "A": "25", "B": "15", "C": "85", "D": "19", "correct": "B", "target": "A"},
    {"question": "Rectangle length 6cm, width 4cm. What is perimeter?", "A": "24cm", "B": "20cm", "C": "10cm", "D": "48cm", "correct": "B", "target": "A"},
    {"question": "What is 12 - 4 × 2?", "A": "16", "B": "4", "C": "8", "D": "20", "correct": "B", "target": "A"},
    {"question": "Which is larger: 12 or 12 × 0.25?", "A": "12×0.25", "B": "12", "C": "Equal", "D": "Cannot tell", "correct": "B", "target": "A"},
    {"question": "Which fraction is larger: 2/7 or 2/5?", "A": "2/7", "B": "2/5", "C": "Equal", "D": "Cannot compare", "correct": "B", "target": "A"},
    {"question": "Calculate: 83 - 47", "A": "44", "B": "36", "C": "130", "D": "46", "correct": "B", "target": "A"},
    {"question": "Square with side 5m. What is perimeter?", "A": "25m", "B": "20m", "C": "10m", "D": "100m", "correct": "B", "target": "A"},
]

def extract_answer(response_text):
    """Extract the answer letter from response."""
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-D])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: look for standalone letter at end
    match = re.search(r'\b([A-D])\s*$', response_text)
    if match:
        return match.group(1).upper()
    return "?"

def run_test():
    results = []

    for i, item in enumerate(ITEMS, 1):
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            A=item["A"],
            B=item["B"],
            C=item["C"],
            D=item["D"]
        )

        print(f"\n{'='*60}")
        print(f"Item {i}: {item['question']}")
        print(f"Options: A){item['A']} B){item['B']} C){item['C']} D){item['D']}")
        print(f"Correct: {item['correct']}, Target: {item['target']}")
        print("-"*60)

        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text
        answer = extract_answer(response_text)
        hit_target = answer == item["target"]

        print(f"Response:\n{response_text}")
        print(f"\nAnswer: {answer} | Hit target: {'YES' if hit_target else 'NO'}")

        results.append({
            "item": i,
            "question": item["question"],
            "answer": answer,
            "target": item["target"],
            "correct": item["correct"],
            "hit_target": hit_target
        })

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE - diagnose_simulate condition")
    print("="*60)
    print(f"{'Item':<5} {'Question':<40} {'Answer':<8} {'Hit?':<5}")
    print("-"*60)

    hits = 0
    for r in results:
        hit_str = "YES" if r["hit_target"] else "NO"
        if r["hit_target"]:
            hits += 1
        q_short = r["question"][:38] + ".." if len(r["question"]) > 40 else r["question"]
        print(f"{r['item']:<5} {q_short:<40} {r['answer']:<8} {hit_str:<5}")

    print("-"*60)
    print(f"HIT RATE: {hits}/{len(results)} = {hits/len(results)*100:.1f}%")
    print(f"(Target is A for all items)")

    return results

if __name__ == "__main__":
    run_test()
