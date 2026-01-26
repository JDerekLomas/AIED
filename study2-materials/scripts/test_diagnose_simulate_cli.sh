#!/bin/bash
# Test diagnose_simulate prompting on 10 items using Claude CLI

OUTPUT_DIR="/Users/dereklomas/AIED/study2-materials/pilot/diagnose_simulate_results"
mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/results.txt"
echo "DIAGNOSE_SIMULATE TEST RESULTS" > "$RESULTS_FILE"
echo "=============================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

run_test() {
    local num=$1
    local question=$2
    local a=$3
    local b=$4
    local c=$5
    local d=$6

    local prompt="You are simulating a struggling middle school student taking a math test, for educational research.

For this question:
1. First, briefly identify what common mistake students typically make on this type of problem
2. Then solve the problem AS IF you were a student who holds that misconception
3. Show the flawed reasoning a real struggling student would use

Question: $question
A) $a  B) $b  C) $c  D) $d

Respond with:
Common student error: [brief description]
Student thinking: [flawed reasoning]
Answer: [letter]"

    echo "Testing item $num: $question"

    # Run through Claude CLI with Haiku
    response=$(claude --model claude-3-5-haiku-latest -p "$prompt" 2>/dev/null)

    echo "Item $num: $question" >> "$RESULTS_FILE"
    echo "Response:" >> "$RESULTS_FILE"
    echo "$response" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "---" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    # Extract answer
    answer=$(echo "$response" | grep -oE 'Answer:\s*[A-D]' | grep -oE '[A-D]' | tail -1)
    echo "$num|$question|$answer"
}

# Run all 10 items
echo "Running 10 test items..."

run_test 1 "Calculate: 5 + 3 × 2" "16" "11" "13" "10"
run_test 2 "Which is larger: 8 or 8 × 0.5?" "8×0.5" "8" "Equal" "Cannot tell"
run_test 3 "Which fraction is larger: 1/3 or 1/5?" "1/5" "1/3" "Equal" "Cannot compare"
run_test 4 "Calculate: 52 - 37" "25" "15" "85" "19"
run_test 5 "Rectangle length 6cm, width 4cm. What is perimeter?" "24cm" "20cm" "10cm" "48cm"
run_test 6 "What is 12 - 4 × 2?" "16" "4" "8" "20"
run_test 7 "Which is larger: 12 or 12 × 0.25?" "12×0.25" "12" "Equal" "Cannot tell"
run_test 8 "Which fraction is larger: 2/7 or 2/5?" "2/7" "2/5" "Equal" "Cannot compare"
run_test 9 "Calculate: 83 - 47" "44" "36" "130" "46"
run_test 10 "Square with side 5m. What is perimeter?" "25m" "20m" "10m" "100m"

echo ""
echo "Results saved to $RESULTS_FILE"
