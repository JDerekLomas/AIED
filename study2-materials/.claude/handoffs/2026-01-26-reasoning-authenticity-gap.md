# Study 2: Reasoning Authenticity Gap Discovery - Session Handoff

## Current State

**Major finding discovered and documented.** The Reasoning Authenticity Gap fundamentally changes the paper's narrative.

### The Key Finding

| Metric | Value |
|--------|-------|
| Target distractor rate | 50.0% (104/208 errors) |
| Misconception alignment rate | 8.7% (18/208 errors) |
| **Gap** | **41.3 percentage points** |

LLMs select the "right wrong answer" at rates significantly above chance, but their reasoning rarely reflects the actual misconception.

### Misconception Coding Results

Auto-coding breakdown:
- DIFFERENT_ERROR: 46.2% - Wrong answer with coherent but non-target reasoning
- UNCLEAR: 45.2% - Minimal reasoning (especially answer_only condition)
- PARTIAL_MATCH: 8.2% - Single misconception indicator detected
- FULL_MATCH: 0.5% - Multiple indicators detected

By Model (Misconception Alignment):
- Mistral-7B: 16.7% (highest alignment)
- Llama-3.1-8B: 10.0%
- GPT-3.5-turbo: 4.0%
- Llama-3.2-3B: 3.2%
- Claude models: 0.0%

### Critical Insight

Chain-of-thought prompting increased target distractor selection (78.4% vs 52.6%) but did NOT increase reasoning alignment. This suggests CoT helps models select common wrong answers without producing authentic misconception-driven reasoning.

## Files Created This Session

- `scripts/code_misconceptions.py` - Misconception coding script
- `pilot/full_collection/misconception_coding/coding_report.txt` - Full report
- `pilot/full_collection/misconception_coding/coded_errors.json` - Coded data

## Google Doc Updates Made

Document: https://docs.google.com/document/d/1hX1uoC8i2kU9QHCIfIp7nSEv9uQs-cT7WfdQMweaNws/edit

1. Added new Section 4.7: "Misconception Coding Results: The Reasoning Authenticity Gap"
2. Added new Section 5.1: "The Reasoning Authenticity Gap: Primary Finding"
3. Updated Abstract with key finding
4. Updated Implications section (confirmed the low-alignment scenario)
5. Updated exploratory hypothesis (marked as CONFIRMED)
6. Added Future Work items 6 & 7 on closing the gap and assessment design implications
7. Marked misconception coding as complete in Next Steps

## Paper Narrative Shift

**Before:** "LLMs show moderate alignment with target distractors (59.8%), supporting use as synthetic students"

**After:** "LLMs select target distractors (50%) but reasoning matches only 8.7% of the time, revealing surface mimicry without authentic error cognition"

## Implications for Future Work

1. **Misconception-Grounded Training**: Fine-tune on datasets with explicit misconception labels
2. **Reasoning Verification**: Methods to verify CoT authentically reflects misconceptions
3. **Hybrid Approaches**: LLM generation + misconception library validation
4. **Alternative Architectures**: Neuro-symbolic systems for stable misconceptions
5. **Assessment Design**: Human validation required even with accurate difficulty estimates

## Deadlines

- Abstract deadline: January 26, 2026 (TODAY)
- Full paper deadline: February 2, 2026

## Next Steps

- [ ] Final review of Google Doc before abstract submission
- [ ] Consider adding a figure visualizing the Reasoning Authenticity Gap
- [ ] Expand misconception coding methodology for full paper
- [ ] Inter-rater reliability analysis (for full paper)
