# Manual Error Analysis: SmartPaper Bias Patterns

**Date:** 2026-02-02
**Condition analyzed:** SmartPaper anchors_t2.0 (ρ=0.877, 20 probe items) + direct prediction full-scale (ρ=0.547, 140 items)

## Summary

LLM predictions are systematically miscalibrated for Indian state-school students. The model universally overestimates item pass rates, never underestimating. The bias is subject-dependent and inversely related to rank correlation quality.

---

## 1. Probe Set Analysis (anchors_t2.0, n=20, ρ=0.877)

### Per-Item Residuals (ranked by |residual|)

| Item | Subject | Actual | Predicted | Residual |
|---|---|---|---|---|
| G7 Maths q3 (triangle area) | Math | 0.290 | 0.680 | +0.390 |
| G6 English q8 (rearrange words) | English | 0.380 | 0.740 | +0.360 |
| G7 English q15 (write paragraph) | English | 0.103 | 0.430 | +0.327 |
| G6 Maths q7 (temperature problem) | Math | 0.296 | 0.620 | +0.324 |
| G8 SocSci q2 (Harappan planning) | SocSci | 0.261 | 0.580 | +0.319 |
| G7 SocSci q8 (occupations) | SocSci | 0.269 | 0.570 | +0.301 |
| G8 Maths q1 (solve y+9=4) | Math | 0.401 | 0.700 | +0.299 |
| G8 English q6 (define fable) | English | 0.176 | 0.470 | +0.294 |
| G6 English q10 (some/any) | English | 0.513 | 0.780 | +0.267 |
| G8 SocSci q7 (Battle of Plassey) | SocSci | 0.146 | 0.410 | +0.264 |
| G6 SocSci q1 (mountain ranges) | SocSci | 0.354 | 0.610 | +0.256 |
| G7 English q4 (king's questions) | English | 0.198 | 0.440 | +0.242 |
| G7 English q6 (poet's writing style) | English | 0.037 | 0.250 | +0.213 |
| G8 Science q7 (composting) | Science | 0.191 | 0.400 | +0.209 |
| G6 SocSci q7 (Rajasthan resources) | SocSci | 0.186 | 0.380 | +0.194 |
| G6 Maths q3 (ascending order) | Math | 0.589 | 0.780 | +0.191 |
| G7 English q12 (active→passive voice) | English | 0.126 | 0.280 | +0.154 |
| G7 English q11 (adjective phrase) | English | 0.251 | 0.400 | +0.149 |
| G6 English q6 (favorite poem line) | English | 0.258 | 0.370 | +0.112 |
| G6 English q2 (who lived in book?) | English | 0.634 | 0.720 | +0.086 |

**All 20 residuals are positive.** Mean bias: +0.248 (25 percentage points).

### Qualitative Analysis of 5 Worst Items

**#1. G7 Maths q3 — "Find the area of a triangle" (Residual: +0.390)**
- Reps: 75%, 60%, 55%, 75%, 75%
- LLM reasoning: Views this as "direct application of a basic geometric formula," compares to a 66%-pass-rate anchor. Treats formula application as trivially easy.
- Reality: Only 29% succeed. Formula recall and application is weaker than factual recall in this population. "Basic by curriculum" ≠ "basic for these students."

**#2. G6 English q8 — "Rearrange: book/reading/a/is/he" (Residual: +0.360)**
- Reps: 75%, 75%, 70%, 75%, 75% (tightest clustering in dataset)
- LLM reasoning: All 5 reps confidently declare this "significantly easier than all anchor items" — basic SVO structure that students "would have practiced extensively."
- Reality: Only 38% succeed. The LLM cannot model English proficiency of Hindi-medium state school students where English is a 2nd/3rd language. Most confident prediction = second worst error.

**#3. G7 English q15 — "Write a paragraph about 'My Favourite Hobby'" (Residual: +0.327)**
- Reps: 45%, 40%, 40%, 55%, 35%
- LLM reasoning: Compares to a "My Favorite Season" anchor (30%) and estimates slightly easier.
- Reality: 10.3% pass rate. Productive writing is catastrophically hard. Rubric requires three separate content elements (what, when, why) — most students can't produce organized multi-sentence output.

**#4. G6 Maths q7 — Temperature word problem (Residual: +0.324)**
- Reps: 40%, 75%, 75%, 75%, 45% (highest inter-rep variance, SD ~16pp)
- LLM reasoning: Split — 2 reps correctly identify difficulty (~40%), 3 reps call it "basic arithmetic" (~75%). The 75% reps evaluate each arithmetic step in isolation.
- Reality: 29.6%. Multi-step word problems (compute elapsed time → multiply → add) are hard even when each operation is simple. The confident reps fail to model composite task burden.

**#5. G8 SocSci q2 — "Describe Harappan urban planning" (Residual: +0.319)**
- Reps: 60%, 60%, 55%, 60%, 55%
- LLM reasoning: Compares to "Fundamental Rights" anchor (66%) — both involve "recall of a specific factual point."
- Reality: 26.1%. The model fails to distinguish contemporary civics (encountered in daily life) from ancient history requiring recall of archaeological details.

---

## 2. Full-Scale Analysis (direct prediction, n=140, ρ=0.547)

### Bias by Subject

| Subject | n | Mean Actual | Mean Pred | Bias | MAE | ρ |
|---|---|---|---|---|---|---|
| Science | 32 | 0.246 | 0.738 | **+0.492** | 0.492 | **0.734** |
| Social Science | 34 | 0.246 | 0.746 | **+0.500** | 0.500 | **0.702** |
| Mathematics | 33 | 0.315 | 0.668 | +0.354 | 0.354 | 0.600 |
| English | 41 | 0.353 | 0.751 | +0.398 | 0.416 | 0.432 |

**Key finding: Bias and rank correlation are inversely related.** Science/SocSci have the worst calibration (+50pp) but the best rank ordering. English has the smallest bias but worst ρ.

Interpretation: Science/SocSci items vary more in cognitive complexity (which the LLM can rank) but the LLM has zero calibration for how hard these topics are for this population. English items are more heterogeneous in what they test (comprehension vs. grammar vs. writing production), making rank-ordering harder.

### Bias by Grade

| Grade | n | Mean Actual | Mean Pred | Bias | MAE |
|---|---|---|---|---|---|
| 6 | 49 | 0.317 | 0.754 | +0.437 | 0.440 |
| 7 | 47 | 0.259 | 0.702 | +0.443 | 0.447 |
| 8 | 44 | 0.304 | 0.725 | +0.421 | 0.429 |

**No meaningful grade effect.** Bias is flat across grades 6-8.

### Subject × Grade Interaction

| Subject | G6 | G7 | G8 |
|---|---|---|---|
| English | +0.418 | +0.398 | +0.375 |
| Mathematics | +0.353 | +0.375 | +0.331 |
| Science | +0.473 | +0.500 | +0.501 |
| Social Science | +0.515 | +0.511 | +0.472 |

Science and Social Science bias is remarkably stable across grades (~0.50). Math and English show a slight decrease at Grade 8 but the effect is small.

### Directionality

- **136 of 140 items** are overestimated (pred > actual)
- Only **4 items** are underestimated (all easy English items with actual > 0.75):
  - G8 English q2: actual 0.825, pred 0.720 (−0.105)
  - G7 English q11: actual 0.251, pred 0.150 (−0.101)
  - G8 English q1: actual 0.784, pred 0.697 (−0.087)
  - G6 English q5: actual 0.768, pred 0.700 (−0.068)

### Prediction Ceiling

The LLM's predictions cluster around 0.70-0.75 regardless of subject. Actual performance ranges 0.04-0.83. The model is essentially saying "most items should be moderately easy" and then differentiating within that narrow band.

### 10 Worst Absolute Errors

| Item | Subject | Actual | Pred | Residual |
|---|---|---|---|---|
| G6 SocSci q11 | Social Science | 0.142 | 0.850 | +0.708 |
| G8 SocSci q10 | Social Science | 0.164 | 0.850 | +0.686 |
| G7 English q16 | English | 0.172 | 0.850 | +0.678 |
| G6 English q16 | English | 0.302 | 0.950 | +0.648 |
| G7 English q15 | English | 0.103 | 0.750 | +0.647 |
| G7 Science q9 | Science | 0.110 | 0.750 | +0.640 |
| G8 English q5 | English | 0.046 | 0.683 | +0.637 |
| G8 English q4 | English | 0.215 | 0.850 | +0.635 |
| G7 SocSci q7 | Social Science | 0.129 | 0.750 | +0.621 |
| G8 English q9 | English | 0.256 | 0.850 | +0.594 |

All worst items: actual pass rate 5-17%, LLM predicts 68-95%.

---

## 3. Cross-Cutting Failure Patterns

1. **Population model deficit.** The single biggest driver. The LLM's implicit "default student" is far more capable than the actual population (Hindi-medium, low-resource government schools, Rajasthan). The +44pp mean bias quantifies this gap.

2. **Complexity vs. calibration dissociation.** The LLM encodes two independent signals: (a) relative cognitive complexity of items (captured by ρ), and (b) absolute population performance (captured by bias). These are separable — subjects where rank ordering is best have the worst calibration.

3. **Production vs. recognition gap.** Writing/production tasks are the most overpredicted. The LLM underweights the production burden relative to recognition/recall.

4. **Math procedural blindness.** The LLM treats formula application and multi-step arithmetic as "basic" without modeling that procedural fluency is often the weakest skill in low-performing populations.

5. **Anchoring to curriculum, not to students.** The LLM reasons about what a student "should" know based on curriculum placement, not what students empirically do know. "Grade 7 curriculum" → "should be accessible" is the dominant reasoning pattern, even when reality is 10% pass rates.

6. **High confidence ≠ accuracy.** Tight inter-rep agreement (low variance across 5 reps) does not predict accuracy. Some of the tightest-clustered predictions are the worst errors.

---

## 4. Implications for Paper

### What this adds to the dissociation argument

The bias analysis strengthens the claim that LLMs capture **text-legible complexity structure** but not **population-specific calibration**:

- Rank ordering (ρ) depends on complexity legibility from item text → LLM succeeds
- Absolute calibration depends on population knowledge → LLM fails systematically
- These are independent: best ρ co-occurs with worst bias (Science, SocSci)

### Novel finding: "Default Student" bias

The uniform +44pp overestimate across 140 items, 4 subjects, and 3 grade levels quantifies the gap between the LLM's implicit student model and the actual population. This is directly relevant to AI-in-education equity: deploying LLM-based difficulty estimation without population-specific calibration would systematically underestimate the challenge these students face.

### Potential paper framing

> "LLMs can rank-order complexity-driven difficulty (ρ=0.55 at scale) but are systematically miscalibrated about absolute performance in low-resource populations. The uniform +44pp bias suggests LLMs encode a 'default student' whose capabilities far exceed those of the target population."

---

## 5. Possible Optimizations (not yet implemented)

If absolute calibration is ever needed:

1. **Population base rate in prompt** — "Mean pass rate is 29%, range 4-83%"
2. **Better low-end anchors** — current anchors don't cover the 5-15% range well
3. **Post-hoc linear recalibration** — `pred_adj = a*pred + b` on 5 anchor items could halve MAE while preserving ρ
4. **Distribution constraint** — "75% of items have pass rates below 40%"
5. **Subject-specific anchors** — especially for math procedural items

These were not pursued because rank ordering was the primary outcome of interest.
