# DBE-KT22 Out-of-Sample Validation

## Dataset

**DBE-KT22** is a knowledge tracing dataset from an introductory database systems course at the Australian National University (Abdi et al., 2023). It contains:

- 212 multiple-choice questions on database concepts (SQL, transactions, normalization, ER diagrams)
- 168 text-only items used in analysis (44 excluded for containing images)
- 161,953 student response transactions from 1,264 students
- 3-4 answer choices per question
- Author-assigned difficulty levels (1/2/3)

Empirical p-correct: mean = 0.764, range [0.106, 0.996].

## IRT Analysis

### Author Difficulty vs Empirical Difficulty

| Author Level | n items | Mean p-correct |
|:---:|:---:|:---:|
| 1 | 82 | 0.805 |
| 2 | 71 | 0.786 |
| 3 | 15 | 0.713 |

**Author-assigned difficulty does not correlate with empirical difficulty:**

- p-correct vs author difficulty: ρ = 0.061, p = 0.431 (ns)
- 2PL b vs author difficulty: ρ similar, ns

Levels 1 and 2 are essentially indistinguishable. Only level 3 trends slightly harder, but with only 15 items this is unreliable. Empirical p-correct from student responses is the only valid ground truth for this dataset.

### Cross-Dataset IRT Comparison (R mirt package)

1PL (Rasch) and 2PL models were fit using R's `mirt` package via MML-EM, with 500 student random subsamples per dataset.

| Dataset | Items | 1PL b vs p-correct | 2PL b vs p-correct | 2PL a mean | 2PL a median | 2PL a sd |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| DBE-KT22 | 212 | 1.000 | 0.800 | 0.80 | 0.71 | 0.58 |
| SmartPaper | 135 | 0.996 | 0.622 | 1.11 | 0.87 | 1.28 |
| Eedi | — | — | — | — | — | — |

**Note on Eedi IRT and ground truth correction:** The Eedi NeurIPS competition shuffled answer positions per student. The `CorrectAnswer` column changes per student to track the shuffle, so `IsCorrect = (AnswerValue == CorrectAnswer)` is internally valid. However, the NeurIPS-derived p-correct (mean 0.618) does not match the aggregate `pct_{CorrectAnswer}` from `eedi_with_student_data.csv` (mean 0.246) because the aggregate file's A/B/C/D mapping uses Kaggle ordering, not the NeurIPS shuffled positions.

**The NeurIPS `IsCorrect` is the correct ground truth** for item difficulty. The aggregate `pct_A/B/C/D` columns are unreliable for computing p-correct. All Eedi experiments were recomputed against NeurIPS-derived p-correct — the results are unchanged: ρ ≈ 0 for the full item set regardless of ground truth source (see Eedi Ground Truth Correction section below).

IRT was not fit for Eedi because the sparse response matrix (500 students across 1,532 items, 3% density) yields unstable estimates.

**Key findings:**

1. **1PL difficulty ≈ p-correct** across both datasets with student-level data (ρ > 0.99). Under Rasch, difficulty is a monotonic transform of p-correct — using classical p-correct as ground truth is equivalent to 1PL IRT.

2. **2PL diverges from p-correct** when item discrimination varies. SmartPaper shows the most divergence (ρ = 0.622), meaning its items differ substantially in discriminating ability.

3. **For the paper's purpose**, classical p-correct is the appropriate benchmark. The LLM prompt asks "what percentage would answer correctly?" — this maps directly to p-correct / 1PL difficulty, not to 2PL difficulty which incorporates discrimination (a separate psychometric property the LLM is not asked to estimate).

## LLM Difficulty Estimation

### Method 1: Direct Estimation (Baseline)

- **Model**: Gemini 2.5 Flash
- **Prompt**: Direct estimation — "What percentage of undergraduate students would answer this correctly?"
- **Design**: Single pass, temperature 0.0, 168 items
- **Ground truth**: Empirical p-correct from 128,845 student responses

| Metric | Value |
|:---|:---|
| Spearman ρ | **0.342** |
| p-value | < 0.001 |
| 95% CI (bootstrap) | [0.194, 0.478] |
| n items | 168 |
| n student responses | 128,845 |

ρ vs author-assigned difficulty: 0.338 (but author difficulty is unreliable; see above).

### Method 2: Optimized Contrastive Pipeline

- **Model**: Gemini 3 Flash Preview
- **Prompt**: Contrastive — predict per-option percentages at 4 ability levels (struggling/average/good/advanced)
- **Design**: 3 replications averaged, temperature 1.5, 168 items
- **Ground truth**: Same empirical p-correct

| Metric | Value |
|:---|:---|
| Averaged Spearman ρ | **0.440** |
| p-value | 2.4e-9 |
| 95% CI (bootstrap) | [0.303, 0.561] |
| Per-rep ρ | 0.409, 0.488, 0.421 |
| ρ vs author difficulty | 0.280 |
| n items | 168 |
| n student responses | 128,845 |

The optimized pipeline improves over direct estimation by ~0.10 points. Every individual rep (0.409–0.488) exceeds the direct baseline (0.342).

### Method Comparison

| Method | Model | Temp | Reps | ρ | 95% CI |
|:---|:---|:---:|:---:|:---:|:---|
| Direct | Gemini 2.5 Flash | 0.0 | 1 | 0.342 | [0.194, 0.478] |
| **Contrastive** | **Gemini 3 Flash** | **1.5** | **3** | **0.440** | **[0.303, 0.561]** |

## Cross-Dataset Comparison

| Dataset | Domain | n | Direct ρ | Optimized ρ | Type |
|:---|:---|:---:|:---:|:---:|:---|
| Eedi MCQ | K-12 Math | 1,869 | ~0 | ~0 | Selectivity-driven (archived) |
| **DBE-KT22 MCQ** | **DB Systems** | **168** | **0.342** | **0.440** | **Complexity-driven** |
| SmartPaper | K-12 Math (India) | 140 | 0.547 | TBD | Complexity-driven |

## Interpretation

DBE-KT22 falls between the two primary datasets, consistent with the paper's complexity-selectivity framework:

1. **Significantly better than Eedi** (ρ = 0.440 vs 0.008) — DB systems questions derive difficulty from cognitive complexity (definitions vs. multi-step transaction reasoning), making difficulty legible from item text.

2. **Lower than SmartPaper** (ρ = 0.440 vs 0.547) — likely because DBE-KT22 is university-level with a narrower ability range and more domain-specific knowledge requirements. Some difficulty may come from prerequisite knowledge gaps rather than item complexity alone.

3. **Optimized pipeline helps**: The contrastive prompt with 3-rep averaging and higher temperature consistently outperforms single-pass direct estimation (+0.10), confirming that the pipeline improvements generalize beyond the Eedi probe items.

4. **Supports the central claim**: LLM difficulty estimation works for complexity-driven items across domains, not just mathematics. The effect generalizes from math to computer science.

## Eedi Ground Truth Correction

### Discovery

The Eedi NeurIPS competition shuffled answer positions per student. The `CorrectAnswer` column in the NeurIPS data varies per student (tracking the shuffle), so `IsCorrect = (AnswerValue == CorrectAnswer)` is valid. However, the aggregate `pct_A/B/C/D` columns in `eedi_with_student_data.csv` use Kaggle-ordering positions, not the NeurIPS shuffled positions. This means `pct_{CorrectAnswer}` from the aggregate file does NOT reliably give true p-correct.

NeurIPS-derived p-correct: mean = 0.618 (2.4M responses, 117,584 students).
Aggregate-derived p-correct: mean = 0.246 (incorrect — scrambled by position shuffle).

### Recomputation Against NeurIPS p-correct

All Eedi LLM difficulty estimation experiments were recomputed using the correct NeurIPS IsCorrect-derived p-correct:

| Experiment | n items | Old ρ | Corrected ρ | Notes |
|:---|:---:|:---:|:---:|:---|
| Full item set | 1,869 | 0.011 | 0.011 | Unchanged — genuinely null |
| Text-only items | 921 | — | 0.007 | Images don't explain the null |
| RSM probe (curated) | 20 | negative | 0.25–0.37 | Within bootstrap noise (see below) |
| B2PL 105 items | 105 | — | 0.182 (p=0.063) | Borderline, not significant |

### Bootstrap Analysis of n=20 Probe Items

The curated 20-item probe subset showed ρ = 0.25–0.37 with corrected ground truth (previously negative). However, bootstrap null distribution analysis shows P(ρ ≥ 0.25) ≈ 15% for random 20-item samples from the full Eedi set. This is within sampling noise — the probe result cannot be distinguished from chance at n = 20.

### Conclusion

The Eedi null result (ρ ≈ 0 for the full item set) is genuine and robust to ground truth correction. The data issue affected only small-n subsets where sampling noise dominates. All primary findings in the paper are unchanged.
