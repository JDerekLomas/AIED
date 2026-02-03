# Experimental Design: DSPy Optimization for Human-Like Student Errors

*Draft: 2026-01-26*

## Research Question

Can DSPy prompt optimization, with an inverted objective targeting error fidelity rather than correctness, produce LLM-simulated students whose responses match real student error patterns?

---

## Core Hypothesis

DSPy optimizers (MIPRO, BootstrapFewShot) can discover prompts that overcome the "competence paradox" by explicitly optimizing for metrics that reward realistic errors rather than correct answers.

---

## Dataset: Eedi Misconception Data

The [Eedi dataset](https://eedi.com/projects/neurips-education-challenge) is ideal because:
- Multiple-choice math questions with **misconception-tagged distractors**
- Each wrong answer maps to a specific misconception (e.g., "adds numerators and denominators")
- Real student response distributions across all options
- ~15M responses from 100K+ students

This lets us measure whether simulated students select the *right wrong answer*.

**Splits**:
- 60% optimization (DSPy training)
- 20% validation (hyperparameter tuning)
- 20% held-out test

---

## DSPy Implementation

### Signature

```python
import dspy

class SimulatedStudent(dspy.Signature):
    """Simulate a student answering a multiple-choice math question."""

    question: str = dspy.InputField(desc="The math question")
    options: list[str] = dspy.InputField(desc="Answer choices A-D")
    student_ability: str = dspy.InputField(desc="struggling/average/advanced")
    target_misconceptions: list[str] = dspy.InputField(
        desc="Misconceptions this student has, if any"
    )

    selected_option: str = dspy.OutputField(desc="The letter chosen: A, B, C, or D")
    reasoning: str = dspy.OutputField(desc="Student's thought process")
```

### Module

```python
class ErrorProneSudent(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(SimulatedStudent)

    def forward(self, question, options, student_ability, target_misconceptions):
        return self.respond(
            question=question,
            options=options,
            student_ability=student_ability,
            target_misconceptions=target_misconceptions
        )
```

### Custom Metrics

**Metric 1: Response Distribution Correlation**

```python
def response_distribution_metric(example, pred, trace=None):
    """
    Reward predictions that match real student response distributions.

    example.real_distribution: [0.15, 0.45, 0.30, 0.10] (% selecting A,B,C,D)
    """
    # Run N=50 simulations with temperature to get distribution
    simulated_counts = simulate_n_responses(example, n=50)
    simulated_dist = simulated_counts / simulated_counts.sum()

    # Pearson correlation
    correlation = np.corrcoef(example.real_distribution, simulated_dist)[0,1]
    return correlation
```

**Metric 2: Misconception Match Rate**

```python
def misconception_match_metric(example, pred, trace=None):
    """
    When wrong, did the student select the distractor matching their misconception?

    example.misconception_mapping: {"B": "adds_num_and_denom", "C": "subtracts_instead", ...}
    example.target_misconception: "adds_num_and_denom"
    example.correct_answer: "A"
    """
    selected = pred.selected_option

    # If correct, return 0.5 (neutral—not rewarding correctness)
    if selected == example.correct_answer:
        return 0.5

    # If wrong, reward matching the expected misconception
    if example.misconception_mapping.get(selected) == example.target_misconception:
        return 1.0
    else:
        return 0.0
```

**Metric 3: Combined (Primary)**

```python
def error_fidelity_metric(example, pred, trace=None):
    """Combined metric balancing distribution match and misconception accuracy."""
    dist_score = response_distribution_metric(example, pred, trace)
    misc_score = misconception_match_metric(example, pred, trace)

    # Weight misconception match higher—it's the key insight
    return 0.4 * dist_score + 0.6 * misc_score
```

### Optimization

```python
from dspy.teleprompt import MIPROv2, BootstrapFewShot

# Option A: MIPRO (instruction + example optimization)
optimizer = MIPROv2(
    metric=error_fidelity_metric,
    num_candidates=10,
    init_temperature=1.0,
)

# Option B: BootstrapFewShot (example-only optimization)
optimizer = BootstrapFewShot(
    metric=error_fidelity_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)

# Compile
optimized_student = optimizer.compile(
    ErrorProneStudent(),
    trainset=train_items,
    valset=val_items,
)

# Save
optimized_student.save("optimized_student_v1.json")
```

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Baseline-ZeroShot** | No optimization, simple ability prompt |
| **Baseline-LuWang** | KC-based prompt from Generative Students paper |
| **DSPy-BootstrapFewShot** | Optimize few-shot examples only |
| **DSPy-MIPRO** | Optimize instructions + examples jointly |
| **DSPy-PerAbility** | Separate MIPRO optimization per ability band |
| **DSPy-PerMisconception** | Separate optimization per misconception cluster |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Response Distribution r** | Correlation with real student distributions | > 0.75 |
| **Misconception Match %** | When wrong, selected expected misconception | > 60% |
| **IRT Difficulty r** | Correlation of item difficulty params | > 0.80 |
| **Accuracy Calibration** | Match real student accuracy per ability band | ±5% |
| **Error Consistency** | Consistent errors on isomorphic items | Match real variance |

---

## Procedure

### Phase 1: Setup (Week 1)

1. Obtain Eedi dataset, filter to items with ≥100 responses
2. Map misconceptions to distractors
3. Cluster students into ability bands via IRT
4. Create train/val/test splits
5. Implement DSPy signatures and metrics

### Phase 2: Baseline Evaluation (Week 2)

1. Run Baseline-ZeroShot and Baseline-LuWang
2. Compute all metrics
3. Identify: Where do simulated students diverge most from real?

### Phase 3: DSPy Optimization (Week 3-4)

1. Run BootstrapFewShot optimization (faster, simpler)
2. Run MIPRO optimization (more powerful)
3. Track metric improvement over optimization steps
4. Compare: Does per-ability or per-misconception optimization help?

### Phase 4: Analysis (Week 5)

1. Evaluate all conditions on held-out test
2. Statistical significance (bootstrap CIs)
3. Qualitative analysis: What prompts did DSPy discover?
4. Ablation: Effect of training set size, optimization steps

---

## What We Expect to Learn

**If DSPy works well**:
- Optimized prompts will include specific phrasing that elicits errors
- Few-shot examples will be selected to demonstrate "how to be wrong"
- We'll have transferable prompts for synthetic student generation

**If DSPy struggles**:
- The competence paradox may be fundamental—prompt optimization can't overcome training
- Would motivate DPO or fine-tuning approaches instead
- Still valuable as a negative result

**Either way**:
- First empirical test of prompt optimization for error fidelity
- Publishable contribution to synthetic students literature

---

## Resources

- **Compute**: ~$200-400 API costs for optimization runs
- **Time**: ~5 weeks
- **Code**: DSPy + custom metrics, will open-source

---

## Next Steps

1. [ ] Confirm Eedi dataset access (or use public ASSISTments subset)
2. [ ] Implement SimulatedStudent signature
3. [ ] Implement custom metrics
4. [ ] Run baselines on 100-item pilot
5. [ ] Run MIPRO optimization
6. [ ] Evaluate and iterate
