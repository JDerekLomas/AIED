# Study 2: Completion Criteria

## Phase-by-Phase Success Criteria

---

## Phase 0: Setup (CURRENT)

### Deliverables
- [x] Study design document
- [x] Directory structure created
- [ ] Eedi dataset downloaded and validated
- [ ] Coding rubric with examples
- [ ] Prompt templates finalized
- [ ] Data collection script functional
- [ ] Probe items generated

### Exit Criteria
- [ ] Can run 1 item through 1 model and code the response
- [ ] All team members understand coding rubric (if applicable)

---

## Phase 1: Pilot Study

### Objective
Validate prompts, coding scheme, and estimate feasibility

### Deliverables
- [ ] 20 Eedi items selected (balanced by misconception type)
- [ ] Responses generated: 20 items × 3 models × 3 prompts × 5 responses = 900 responses
- [ ] Responses coded by 2 coders (subset: 100 responses)
- [ ] Inter-rater reliability computed

### Success Criteria
| Metric | Target | Action if Not Met |
|--------|--------|-------------------|
| IRR (Cohen's κ) | ≥ 0.60 | Revise rubric, retrain coders |
| Coding completion | 100% codeable | Add categories or clarify edge cases |
| API success rate | ≥ 95% | Debug prompts, check rate limits |
| Cost estimate accuracy | Within 20% of projected | Adjust sample size |

### Exit Criteria
- [ ] κ ≥ 0.60 achieved
- [ ] No systematic coding ambiguities
- [ ] Cost projection validated
- [ ] Prompts produce usable reasoning (for explain/persona conditions)

---

## Phase 2: Full Data Collection

### Objective
Generate all responses for main analysis

### Deliverables
- [ ] Eedi sample: 400 items selected
- [ ] ASSISTments sample: 200 items selected (if including)
- [ ] Probe items: 50 items validated
- [ ] All responses generated and stored

### Sample Size
```
Eedi:        400 items × 3 models × 3 prompts × 20 responses = 72,000 responses
Probes:       50 items × 3 models × 3 prompts × 20 responses =  9,000 responses
────────────────────────────────────────────────────────────────────────────────
Total:                                                          81,000 responses
```

### Success Criteria
| Metric | Target | Action if Not Met |
|--------|--------|-------------------|
| Response collection | 100% complete | Retry failed calls |
| Data integrity | No missing fields | Validate schema |
| Cost | ≤ $1,000 | Reduce N or models |

### Exit Criteria
- [ ] All 81,000 responses collected
- [ ] Data exported to analysis-ready format
- [ ] Backup created

---

## Phase 3: Coding & Analysis

### Objective
Code qualitative sample, run statistical analyses

### Deliverables

#### Quantitative
- [ ] Distractor match rates computed (all conditions)
- [ ] Misconception match rates computed (coded sample)
- [ ] Statistical model fitted (mixed-effects logistic regression)
- [ ] Figures generated (interaction plots, bar charts)

#### Qualitative (Light Touch)
- [ ] 50 items coded by primary coder
- [ ] 25 items double-coded for reliability
- [ ] Themes identified and documented
- [ ] Illustrative examples selected

### Success Criteria
| Metric | Target | Notes |
|--------|--------|-------|
| Final IRR (κ) | ≥ 0.70 | On 25 double-coded items |
| Model convergence | Yes | If not, simplify random effects |
| Effect sizes | Report regardless | No minimum; report actual |
| Qualitative saturation | Themes stabilize | Add items if new themes emerge |

### Exit Criteria
- [ ] All planned analyses complete
- [ ] Results tables and figures finalized
- [ ] Key findings summarized

---

## Phase 4: Probe Item Validation

### Objective
Test whether generated items reliably elicit target misconceptions

### Deliverables
- [ ] Elicitation rates computed per misconception
- [ ] Successful vs. failed probes identified
- [ ] Recommendations for probe design

### Success Criteria
| Metric | Target | Interpretation |
|--------|--------|----------------|
| Mean elicitation rate | ≥ 50% | Probes are somewhat effective |
| Best misconceptions | ≥ 3 with >70% | Some misconceptions reliably elicitable |
| Worst misconceptions | Identify | Document which can't be elicited |

### Exit Criteria
- [ ] Elicitation rates computed
- [ ] Taxonomy of "elicitable" vs. "non-elicitable" misconceptions

---

## Phase 5: Writing & Submission

### Objective
Complete manuscript for target venue

### Deliverables
- [ ] Abstract (250 words)
- [ ] Introduction with RQs
- [ ] Methods (reproducible detail)
- [ ] Results (quant + qual)
- [ ] Discussion (implications, limitations)
- [ ] Supplementary materials (rubric, prompts, sample data)

### Success Criteria
| Metric | Target |
|--------|--------|
| Word count | Within venue limit |
| All figures/tables | Camera-ready |
| Co-author review | Approved |

### Target Venues & Deadlines
| Venue | Deadline | Format |
|-------|----------|--------|
| EDM 2026 | Feb 9 | 10-page full or 6-page short |
| L@S 2026 | Feb 16 | 10-page research |
| IJAIED | Rolling | Journal (no limit) |

---

## Overall Study Success Criteria

### Minimum Viable Study
- [ ] ≥200 Eedi items analyzed
- [ ] ≥2 models compared
- [ ] ≥2 prompting conditions
- [ ] Misconception match rate reported with CI
- [ ] At least 1 moderator tested (misconception type OR prompting)

### Full Study
- [ ] 400 Eedi items + 50 probes
- [ ] 3 models × 3 prompts
- [ ] Full factorial analysis
- [ ] Qualitative themes documented
- [ ] Probe validation complete

### Stretch Goals
- [ ] ASSISTments cross-validation
- [ ] Demographic subgroup analysis (if data available)
- [ ] Open dataset/code release

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Low IRR | Medium | High | Simplify rubric; use top-level categories only |
| API costs exceed budget | Low | Medium | Reduce N per condition from 20 to 10 |
| Eedi access issues | Low | High | Fall back to ASSISTments + generated items |
| No significant effects | Medium | Medium | Report as valid null finding; emphasize descriptives |
| Tight deadline | High | High | Prioritize core analyses; defer probes to Phase 2 |

---

## Decision Points

### After Pilot
- **Continue** if κ ≥ 0.60 and prompts work
- **Revise** if κ < 0.60 but fixable
- **Pivot** if fundamental issues with approach

### After Data Collection
- **Full analysis** if N ≥ 300 items complete
- **Reduced scope** if 200-300 items
- **Abort** if < 200 items (unlikely)

### After Analysis
- **Submit to EDM** if results ready by Feb 7
- **Submit to L@S** if need extra week
- **Target journal** if results warrant longer format

---

*Last updated: January 26, 2026*
