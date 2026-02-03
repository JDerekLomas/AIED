# It's Hard to Know How Hard It Is

**Mapping the Design Space of LLM Item Difficulty Estimation**

Derek Lomas | AIED 2026 | February 2026

## Paper

- **LaTeX source:** `paper/main.tex` (compile with `pdflatex main.tex`)
- **Interactive version:** `results-site/paper.html` (open in browser)

## Key Results

- **Best performance:** ρ=0.69 with prerequisite_chain prompt (SmartPaper, n=140)
- **Prompt framing is the primary variable:** Item-analysis prompts (ρ=0.61 mean) outperform population-only prompts (ρ=0.47 mean)
- **Model matters less than prompt:** ANOVA shows prompt η²=0.31 vs. model η²=0.08
- **Temperature doesn't matter:** No significant difference between t=0.5, 1.0, 2.0
- **Simulation-based approaches fail:** synthetic_students ρ=0.19 (worst performer)
- **Cross-dataset generalization:** Correlations transfer to MCQ datasets (DBE-KT22 ρ=0.35, BEA 2024 ρ=0.45)

Total cost: ~$100 in API calls across ~120 experimental configurations (5 models × 4 prompts × 3 datasets × 3 reps).

## Directory Structure

```
study2-materials/
├── paper/
│   ├── main.tex                     # LaTeX source (LNCS format)
│   ├── main.pdf                     # Compiled paper
│   └── figures/                     # Generated figures
│
├── data/
│   ├── smartpaper/                  # 140 open-ended items (India, Grades 6-8)
│   └── dbe-kt22/                    # South African maths MCQs
│
├── scripts/
│   ├── run_prompt_screening_g3f.py  # Phase 1: 15-prompt screening (CONTAINS ALL PROMPTS)
│   ├── run_model_survey.py          # Phase 2: 5-model comparison
│   └── run_bea2024_validation.py    # Phase 3: BEA 2024 benchmark
│
├── pilot/
│   ├── prompt_framing_experiment/   # Phase 1 results: 15 framings × 3 temps × 140 items
│   │   └── results.json             # Main screening results
│   ├── model_survey/
│   │   └── survey_results.json      # 5 models × 4 prompts × 3 reps
│   └── anova_analysis.py            # Statistical analysis script
│
└── .claude/handoffs/                # Session-by-session research notes
```

## Prompts

All 16 prompts are defined in `scripts/run_prompt_screening_g3f.py`:

| Prompt | Type | Best ρ | Description |
|--------|------|--------|-------------|
| prerequisite_chain | Item | 0.69 | Count knowledge/skill prerequisites |
| cognitive_load | Item | 0.67 | Estimate working memory demands |
| buggy_rules | Item+Pop | 0.66 | Identify systematic procedural errors |
| misconception_holistic | Item+Pop | 0.64 | Predict misconceptions holistically |
| error_analysis | Item+Pop | 0.60 | Analyze likely student errors |
| devil_advocate | Pop | 0.60 | Challenge overconfidence bias |
| cognitive_profile | Item | 0.59 | Analyze cognitive demands |
| contrastive | Item | 0.58 | Compare to similar content |
| classroom_sim | Pop | 0.56 | Simulate 20 students |
| teacher | Baseline | 0.56 | Expert teacher judgment |
| verbalized_sampling | Pop | <0.55 | Multiple perspective estimates |
| familiarity_gradient | Item | <0.55 | Distance from textbook drills |
| imagine_classroom | Pop | <0.55 | Visualize classroom distribution |
| teacher_decomposed | Pop | <0.55 | Estimate per proficiency level |
| error_affordance | Item | <0.55 | Count plausible error paths |
| synthetic_students | Pop | 0.19 | Two-stage persona simulation (worst) |

## Reproduction

```bash
pip install google-generativeai scipy numpy pandas tqdm python-dotenv

# Set API key
export GOOGLE_API_KEY="your-key"

# Phase 1: Prompt screening (15 prompts × 3 temps × 140 items)
python scripts/run_prompt_screening_g3f.py

# Phase 2: Model survey (requires additional API keys)
python scripts/run_model_survey.py

# Analyze results
python pilot/anova_analysis.py
```

## Datasets

- **SmartPaper** (India): 140 open-ended items, Grades 6-8, 4 subjects. Difficulty = proportion correct.
- **DBE-KT22** (South Africa): MCQ maths items with known p-correct. Used for cross-dataset validation.
- **BEA 2024** (USA): USMLE medical exam MCQs from shared task. Used for benchmark comparison.

## Citation

```
@inproceedings{lomas2026difficulty,
  title={It's Hard to Know How Hard It Is: Mapping the Design Space of LLM Item Difficulty Estimation},
  author={Lomas, Derek},
  booktitle={Proceedings of AIED 2026},
  year={2026}
}
```

*Last updated: February 3, 2026*
