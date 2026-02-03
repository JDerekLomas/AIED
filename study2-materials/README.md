# When Does LLM Difficulty Estimation Work?

**A Systematic Search Across Methods, Models, and Datasets**

Derek Lomas | AIED 2026 | February 2026

## Paper

Open `results-site/paper.html` in a browser for the full paper with interactive figures.

## Key Results

- **Open-ended items (SmartPaper):** rho=0.55-0.67 depending on prompt framing (n=140, p<0.0001)
- **MCQ items (Eedi):** rho=0.11, non-significant on 105-item confirmation (95% CI [-0.07, 0.30])
- **Prompt framing is the primary variable:** prerequisite_chain rho=0.67 vs. simulation rho<0.45
- **Temperature x framing interaction:** structural framings peak at low temp; cognitive load peaks at high temp
- **More cognitive scaffolding hurts:** misconception hints rho=0.19 vs. direct prediction rho=0.50
- **Model size doesn't predict success:** Scout 17B (rho=0.37) > GPT-4o (rho=0.17)

Total cost: ~$50 in API calls across ~200 experimental conditions.

## Directory Structure

```
study2-materials/
├── results-site/paper.html         # Full paper (open in browser)
├── paper/sections/                  # Paper-ready section drafts
│
├── data/
│   ├── smartpaper/                  # 140 open-ended items, 4 subjects (India)
│   ├── dbe-kt22/                   # South African maths assessment
│   └── indian_state_assessment/     # Indian state exam data
│
├── scripts/                         # Active experiment scripts (~27 files)
│   ├── run_prompt_screening_g3f.py  # Prompt framing RSM experiment
│   ├── rsm_analysis.py             # RSM analysis and visualization
│   ├── run_confirmation.py          # 105-item Eedi confirmation
│   ├── run_smartpaper_expansion.py  # 140-item SmartPaper expansion
│   ├── run_model_survey.py          # 15-model cross-model survey
│   ├── analyze_results.py           # Main analysis pipeline
│   ├── prompts_mcq.py              # Prompt library
│   └── ...                          # Phase-specific experiment runners
│
├── pilot/                           # Active experiment outputs
│   ├── prompt_framing_experiment/   # 15 framings x temps x 140 items (main result)
│   ├── replications/                # Literature replication attempts
│   ├── smartpaper_expansion/        # 140-item SmartPaper results
│   ├── confirmation_experiment/     # 105-item Eedi confirmation
│   ├── model_survey/                # 15-model comparison
│   ├── rsm_experiment/              # Initial RSM screening
│   └── ...
│
├── archive/                         # Earlier phases (drafts, scripts, pilot dirs)
│   ├── drafts/                      # Old abstracts, design docs, paper drafts
│   ├── scripts/                     # Earlier experiment scripts (~57 files)
│   └── pilot/                       # Old pilot test directories (~40 dirs)
│
├── docs/                            # Literature review, prompt design rationale
└── .claude/handoffs/                # Session-by-session research documentation
```

## Reproduction

```bash
pip install openai google-generativeai groq scipy numpy pandas tqdm

# Set API keys
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"

# Run the prompt framing experiment (main result)
python scripts/run_prompt_screening_g3f.py

# Run RSM analysis
python scripts/rsm_analysis.py
```

## Datasets

- **SmartPaper** (India): 140 open-ended items, Grades 6-8, 4 subjects. Difficulty = proportion correct.
- **Eedi** (UK): 1,869 MCQ maths items with misconception-mapped distractors. Difficulty = IRT b_2PL.
- **DBE-KT22** (South Africa): Maths assessment for validation.

## Citation

```
@inproceedings{lomas2026difficulty,
  title={When Does LLM Difficulty Estimation Work? A Systematic Search Across Methods, Models, and Datasets},
  author={Lomas, Derek},
  booktitle={Proceedings of AIED 2026},
  year={2026}
}
```

*Last updated: February 3, 2026*
