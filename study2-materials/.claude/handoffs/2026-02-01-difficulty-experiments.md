# Difficulty Experiments Handoff — 2026-02-01

## Current State

### Teacher Perspective (COMPLETE)
- 500 calls finished (100 items × 5 reps), Gemini 3 Flash
- Results: `pilot/teacher_perspective/gemini-3-flash/aggregated.csv`
- **Disappointing**: Pearson r=0.095 with classical difficulty (p=0.35), essentially no correlation
- Tercile analysis shows no differentiation (easy=52.9%, medium=51.9%, hard=55.3%)
- The teacher-prediction prompt doesn't discriminate difficulty with this model

### Student Simulation (RUNNING)
- 20,000 calls (100 items × 200 simulated students), Gemini 3 Flash
- 4 proficiency levels: below_basic(50), basic(70), proficient(50), advanced(30)
- Weighted distribution: 25/35/25/15%
- Output: `pilot/student_simulation_v2/gemini-3-flash/`
- Resume support: just re-run the same command if interrupted
- Command: `python3 scripts/run_difficulty_experiments.py --experiment simulation --model gemini-3-flash`
- ~70 calls/min, estimated 4-5 hours total

## Next Steps
1. Check if simulation finished: `ls pilot/student_simulation_v2/gemini-3-flash/raw_responses/ | wc -l` (should be 20,000)
2. If interrupted, re-run the command (resumes from cache)
3. Run analysis: `python3 scripts/run_difficulty_experiments.py --experiment both --model gemini-3-flash` (teacher cached, sim cached, just runs analysis)
4. Compare simulation correlations to teacher perspective and structured estimation results
5. If simulation also shows low correlation, consider whether Gemini Flash is too capable to simulate weak students

## Key Decisions
- Used temp=0.9 for simulation (higher variance, more realistic errors)
- Used temp=0.7 for teacher perspective (consistent with other experiments)
- Same 100 test items and 5 calibration items as structured estimation
- Student simulation does NOT reveal correct answer in prompt (unlike teacher)

## Files
- **Script**: `scripts/run_difficulty_experiments.py`
- **Teacher results**: `pilot/teacher_perspective/gemini-3-flash/`
- **Simulation results**: `pilot/student_simulation_v2/gemini-3-flash/`
