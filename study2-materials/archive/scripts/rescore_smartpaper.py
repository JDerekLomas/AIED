"""
Re-score existing SmartPaper RSM responses using LLM-as-judge.
Reads cached raw responses from the sweep, scores them, and recomputes correlations.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

OUTPUT_DIR = Path("pilot/smartpaper_rsm")
SCORE_CACHE_PATH = OUTPUT_DIR / "score_cache.json"

PROFICIENCY_DISTRIBUTION = [
    ("struggling", 0.30),
    ("basic", 0.35),
    ("competent", 0.25),
    ("advanced", 0.10),
]


def load_score_cache():
    if SCORE_CACHE_PATH.exists():
        return json.loads(SCORE_CACHE_PATH.read_text())
    return {}


def save_score_cache(cache):
    SCORE_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def llm_score(client, question, rubric, max_score, answer):
    """Score using LLM. Returns 0 or 1."""
    prompt = f"""Score this student answer. Reply ONLY "1" (adequate) or "0" (inadequate).

Question: {question}
Rubric: {rubric}
Max score: {max_score}
Student answer: {answer}

Score:"""

    from google.genai import types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text.strip()
    return 1 if "1" in text[:3] else 0


def parse_classroom_answers(text, n_students):
    answers = []
    for i in range(1, n_students + 1):
        pattern = rf'Student\s*{i}\s*\([^)]*\)\s*:\s*(.+?)(?=\n\*?\*?Student\s*\d|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            ans = match.group(1).strip().strip('*').strip()
            answers.append(ans)
        else:
            answers.append("")
    return answers


def parse_teacher_prediction(text):
    results = {}
    for level, _ in PROFICIENCY_DISTRIBUTION:
        pattern = rf'{level}\s*:\s*(\d+)\s*%'
        match = re.search(pattern, text, re.IGNORECASE)
        results[level] = int(match.group(1)) / 100.0 if match else 0.5
    return results


def rescore_config(config_id, probe_items, client, score_cache, design_row):
    """Re-score a single config's responses."""
    raw_dir = OUTPUT_DIR / "raw_responses" / f"config_{config_id}"
    if not raw_dir.exists():
        return None

    prompt_style = design_row["prompt_style"]
    spc = int(design_row["students_per_call"])

    item_scores = {}
    n_scored = 0
    n_cached = 0

    for item in probe_items:
        ik = f"{item['assessment']}_q{item['question_number']}"
        item_scores[ik] = []

        if prompt_style == "teacher_prediction":
            raw_path = raw_dir / f"{ik}_teacher.txt"
            if not raw_path.exists():
                continue
            text = raw_path.read_text()
            preds = parse_teacher_prediction(text)
            for level, weight in PROFICIENCY_DISTRIBUTION:
                n_students = max(1, round(20 * weight))
                p_correct = preds.get(level, 0.5)
                for _ in range(n_students):
                    item_scores[ik].append(1 if np.random.random() < p_correct else 0)

        elif prompt_style == "classroom_batch":
            n_calls = max(1, 20 // spc)
            for batch_idx in range(n_calls):
                raw_path = raw_dir / f"{ik}_batch{batch_idx}.txt"
                if not raw_path.exists():
                    continue
                text = raw_path.read_text()
                answers = parse_classroom_answers(text, spc)
                for ans in answers:
                    if not ans or ans.lower() in ["", "blank", "[blank]"]:
                        item_scores[ik].append(0)
                        continue
                    # Check cache
                    cache_key = f"{ik}|{ans[:200]}"
                    if cache_key in score_cache:
                        item_scores[ik].append(score_cache[cache_key])
                        n_cached += 1
                    else:
                        score = llm_score(client, item["question_text"], item["rubric"],
                                         item["max_score"], ans)
                        score_cache[cache_key] = score
                        item_scores[ik].append(score)
                        n_scored += 1
                        if n_scored % 50 == 0:
                            save_score_cache(score_cache)
                        time.sleep(0.05)

        elif prompt_style == "individual_roleplay":
            for si in range(20):
                raw_path = raw_dir / f"{ik}_s{si}.txt"
                if not raw_path.exists():
                    continue
                text = raw_path.read_text()
                # Extract answer
                match = re.search(r'(?:YOUR ANSWER|ANSWER)\s*:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                ans = match.group(1).strip() if match else text.strip().split('\n')[-1]

                if not ans or ans.lower() in ["", "blank", "[blank]"]:
                    item_scores[ik].append(0)
                    continue

                cache_key = f"{ik}|{ans[:200]}"
                if cache_key in score_cache:
                    item_scores[ik].append(score_cache[cache_key])
                    n_cached += 1
                else:
                    score = llm_score(client, item["question_text"], item["rubric"],
                                     item["max_score"], ans)
                    score_cache[cache_key] = score
                    item_scores[ik].append(score)
                    n_scored += 1
                    if n_scored % 50 == 0:
                        save_score_cache(score_cache)
                    time.sleep(0.05)

    # Compute mean score proportion per item
    mean_scores = {}
    for ik, scores in item_scores.items():
        if scores:
            mean_scores[ik] = sum(scores) / len(scores)

    print(f"  Config {config_id}: scored={n_scored}, cached={n_cached}, items={len(mean_scores)}")
    return mean_scores


def evaluate(mean_scores, probe_items):
    sim_vals, actual_vals = [], []
    for item in probe_items:
        ik = f"{item['assessment']}_q{item['question_number']}"
        if ik in mean_scores:
            sim_vals.append(mean_scores[ik])
            actual_vals.append(item["classical_difficulty"])
    if len(sim_vals) < 5:
        return {"spearman_rho": float('nan'), "pearson_r": float('nan'), "n": len(sim_vals)}
    rho, p_rho = stats.spearmanr(sim_vals, actual_vals)
    r, p_r = stats.pearsonr(sim_vals, actual_vals)
    return {"spearman_rho": rho, "spearman_p": p_rho, "pearson_r": r, "pearson_p": p_r, "n": len(sim_vals)}


def main():
    import pandas as pd
    from google import genai

    os.chdir(Path(__file__).parent.parent)
    np.random.seed(42)

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    probe = json.loads((OUTPUT_DIR / "probe_items.json").read_text())
    design = pd.read_csv(OUTPUT_DIR / "design_matrix.csv")
    score_cache = load_score_cache()

    # Find which configs have raw responses
    raw_base = OUTPUT_DIR / "raw_responses"
    config_dirs = sorted([d.name for d in raw_base.iterdir() if d.is_dir()])
    print(f"Found {len(config_dirs)} config directories")

    results = []
    for config_dir in config_dirs:
        cid = int(config_dir.replace("config_", ""))
        row = design[design["config_id"] == cid].iloc[0]

        mean_scores = rescore_config(cid, probe, client, score_cache, row)
        if mean_scores is None:
            continue

        metrics = evaluate(mean_scores, probe)
        result = row.to_dict()
        result.update(metrics)
        results.append(result)

        print(f"    rho={metrics['spearman_rho']:.3f}, r={metrics['pearson_r']:.3f}")

    save_score_cache(score_cache)

    # Save rescored results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results_rescored.csv", index=False)
    print(f"\nSaved {len(df)} rescored results")
    print(f"Score cache: {len(score_cache)} entries")

    if len(df) > 0:
        print(f"\nRescored Spearman rho: mean={df['spearman_rho'].mean():.3f}, "
              f"range=[{df['spearman_rho'].min():.3f}, {df['spearman_rho'].max():.3f}]")


if __name__ == "__main__":
    main()
