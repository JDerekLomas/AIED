#!/usr/bin/env python3
"""Parse rate transparency analysis for AIED paper revision.

Scans all model_survey and prompt_framing_experiment directories to report
parse success rates per model x prompt combination.

Reviewer 3 concern: we never report parsing success rates for retained models.
"""

import os
import re
import glob
import json
from collections import defaultdict

BASE = "/Users/dereklomas/AIED/study2-materials"
MODEL_SURVEY_DIR = f"{BASE}/pilot/model_survey"
PROMPT_FRAMING_DIR = f"{BASE}/pilot/prompt_framing_experiment"

EXPECTED_PER_REP = 140
N_REPS = 3

# Model display names
MODEL_NAMES = {
    "gemini": "Gemini 2.0 Flash",
    "gemini_batch": "Gemini 2.0 Flash (batch)",
    "gemma3_27b": "Gemma 3 27B",
    "gpt4o": "GPT-4o",
    "llama31_8b": "Llama 3.1 8B",
    "llama33_70b": "Llama 3.3 70B",
    "maverick": "Llama 4 Maverick",
    "opus45": "Claude Opus 4.5",
    "opus45_dbe": "Claude Opus 4.5 (DBE)",
    "qwen3_32b": "Qwen 3 32B",
    "scout": "Llama 4 Scout",
}


def parse_value(text):
    """Extract a float prediction from response text.

    Matches the logic used in analyze_teacher.py:
    - Searches for numbers in the text
    - Returns first number in [0, 1] range
    - If number is in (1, 100], divides by 100
    """
    text = text.strip()
    # Try last non-empty line first (structured formats end with the number)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    # Try parsing last line as a bare float
    last_line = lines[-1]
    try:
        v = float(last_line)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100.0
    except ValueError:
        pass

    # Fall back: search for numbers in last line, then full text
    for search_text in [last_line, text]:
        nums = re.findall(r"\b(\d+\.?\d*)\b", search_text)
        for n in nums:
            v = float(n)
            if 0 <= v <= 1:
                return v
            if 1 < v <= 100:
                return v / 100.0

    return None


def analyze_directory(base_dir, prompt_dir, n_reps=3):
    """Analyze a single prompt directory with rep0, rep1, rep2 subdirs."""
    total_expected = 0
    total_present = 0
    total_parsed = 0

    for r in range(n_reps):
        rep_dir = os.path.join(base_dir, prompt_dir, f"rep{r}")
        if not os.path.isdir(rep_dir):
            total_expected += EXPECTED_PER_REP
            continue

        total_expected += EXPECTED_PER_REP
        files = glob.glob(os.path.join(rep_dir, "*.txt"))
        total_present += len(files)

        for f in files:
            try:
                with open(f) as fh:
                    content = fh.read()
                val = parse_value(content)
                if val is not None:
                    total_parsed += 1
            except Exception:
                pass

    return total_expected, total_present, total_parsed


def main():
    results = []

    # ── Phase 2: Model Survey ──
    print("=" * 90)
    print("PARSE RATE ANALYSIS")
    print("=" * 90)

    print("\n── Phase 2: Model Survey (SmartPaper, 140 items × 3 reps = 420 expected) ──\n")
    print(f"{'Model':<28} {'Prompt':<22} {'Expected':>8} {'Present':>8} {'Parsed':>8} {'File%':>7} {'Parse%':>7}")
    print("-" * 90)

    model_dirs = sorted(
        [d for d in os.listdir(MODEL_SURVEY_DIR)
         if os.path.isdir(os.path.join(MODEL_SURVEY_DIR, d))
         and d not in ("figures",)]
    )

    for model in model_dirs:
        model_path = os.path.join(MODEL_SURVEY_DIR, model)
        prompt_dirs = sorted(
            [d for d in os.listdir(model_path)
             if os.path.isdir(os.path.join(model_path, d))
             and d not in ("figures",)
             and os.path.isdir(os.path.join(model_path, d, "rep0"))]
        )

        for prompt in prompt_dirs:
            expected, present, parsed = analyze_directory(model_path, prompt)
            file_pct = 100.0 * present / expected if expected > 0 else 0
            parse_pct = 100.0 * parsed / present if present > 0 else 0
            display_model = MODEL_NAMES.get(model, model)
            results.append({
                "phase": "Model Survey",
                "model": display_model,
                "model_key": model,
                "prompt": prompt,
                "expected": expected,
                "present": present,
                "parsed": parsed,
                "file_pct": file_pct,
                "parse_pct": parse_pct,
            })
            print(f"{display_model:<28} {prompt:<22} {expected:>8} {present:>8} {parsed:>8} {file_pct:>6.1f}% {parse_pct:>6.1f}%")

    # ── Phase 1: Prompt Framing Experiment (Gemini Flash) ──
    print(f"\n\n── Phase 1: Prompt Framing Experiment (Gemini 2.0 Flash, 140 items × 3 reps = 420 expected) ──\n")
    print(f"{'Prompt Config':<38} {'Expected':>8} {'Present':>8} {'Parsed':>8} {'File%':>7} {'Parse%':>7}")
    print("-" * 90)

    prompt_configs = sorted(
        [d for d in os.listdir(PROMPT_FRAMING_DIR)
         if os.path.isdir(os.path.join(PROMPT_FRAMING_DIR, d))
         and d not in ("figures", "llama-4-maverick-17b-128e", "llama-4-scout-17b-16e")
         and os.path.isdir(os.path.join(PROMPT_FRAMING_DIR, d, "rep0"))]
    )

    for config in prompt_configs:
        expected, present, parsed = analyze_directory(PROMPT_FRAMING_DIR, config)
        file_pct = 100.0 * present / expected if expected > 0 else 0
        parse_pct = 100.0 * parsed / present if present > 0 else 0
        results.append({
            "phase": "Prompt Screening",
            "model": "Gemini 2.0 Flash",
            "model_key": "gemini",
            "prompt": config,
            "expected": expected,
            "present": present,
            "parsed": parsed,
            "file_pct": file_pct,
            "parse_pct": parse_pct,
        })
        print(f"{config:<38} {expected:>8} {present:>8} {parsed:>8} {file_pct:>6.1f}% {parse_pct:>6.1f}%")

    # ── Summary Statistics ──
    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Phase 2 summary by model
    print("\n── Phase 2: Per-model aggregates ──\n")
    print(f"{'Model':<28} {'Expected':>8} {'Present':>8} {'Parsed':>8} {'File%':>7} {'Parse%':>7}")
    print("-" * 75)

    model_agg = defaultdict(lambda: [0, 0, 0])
    for r in results:
        if r["phase"] == "Model Survey":
            k = r["model"]
            model_agg[k][0] += r["expected"]
            model_agg[k][1] += r["present"]
            model_agg[k][2] += r["parsed"]

    for model, (exp, pres, par) in sorted(model_agg.items()):
        fp = 100.0 * pres / exp if exp > 0 else 0
        pp = 100.0 * par / pres if pres > 0 else 0
        print(f"{model:<28} {exp:>8} {pres:>8} {par:>8} {fp:>6.1f}% {pp:>6.1f}%")

    # Phase 1 aggregate
    p1 = [r for r in results if r["phase"] == "Prompt Screening"]
    p1_exp = sum(r["expected"] for r in p1)
    p1_pres = sum(r["present"] for r in p1)
    p1_par = sum(r["parsed"] for r in p1)
    print(f"\n── Phase 1: Overall ──")
    print(f"  Expected: {p1_exp}, Present: {p1_pres} ({100*p1_pres/p1_exp:.1f}%), Parsed: {p1_par} ({100*p1_par/p1_pres:.1f}%)")

    # Phase 2 aggregate
    p2 = [r for r in results if r["phase"] == "Model Survey"]
    p2_exp = sum(r["expected"] for r in p2)
    p2_pres = sum(r["present"] for r in p2)
    p2_par = sum(r["parsed"] for r in p2)
    print(f"\n── Phase 2: Overall ──")
    print(f"  Expected: {p2_exp}, Present: {p2_pres} ({100*p2_pres/p2_exp:.1f}%), Parsed: {p2_par} ({100*p2_par/p2_pres:.1f}%)")

    # Grand total
    g_exp = p1_exp + p2_exp
    g_pres = p1_pres + p2_pres
    g_par = p1_par + p2_par
    print(f"\n── Grand Total ──")
    print(f"  Expected: {g_exp}, Present: {g_pres} ({100*g_pres/g_exp:.1f}%), Parsed: {g_par} ({100*g_par/g_pres:.1f}%)")

    # Save JSON for downstream use
    out_path = os.path.join(os.path.dirname(__file__), "parse_rate_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
