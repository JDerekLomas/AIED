#!/usr/bin/env python3
"""Compare strict vs lenient parsing of Eedi RSM response files."""

import csv
import json
import os
import re
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np

BASE = Path("/Users/dereklomas/AIED/study2-materials")
SWEEP_DIR = BASE / "pilot/rsm_experiment/metaprompt_sweep"
CROSS_DIR = BASE / "pilot/rsm_experiment/cross_model"
PROBE_CSV = BASE / "pilot/rsm_experiment/probe_items.csv"
IRT_JSON = BASE / "results/irt_proper_statistics.json"

WEIGHTS = {"below_basic": 0.25, "basic": 0.35, "proficient": 0.25, "advanced": 0.15}
LEVELS = ["below_basic", "basic", "proficient", "advanced"]


def load_ground_truth():
    with open(PROBE_CSV) as f:
        probes = list(csv.DictReader(f))
    correct = {row["QuestionId"]: row["correct_answer_kaggle"] for row in probes}
    qids = [row["QuestionId"] for row in probes]
    with open(IRT_JSON) as f:
        irt = json.load(f)
    b_2pl = {}
    for qid in qids:
        if qid in irt.get("items", {}):
            b_2pl[qid] = irt["items"][qid]["b_2pl"]
    return qids, correct, b_2pl


def parse_strict(text):
    results = {}
    for level in LEVELS:
        pat = re.compile(
            rf'{level}\s*:\s*A\s*=\s*(\d+)%\s*B\s*=\s*(\d+)%\s*C\s*=\s*(\d+)%\s*D\s*=\s*(\d+)%',
            re.IGNORECASE
        )
        m = pat.search(text)
        if m:
            results[level] = {
                "A": int(m.group(1)), "B": int(m.group(2)),
                "C": int(m.group(3)), "D": int(m.group(4)),
            }
    return results if len(results) == 4 else None


def parse_lenient(text):
    results = {}
    for level in LEVELS:
        level_variants = [level, level.replace("_", " "), level.replace("_", "-")]
        
        # Find section for this level
        section = None
        for variant in level_variants:
            idx = text.lower().find(variant.lower())
            if idx >= 0:
                section = text[idx:idx+500]
                break
        
        if section is None:
            continue

        # Try to extract percentages for A, B, C, D
        pcts = {}
        for letter in "ABCD":
            patterns = [
                rf'{letter}\s*[=:]\s*(\d+)\s*%',
                rf'\*\*{letter}\*\*[^0-9]*?(\d+)\s*%',
                rf'{letter}\s*[-–]\s*(\d+)\s*%',
                rf'{letter}\)?\s+(\d+)\s*%',
            ]
            for pat in patterns:
                m = re.search(pat, section, re.IGNORECASE)
                if m:
                    pcts[letter] = int(m.group(1))
                    break
        
        if len(pcts) == 4:
            results[level] = pcts
            continue

        # Fallback: markdown table row
        # | Below Basic | 30% | 25% | 15% | 30% |
        # Headers should be A B C D - find them first
        # Or: | level | num | num | num | num |
        for variant in level_variants:
            table_pat = re.compile(
                rf'{re.escape(variant)}[^|]*\|\s*(\d+)\s*%?\s*\|\s*(\d+)\s*%?\s*\|\s*(\d+)\s*%?\s*\|\s*(\d+)\s*%?',
                re.IGNORECASE
            )
            m = table_pat.search(text)
            if m:
                results[level] = {
                    "A": int(m.group(1)), "B": int(m.group(2)),
                    "C": int(m.group(3)), "D": int(m.group(4)),
                }
                break

    return results if len(results) == 4 else None


def compute_weighted_p_incorrect(parsed, correct_answer):
    total = 0.0
    for level, weight in WEIGHTS.items():
        if level not in parsed:
            return None
        p_correct = parsed[level].get(correct_answer, 0) / 100.0
        total += weight * (1.0 - p_correct)
    return total


def process_config(config_dir, qids, correct, b_2pl):
    reps = sorted([d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith("rep")])
    if not reps:
        return None

    strict_vals = {}
    lenient_vals = {}

    for qid in qids:
        strict_pis = []
        lenient_pis = []
        for rep_dir in reps:
            fpath = rep_dir / f"qid{qid}.txt"
            if not fpath.exists():
                continue
            text = fpath.read_text(errors="replace")

            s = parse_strict(text)
            if s:
                pi = compute_weighted_p_incorrect(s, correct[qid])
                if pi is not None:
                    strict_pis.append(pi)

            l = parse_lenient(text)
            if l:
                pi = compute_weighted_p_incorrect(l, correct[qid])
                if pi is not None:
                    lenient_pis.append(pi)

        if strict_pis:
            strict_vals[qid] = np.mean(strict_pis)
        if lenient_pis:
            lenient_vals[qid] = np.mean(lenient_pis)

    def compute_rho(vals_dict):
        shared = [q for q in vals_dict if q in b_2pl]
        if len(shared) < 4:
            return len(shared), None
        x = [b_2pl[q] for q in shared]
        y = [vals_dict[q] for q in shared]
        rho, _ = spearmanr(x, y)
        return len(shared), rho

    n_strict, rho_strict = compute_rho(strict_vals)
    n_lenient, rho_lenient = compute_rho(lenient_vals)

    return {
        "n_strict": n_strict,
        "n_lenient": n_lenient,
        "rho_strict": rho_strict,
        "rho_lenient": rho_lenient,
    }


def main():
    qids, correct, b_2pl = load_ground_truth()
    print(f"Loaded {len(qids)} probe items, {len(b_2pl)} with IRT b_2pl\n")

    configs = []
    if SWEEP_DIR.exists():
        for d in sorted(SWEEP_DIR.iterdir()):
            if d.is_dir():
                configs.append(("sweep", d.name, d))
    if CROSS_DIR.exists():
        for d in sorted(CROSS_DIR.iterdir()):
            if d.is_dir():
                configs.append(("cross", d.name, d))

    print(f"{'type':<7} {'config':<45} {'n_str':>5} {'n_len':>5} {'rho_str':>8} {'rho_len':>8} {'delta':>7}")
    print("-" * 93)

    for ctype, name, path in configs:
        result = process_config(path, qids, correct, b_2pl)
        if result is None:
            continue
        rs = result["rho_strict"]
        rl = result["rho_lenient"]
        rs_s = f"{rs:.3f}" if rs is not None else "  N/A"
        rl_s = f"{rl:.3f}" if rl is not None else "  N/A"
        if rs is not None and rl is not None:
            delta = f"{rl - rs:+.3f}"
        else:
            delta = "  N/A"
        print(f"{ctype:<7} {name:<45} {result['n_strict']:>5} {result['n_lenient']:>5} {rs_s:>8} {rl_s:>8} {delta:>7}")


if __name__ == "__main__":
    main()
