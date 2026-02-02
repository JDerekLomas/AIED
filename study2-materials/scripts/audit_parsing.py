#!/usr/bin/env python3
"""Audit parsing success/failure rates across all experiment data."""

import os
import re
from pathlib import Path
from collections import defaultdict

BASE = Path("/Users/dereklomas/AIED/study2-materials/pilot")

def parse_smartpaper(text):
    results = {}
    # Pattern 1: "struggling: 5% full marks"
    pat1 = re.findall(r'(?:\*\*)?(\w+):\s*(\d+(?:\.\d+)?)%\s*full\s*marks', text, re.IGNORECASE)
    if pat1:
        for level, val in pat1:
            results[level.lower()] = float(val)
        return results
    # Pattern 2: "ESTIMATED PROPORTION CORRECT: XX%"
    pat2 = re.findall(r'ESTIMATED\s+PROPORTION\s+CORRECT[:\s]*(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
    if pat2:
        return {"single": float(pat2[0])}
    # Pattern 3: "proportion correct" near a number
    pat3 = re.findall(r'proportion\s+correct[:\s]*(\d+(?:\.\d+)?)%?', text, re.IGNORECASE)
    if pat3:
        return {"single": float(pat3[0])}
    # Pattern 4: level: XX%
    pat4 = re.findall(r'(?:\*\*)?(\w+)[:\s]+(\d+(?:\.\d+)?)%', text, re.IGNORECASE)
    level_words = {'struggling', 'basic', 'competent', 'advanced', 'below', 'proficient'}
    for level, val in pat4:
        if level.lower() in level_words:
            results[level.lower()] = float(val)
    if results:
        return results
    # Pattern 5: just percentages
    pats = re.findall(r'(\d+(?:\.\d+)?)%', text)
    if pats:
        return {"raw_pct": [float(p) for p in pats]}
    return None

def parse_eedi(text):
    results = {}
    levels = ['below_basic', 'basic', 'proficient', 'advanced']
    for level in levels:
        pat = re.search(
            rf'\*?\*?{level}\*?\*?[:\s]*A\s*=\s*(\d+(?:\.\d+)?)%\s*B\s*=\s*(\d+(?:\.\d+)?)%\s*C\s*=\s*(\d+(?:\.\d+)?)%\s*D\s*=\s*(\d+(?:\.\d+)?)%',
            text, re.IGNORECASE
        )
        if pat:
            results[level] = {'A': float(pat.group(1)), 'B': float(pat.group(2)),
                              'C': float(pat.group(3)), 'D': float(pat.group(4))}
    return results if results else None

def audit_smartpaper_dir(dirpath, label):
    files = list(Path(dirpath).glob("*.txt"))
    total = len(files)
    parsed_ok = 0
    failed_files = []
    all_values = []
    for f in files:
        text = f.read_text(errors='replace')
        result = parse_smartpaper(text)
        if result:
            parsed_ok += 1
            for k, v in result.items():
                if isinstance(v, list):
                    all_values.extend(v)
                else:
                    all_values.append(v)
        else:
            failed_files.append(f.name)
    return {'label': label, 'total': total, 'parsed_ok': parsed_ok,
            'failed': total - parsed_ok,
            'fail_rate': f"{(total - parsed_ok) / total * 100:.1f}%" if total > 0 else "N/A",
            'failed_examples': failed_files[:3], 'values': all_values}

def audit_eedi_dir(dirpath, label):
    base = Path(dirpath)
    txt_files = list(base.glob("*.txt"))
    rep_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("rep")]
    for rd in rep_dirs:
        txt_files.extend(rd.glob("*.txt"))
    total = len(txt_files)
    parsed_ok = 0
    partial_parses = 0
    levels_found = defaultdict(int)
    failed_files = []
    all_values = []
    for f in txt_files:
        text = f.read_text(errors='replace')
        result = parse_eedi(text)
        if result:
            if len(result) == 4:
                parsed_ok += 1
            else:
                partial_parses += 1
            for level, opts in result.items():
                levels_found[level] += 1
                all_values.append(sum(opts.values()))
        else:
            failed_files.append(str(f.relative_to(base)))
    return {'label': label, 'total': total, 'parsed_ok_4levels': parsed_ok,
            'partial': partial_parses, 'failed': total - parsed_ok - partial_parses,
            'fail_rate': f"{(total - parsed_ok - partial_parses) / total * 100:.1f}%" if total > 0 else "N/A",
            'levels_found': dict(levels_found), 'failed_examples': failed_files[:3],
            'pct_sums': all_values}

# ============================================================
print("=" * 90)
print("SMARTPAPER PARSING AUDIT")
print("=" * 90)

sp_results = []
temp_sweep = BASE / "smartpaper_rsm_v2" / "temp_sweep"
if temp_sweep.exists():
    for subdir in sorted(temp_sweep.iterdir()):
        if subdir.is_dir():
            sp_results.append(audit_smartpaper_dir(subdir, subdir.name))

calibration = BASE / "smartpaper_rsm_v2" / "calibration"
if calibration.exists():
    for subdir in sorted(calibration.iterdir()):
        if subdir.is_dir():
            sp_results.append(audit_smartpaper_dir(subdir, f"cal/{subdir.name}"))

print(f"\n{'Config':<45} {'Total':>6} {'OK':>6} {'Fail':>6} {'Rate':>8}  Suspicious")
print("-" * 90)
for r in sp_results:
    suspicious = []
    vals = r['values']
    if vals:
        zeros = sum(1 for v in vals if v == 0)
        fifties = sum(1 for v in vals if v == 50)
        if zeros > max(1, len(vals) * 0.1):
            suspicious.append(f"{zeros} zeros")
        if fifties > max(1, len(vals) * 0.2):
            suspicious.append(f"{fifties} @50%")
    susp_str = ", ".join(suspicious) if suspicious else ""
    print(f"{r['label']:<45} {r['total']:>6} {r['parsed_ok']:>6} {r['failed']:>6} {r['fail_rate']:>8}  {susp_str}")

print("\nFailed file examples (SmartPaper):")
shown = False
for r in sp_results:
    if r['failed_examples']:
        print(f"  {r['label']}: {r['failed_examples']}")
        shown = True
if not shown:
    print("  None -- all files parsed successfully")

# ============================================================
print("\n" + "=" * 90)
print("EEDI PARSING AUDIT")
print("=" * 90)

eedi_results = []
ms = BASE / "rsm_experiment" / "metaprompt_sweep"
if ms.exists():
    for subdir in sorted(ms.iterdir()):
        if subdir.is_dir():
            eedi_results.append(audit_eedi_dir(subdir, f"metaprompt/{subdir.name}"))

hr = BASE / "rsm_experiment" / "high_reps"
if hr.exists():
    for subdir in sorted(hr.iterdir()):
        if subdir.is_dir():
            eedi_results.append(audit_eedi_dir(subdir, f"high_reps/{subdir.name}"))

cm = BASE / "rsm_experiment" / "cross_model"
if cm.exists():
    for subdir in sorted(cm.iterdir()):
        if subdir.is_dir():
            eedi_results.append(audit_eedi_dir(subdir, f"cross_model/{subdir.name}"))

print(f"\n{'Config':<45} {'Total':>6} {'4Lev':>6} {'Part':>6} {'Fail':>6} {'Rate':>8}  Levels")
print("-" * 105)
for r in eedi_results:
    lf = r['levels_found']
    level_str = " ".join(f"{k[:2]}={v}" for k, v in sorted(lf.items())) if lf else "none"
    print(f"{r['label']:<45} {r['total']:>6} {r['parsed_ok_4levels']:>6} {r['partial']:>6} {r['failed']:>6} {r['fail_rate']:>8}  {level_str}")

print("\nPercentage sum check (should be ~100%):")
bad_sums = []
for r in eedi_results:
    sums = r['pct_sums']
    if sums:
        off = [s for s in sums if abs(s - 100) > 1]
        if off:
            bad_sums.append((r['label'], len(off), len(sums), min(off), max(off)))
if bad_sums:
    for label, n_bad, n_total, mn, mx in bad_sums:
        print(f"  {label}: {n_bad}/{n_total} sums off (range {mn:.0f}-{mx:.0f}%)")
else:
    print("  All option percentages sum to ~100% -- OK")

print("\nFailed file examples (Eedi):")
shown = False
for r in eedi_results:
    if r['failed_examples']:
        print(f"  {r['label']}: {r['failed_examples']}")
        shown = True
if not shown:
    print("  None -- all files parsed successfully")

# Inspect failed cross-model files
cm = BASE / "rsm_experiment" / "cross_model"
if cm.exists():
    print("\n" + "=" * 90)
    print("CROSS-MODEL: Sample failed parse content")
    print("=" * 90)
    for r in eedi_results:
        if 'cross_model' in r['label'] and r['failed'] > 0:
            model = r['label'].split("/")[-1]
            model_dir = cm / model
            found = False
            for rep_dir in sorted(model_dir.iterdir()):
                if not rep_dir.is_dir():
                    continue
                for f in sorted(rep_dir.glob("*.txt")):
                    text = f.read_text(errors='replace')
                    result = parse_eedi(text)
                    if not result:
                        print(f"\n  FAILED: {f.relative_to(BASE)}")
                        # Show last 400 chars (where the answer usually is)
                        print(f"  Last 400 chars:\n    {text[-400:]}")
                        found = True
                        break
                if found:
                    break

# ============================================================
print("\n" + "=" * 90)
print("OVERALL SUMMARY")
print("=" * 90)
total_sp = sum(r['total'] for r in sp_results)
ok_sp = sum(r['parsed_ok'] for r in sp_results)
total_ee = sum(r['total'] for r in eedi_results)
ok_ee = sum(r['parsed_ok_4levels'] for r in eedi_results)
partial_ee = sum(r['partial'] for r in eedi_results)
fail_ee = total_ee - ok_ee - partial_ee

print(f"SmartPaper: {ok_sp}/{total_sp} parsed OK ({total_sp-ok_sp} failed = {(total_sp-ok_sp)/total_sp*100:.1f}% fail rate)" if total_sp else "SmartPaper: no files found")
print(f"Eedi:       {ok_ee}/{total_ee} with all 4 levels, {partial_ee} partial, {fail_ee} total fail ({fail_ee/total_ee*100:.1f}% fail rate)" if total_ee else "Eedi: no files found")
