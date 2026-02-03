"""Analyze SmartPaper RSM v2 temperature sweep results for DeepSeek and Gemini configs."""
import json, os, re, glob
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import numpy as np

BASE = "/Users/dereklomas/AIED/study2-materials"
SWEEP_DIR = os.path.join(BASE, "pilot/smartpaper_rsm_v2/temp_sweep")

with open(os.path.join(BASE, "data/smartpaper/item_statistics.json")) as f:
    all_items = json.load(f)

with open(os.path.join(BASE, "pilot/smartpaper_rsm_v2/probe_items.json")) as f:
    probe_items = json.load(f)

gt = {}
for item in all_items:
    key = f"{item['assessment']}_q{item['question_number']}"
    gt[key] = item['classical_difficulty']

probe_keys = [f"{item['assessment']}_q{item['question_number']}" for item in probe_items]
print(f"Probe items: {len(probe_keys)}, Ground truth available: {sum(1 for k in probe_keys if k in gt)}")


def extract_proportion(text):
    text_lower = text.lower()
    
    m = re.search(r'estimated\s+proportion\s+correct[:\s]*(\d+(?:\.\d+)?)\s*%', text_lower)
    if m:
        return float(m.group(1)) / 100.0
    
    m = re.search(r'proportion\s+correct[:\s]*(?:approximately\s+|about\s+|~\s*)?(\d+(?:\.\d+)?)\s*%', text_lower)
    if m:
        return float(m.group(1)) / 100.0
    
    m = re.search(r'proportion\s+correct[:\s]*(?:approximately\s+|about\s+|~\s*)?0?\.(\d+)', text_lower)
    if m:
        return float(f"0.{m.group(1)}")
    
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        for kw in ['overall', 'estimate', 'proportion', 'expect', 'predict', 'approximately', 'around', 'about']:
            for m2 in re.finditer(rf'{kw}[^%]*?(\d+(?:\.\d+)?)\s*%', text_lower):
                val = float(m2.group(1))
                if 0 <= val <= 100:
                    return val / 100.0
        val = float(matches[-1])
        if 0 <= val <= 100:
            return val / 100.0
    
    matches = re.findall(r'\b(0\.\d+)\b', text)
    if matches:
        return float(matches[-1])
    
    return None


def analyze_config(config_dir):
    files = glob.glob(os.path.join(config_dir, "*.txt"))
    if len(files) != 100:
        return None
    
    estimates = defaultdict(list)
    parse_failures = 0
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(r'(.+)_rep(\d+)\.txt$', fname)
        if not m:
            continue
        item_key = m.group(1)
        
        with open(fpath) as f:
            text = f.read()
        
        val = extract_proportion(text)
        if val is not None:
            estimates[item_key].append(val)
        else:
            parse_failures += 1
    
    if parse_failures > 0:
        print(f"  Note: {parse_failures} parse failures in {os.path.basename(config_dir)}")
    
    means = {k: np.mean(v) for k, v in estimates.items()}
    return means


results = []
for config_name in sorted(os.listdir(SWEEP_DIR)):
    config_path = os.path.join(SWEEP_DIR, config_name)
    if not os.path.isdir(config_path):
        continue
    
    files = glob.glob(os.path.join(config_path, "*.txt"))
    if len(files) != 100:
        print(f"  Skipping {config_name}: {len(files)} files (need 100)")
        continue
    
    means = analyze_config(config_path)
    if means is None:
        continue
    
    gt_vals = []
    est_vals = []
    for key in probe_keys:
        if key in means and key in gt:
            gt_vals.append(gt[key])
            est_vals.append(means[key])
    
    if len(gt_vals) < 5:
        print(f"  Skipping {config_name}: only {len(gt_vals)} matched items")
        continue
    
    rho, rho_p = spearmanr(gt_vals, est_vals)
    r, r_p = pearsonr(gt_vals, est_vals)
    mae = np.mean(np.abs(np.array(gt_vals) - np.array(est_vals)))
    rmse = np.sqrt(np.mean((np.array(gt_vals) - np.array(est_vals))**2))
    
    if config_name.startswith("deepseek_chat_"):
        model = "DeepSeek"
    elif config_name.startswith("gemini"):
        model = "Gemini"
    else:
        model = "Gemini"  # the non-prefixed ones are gemini
    
    results.append({
        'config': config_name,
        'model': model,
        'n_items': len(gt_vals),
        'rho': rho,
        'rho_p': rho_p,
        'r': r,
        'r_p': r_p,
        'mae': mae,
        'rmse': rmse,
    })

results.sort(key=lambda x: x['rho'], reverse=True)

print(f"\n{'Config':<40} {'Model':<10} {'N':>3} {'Rho':>6} {'p(rho)':>8} {'r':>6} {'p(r)':>8} {'MAE':>6} {'RMSE':>6}")
print("-" * 100)
for r in results:
    sig_rho = "*" if r['rho_p'] < 0.05 else " "
    sig_r = "*" if r['r_p'] < 0.05 else " "
    print(f"{r['config']:<40} {r['model']:<10} {r['n_items']:>3} {r['rho']:>6.3f}{sig_rho} {r['rho_p']:>7.4f} {r['r']:>6.3f}{sig_r} {r['r_p']:>7.4f} {r['mae']:>6.3f} {r['rmse']:>6.3f}")

print("\n\nSummary by model:")
for model in ['DeepSeek', 'Gemini']:
    subset = [r2 for r2 in results if r2['model'] == model]
    if subset:
        avg_rho = np.mean([r2['rho'] for r2 in subset])
        avg_r = np.mean([r2['r'] for r2 in subset])
        avg_mae = np.mean([r2['mae'] for r2 in subset])
        best = max(subset, key=lambda x: x['rho'])
        print(f"  {model}: {len(subset)} configs, avg rho={avg_rho:.3f}, avg r={avg_r:.3f}, avg MAE={avg_mae:.3f}")
        print(f"    Best: {best['config']} (rho={best['rho']:.3f}, r={best['r']:.3f})")
