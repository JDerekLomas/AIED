"""
Probe: Ask Gemini 2.5 Flash to predict both difficulty (p-correct) and 
discrimination (IRT a-parameter) for 168 DBE-KT22 items.
"""

import json, os, time, sys
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import google.generativeai as genai

# --- Config ---
CACHE_FILE = Path(__file__).parent / "probe_diff_disc_cache.json"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dbe-kt22"
IRT_FILE = Path(__file__).parent / "irt_params.npz"

# --- Load data ---
items = json.load(open(DATA_DIR / "item_statistics.json"))
irt = np.load(IRT_FILE, allow_pickle=True)
irt_map = dict(zip(irt['item_ids'].tolist(), irt['a_2pl'].tolist()))

# Filter to items that have IRT discrimination
items_with_disc = []
for item in items:
    qid = item['question_id']
    if qid in irt_map:
        item['a_2pl'] = irt_map[qid]
        items_with_disc.append(item)

print(f"Items with both text and IRT discrimination: {len(items_with_disc)}")
print(f"Items in item_statistics.json: {len(items)}")
print(f"Items in IRT params: {len(irt_map)}")

# --- Build prompts ---
def make_prompt(item):
    opts = "\n".join(f"  {o['label']}) {o['text']}" for o in item['options'])
    correct = next(o['label'] for o in item['options'] if o['is_correct'])
    return f"""You are a psychometrician analyzing test items. For this multiple choice question, estimate TWO parameters:

1. DIFFICULTY: What proportion of undergraduate students would answer correctly? (0.0 to 1.0)
2. DISCRIMINATION: How well does this item distinguish strong from weak students? (0.0 = no discrimination, 1.0 = moderate, 2.0+ = high discrimination)

Question: {item['question_text']}
Options:
{opts}
Correct: {correct}

Respond in this exact format:
DIFFICULTY: 0.XX
DISCRIMINATION: X.XX"""


def parse_response(text):
    diff, disc = None, None
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('DIFFICULTY:'):
            try:
                diff = float(line.split(':')[1].strip())
            except:
                pass
        elif line.upper().startswith('DISCRIMINATION:'):
            try:
                disc = float(line.split(':')[1].strip())
            except:
                pass
    return diff, disc


# --- Load cache or run ---
if CACHE_FILE.exists():
    print("Loading cached predictions...")
    cache = json.load(open(CACHE_FILE))
else:
    cache = {}

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')

need_run = [it for it in items_with_disc if str(it['question_id']) not in cache]
print(f"Need to query: {len(need_run)} items")

for i, item in enumerate(need_run):
    qid = str(item['question_id'])
    prompt = make_prompt(item)
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0)
        )
        text = resp.text
        diff, disc = parse_response(text)
        cache[qid] = {'raw': text, 'pred_difficulty': diff, 'pred_discrimination': disc}
    except Exception as e:
        print(f"  Error qid={qid}: {e}")
        cache[qid] = {'raw': str(e), 'pred_difficulty': None, 'pred_discrimination': None}
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(need_run)} done")
        json.dump(cache, open(CACHE_FILE, 'w'), indent=2)
    
    time.sleep(0.1)  # rate limit

json.dump(cache, open(CACHE_FILE, 'w'), indent=2)
print(f"Cache saved: {len(cache)} items")

# --- Analyze ---
actual_diff, pred_diff = [], []
actual_disc, pred_disc = [], []
parse_fails = 0

for item in items_with_disc:
    qid = str(item['question_id'])
    if qid not in cache:
        continue
    c = cache[qid]
    if c['pred_difficulty'] is not None:
        actual_diff.append(item['p_correct'])
        pred_diff.append(c['pred_difficulty'])
    else:
        parse_fails += 1
    if c['pred_discrimination'] is not None:
        actual_disc.append(item['a_2pl'])
        pred_disc.append(c['pred_discrimination'])

actual_diff = np.array(actual_diff)
pred_diff = np.array(pred_diff)
actual_disc = np.array(actual_disc)
pred_disc = np.array(pred_disc)

print("\n" + "="*60)
print("PROBE RESULTS: Gemini 2.5 Flash - Difficulty & Discrimination")
print("="*60)

print(f"\nItems analyzed: {len(items_with_disc)}")
print(f"Parse failures: {parse_fails}")

# Difficulty
rho_d, p_d = spearmanr(actual_diff, pred_diff)
print(f"\n--- DIFFICULTY (p-correct) ---")
print(f"  N = {len(actual_diff)}")
print(f"  Actual:    mean={actual_diff.mean():.3f}  std={actual_diff.std():.3f}  range=[{actual_diff.min():.3f}, {actual_diff.max():.3f}]")
print(f"  Predicted: mean={pred_diff.mean():.3f}  std={pred_diff.std():.3f}  range=[{pred_diff.min():.3f}, {pred_diff.max():.3f}]")
print(f"  Spearman rho = {rho_d:.4f}  (p = {p_d:.2e})")
mae_d = np.abs(actual_diff - pred_diff).mean()
print(f"  MAE = {mae_d:.4f}")

# Discrimination
rho_disc, p_disc = spearmanr(actual_disc, pred_disc)
print(f"\n--- DISCRIMINATION (IRT 2PL a-parameter) ---")
print(f"  N = {len(actual_disc)}")
print(f"  Actual:    mean={actual_disc.mean():.3f}  std={actual_disc.std():.3f}  range=[{actual_disc.min():.3f}, {actual_disc.max():.3f}]")
print(f"  Predicted: mean={pred_disc.mean():.3f}  std={pred_disc.std():.3f}  range=[{pred_disc.min():.3f}, {pred_disc.max():.3f}]")
print(f"  Spearman rho = {rho_disc:.4f}  (p = {p_disc:.2e})")
mae_disc = np.abs(actual_disc - pred_disc).mean()
print(f"  MAE = {mae_disc:.4f}")

# Verdict
print(f"\n--- VERDICT ---")
if rho_disc > 0.3 and p_disc < 0.01:
    print("  Discrimination prediction: PROMISING (rho > 0.3, significant)")
elif rho_disc > 0.15 and p_disc < 0.05:
    print("  Discrimination prediction: WEAK SIGNAL (rho > 0.15, significant)")
else:
    print("  Discrimination prediction: NOT WORKING (rho too low or not significant)")

if rho_d > 0.5:
    print(f"  Difficulty prediction: GOOD (rho = {rho_d:.3f})")
elif rho_d > 0.3:
    print(f"  Difficulty prediction: MODERATE (rho = {rho_d:.3f})")
else:
    print(f"  Difficulty prediction: POOR (rho = {rho_d:.3f})")
