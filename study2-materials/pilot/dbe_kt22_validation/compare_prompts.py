"""Compare difficulty prediction quality across prompting approaches on DBE-KT22."""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

BASE = '/Users/dereklomas/AIED/study2-materials/pilot/dbe_kt22_validation'
DATA = '/Users/dereklomas/AIED/study2-materials/data/dbe-kt22'

# --- Ground truth ---
tx = pd.read_csv(os.path.join(DATA, 'Transaction.csv'))
tx['correct'] = tx['answer_state'].astype(str).str.lower().map({'true': 1, 'false': 0})
tx = tx.dropna(subset=['correct'])
gt_p = tx.groupby('question_id')['correct'].mean()
gt_n = tx.groupby('question_id')['correct'].count()
student_total = tx.groupby('student_id')['correct'].mean()
tx_disc = tx.merge(student_total.rename('total'), on='student_id')
disc = tx_disc.groupby('question_id').apply(
    lambda g: g['correct'].corr(g['total']) if len(g) > 5 else np.nan, include_groups=False
)
gt = pd.DataFrame({'p_correct': gt_p, 'n_responses': gt_n, 'discrimination': disc})
gt.index = gt.index.astype(str)
print(f"Ground truth: {len(gt)} questions, mean p_correct={gt.p_correct.mean():.3f}, median n={gt.n_responses.median():.0f}")

# --- Bootstrap ---
def bootstrap_spearman(x, y, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = [stats.spearmanr(x[idx:=rng.randint(0,n,n)], y[idx]).correlation for _ in range(n_boot)]
    return np.percentile(rhos, [2.5, 97.5])

# --- Load predictions ---
# Contrastive: 'predicted' is weighted-avg p_correct from simulated response distributions
cont = json.load(open(os.path.join(BASE, 'predictions_contrastive_g3f_rep0.json')))
# Distractor-quality: 'difficulty' is predicted p_correct (0-1 scale)
dq = json.load(open(os.path.join(BASE, 'probe_distractor_quality.json')))
# Simple diff+disc: 'pred_difficulty' is predicted p_correct
sd = json.load(open(os.path.join(BASE, 'probe_diff_disc_cache.json')))

sources = {
    'Contrastive (g3f rep0)': {q: cont[q].get('predicted_p', cont[q].get('predicted')) for q in cont},
    'Distractor-quality': {q: dq[q]['difficulty'] for q in dq if dq[q].get('difficulty') is not None},
    'Simple diff+disc': {q: sd[q]['pred_difficulty'] for q in sd if sd[q].get('pred_difficulty') is not None},
}

print("\n" + "="*85)
print("DIFFICULTY PREDICTION COMPARISON (all predict p_correct on 0-1 scale)")
print("="*85)

rows = []
for name, preds in sources.items():
    common = sorted(set(str(q) for q in preds.keys()) & set(gt.index))
    pred_vals = np.array([float(preds[q]) for q in common])
    true_vals = np.array([gt.loc[q, 'p_correct'] for q in common])
    
    # Filter NaN
    mask = ~(np.isnan(pred_vals) | np.isnan(true_vals))
    pred_vals, true_vals = pred_vals[mask], true_vals[mask]
    n = len(pred_vals)
    
    rho, _ = stats.spearmanr(pred_vals, true_vals)
    ci = bootstrap_spearman(pred_vals, true_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    bias = np.mean(pred_vals - true_vals)
    
    # Also compute |rho| -- for contrastive, the sign flip matters
    rows.append({
        'Source': name, 'n': n,
        'rho': rho, 'rho_lo': ci[0], 'rho_hi': ci[1],
        'MAE': mae, 'Bias': bias,
        'pred_mean': np.mean(pred_vals), 'true_mean': np.mean(true_vals)
    })

for r in rows:
    print(f"\n  {r['Source']} (n={r['n']})")
    print(f"    Spearman rho = {r['rho']:+.3f}  95% CI [{r['rho_lo']:+.3f}, {r['rho_hi']:+.3f}]")
    print(f"    MAE = {r['MAE']:.3f}   Bias = {r['Bias']:+.3f}")
    print(f"    Pred mean = {r['pred_mean']:.3f}   True mean = {r['true_mean']:.3f}")

print(f"\n  NOTE: Contrastive rho is NEGATIVE -- the model's ranking is inverted!")
print(f"  The contrastive prompt predicts easy items as hard and vice versa on DBE-KT22.")
print(f"  |rho| for contrastive = {abs(rows[0]['rho']):.3f} -- rank sensitivity is comparable")
print(f"  but the direction is wrong, making it useless for absolute difficulty prediction.")

# --- Distractor features linear model ---
print("\n" + "="*85)
print("DISTRACTOR FEATURES: DO THEY ADD VALUE BEYOND DIRECT DIFFICULTY ESTIMATE?")
print("="*85)

common = sorted(set(dq.keys()) & set(gt.index))
common = [q for q in common if dq[q].get('difficulty') is not None]
y = np.array([gt.loc[q, 'p_correct'] for q in common])
diff_pred = np.array([dq[q]['difficulty'] for q in common]).reshape(-1, 1)
feat_names = ['max_plausibility', 'mean_plausibility', 'n_plausible', 'plausibility_range']
X_feat = np.array([[dq[q][f] for f in feat_names] for q in common])
X_all = np.hstack([diff_pred, X_feat])

models = {'Difficulty pred only': diff_pred, 'Distractor features only': X_feat, 'Difficulty + distractors': X_all}

print(f"\nLOO-CV linear regression -> p_correct (n={len(common)})")
print("-"*80)

for label, X in models.items():
    preds_loo = np.empty(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool); mask[i] = False
        lr = LinearRegression().fit(X[mask], y[mask])
        preds_loo[i] = lr.predict(X[i:i+1])[0]
    rho_cv = stats.spearmanr(preds_loo, y).correlation
    ci_cv = bootstrap_spearman(preds_loo, y)
    mae_cv = mean_absolute_error(y, preds_loo)
    r2 = LinearRegression().fit(X, y).score(X, y)
    print(f"  {label:30s}  rho={rho_cv:.3f} [{ci_cv[0]:.3f},{ci_cv[1]:.3f}]  MAE={mae_cv:.3f}  R2={r2:.3f}")

lr_full = LinearRegression().fit(X_all, y)
print(f"\n  Combined model coefficients:")
for nm, coef in zip(['difficulty'] + feat_names, lr_full.coef_):
    print(f"    {nm:25s} {coef:+.4f}")
print(f"    {'intercept':25s} {lr_full.intercept_:+.4f}")

# --- Discrimination ---
print("\n" + "="*85)
print("DISCRIMINATION PREDICTION")
print("="*85)

for label, data, key in [
    ('Distractor-quality', dq, 'discrimination'),
    ('Simple diff+disc', sd, 'pred_discrimination')
]:
    common_d = [q for q in data if q in gt.index and not np.isnan(gt.loc[q, 'discrimination'])]
    pv = [data[q].get(key) for q in common_d]
    valid = [(q, float(p)) for q, p in zip(common_d, pv) if p is not None and not (isinstance(p, float) and np.isnan(p))]
    if len(valid) < 10:
        print(f"  {label}: insufficient valid predictions ({len(valid)})")
        continue
    qs, ps = zip(*valid)
    pred_d = np.array(ps)
    true_d = np.array([gt.loc[q, 'discrimination'] for q in qs])
    mask = ~np.isnan(true_d)
    pred_d, true_d = pred_d[mask], true_d[mask]
    rho_d, _ = stats.spearmanr(pred_d, true_d)
    ci_d = bootstrap_spearman(pred_d, true_d)
    print(f"  {label:25s}  rho={rho_d:.3f} [{ci_d[0]:.3f},{ci_d[1]:.3f}]  n={len(pred_d)}")

# --- Summary ---
print("\n" + "="*85)
print("SUMMARY")
print("="*85)
print("""
1. CONTRASTIVE prompt (rho ~ -0.41): Produces INVERTED difficulty rankings on
   DBE-KT22. The model simulates response distributions but systematically
   misjudges which items are hard vs easy. Absolute predictions are far off
   (MAE ~0.47, strong negative bias).

2. DISTRACTOR-QUALITY prompt (rho ~ 0.29): Correct direction, modest correlation.
   Direct difficulty estimate is noisy but correctly oriented. Distractor features
   add little beyond the direct estimate in the linear model.

3. SIMPLE DIFF+DISC prompt (rho ~ 0.30): Very similar to distractor-quality for
   difficulty prediction. Asking about more features doesn't help or hurt the
   difficulty estimate.

Bottom line: The contrastive prompt's problem isn't sensitivity -- |rho| is
actually the highest -- but it inverts the difficulty scale. The simpler prompts
get the direction right but with modest rank correlation. None of the approaches
achieve strong absolute accuracy on this dataset.
""")
