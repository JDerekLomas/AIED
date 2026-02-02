"""
Investigate why the contrastive prompt produces INVERTED difficulty predictions
on DBE-KT22 (rho ~ -0.409).
"""
import json, re, os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir("/Users/dereklomas/AIED/study2-materials")

contrastive = json.load(open("pilot/dbe_kt22_validation/predictions_contrastive_g3f_rep0.json"))
distractor = json.load(open("pilot/dbe_kt22_validation/probe_distractor_quality.json"))
trans = pd.read_csv("data/dbe-kt22/Transaction.csv")
questions = pd.read_csv("data/dbe-kt22/Questions.csv")
choices = pd.read_csv("data/dbe-kt22/Question_Choices.csv")

p_correct = trans.groupby("question_id")["answer_state"].mean().rename("p_correct")
n_responses = trans.groupby("question_id")["answer_state"].count().rename("n_responses")

q_text = {}
for _, row in questions.iterrows():
    txt = row["question_text"] if pd.notna(row["question_text"]) else row["question_rich_text"]
    q_text[str(row["id"])] = str(txt) if pd.notna(txt) else ""

choice_text = {}
for qid, grp in choices.groupby("question_id"):
    opts = {}
    for _, r in grp.iterrows():
        label = chr(65 + len(opts))
        opts[label] = {"text": r["choice_text"], "correct": r["is_correct"]}
    choice_text[str(qid)] = opts

rows = []
for qid in contrastive:
    qid_int = int(qid)
    if qid_int in p_correct.index:
        rows.append({
            "qid": qid,
            "p_actual": p_correct[qid_int],
            "n": n_responses.get(qid_int, 0),
            "p_contrastive": contrastive[qid].get("predicted", np.nan),
            "p_distractor": distractor[qid].get("difficulty", np.nan) if qid in distractor else np.nan,
            "raw_contrastive": contrastive[qid].get("raw", ""),
            "raw_distractor": distractor[qid].get("raw", "") if qid in distractor else "",
            "q_text": q_text.get(qid, "")[:300],
        })

df = pd.DataFrame(rows).dropna(subset=["p_contrastive", "p_distractor"])
df["inversion_gap"] = df["p_actual"] - df["p_contrastive"]
df["abs_error_contr"] = (df["p_actual"] - df["p_contrastive"]).abs()
df["abs_error_distr"] = (df["p_actual"] - df["p_distractor"]).abs()

print(f"Items with all three measures: {len(df)}")
print()

# A ========================================================================
print("=" * 80)
print("A. DISTRIBUTION COMPARISON")
print("=" * 80)

for name, col in [("Actual p_correct", "p_actual"),
                   ("Contrastive pred", "p_contrastive"),
                   ("Distractor pred",  "p_distractor")]:
    s = df[col]
    print(f"\n  {name}:")
    print(f"    mean={s.mean():.3f}  sd={s.std():.3f}  skew={s.skew():.3f}")
    print(f"    min={s.min():.3f}  Q1={s.quantile(.25):.3f}  med={s.median():.3f}  Q3={s.quantile(.75):.3f}  max={s.max():.3f}")

rho_c, p_c = stats.spearmanr(df["p_actual"], df["p_contrastive"])
rho_d, p_d = stats.spearmanr(df["p_actual"], df["p_distractor"])
print(f"\n  Spearman correlations with actual:")
print(f"    Contrastive:  rho={rho_c:.3f}  p={p_c:.4f}")
print(f"    Distractor:   rho={rho_d:.3f}  p={p_d:.4f}")
print(f"\n  KEY: Contrastive mean ({df['p_contrastive'].mean():.3f}) is {df['p_actual'].mean() - df['p_contrastive'].mean():.3f} below actual ({df['p_actual'].mean():.3f})")

# B ========================================================================
print("\n" + "=" * 80)
print("B. RAW RESPONSE ANALYSIS -- 5 worst inversions + 5 least inverted")
print("=" * 80)

df_sorted = df.sort_values("inversion_gap", ascending=False)

print("\n-- TOP 5 WORST INVERSIONS (easy items predicted as hard) --")
for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
    print(f"\n  [{i+1}] QID {row['qid']}  actual={row['p_actual']:.3f}  contrastive={row['p_contrastive']:.3f}  gap={row['inversion_gap']:+.3f}")
    print(f"  Question: {row['q_text'][:150]}...")
    print(f"  RAW (first 500 chars):")
    for line in row['raw_contrastive'][:500].split('\n'):
        print(f"    {line}")

print("\n-- 5 LEAST INVERTED (or correctly ordered) --")
for i, (_, row) in enumerate(df_sorted.tail(5).iterrows()):
    print(f"\n  [{i+1}] QID {row['qid']}  actual={row['p_actual']:.3f}  contrastive={row['p_contrastive']:.3f}  gap={row['inversion_gap']:+.3f}")
    print(f"  Question: {row['q_text'][:150]}...")
    print(f"  RAW (first 500 chars):")
    for line in row['raw_contrastive'][:500].split('\n'):
        print(f"    {line}")

# C ========================================================================
print("\n" + "=" * 80)
print("C. 10 MOST INVERTED ITEMS")
print("=" * 80)

for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
    print(f"\n  [{i+1}] QID {row['qid']}")
    print(f"    Actual p_correct:  {row['p_actual']:.3f}  (n={row['n']:.0f})")
    print(f"    Contrastive pred:  {row['p_contrastive']:.3f}  (error={row['inversion_gap']:+.3f})")
    print(f"    Distractor pred:   {row['p_distractor']:.3f}  (error={row['p_actual'] - row['p_distractor']:+.3f})")
    print(f"    Question: {row['q_text'][:200]}")

# D ========================================================================
print("\n" + "=" * 80)
print("D. PREDICTIONS BY ACTUAL DIFFICULTY QUARTILE")
print("=" * 80)

df["quartile"] = pd.qcut(df["p_actual"], 4, labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"])
qt = df.groupby("quartile", observed=True).agg(
    n=("qid", "count"),
    actual_mean=("p_actual", "mean"),
    contrastive_mean=("p_contrastive", "mean"),
    distractor_mean=("p_distractor", "mean"),
    contrastive_err=("inversion_gap", "mean"),
).round(3)
print(f"\n{qt.to_string()}")

q_means = df.groupby("quartile", observed=True)["p_contrastive"].mean()
print(f"\n  Contrastive across quartiles: {[f'{v:.3f}' for v in q_means.values]}")
if q_means.iloc[0] > q_means.iloc[-1]:
    print("  --> INVERTED: predicts hardest quartile as EASIEST")
else:
    print("  --> Follows correct direction but wrong magnitude")

# E ========================================================================
print("\n" + "=" * 80)
print("E. PROMPT FRAMING HYPOTHESIS")
print("=" * 80)

print(f"\n  Variance: actual={df['p_actual'].var():.4f}  contrastive={df['p_contrastive'].var():.4f}  distractor={df['p_distractor'].var():.4f}")
print(f"  Median:   actual={df['p_actual'].median():.3f}  contrastive={df['p_contrastive'].median():.3f}  distractor={df['p_distractor'].median():.3f}")

below_50 = (df["p_contrastive"] < 0.5).mean()
print(f"\n  Contrastive preds < 0.5: {below_50:.1%}   Actual < 0.5: {(df['p_actual'] < 0.5).mean():.1%}")

difficulty_words = ["hard", "difficult", "challenging", "tricky", "complex", "confus", "struggle"]
ease_words = ["easy", "simple", "straightforward", "basic", "trivial"]
diff_counts = sum(1 for r in df["raw_contrastive"] if any(w in r.lower() for w in difficulty_words))
ease_counts = sum(1 for r in df["raw_contrastive"] if any(w in r.lower() for w in ease_words))
print(f"\n  Difficulty language in responses: {diff_counts}/{len(df)} ({diff_counts/len(df):.0%})")
print(f"  Ease language in responses:       {ease_counts}/{len(df)} ({ease_counts/len(df):.0%})")

# Parse per-level correct-answer rates from contrastive responses
print("\n  Per-level correct-answer rates from contrastive prompt:")
correct_by_level = {level: [] for level in ["struggling", "average", "good", "advanced"]}
level_series = {level: [] for level in correct_by_level}

for _, row in df.iterrows():
    qid = row["qid"]
    raw = row["raw_contrastive"]
    correct_label = None
    if qid in choice_text:
        for lbl, info in choice_text[qid].items():
            if info["correct"]:
                correct_label = lbl
                break
    if not correct_label and qid in distractor and "correct_label" in distractor[qid]:
        correct_label = distractor[qid]["correct_label"]
    
    for level in correct_by_level:
        val = np.nan
        if correct_label:
            m = re.search(rf"{level}:\s*(.*)", raw, re.IGNORECASE)
            if m:
                m2 = re.search(rf"{correct_label}\s*=\s*(\d+)%", m.group(1))
                if m2:
                    val = int(m2.group(1)) / 100
                    correct_by_level[level].append(val)
        level_series[level].append(val)

for level, vals in correct_by_level.items():
    if vals:
        print(f"    {level:12s}: mean={np.mean(vals):.3f}  sd={np.std(vals):.3f}  n={len(vals)}")

if all(len(v) > 0 for v in correct_by_level.values()):
    lm = np.array([np.mean(correct_by_level[l]) for l in ["struggling", "average", "good", "advanced"]])
    print(f"\n  Equal-weight average of 4 levels: {lm.mean():.3f}")
    print(f"  Actual contrastive pred mean:     {df['p_contrastive'].mean():.3f}")
    rw = np.array([0.10, 0.40, 0.35, 0.15])
    print(f"  Realistic-weight (10/40/35/15):   {np.dot(lm, rw):.3f}")
    print(f"  Using only 'good' students:       {lm[2]:.3f}")
    print(f"  Using only 'average' students:    {lm[1]:.3f}")

# Per-level correlations
print("\n  Per-level Spearman correlations with actual p_correct:")
for level in ["struggling", "average", "good", "advanced"]:
    s = pd.Series(level_series[level])
    valid = s.notna() & df["p_actual"].notna()
    if valid.sum() > 10:
        r, p = stats.spearmanr(df.loc[valid.values, "p_actual"], s[valid.values])
        print(f"    {level:12s}: rho={r:.3f}  p={p:.4f}  (n={valid.sum()})")

# F ========================================================================
print("\n" + "=" * 80)
print("F. DOMAIN MISMATCH -- TECHNICAL JARGON")
print("=" * 80)

sql_terms = ["SELECT", "INSERT", "UPDATE", "DELETE", "JOIN", "WHERE", "FROM", "GROUP BY",
             "ORDER BY", "HAVING", "CREATE", "ALTER", "DROP", "INDEX", "COMMIT", "ROLLBACK",
             "TRANSACTION", "LOCK", "GRANT", "REVOKE", "VIEW", "TRIGGER", "PROCEDURE",
             "SQL", "TABLE", "PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "SCHEMA",
             "NORMALIZE", "NORMAL FORM", "BCNF", "3NF", "2NF", "1NF",
             "ER ", "ENTITY", "RELATION", "ATTRIBUTE", "TUPLE", "B-TREE", "HASH"]

df["tech_count"] = df["q_text"].apply(lambda t: sum(1 for term in sql_terms if term in t.upper()))
df["high_tech"] = df["tech_count"] >= df["tech_count"].median()

for label, mask in [("Low-tech items", ~df["high_tech"]), ("High-tech items", df["high_tech"])]:
    sub = df[mask]
    if len(sub) > 5:
        r, p = stats.spearmanr(sub["p_actual"], sub["p_contrastive"])
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    Mean inversion gap: {sub['inversion_gap'].mean():+.3f}")
        print(f"    Spearman rho:       {r:.3f}  (p={p:.4f})")
        print(f"    Mean actual:        {sub['p_actual'].mean():.3f}   Mean contrastive: {sub['p_contrastive'].mean():.3f}")

# SYNTHESIS ================================================================
print("\n" + "=" * 80)
print("SYNTHESIS: WHY DOES THE CONTRASTIVE PROMPT INVERT?")
print("=" * 80)
print("""
The contrastive prompt predicts response distributions for 4 proficiency levels
(struggling, average, good, advanced) and averages them equally.

MECHANISM 1 -- POPULATION WEIGHTING:
  Equal-weighting 4 levels gives 25% weight to "struggling" students who the
  model predicts will get almost everything wrong. This drags the average down,
  making all items look harder than they are for the actual (self-selected,
  prepared) university population.

MECHANISM 2 -- DIFFICULTY PRIMING:
  The contrastive frame ("what makes this SPECIFIC question easy/hard?") primes
  the model to generate difficulty narratives. For EVERY item, including easy
  ones, the model finds plausible reasons students could fail (edge cases,
  terminology confusion, etc.), lowering its predictions.

MECHANISM 3 -- INVERSE RELATIONSHIP:
  Items that are actually EASY are often standard, well-taught concepts. The
  model sees standard concepts and generates rich misconception narratives
  (because textbook concepts have well-known misconceptions). Items that are
  actually HARD may be novel/unusual, giving the model less to say about
  specific misconceptions -- paradoxically predicting higher success.

MECHANISM 4 -- DOMAIN ANCHOR MISMATCH:
  The model's "struggling student" imagines someone unfamiliar with databases,
  not a CS undergrad who attended lectures. This population mismatch is more
  severe for standard DB concepts (which real students learned) than for
  genuinely tricky items.
""")
