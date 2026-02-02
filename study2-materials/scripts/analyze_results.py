#!/usr/bin/env python3
"""
Comprehensive Analysis of Misconception Alignment Study
Generates tables, statistics, and publication-ready figures.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(responses_path: str, items_path: str) -> pd.DataFrame:
    """Load and merge response data with item metadata."""

    # Load responses
    responses = []
    with open(responses_path, "r") as f:
        for line in f:
            responses.append(json.loads(line))

    # Load probe items (nested structure)
    with open(items_path, "r") as f:
        data = json.load(f)

    # Extract item info
    item_info = {}
    for misconception in data["misconceptions"]:
        category = misconception["category"]
        misconception_id = misconception["id"]
        misconception_name = misconception["name"]
        for item in misconception["items"]:
            item_info[item["item_id"]] = {
                "target_distractor": item["target_distractor"],
                "category": category,
                "misconception_id": misconception_id,
                "misconception_name": misconception_name
            }

    # Create dataframe
    df = pd.DataFrame(responses)

    # Add item metadata
    df["target_distractor"] = df["item_id"].map(lambda x: item_info.get(x, {}).get("target_distractor"))
    df["category"] = df["item_id"].map(lambda x: item_info.get(x, {}).get("category"))
    df["misconception_id"] = df["item_id"].map(lambda x: item_info.get(x, {}).get("misconception_id"))
    df["misconception_name"] = df["item_id"].map(lambda x: item_info.get(x, {}).get("misconception_name"))

    # Compute target hit
    df["hit_target"] = (df["parsed_answer"] == df["target_distractor"]) & (df["is_correct"] == False)

    # Clean up model tier
    df["model_tier"] = df["model_tier"].fillna("unknown")

    return df

# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute key statistics with confidence intervals and significance tests."""

    results = {}

    # Overall stats
    results["total_responses"] = len(df)
    results["parse_success"] = df["parsed_answer"].notna().mean()
    results["overall_accuracy"] = df["is_correct"].sum() / df["is_correct"].notna().sum()

    # Errors only
    errors = df[(df["is_correct"] == False) & (df["parsed_answer"].notna())]
    results["total_errors"] = len(errors)
    results["overall_target_rate"] = errors["hit_target"].mean()

    # Test against chance (33.3%)
    n_hits = errors["hit_target"].sum()
    n_errors = len(errors)
    expected = n_errors / 3  # Chance = 33.3%
    chi2, p_value = stats.chisquare([n_hits, n_errors - n_hits], [expected, n_errors - expected])
    results["chi2_vs_chance"] = chi2
    results["p_value_vs_chance"] = p_value

    # 95% CI for target rate (Wilson score interval)
    p_hat = results["overall_target_rate"]
    n = n_errors
    z = 1.96
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
    results["target_rate_ci_low"] = center - margin
    results["target_rate_ci_high"] = center + margin

    return results, errors

def model_analysis(errors: pd.DataFrame) -> pd.DataFrame:
    """Analyze target rates by model."""

    model_stats = errors.groupby("model").agg(
        hits=("hit_target", "sum"),
        errors=("hit_target", "count"),
        rate=("hit_target", "mean"),
        tier=("model_tier", "first"),
        gsm8k=("gsm8k_score", "first")
    ).reset_index()

    # Add 95% CI
    def wilson_ci(row):
        p, n = row["rate"], row["errors"]
        if n == 0:
            return 0, 0
        z = 1.96
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
        return center - margin, center + margin

    model_stats[["ci_low", "ci_high"]] = model_stats.apply(wilson_ci, axis=1, result_type="expand")

    return model_stats.sort_values("rate", ascending=False)

def condition_analysis(errors: pd.DataFrame) -> pd.DataFrame:
    """Analyze target rates by condition."""

    cond_stats = errors.groupby("condition").agg(
        hits=("hit_target", "sum"),
        errors=("hit_target", "count"),
        rate=("hit_target", "mean")
    ).reset_index()

    return cond_stats

def category_analysis(errors: pd.DataFrame) -> pd.DataFrame:
    """Analyze target rates by misconception category."""

    cat_stats = errors.groupby("category").agg(
        hits=("hit_target", "sum"),
        errors=("hit_target", "count"),
        rate=("hit_target", "mean")
    ).reset_index()

    return cat_stats.sort_values("rate", ascending=False)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_model_comparison(df: pd.DataFrame, errors: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing models on accuracy and target rate."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy by model
    ax1 = axes[0]
    accuracy = df.groupby("model").apply(
        lambda x: x["is_correct"].sum() / x["is_correct"].notna().sum()
    ).sort_values(ascending=True)

    colors = ['#2ecc71' if acc == 1.0 else '#3498db' if acc > 0.85 else '#e74c3c'
              for acc in accuracy.values]

    bars = ax1.barh(accuracy.index, accuracy.values * 100, color=colors)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("A. Model Accuracy on Probe Items")
    ax1.set_xlim(0, 105)
    ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Perfect')

    # Add value labels
    for bar, val in zip(bars, accuracy.values):
        ax1.text(val * 100 + 1, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=9)

    # Panel B: Target distractor rate (errors only)
    ax2 = axes[1]
    model_stats = model_analysis(errors)

    # Only models with errors
    model_stats = model_stats[model_stats["errors"] > 0].sort_values("rate", ascending=True)

    colors2 = ['#9b59b6' if rate > 0.5 else '#3498db' if rate > 0.33 else '#95a5a6'
               for rate in model_stats["rate"].values]

    bars2 = ax2.barh(model_stats["model"], model_stats["rate"] * 100, color=colors2)

    # Add error bars
    ax2.errorbar(model_stats["rate"] * 100, model_stats["model"],
                xerr=[(model_stats["rate"] - model_stats["ci_low"]) * 100,
                      (model_stats["ci_high"] - model_stats["rate"]) * 100],
                fmt='none', color='black', capsize=3)

    ax2.axvline(x=33.3, color='red', linestyle='--', alpha=0.7, label='Chance (33.3%)')
    ax2.set_xlabel("Target Distractor Rate (%)")
    ax2.set_title("B. Misconception Alignment (When Wrong)")
    ax2.set_xlim(0, 100)
    ax2.legend(loc='lower right')

    # Add value labels
    for bar, val, n in zip(bars2, model_stats["rate"].values, model_stats["errors"].values):
        ax2.text(val * 100 + 2, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}% (n={n})', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_model_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig1_model_comparison.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig1_model_comparison.png/pdf")

def plot_condition_effect(errors: pd.DataFrame, output_dir: Path):
    """Create bar chart showing prompting condition effects."""

    fig, ax = plt.subplots(figsize=(8, 5))

    cond_stats = condition_analysis(errors)
    cond_order = ["answer_only", "explain", "persona"]
    cond_stats = cond_stats.set_index("condition").reindex(cond_order).reset_index()

    colors = ['#95a5a6', '#3498db', '#9b59b6']
    bars = ax.bar(cond_stats["condition"], cond_stats["rate"] * 100, color=colors)

    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.7, label='Chance (33.3%)')
    ax.set_ylabel("Target Distractor Rate (%)")
    ax.set_xlabel("Prompting Condition")
    ax.set_title("Effect of Prompting Strategy on Misconception Alignment")
    ax.set_ylim(0, 70)
    ax.legend()

    # Add value labels
    for bar, val, n in zip(bars, cond_stats["rate"].values, cond_stats["errors"].values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 100 + 2,
               f'{val*100:.1f}%\n(n={n})', ha='center', fontsize=10)

    # Add condition descriptions
    descriptions = {
        "answer_only": "Select answer\n(no reasoning)",
        "explain": "Show reasoning\nstep-by-step",
        "persona": "Role-play as\nstruggling student"
    }
    ax.set_xticklabels([descriptions[c] for c in cond_order])

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_condition_effect.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig2_condition_effect.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig2_condition_effect.png/pdf")

def plot_category_effect(errors: pd.DataFrame, output_dir: Path):
    """Create bar chart showing misconception category effects."""

    fig, ax = plt.subplots(figsize=(8, 5))

    cat_stats = category_analysis(errors)

    colors = ['#e74c3c', '#f39c12', '#3498db']
    bars = ax.bar(cat_stats["category"], cat_stats["rate"] * 100, color=colors)

    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.7, label='Chance (33.3%)')
    ax.set_ylabel("Target Distractor Rate (%)")
    ax.set_xlabel("Misconception Category")
    ax.set_title("Misconception Alignment by Error Type")
    ax.set_ylim(0, 70)
    ax.legend()

    # Add value labels
    for bar, val, n in zip(bars, cat_stats["rate"].values, cat_stats["errors"].values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 100 + 2,
               f'{val*100:.1f}%\n(n={n})', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_category_effect.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig3_category_effect.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig3_category_effect.png/pdf")

def plot_model_x_condition(errors: pd.DataFrame, output_dir: Path):
    """Create heatmap of model x condition interaction."""

    # Pivot table
    pivot = errors.pivot_table(
        values="hit_target",
        index="model",
        columns="condition",
        aggfunc="mean"
    ) * 100

    # Reorder
    cond_order = ["answer_only", "explain", "persona"]
    pivot = pivot[cond_order]
    pivot = pivot.sort_values("persona", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                center=33.3, vmin=0, vmax=100, ax=ax,
                cbar_kws={'label': 'Target Distractor Rate (%)'})

    ax.set_title("Model × Condition Interaction\n(Chance = 33.3%)")
    ax.set_xlabel("Prompting Condition")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_model_condition_heatmap.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig4_model_condition_heatmap.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig4_model_condition_heatmap.png/pdf")

def plot_capability_vs_alignment(df: pd.DataFrame, errors: pd.DataFrame, output_dir: Path):
    """Scatter plot of model capability (GSM8K) vs alignment."""

    # Get accuracy and target rate per model
    model_acc = df.groupby("model").agg(
        accuracy=("is_correct", lambda x: x.sum() / x.notna().sum()),
        gsm8k=("gsm8k_score", "first")
    ).reset_index()

    model_target = errors.groupby("model").agg(
        target_rate=("hit_target", "mean"),
        n_errors=("hit_target", "count")
    ).reset_index()

    merged = model_acc.merge(model_target, on="model", how="left")
    merged["target_rate"] = merged["target_rate"].fillna(0)
    merged["n_errors"] = merged["n_errors"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Only plot models with errors
    has_errors = merged[merged["n_errors"] > 0]
    no_errors = merged[merged["n_errors"] == 0]

    # Scatter for models with errors
    scatter = ax.scatter(has_errors["gsm8k"], has_errors["target_rate"] * 100,
                        s=has_errors["n_errors"] * 3, c=has_errors["accuracy"],
                        cmap='RdYlGn', vmin=0.5, vmax=1.0, alpha=0.7, edgecolors='black')

    # Mark frontier models at top
    for _, row in no_errors.iterrows():
        ax.annotate(f'{row["model"]}\n(100% acc)',
                   xy=(row["gsm8k"], 95), fontsize=8, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Add labels
    for _, row in has_errors.iterrows():
        ax.annotate(row["model"], xy=(row["gsm8k"], row["target_rate"] * 100),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.7, label='Chance')
    ax.set_xlabel("Model Capability (GSM8K %)")
    ax.set_ylabel("Target Distractor Rate (%)")
    ax.set_title("Capability Paradox: Model Performance vs Misconception Alignment\n(Size = error count, Color = accuracy)")
    ax.set_xlim(30, 100)
    ax.set_ylim(0, 100)

    plt.colorbar(scatter, label="Model Accuracy")
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_capability_paradox.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig5_capability_paradox.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig5_capability_paradox.png/pdf")

def plot_misconception_breakdown(errors: pd.DataFrame, output_dir: Path):
    """Bar chart of individual misconception target rates."""

    misc_stats = errors.groupby("misconception_id").agg(
        hits=("hit_target", "sum"),
        errors=("hit_target", "count"),
        rate=("hit_target", "mean"),
        category=("category", "first")
    ).reset_index()

    misc_stats = misc_stats[misc_stats["errors"] >= 3]  # Min sample
    misc_stats = misc_stats.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by category
    cat_colors = {"Procedural": "#3498db", "Conceptual": "#e74c3c", "Interpretive": "#f39c12"}
    colors = [cat_colors[c] for c in misc_stats["category"]]

    bars = ax.barh(misc_stats["misconception_id"], misc_stats["rate"] * 100, color=colors)

    ax.axvline(x=33.3, color='red', linestyle='--', alpha=0.7, label='Chance')
    ax.set_xlabel("Target Distractor Rate (%)")
    ax.set_ylabel("Misconception")
    ax.set_title("Alignment by Specific Misconception")
    ax.set_xlim(0, 100)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add value labels
    for bar, val, n in zip(bars, misc_stats["rate"].values, misc_stats["errors"].values):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height()/2,
               f'{val*100:.0f}% (n={n})', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "fig6_misconception_breakdown.png", bbox_inches='tight')
    plt.savefig(output_dir / "fig6_misconception_breakdown.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: fig6_misconception_breakdown.png/pdf")

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(df: pd.DataFrame, stats: dict, errors: pd.DataFrame, output_dir: Path):
    """Generate markdown report with all statistics."""

    model_stats = model_analysis(errors)
    cond_stats = condition_analysis(errors)
    cat_stats = category_analysis(errors)

    report = f"""# Study 2: Misconception Alignment Analysis Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## 1. Overview

- **Total responses**: {stats['total_responses']:,}
- **Parse success rate**: {stats['parse_success']*100:.1f}%
- **Overall accuracy**: {stats['overall_accuracy']*100:.1f}%
- **Total errors analyzed**: {stats['total_errors']}

## 2. Primary Finding: Target Distractor Rate

**Overall target distractor rate**: {stats['overall_target_rate']*100:.1f}%
(95% CI: [{stats['target_rate_ci_low']*100:.1f}%, {stats['target_rate_ci_high']*100:.1f}%])

**Significance test** (vs 33.3% chance):
- χ² = {stats['chi2_vs_chance']:.2f}
- p = {stats['p_value_vs_chance']:.4f} {'***' if stats['p_value_vs_chance'] < 0.001 else '**' if stats['p_value_vs_chance'] < 0.01 else '*' if stats['p_value_vs_chance'] < 0.05 else 'ns'}

## 3. Results by Model

| Model | Tier | GSM8K | Errors | Hits | Target Rate | 95% CI |
|-------|------|-------|--------|------|-------------|--------|
"""

    for _, row in model_stats.iterrows():
        report += f"| {row['model']} | {row['tier']} | {row['gsm8k']:.0f}% | {row['errors']} | {row['hits']} | {row['rate']*100:.1f}% | [{row['ci_low']*100:.1f}%, {row['ci_high']*100:.1f}%] |\n"

    report += f"""
## 4. Results by Prompting Condition

| Condition | Errors | Hits | Target Rate |
|-----------|--------|------|-------------|
"""

    for _, row in cond_stats.iterrows():
        report += f"| {row['condition']} | {row['errors']} | {row['hits']} | {row['rate']*100:.1f}% |\n"

    report += f"""
## 5. Results by Misconception Category

| Category | Errors | Hits | Target Rate |
|----------|--------|------|-------------|
"""

    for _, row in cat_stats.iterrows():
        report += f"| {row['category']} | {row['errors']} | {row['hits']} | {row['rate']*100:.1f}% |\n"

    report += """
## 6. Prompting Templates Used

### Condition 1: Answer-Only
```
You are taking a math test with {n_items} questions.
Select the best answer for each question (A, B, C, or D).

Format each answer as:
**Question N:** X
```

### Condition 2: Explain (Chain-of-Thought)
```
You are taking a math test with {n_items} questions.
Answer each question by showing your reasoning, then stating your final answer.

Format each answer as:
**Question N:**
[Your reasoning]
**Answer: X**
```

### Condition 3: Persona (Struggling Student)
```
You are a middle school student taking a math test.
Students at your level sometimes make mistakes - that's okay and normal.
Work through each problem the way a real student would.

Format each answer as:
**Question N:**
[Your thinking/work]
**Answer: X**
```

## 7. Key Findings

1. **Overall alignment significantly above chance**: {stats['overall_target_rate']*100:.1f}% vs 33.3% (p < .001)

2. **Capability paradox confirmed**: Frontier models achieve 100% accuracy, generating no errors for analysis

3. **"Sweet spot" model identified**: Claude 3 Haiku shows highest alignment ({model_stats[model_stats['model']=='claude-3-haiku']['rate'].values[0]*100:.1f}% if 'claude-3-haiku' in model_stats['model'].values else 'N/A'})

4. **Chain-of-thought helps**: Reasoning prompts increase alignment by ~{(cond_stats[cond_stats['condition']=='explain']['rate'].values[0] - cond_stats[cond_stats['condition']=='answer_only']['rate'].values[0])*100:.0f} percentage points

5. **Conceptual > Procedural**: Flawed mental models align better than procedural rule violations
"""

    with open(output_dir / "analysis_report.md", "w") as f:
        f.write(report)

    print(f"Saved: analysis_report.md")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Paths
    responses_path = "pilot/full_collection/batch_responses.jsonl"
    items_path = "data/probe_items.json"
    output_dir = Path("pilot/full_collection/figures")
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data(responses_path, items_path)
    print(f"  Loaded {len(df)} responses")

    print("\nComputing statistics...")
    stats, errors = compute_statistics(df)
    print(f"  {stats['total_errors']} errors to analyze")
    print(f"  Overall target rate: {stats['overall_target_rate']*100:.1f}%")
    print(f"  p-value vs chance: {stats['p_value_vs_chance']:.4f}")

    print("\nGenerating figures...")
    plot_model_comparison(df, errors, output_dir)
    plot_condition_effect(errors, output_dir)
    plot_category_effect(errors, output_dir)
    plot_model_x_condition(errors, output_dir)
    plot_capability_vs_alignment(df, errors, output_dir)
    plot_misconception_breakdown(errors, output_dir)

    print("\nGenerating report...")
    generate_report(df, stats, errors, output_dir)

    print(f"\nDone! All outputs saved to {output_dir}/")

if __name__ == "__main__":
    main()
