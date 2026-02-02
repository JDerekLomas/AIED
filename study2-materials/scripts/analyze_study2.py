#!/usr/bin/env python3
"""
Statistical Analysis for Study 2: Reasoning Authenticity Gap

Performs:
1. 3-way ANOVA: Specification Level × Model Tier × Misconception Type
2. Planned contrasts (S4 vs S1, tier interactions)
3. Reasoning Authenticity Gap analysis by cell
4. Visualization for paper figures
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
try:
    import scipy.stats as stats
    from scipy.stats import f_oneway, ttest_ind, chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# Model tier mapping
MODEL_TIERS = {
    "claude-sonnet-4": "frontier",
    "gpt-4o": "frontier",
    "gpt-4o-mini": "mid",
    "claude-3-haiku": "mid",
    "llama-3.1-8b": "weak",
    "mixtral-8x7b": "weak",
}


def load_coded_data(filepath: Path) -> pd.DataFrame:
    """Load coded errors into a DataFrame."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Add derived columns
    df['model_tier'] = df['model'].map(MODEL_TIERS)
    df['misconception_type'] = df['misconception_id'].apply(
        lambda x: 'procedural' if x.startswith('PROC') else 'conceptual'
    )
    df['is_aligned'] = df['code'].isin(['FULL_MATCH', 'PARTIAL_MATCH'])
    df['is_full_match'] = df['code'] == 'FULL_MATCH'

    return df


def compute_cell_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for each cell of the design."""

    cells = df.groupby(['spec_level', 'model_tier', 'misconception_type']).agg({
        'hit_target': ['sum', 'count'],
        'is_aligned': 'sum',
        'is_full_match': 'sum'
    }).reset_index()

    cells.columns = ['spec_level', 'model_tier', 'misconception_type',
                     'target_hits', 'n_errors', 'aligned', 'full_match']

    cells['target_rate'] = cells['target_hits'] / cells['n_errors']
    cells['alignment_rate'] = cells['aligned'] / cells['n_errors']
    cells['full_match_rate'] = cells['full_match'] / cells['n_errors']
    cells['gap'] = cells['target_rate'] - cells['alignment_rate']

    return cells


def run_anova(df: pd.DataFrame) -> dict:
    """Run 3-way ANOVA on misconception alignment."""

    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}

    # Create binary outcome for each observation
    # Note: For proportion data, we should use logistic regression
    # but ANOVA gives interpretable F-tests for the factorial design

    # Aggregate to cell means first (more appropriate for this design)
    cells = df.groupby(['spec_level', 'model_tier', 'misconception_type']).agg({
        'is_aligned': 'mean'
    }).reset_index()
    cells.columns = ['spec_level', 'model_tier', 'misconception_type', 'alignment_rate']

    # 3-way ANOVA
    model = ols('alignment_rate ~ C(spec_level) * C(model_tier) * C(misconception_type)',
                data=cells).fit()
    anova_table = anova_lm(model, typ=2)

    return {
        'model_summary': model.summary().as_text(),
        'anova_table': anova_table.to_dict(),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }


def run_planned_contrasts(df: pd.DataFrame) -> dict:
    """Run planned contrasts from the analysis plan."""

    results = {}

    # Contrast 1: S4 vs S1 overall
    s1_data = df[df['spec_level'] == 'S1']['is_aligned']
    s4_data = df[df['spec_level'] == 'S4']['is_aligned']

    if len(s1_data) > 0 and len(s4_data) > 0:
        t_stat, p_val = ttest_ind(s4_data, s1_data)
        s1_rate = s1_data.mean()
        s4_rate = s4_data.mean()
        results['s4_vs_s1'] = {
            's1_alignment': s1_rate,
            's4_alignment': s4_rate,
            'difference': s4_rate - s1_rate,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    # Contrast 2: S4 vs S3 within procedural misconceptions
    proc_s3 = df[(df['spec_level'] == 'S3') & (df['misconception_type'] == 'procedural')]['is_aligned']
    proc_s4 = df[(df['spec_level'] == 'S4') & (df['misconception_type'] == 'procedural')]['is_aligned']

    if len(proc_s3) > 0 and len(proc_s4) > 0:
        t_stat, p_val = ttest_ind(proc_s4, proc_s3)
        results['s4_vs_s3_procedural'] = {
            's3_alignment': proc_s3.mean(),
            's4_alignment': proc_s4.mean(),
            'difference': proc_s4.mean() - proc_s3.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    # Contrast 3: S3 vs S4 within conceptual misconceptions
    conc_s3 = df[(df['spec_level'] == 'S3') & (df['misconception_type'] == 'conceptual')]['is_aligned']
    conc_s4 = df[(df['spec_level'] == 'S4') & (df['misconception_type'] == 'conceptual')]['is_aligned']

    if len(conc_s3) > 0 and len(conc_s4) > 0:
        t_stat, p_val = ttest_ind(conc_s3, conc_s4)
        results['s3_vs_s4_conceptual'] = {
            's3_alignment': conc_s3.mean(),
            's4_alignment': conc_s4.mean(),
            'difference': conc_s3.mean() - conc_s4.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    # Contrast 4: Frontier-S4 vs Weak-S1 (capability paradox test)
    frontier_s4 = df[(df['model_tier'] == 'frontier') & (df['spec_level'] == 'S4')]['is_aligned']
    weak_s1 = df[(df['model_tier'] == 'weak') & (df['spec_level'] == 'S1')]['is_aligned']

    if len(frontier_s4) > 0 and len(weak_s1) > 0:
        t_stat, p_val = ttest_ind(frontier_s4, weak_s1)
        results['frontier_s4_vs_weak_s1'] = {
            'frontier_s4_alignment': frontier_s4.mean(),
            'weak_s1_alignment': weak_s1.mean(),
            'difference': frontier_s4.mean() - weak_s1.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    # Linear trend test across specification levels
    spec_order = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4}
    df_trend = df.copy()
    df_trend['spec_numeric'] = df_trend['spec_level'].map(spec_order)

    correlation = df_trend['spec_numeric'].corr(df_trend['is_aligned'].astype(float))
    results['linear_trend'] = {
        'correlation': correlation,
        'interpretation': 'positive' if correlation > 0 else 'negative'
    }

    return results


def compute_gap_analysis(df: pd.DataFrame) -> dict:
    """Compute Reasoning Authenticity Gap for each condition."""

    gaps = {}

    # Overall gap
    overall_target = df['hit_target'].mean()
    overall_align = df['is_aligned'].mean()
    gaps['overall'] = {
        'target_rate': overall_target,
        'alignment_rate': overall_align,
        'gap': overall_target - overall_align,
        'n': len(df)
    }

    # By specification level
    gaps['by_spec'] = {}
    for spec in ['S1', 'S2', 'S3', 'S4']:
        subset = df[df['spec_level'] == spec]
        if len(subset) > 0:
            gaps['by_spec'][spec] = {
                'target_rate': subset['hit_target'].mean(),
                'alignment_rate': subset['is_aligned'].mean(),
                'gap': subset['hit_target'].mean() - subset['is_aligned'].mean(),
                'n': len(subset)
            }

    # By model tier
    gaps['by_tier'] = {}
    for tier in ['frontier', 'mid', 'weak']:
        subset = df[df['model_tier'] == tier]
        if len(subset) > 0:
            gaps['by_tier'][tier] = {
                'target_rate': subset['hit_target'].mean(),
                'alignment_rate': subset['is_aligned'].mean(),
                'gap': subset['hit_target'].mean() - subset['is_aligned'].mean(),
                'n': len(subset)
            }

    # By misconception type
    gaps['by_type'] = {}
    for mtype in ['procedural', 'conceptual']:
        subset = df[df['misconception_type'] == mtype]
        if len(subset) > 0:
            gaps['by_type'][mtype] = {
                'target_rate': subset['hit_target'].mean(),
                'alignment_rate': subset['is_aligned'].mean(),
                'gap': subset['hit_target'].mean() - subset['is_aligned'].mean(),
                'n': len(subset)
            }

    return gaps


def create_visualizations(df: pd.DataFrame, cells: pd.DataFrame, output_dir: Path):
    """Create publication-quality figures."""

    if not HAS_PLOTTING:
        print("matplotlib/seaborn not installed, skipping visualizations")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")

    # Figure 1: Reasoning Authenticity Gap by Specification Level
    fig, ax = plt.subplots(figsize=(10, 6))

    spec_data = df.groupby('spec_level').agg({
        'hit_target': 'mean',
        'is_aligned': 'mean'
    }).reindex(['S1', 'S2', 'S3', 'S4'])

    x = np.arange(4)
    width = 0.35

    bars1 = ax.bar(x - width/2, spec_data['hit_target'] * 100, width,
                   label='Target Distractor Rate', color='steelblue')
    bars2 = ax.bar(x + width/2, spec_data['is_aligned'] * 100, width,
                   label='Misconception Alignment', color='coral')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Specification Level', fontsize=12)
    ax.set_title('Reasoning Authenticity Gap by Specification Level', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['S1\n(Persona)', 'S2\n(Knowledge)', 'S3\n(Mental Model)', 'S4\n(Production)'])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)

    # Add gap annotations
    for i, (target, align) in enumerate(zip(spec_data['hit_target'], spec_data['is_aligned'])):
        gap = (target - align) * 100
        ax.annotate(f'Gap: {gap:.1f}pp',
                   xy=(i, max(target, align) * 100 + 3),
                   ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_gap_by_spec.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_gap_by_spec.pdf', bbox_inches='tight')
    plt.close()

    # Figure 2: Heatmap of Alignment by Spec × Tier
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = df.groupby(['spec_level', 'model_tier'])['is_aligned'].mean().unstack()
    pivot = pivot.reindex(['S1', 'S2', 'S3', 'S4'])[['frontier', 'mid', 'weak']]

    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=50, ax=ax, cbar_kws={'label': 'Alignment Rate (%)'})
    ax.set_xlabel('Model Tier', fontsize=12)
    ax.set_ylabel('Specification Level', fontsize=12)
    ax.set_title('Misconception Alignment Rate: Spec Level × Model Tier', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_heatmap_spec_tier.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_heatmap_spec_tier.pdf', bbox_inches='tight')
    plt.close()

    # Figure 3: Interaction plot - Type × Spec
    fig, ax = plt.subplots(figsize=(10, 6))

    for mtype in ['procedural', 'conceptual']:
        subset = df[df['misconception_type'] == mtype]
        means = subset.groupby('spec_level')['is_aligned'].mean().reindex(['S1', 'S2', 'S3', 'S4'])
        linestyle = '-' if mtype == 'procedural' else '--'
        marker = 'o' if mtype == 'procedural' else 's'
        ax.plot(['S1', 'S2', 'S3', 'S4'], means * 100, marker=marker,
                linestyle=linestyle, linewidth=2, markersize=8, label=mtype.capitalize())

    ax.set_ylabel('Misconception Alignment Rate (%)', fontsize=12)
    ax.set_xlabel('Specification Level', fontsize=12)
    ax.set_title('Specification × Misconception Type Interaction', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_interaction_type_spec.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_interaction_type_spec.pdf', bbox_inches='tight')
    plt.close()

    # Figure 4: Gap by Model (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    model_data = df.groupby('model').agg({
        'hit_target': 'mean',
        'is_aligned': 'mean'
    }).sort_values('hit_target', ascending=False)

    x = np.arange(len(model_data))
    width = 0.35

    ax.bar(x - width/2, model_data['hit_target'] * 100, width,
           label='Target Rate', color='steelblue')
    ax.bar(x + width/2, model_data['is_aligned'] * 100, width,
           label='Alignment Rate', color='coral')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Target Rate vs Alignment Rate by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_data.index, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_gap_by_model.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_gap_by_model.pdf', bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {output_dir}")


def generate_results_summary(df: pd.DataFrame, anova_results: dict,
                             contrasts: dict, gaps: dict) -> str:
    """Generate a summary suitable for the paper results section."""

    lines = []
    lines.append("=" * 80)
    lines.append("STUDY 2 RESULTS SUMMARY")
    lines.append("For Paper Draft")
    lines.append("=" * 80)
    lines.append("")

    # Overall finding
    lines.append("## OVERALL FINDING")
    lines.append("-" * 60)
    overall = gaps['overall']
    lines.append(f"Total errors analyzed: N = {overall['n']}")
    lines.append(f"Target distractor rate: {100*overall['target_rate']:.1f}%")
    lines.append(f"Misconception alignment rate: {100*overall['alignment_rate']:.1f}%")
    lines.append(f"REASONING AUTHENTICITY GAP: {100*overall['gap']:.1f} percentage points")
    lines.append("")

    # Main effects
    lines.append("## MAIN EFFECTS (ANOVA)")
    lines.append("-" * 60)
    if 'anova_table' in anova_results:
        table = anova_results['anova_table']
        for effect in ['C(spec_level)', 'C(model_tier)', 'C(misconception_type)']:
            if effect in table.get('F', {}):
                f_val = table['F'].get(effect, 0)
                p_val = table['PR(>F)'].get(effect, 1)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                lines.append(f"{effect}: F = {f_val:.2f}, p = {p_val:.4f} {sig}")
    lines.append("")

    # Planned contrasts
    lines.append("## PLANNED CONTRASTS")
    lines.append("-" * 60)

    if 's4_vs_s1' in contrasts:
        c = contrasts['s4_vs_s1']
        sig = "*" if c['significant'] else ""
        lines.append(f"H1 - S4 vs S1 (overall specification effect):")
        lines.append(f"  S1 alignment: {100*c['s1_alignment']:.1f}%")
        lines.append(f"  S4 alignment: {100*c['s4_alignment']:.1f}%")
        lines.append(f"  Difference: {100*c['difference']:.1f}pp, t = {c['t_statistic']:.2f}, p = {c['p_value']:.4f} {sig}")
        lines.append("")

    if 's4_vs_s3_procedural' in contrasts:
        c = contrasts['s4_vs_s3_procedural']
        sig = "*" if c['significant'] else ""
        lines.append(f"H2a - S4 vs S3 for procedural misconceptions:")
        lines.append(f"  S3: {100*c['s3_alignment']:.1f}%, S4: {100*c['s4_alignment']:.1f}%")
        lines.append(f"  p = {c['p_value']:.4f} {sig}")
        lines.append("")

    if 's3_vs_s4_conceptual' in contrasts:
        c = contrasts['s3_vs_s4_conceptual']
        sig = "*" if c['significant'] else ""
        lines.append(f"H2b - S3 vs S4 for conceptual misconceptions:")
        lines.append(f"  S3: {100*c['s3_alignment']:.1f}%, S4: {100*c['s4_alignment']:.1f}%")
        lines.append(f"  p = {c['p_value']:.4f} {sig}")
        lines.append("")

    if 'frontier_s4_vs_weak_s1' in contrasts:
        c = contrasts['frontier_s4_vs_weak_s1']
        sig = "*" if c['significant'] else ""
        lines.append(f"H3 - Capability paradox (Frontier-S4 vs Weak-S1):")
        lines.append(f"  Frontier-S4: {100*c['frontier_s4_alignment']:.1f}%")
        lines.append(f"  Weak-S1: {100*c['weak_s1_alignment']:.1f}%")
        lines.append(f"  p = {c['p_value']:.4f} {sig}")
        lines.append("")

    # Gap by condition
    lines.append("## REASONING AUTHENTICITY GAP BY CONDITION")
    lines.append("-" * 60)

    lines.append("\nBy Specification Level:")
    for spec in ['S1', 'S2', 'S3', 'S4']:
        if spec in gaps['by_spec']:
            g = gaps['by_spec'][spec]
            lines.append(f"  {spec}: Gap = {100*g['gap']:.1f}pp (target={100*g['target_rate']:.1f}%, align={100*g['alignment_rate']:.1f}%)")

    lines.append("\nBy Model Tier:")
    for tier in ['frontier', 'mid', 'weak']:
        if tier in gaps['by_tier']:
            g = gaps['by_tier'][tier]
            lines.append(f"  {tier}: Gap = {100*g['gap']:.1f}pp")

    lines.append("\nBy Misconception Type:")
    for mtype in ['procedural', 'conceptual']:
        if mtype in gaps['by_type']:
            g = gaps['by_type'][mtype]
            lines.append(f"  {mtype}: Gap = {100*g['gap']:.1f}pp")

    lines.append("")

    # Interpretation based on outcome patterns from paper
    lines.append("## INTERPRETATION")
    lines.append("-" * 60)

    # Determine which outcome pattern best matches
    s1_align = gaps['by_spec'].get('S1', {}).get('alignment_rate', 0)
    s4_align = gaps['by_spec'].get('S4', {}).get('alignment_rate', 0)
    s4_gap = gaps['by_spec'].get('S4', {}).get('gap', 1)

    if s4_align > s1_align * 2 and s4_gap < 0.2:
        lines.append("Pattern: OUTCOME A (Specification Works)")
        lines.append("Misconception alignment increases with specification level.")
        lines.append("Gap shrinks significantly at S4.")
    elif s4_align < 0.15:
        lines.append("Pattern: OUTCOME C (Nothing Works)")
        lines.append("Gap persists across all specification levels.")
        lines.append("Even detailed prompts fail to induce authentic misconceptions.")
    else:
        lines.append("Pattern: Mixed/Unclear")
        lines.append("Results require further interpretation.")

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Statistical analysis for Study 2")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file with coded errors")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "analysis"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading coded data from {input_path}...")
    df = load_coded_data(input_path)
    print(f"Loaded {len(df)} coded errors")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Spec levels: {df['spec_level'].unique().tolist()}")

    print("\nComputing cell statistics...")
    cells = compute_cell_statistics(df)
    cells.to_csv(output_dir / 'cell_statistics.csv', index=False)

    print("Running ANOVA...")
    anova_results = run_anova(df)

    print("Running planned contrasts...")
    contrasts = run_planned_contrasts(df)

    print("Computing gap analysis...")
    gaps = compute_gap_analysis(df)

    if not args.no_plots:
        print("Creating visualizations...")
        create_visualizations(df, cells, output_dir)

    print("Generating results summary...")
    summary = generate_results_summary(df, anova_results, contrasts, gaps)

    # Save all results
    summary_path = output_dir / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    with open(output_dir / 'anova_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({k: convert(v) if not isinstance(v, dict) else
                   {k2: convert(v2) for k2, v2 in v.items()}
                   for k, v in anova_results.items() if k != 'model_summary'}, f, indent=2)

    with open(output_dir / 'contrasts.json', 'w') as f:
        json.dump(contrasts, f, indent=2, default=float)

    with open(output_dir / 'gaps.json', 'w') as f:
        json.dump(gaps, f, indent=2, default=float)

    # Print summary to console
    print("\n" + summary)


if __name__ == "__main__":
    main()
