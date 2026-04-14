#!/usr/bin/env python3
"""Generate figures for AIED 2026 paper."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load data
data_path = Path(__file__).parent.parent.parent / 'pilot' / 'prompt_framing_experiment' / 'rsm_data.json'
with open(data_path) as f:
    rsm_data = json.load(f)

# Factor coding for prompts (from paper Table 2)
PROMPT_FACTORS = {
    'teacher': {'item': False, 'pop': False},
    'verbalized_sampling': {'item': False, 'pop': False},
    'familiarity_gradient': {'item': True, 'pop': False},
    'contrastive': {'item': True, 'pop': False},
    'prerequisite_chain': {'item': True, 'pop': False},
    'cognitive_load': {'item': True, 'pop': False},
    'cognitive_profile': {'item': True, 'pop': False},
    'devil_advocate': {'item': False, 'pop': True},
    'teacher_decomposed': {'item': False, 'pop': True},
    'classroom_sim': {'item': False, 'pop': True},
    'imagine_classroom': {'item': False, 'pop': True},
    'synthetic_students': {'item': False, 'pop': True},
    'error_analysis': {'item': True, 'pop': True},
    'error_affordance': {'item': True, 'pop': True},
    'buggy_rules': {'item': True, 'pop': True},
    'misconception_holistic': {'item': True, 'pop': True},
}

def get_best_result_per_prompt(configs):
    """Get best (max rho) result for each prompt, using n_reps=3."""
    best = {}
    for c in configs:
        if c['n_reps'] != 3:
            continue
        framing = c['framing']
        if framing not in best or c['rho'] > best[framing]['rho']:
            best[framing] = c
    return best

def get_color(prompt):
    """Return color based on factor membership."""
    factors = PROMPT_FACTORS.get(prompt, {'item': False, 'pop': False})
    if factors['item'] and not factors['pop']:
        return '#2ecc71'  # Green: item only
    elif factors['item'] and factors['pop']:
        return '#3498db'  # Blue: item + pop
    elif not factors['item'] and factors['pop']:
        return '#e74c3c'  # Red: pop only
    else:
        return '#95a5a6'  # Grey: neither

def figure1_screening():
    """Generate Figure 1: Prompt screening results bar chart."""
    best = get_best_result_per_prompt(rsm_data['configs'])

    # Sort by rho descending
    sorted_prompts = sorted(best.keys(), key=lambda p: best[p]['rho'], reverse=True)

    # Filter to main 15 prompts (exclude experimental ones)
    main_prompts = [p for p in sorted_prompts if p in PROMPT_FACTORS][:15]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(main_prompts))
    colors = [get_color(p) for p in main_prompts]
    rhos = [best[p]['rho'] for p in main_prompts]

    bars = ax.bar(x, rhos, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=8)

    # Teacher baseline
    teacher_rho = best['teacher']['rho']
    ax.axhline(y=teacher_rho, color='black', linestyle='--', linewidth=1, label=f'teacher baseline (ρ={teacher_rho:.3f})')

    # Format x-axis labels
    labels = [p.replace('_', '\n') for p in main_prompts]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    ax.set_ylabel('Spearman ρ')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.5, len(main_prompts) - 0.5)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Item analysis only'),
        Patch(facecolor='#3498db', label='Item + Population'),
        Patch(facecolor='#e74c3c', label='Population only'),
        Patch(facecolor='#95a5a6', label='Neither (baseline)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig1_screening.pdf')
    plt.savefig(Path(__file__).parent / 'fig1_screening.png')
    print('Saved fig1_screening.pdf')

def figure2_temperature():
    """Generate Figure 2: Temperature effects for selected prompts."""
    # Get data for selected prompts across temperatures
    prompts_to_show = ['prerequisite_chain', 'cognitive_load', 'buggy_rules', 'teacher']

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'prerequisite_chain': '#2ecc71', 'cognitive_load': '#2ecc71',
              'buggy_rules': '#3498db', 'teacher': '#95a5a6'}
    markers = {'prerequisite_chain': 'o', 'cognitive_load': 's',
               'buggy_rules': '^', 'teacher': 'D'}

    for prompt in prompts_to_show:
        # Get all temperature points for this prompt (n_reps=3)
        temps = []
        rhos = []
        for c in rsm_data['configs']:
            if c['framing'] == prompt and c['n_reps'] == 3:
                temps.append(c['temperature'])
                rhos.append(c['rho'])

        # Sort by temperature
        sorted_data = sorted(zip(temps, rhos))
        temps, rhos = zip(*sorted_data)

        label = prompt.replace('_', ' ')
        ax.plot(temps, rhos, marker=markers[prompt], color=colors[prompt],
                label=label, linewidth=1.5, markersize=6)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Spearman ρ')
    ax.set_ylim(0.5, 0.75)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig2_temperature.pdf')
    plt.savefig(Path(__file__).parent / 'fig2_temperature.png')
    print('Saved fig2_temperature.pdf')

def figure3_model_survey():
    """Generate Figure 3: Model survey heatmap."""
    # Load model survey data
    survey_path = Path(__file__).parent.parent.parent / 'pilot' / 'model_survey' / 'survey_results.json'
    with open(survey_path) as f:
        survey_data = json.load(f)

    # Build matrix of model x prompt
    models = ['Claude Opus 4.5', 'Gemini 3 Flash', 'Gemma-3-27B', 'GPT-4o',
              'Llama-3.3-70B', 'Llama-4-Scout', 'Llama-3.1-8B']
    prompts = ['teacher', 'cognitive_load', 'prerequisite_chain']

    model_name_map = {
        'Claude Opus 4.5': 'opus45',
        'Gemini 3 Flash': 'gemini',
        'Gemma-3-27B': 'gemma3_27b',
        'GPT-4o': 'gpt4o',
        'Llama-3.3-70B': 'llama33_70b',
        'Llama-4-Scout': 'scout',
        'Llama-3.1-8B': 'llama31_8b',
    }

    # Build lookup from survey data
    lookup = {}
    for entry in survey_data:
        key = (entry['model'], entry['prompt'])
        if entry['n_items'] >= 50:  # Only use entries with sufficient items
            lookup[key] = entry['avg_pred_rho']

    # Build matrix
    matrix = np.zeros((len(models), len(prompts)))
    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            key = (model_name_map[model], prompt)
            if key in lookup:
                matrix[i, j] = lookup[key]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=0.7, aspect='auto')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(prompts)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.35 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            else:
                ax.text(j, i, '—', ha='center', va='center', color='gray', fontsize=9)

    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([p.replace('_', '\n') for p in prompts])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig3_model_survey.pdf')
    plt.savefig(Path(__file__).parent / 'fig3_model_survey.png')
    print('Saved fig3_model_survey.pdf')

def figure4_cross_dataset():
    """Generate Figure 4: Cross-dataset comparison."""
    # Data from paper Table 5
    datasets = ['SmartPaper\n(open-ended)', 'DBE-KT22\n(CS MCQ)', 'BEA 2024\n(USMLE MCQ)']
    prompts = ['teacher', 'prerequisite_chain', 'buggy_rules']

    # Values from paper Table 5
    data = {
        'teacher': [0.56, 0.52, 0.45],
        'prerequisite_chain': [0.69, 0.53, 0.44],
        'buggy_rules': [0.66, np.nan, 0.45],  # buggy not tested on DBE-KT22
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(datasets))
    width = 0.25

    colors = {'teacher': '#95a5a6', 'prerequisite_chain': '#2ecc71', 'buggy_rules': '#3498db'}

    for i, prompt in enumerate(prompts):
        offset = (i - 1) * width
        vals = data[prompt]
        bars = ax.bar(x + offset, vals, width, label=prompt.replace('_', ' '), color=colors[prompt])

        # Add value labels
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Spearman ρ')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig4_cross_dataset.pdf')
    plt.savefig(Path(__file__).parent / 'fig4_cross_dataset.png')
    print('Saved fig4_cross_dataset.pdf')

def figure5_factor_analysis():
    """Generate Figure 5: Factor analysis (mean rho by factor combination)."""
    best = get_best_result_per_prompt(rsm_data['configs'])

    # Group by factors
    groups = {
        'Item only': [],
        'Item + Pop': [],
        'Pop only': [],
        'Neither': [],
    }

    for prompt, result in best.items():
        if prompt not in PROMPT_FACTORS:
            continue
        factors = PROMPT_FACTORS[prompt]
        if factors['item'] and not factors['pop']:
            groups['Item only'].append(result['rho'])
        elif factors['item'] and factors['pop']:
            groups['Item + Pop'].append(result['rho'])
        elif not factors['item'] and factors['pop']:
            groups['Pop only'].append(result['rho'])
        else:
            groups['Neither'].append(result['rho'])

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = list(groups.keys())
    means = [np.mean(groups[k]) for k in labels]
    stds = [np.std(groups[k]) for k in labels]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='white')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Mean Spearman ρ')
    ax.set_ylim(0, 0.75)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'fig5_factors.pdf')
    plt.savefig(Path(__file__).parent / 'fig5_factors.png')
    print('Saved fig5_factors.pdf')

if __name__ == '__main__':
    figure1_screening()
    figure2_temperature()
    figure3_model_survey()
    figure4_cross_dataset()
    figure5_factor_analysis()
    print('\nAll figures generated!')
