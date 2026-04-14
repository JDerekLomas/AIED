#!/usr/bin/env python3
"""Generate Figure 1: Prompt screening results (Spearman rho) for AIED paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data (ordered by rho descending) ---
prompts = [
    ("Prerequisite\nChain",        0.686, "item",     (0.576, 0.775)),
    ("Cognitive\nLoad",            0.673, "item",     (0.550, 0.766)),
    ("Buggy\nRules",               0.655, "item+pop", (0.532, 0.752)),
    ("Misconception\nHolistic",    0.636, "item+pop", None),
    ("Error\nAnalysis",            0.596, "item+pop", None),
    ("Devil's\nAdvocate",          0.596, "pop",      None),
    ("Cognitive\nProfile",         0.586, "item",     (0.456, 0.693)),
    ("Contrastive",                0.584, "item",     None),
    ("Classroom\nSim",             0.562, "pop",      None),
    ("Teacher\n(baseline)",        0.555, "neither",  (0.422, 0.664)),
    ("Verbalized\nSampling",       0.52,  "neither",  None),
    ("Error\nAffordance",          0.51,  "item+pop", None),
    ("Familiarity\nGradient",      0.50,  "item",     None),
    ("Imagine\nClassroom",         0.48,  "pop",      None),
    ("Teacher\nDecomposed",        0.45,  "pop",      None),
]

# Colorblind-friendly palette (Okabe-Ito inspired)
COLOR_MAP = {
    "item":     "#009E73",  # bluish green
    "item+pop": "#0072B2",  # blue
    "pop":      "#D55E00",  # vermillion
    "neither":  "#999999",  # grey
}
LABEL_MAP = {
    "item":     "Item analysis only",
    "item+pop": "Item + population",
    "pop":      "Population only",
    "neither":  "Neither factor",
}

# --- Build arrays ---
names  = [p[0] for p in prompts]
rhos   = np.array([p[1] for p in prompts])
cats   = [p[2] for p in prompts]
cis    = [p[3] for p in prompts]
colors = [COLOR_MAP[c] for c in cats]

# Error bars (asymmetric): lower = rho - ci_lo, upper = ci_hi - rho
yerr_lo = np.array([rho - ci[0] if ci else 0 for rho, ci in zip(rhos, cis)])
yerr_hi = np.array([ci[1] - rho if ci else 0 for rho, ci in zip(rhos, cis)])

# --- Plot ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(prompts))

# Bars
bars = ax.bar(x, rhos, width=0.65, color=colors, edgecolor="white", linewidth=0.5,
              zorder=3)

# Error bars only where CI exists
for i, (lo, hi) in enumerate(zip(yerr_lo, yerr_hi)):
    if lo > 0:
        ax.errorbar(x[i], rhos[i], yerr=[[lo], [hi]],
                     fmt="none", ecolor="black", elinewidth=1.0,
                     capsize=3, capthick=0.8, zorder=4)

# Teacher baseline
ax.axhline(y=0.555, color="#666666", linestyle="--", linewidth=1.0, zorder=2,
           label="Teacher baseline ($\\rho$ = .555)")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, linespacing=0.9)
ax.set_ylabel("Spearman $\\rho$ (SmartPaper, $n$=140)", fontsize=10)
ax.set_ylim(0.35, 0.85)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax.tick_params(axis="y", labelsize=8)

# Light grid
ax.yaxis.grid(True, which="major", linewidth=0.4, color="#cccccc", zorder=1)
ax.set_axisbelow(True)

# Legend: one entry per category + baseline
from matplotlib.patches import Patch
handles = [Patch(facecolor=COLOR_MAP[k], label=LABEL_MAP[k])
           for k in ["item", "item+pop", "pop", "neither"]]
handles.append(plt.Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.0,
                           label="Teacher baseline ($\\rho$ = .555)"))
ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.9,
          edgecolor="#cccccc", handlelength=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_dir = "/Users/dereklomas/AIED/study2-materials/paper/figures"
fig.savefig(f"{out_dir}/fig1_screening.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_dir}/fig1_screening.png", dpi=300, bbox_inches="tight")
print("Saved fig1_screening.pdf and fig1_screening.png")
