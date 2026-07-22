#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume graphique (Elsevier, format large ~1328x531 px, ratio 2.5:1) :
geometrie de bobine -> [Lemme 1] -> spectre quantique garanti.
Sortie directement dans le dossier de soumission LaTeX.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

SUB = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 11})

fig, ax = plt.subplots(figsize=(10.0, 4.0))   # ratio 2.5:1 -> 3000x1200 at 300 dpi
ax.set_xlim(0, 25); ax.set_ylim(0, 10); ax.axis("off")
BLUE = "#2a5d8f"; GREEN = "#2e7d4f"; GOLD = "#8a6508"; GREY = "#555555"

ax.text(12.5, 9.5, "From coil geometry to a guaranteed quantum spectrum",
        ha="center", fontsize=14, fontweight="bold")

# --- panel 1 : coil + dot + certified field ---
ax.add_patch(FancyBboxPatch((0.4, 1.2), 7.2, 6.7,
             boxstyle="round,pad=0.02,rounding_size=0.2", fc="#eaf1f8", ec=BLUE, lw=1.6))
cx, cy = 4.0, 5.2
ax.add_patch(Circle((cx, cy), 2.3, fill=False, ec=BLUE, lw=2.4))
ax.add_patch(Rectangle((cx - 0.7, cy - 0.7), 1.4, 1.4, fill=False, ec=GREEN, lw=1.8))
ax.annotate("", xy=(cx + 2.3, cy - 1.6), xytext=(cx + 3.0, cy - 2.2),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax.text(cx, 8.05, "Magnetostatics (coil)", ha="center", fontsize=10.5,
        color=BLUE, fontweight="bold")
ax.text(cx, cy - 0.05, "dot", ha="center", va="center", fontsize=8, color=GREEN)
ax.text(cx, 2.15, r"certified $A$: bound $\delta$", ha="center", fontsize=10, color=GOLD)

# --- arrow with Lemma 1 ---
ax.add_patch(FancyArrowPatch((7.9, 5.0), (10.1, 5.0), arrowstyle="-|>",
             mutation_scale=22, lw=2.6, color=GOLD))
ax.text(9.0, 6.35, "Lemma 1", ha="center", fontsize=10.5, color=GOLD, fontweight="bold")
ax.text(9.0, 3.7,
        r"$|\sqrt{E_k(A)}-\sqrt{E_k(A')}|$" + "\n" + r"$\leq\|A-A'\|_\infty$",
        ha="center", va="center", fontsize=9.5, color=GOLD)

# --- panel 3 : guaranteed spectrum ladder ---
ax.add_patch(FancyBboxPatch((10.6, 1.2), 14.0, 6.7,
             boxstyle="round,pad=0.02,rounding_size=0.2", fc="#eaf7f2", ec=GREEN, lw=1.6))
ax.text(17.6, 8.05, "Certified quantum spectrum", ha="center", fontsize=10.5,
        color=GREEN, fontweight="bold")
levels = [3.3, 4.4, 5.4, 6.4, 7.2]
widths = [0.10, 0.16, 0.22, 0.30, 0.42]
x0, x1 = 11.4, 16.0
for E, w in zip(levels, widths):
    ax.add_patch(Rectangle((x0, E - w), x1 - x0, 2 * w, color="tab:green", alpha=0.55))
    ax.plot([x0, x1], [E, E], color=GREEN, lw=0.6)
ax.annotate("", xy=(x1 + 0.25, levels[2]), xytext=(x1 + 0.25, levels[2] + widths[2]),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
ax.text(x1 + 0.5, levels[2] + 0.5 * widths[2], "guaranteed\ninterval", fontsize=8.5,
        color=GREY, va="center")
ax.text(20.9, 5.5, r"8 levels enclosed" + "\n" + r"end-to-end," + "\n" + r"$\sim 0.4\%$ relative",
        ha="center", va="center", fontsize=10, color=GREEN)
ax.text(17.6, 2.15, r"$(-\mathrm{i}\nabla-A)^2\psi=E\psi$ : two-sided bounds",
        ha="center", fontsize=9.5, color=GREY)

fig.tight_layout()
out = os.path.join(SUB, "GraphicalAbstract.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
from PIL import Image
print("GraphicalAbstract.png", Image.open(out).size)
