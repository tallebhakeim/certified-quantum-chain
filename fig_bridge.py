#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure conceptuelle : pont logique entre l'electromagnetisme certifie
(programme DGM/TLM de l'auteur) et le spectre quantique certifie.
Deux liens : (i) ANALOGIE structurelle (meme probleme elliptique auto-adjoint
=> la certification a bornes complementaires se transpose en bornes
bilaterales de valeurs propres) ; (ii) COUPLAGE physique (le potentiel vecteur
A certifie du probleme magnetostatique ENTRE dans l'operateur quantique).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 11})

fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis("off")

BLUE = "#2a5d8f"
GREEN = "#2e7d4f"
GREY = "#555555"


def box(x0, y0, x1, y1, fc, ec):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=fc, ec=ec, lw=1.6, zorder=2))


def arrow(p0, p1, color, style="-|>", lw=2.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18,
                 lw=lw, color=color, ls=ls, zorder=3,
                 connectionstyle="arc3,rad=%g" % rad))


# ---- column headers ----
ax.text(2.4, 6.95, "Classical electromagnetics", ha="center", fontsize=12.5,
        color=BLUE, fontweight="bold")
ax.text(2.4, 6.62, "(author's certified program: DGM / TLM)", ha="center",
        fontsize=9.5, color=BLUE, style="italic")
ax.text(7.6, 6.95, "Quantum spectrum", ha="center", fontsize=12.5,
        color=GREEN, fontweight="bold")
ax.text(7.6, 6.62, "(this work)", ha="center", fontsize=9.5, color=GREEN,
        style="italic")

# ---- top row : the PDE / operator ----
box(0.3, 4.35, 4.55, 6.25, "#eaf1f8", BLUE)
ax.text(2.42, 5.9, "Magnetostatics (source problem)", ha="center", fontsize=10,
        color=BLUE, fontweight="bold")
ax.text(2.42, 5.42, r"$\nabla\times(\nu\,\nabla\times A)=J$", ha="center", fontsize=13)
ax.text(2.42, 4.95, r"$-\Delta A=\mu_0 J$  (Coulomb gauge)", ha="center", fontsize=11)
ax.text(2.42, 4.58, "self-adjoint elliptic", ha="center", fontsize=9,
        color=GREY, style="italic")

box(5.45, 4.35, 9.7, 6.25, "#eaf7f2", GREEN)
ax.text(7.57, 5.9, "Magnetic Schrödinger (eigenproblem)", ha="center",
        fontsize=10, color=GREEN, fontweight="bold")
ax.text(7.57, 5.38, r"$(-i\nabla-A)^2\,\psi=E\,\psi$", ha="center", fontsize=13.5)
ax.text(7.57, 4.86, r"same $A$, self-adjoint elliptic", ha="center", fontsize=9,
        color=GREY, style="italic")

# ---- physical coupling arrow (A feeds the operator) ----
arrow((4.6, 5.3), (5.4, 5.3), "#b8860b", lw=2.6)
ax.text(5.0, 5.62, r"$A$", ha="center", fontsize=12, color="#8a6508")
ax.text(5.0, 3.98, "PHYSICAL COUPLING", ha="center", fontsize=8.5,
        color="#8a6508", fontweight="bold")
ax.text(5.0, 3.66,
        r"certified $A$ (bound $\delta$) enters the operator;",
        ha="center", fontsize=8.6, color="#8a6508")
ax.text(5.0, 3.36,
        r"Lemma 1: $|\sqrt{E_k(A)}-\sqrt{E_k(A')}|\leq\|A-A'\|_\infty$",
        ha="center", fontsize=8.8, color="#8a6508")

# ---- bottom row : the certification ----
box(0.3, 0.5, 4.55, 2.6, "#eef6f0", BLUE)
ax.text(2.42, 2.28, "Complementary (dual) bounds", ha="center", fontsize=10,
        color=BLUE, fontweight="bold")
ax.text(2.42, 1.86, "Prager–Synge", ha="center", fontsize=10.5)
ax.text(2.42, 1.45, r"primal $A$ $\oplus$ dual $H$ bracket the energy", ha="center",
        fontsize=8.8, color=GREY)
ax.text(2.42, 1.08, "guaranteed two-sided field / energy", ha="center",
        fontsize=8.8, color=GREY, style="italic")

box(5.45, 0.5, 9.7, 2.6, "#eef6f0", GREEN)
ax.text(7.57, 2.28, "Two-sided eigenvalue bounds", ha="center", fontsize=10,
        color=GREEN, fontweight="bold")
ax.text(7.57, 1.86, "Temple – Lehmann – Liu", ha="center", fontsize=10.5)
ax.text(7.57, 1.45, "Rayleigh–Ritz (upper) + Liu (lower)",
        ha="center", fontsize=8.8, color=GREY)
ax.text(7.57, 1.08, "guaranteed two-sided spectrum", ha="center", fontsize=8.8,
        color=GREY, style="italic")

# ---- vertical "certified by" arrows ----
arrow((2.42, 4.3), (2.42, 2.65), BLUE, lw=1.6, style="-|>")
arrow((7.57, 4.3), (7.57, 2.65), GREEN, lw=1.6, style="-|>")
ax.text(2.62, 3.5, "certified by", ha="left", fontsize=8, color=BLUE, rotation=90,
        va="center")
ax.text(7.77, 3.5, "certified by", ha="left", fontsize=8, color=GREEN, rotation=90,
        va="center")

# ---- structural analogy arrow (dashed, bidirectional) ----
arrow((4.6, 1.55), (5.4, 1.55), GREY, lw=1.8, style="<|-|>", ls="--")
ax.text(5.0, 0.72, "STRUCTURAL ANALOGY", ha="center", fontsize=8.5,
        color=GREY, fontweight="bold")
ax.text(5.0, 0.4, "same variational structure", ha="center", fontsize=8,
        color=GREY, style="italic")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig0_bridge.png"), dpi=300, bbox_inches="tight")
print("Figure : fig0_bridge.png")
