#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure signature : eventail magneto-spectroscopique CERTIFIE.
Spectre du dot carre en fonction de B (champ uniforme, jauge symetrique),
chaque niveau trace comme un ruban [borne inf, borne sup] garanti.
Reutilise l'assemblage et la borne de Liu 2D de poc_schrodinger_magnetique_2d.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import poc_schrodinger_magnetique_2d as m

OUT = os.path.dirname(os.path.abspath(__file__))
N = 128
NEV = 8
BS = np.linspace(0.0, 10.0, 11)


def certify_B(B):
    Afun = lambda x, y: (-0.5 * B * y, 0.5 * B * x)
    aoo = 0.5 * B * np.sqrt(2.0) * m.W
    A, M, h = m.assemble_2d(N, Afun)
    vals, _, res = m.eig_low(A, M, NEV)
    lbs = np.array([m.liu_lower_2d(v, h, aoo, m.FLOOR, m.S_GRID) for v in vals])
    return lbs, vals, res


def main():
    lbs_all, ubs_all = [], []
    for B in BS:
        lbs, ubs, res = certify_B(B)
        lbs_all.append(lbs)
        ubs_all.append(ubs)
        print("B = %5.1f : E1 dans [%.4f, %.4f], splitting 2-3 dans [%.4f, %.4f]  (res %.0e)"
              % (B, lbs[0], ubs[0], lbs[2] - ubs[1], ubs[2] - lbs[1], res))
    lbs_all = np.array(lbs_all)
    ubs_all = np.array(ubs_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    for k in range(NEV):
        ax1.fill_between(BS, lbs_all[:, k], ubs_all[:, k],
                         color="C%d" % (k % 10), alpha=0.75)
    ax1.set_xlabel("B")
    ax1.set_ylabel("E")
    ax1.set_title("Magneto-spectroscopic fan with guaranteed bounds\n"
                  "(square dot, 8 levels, ribbons = guaranteed intervals, h = %.4g)"
                  % (m.L / N))
    ax1.grid(alpha=0.3)

    sp_lb = lbs_all[:, 2] - ubs_all[:, 1]
    sp_ub = ubs_all[:, 2] - lbs_all[:, 1]
    ax2.fill_between(BS, sp_lb, sp_ub, color="tab:red", alpha=0.6)
    ax2.axhline(0.0, color="k", lw=0.6)
    ax2.set_xlabel("B")
    ax2.set_ylabel(r"$E_3 - E_2$")
    ax2.set_title("Splitting of the (1,2)/(2,1) pair\n(guaranteed interval)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2d_eventail_B.png"), dpi=300)
    print("Figure : fig2d_eventail_B.png")


if __name__ == "__main__":
    main()
