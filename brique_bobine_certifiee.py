#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brique bobine reelle : chaine certifiee COMPLETE
    geometrie de spire -> delta garanti sur A -> intervalles garantis sur le spectre.

Geometrie : spire circulaire de rayon R = 5 coplanaire au dot (boucle de
courant microfabriquee autour d'un dot [-w,w]^2, w = 0.5, dans le plan z = 0).
Courant fixe tel que B au centre = mu0 I/(2R) = 6 (unites hbar^2/2m = q = 1).

Potentiel vecteur exact (azimutal, div A = 0, meme famille de jauge que A*) :
    A_phi(r) = (mu0 I R / 4 pi) INT_0^2pi  cos(t) / sqrt(R^2 + r^2 - 2 R r cos t) dt

ENCADREMENT RIGOUREUX de A_phi (aucune boite noire) :
  * trapeze compose a m sous-intervalles + borne d'erreur explicite
        |erreur| <= (2 pi)^3 max|f''| / (12 m^2),
    avec la borne elementaire, valide pour tout r <= rmax < R :
        max|f''| <= Dm^-1/2 + 3 R r Dm^-3/2 + 3 (R r)^2 Dm^-5/2,  Dm = (R - rmax)^2
    (obtenue en derivant f = cos t (R^2+r^2-2Rr cos t)^-1/2 deux fois et en
    majorant |sin|,|cos| par 1) ;
  * gonflement flottant 1e-12 relatif (meme reserve que le reste du PoC).

delta certifie : delta(Bbar) = max_j sup|A_phi(r_j) - Bbar r_j/2| + L_g dr/2,
grille fine en r, constante de Lipschitz explicite
    L_g <= (mu0 I R/4pi) 2pi (R+rmax)/(R-rmax)^3 + Bbar/2,
puis optimisation de Bbar (chaque Bbar donne un delta valide, on prend le
meilleur : equioscillation).

Ensuite : certification de H*(Bbar) (module etape 2) + lemme de propagation
    E_k(A_spire) dans [ (sqrt(lb)-delta)^2, (sqrt(ub)+delta)^2 ],
et validation de contenance contre une reference fine (Richardson 256/512)
du VRAI operateur a champ de spire (A_phi interpole sur grille dense).

Pour une geometrie de bobine quelconque, cette brique est remplacee par le
solveur magnetostatique certifie DGM qui fournit delta ; pour la spire
canonique, l'enclosure Biot-Savart ci-dessus est autonome et exacte.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import poc_schrodinger_magnetique_2d as m

OUT = os.path.dirname(os.path.abspath(__file__))
PI = np.pi

R_LOOP = 5.0
B0_CENTRE = 6.0
MU0I = 2.0 * R_LOOP * B0_CENTRE          # mu0 I tel que B(0) = 6
PREF = MU0I * R_LOOP / (4.0 * PI)
RMAX = np.sqrt(2.0) * m.W                # rayon du coin du dot
MTH = 4096                                # sous-intervalles en theta
MR = 20001                                # grille en r


def aphi_enclose(rs):
    """Encadrement [lo, hi] de A_phi(r) pour un tableau de rayons rs < R."""
    dm = (R_LOOP - RMAX) ** 2
    fpp = dm ** -0.5 + 2 * R_LOOP * RMAX * dm ** -1.5 \
        + 3 * (R_LOOP * RMAX) ** 2 * dm ** -2.5
    quaderr = PREF * (2 * PI) ** 3 * fpp / (12.0 * MTH ** 2)
    th = np.linspace(0.0, 2 * PI, MTH + 1)
    ct = np.cos(th)
    dth = th[1] - th[0]
    vals = np.empty(len(rs))
    chunk = 2000
    for i0 in range(0, len(rs), chunk):
        r = rs[i0:i0 + chunk, None]
        f = ct[None, :] / np.sqrt(R_LOOP ** 2 + r * r - 2 * R_LOOP * r * ct[None, :])
        T = dth * (f.sum(axis=1) - 0.5 * f[:, 0] - 0.5 * f[:, -1])
        vals[i0:i0 + chunk] = PREF * T
    infl = 1e-12 * np.abs(vals) + 1e-14
    return vals - quaderr - infl, vals + quaderr + infl


def main():
    t0 = time.time()
    print("=" * 78)
    print("Brique bobine reelle : spire R=%.0f coplanaire, B(0)=%.0f" % (R_LOOP, B0_CENTRE))
    print("=" * 78)

    # --- 1. enclosure de A_phi et delta certifie -------------------------
    rs = np.linspace(1e-6, RMAX, MR)
    lo, hi = aphi_enclose(rs)
    print("Enclosure Biot-Savart : m_theta = %d, largeur max = %.2e" % (MTH, np.max(hi - lo)))

    bbars = np.linspace(5.90, 6.30, 801)
    lg = PREF * 2 * PI * (R_LOOP + RMAX) / (R_LOOP - RMAX) ** 3 + bbars.max() / 2
    dr = rs[1] - rs[0]
    g_lo = lo[None, :] - 0.5 * bbars[:, None] * rs[None, :]
    g_hi = hi[None, :] - 0.5 * bbars[:, None] * rs[None, :]
    g_abs = np.maximum(np.abs(g_lo), np.abs(g_hi))
    deltas = g_abs.max(axis=1) + lg * dr / 2.0
    i_opt = int(np.argmin(deltas))
    BBAR, DELTA = float(bbars[i_opt]), float(deltas[i_opt])
    print("Bbar optimal = %.5f, delta certifie = %.6f" % (BBAR, DELTA))
    print("(heuristique (3/8)(rmax/R)^2 * Bbar rmax/2 = %.6f ; Lipschitz slack %.1e)"
          % (0.375 * (RMAX / R_LOOP) ** 2 * 0.5 * BBAR * RMAX, lg * dr / 2))

    # --- 2. certification de H*(Bbar) ------------------------------------
    aoo = 0.5 * BBAR * np.sqrt(2.0) * m.W
    Afun_star = lambda x, y: (-0.5 * BBAR * y, 0.5 * BBAR * x)
    t = time.time()
    A, M, h = m.assemble_2d(256, Afun_star)
    ubs, _, res = m.eig_low(A, M, m.NEV)
    lbs = np.array([m.liu_lower_2d(v, h, aoo, m.FLOOR, m.S_GRID) for v in ubs])
    print("H*(Bbar) certifie n=256 : res alg %.1e  [%.1fs]" % (res, time.time() - t))

    for k in range(m.NEV):
        print("  H*(Bbar) k=%d : [%.6f, %.6f]" % (k + 1, lbs[k], ubs[k]))
    prop_lb = np.maximum(np.sqrt(np.maximum(lbs, 0.0)) - DELTA, 0.0) ** 2
    prop_ub = (np.sqrt(ubs) + DELTA) ** 2

    # --- 3. reference fine du vrai operateur (champ de spire) ------------
    qs = 0.5 * (lo + hi) / rs                    # A_phi(r)/r, interpolation dense

    def Afun_spire(x, y):
        r = np.hypot(x, y)
        q = np.interp(r, rs, qs)
        return -y * q, x * q

    t = time.time()
    refs = {}
    for n in (256, 512):
        A2, M2, _ = m.assemble_2d(n, Afun_spire)
        v, _, r2 = m.eig_low(A2, M2, m.NEV)
        refs[n] = v
    ref = m.richardson(refs[256], refs[512], r=2.0)
    print("Reference spire : Richardson(256, 512)  [%.1fs]" % (time.time() - t))

    # --- 4. contenance ----------------------------------------------------
    print("\nChaine complete : spire -> delta = %.5f -> spectre garanti" % DELTA)
    print("%-4s %-14s %-14s %-14s %-9s %-8s"
          % ("k", "inf propage", "ref spire", "sup propage", "largeur", "contenu"))
    ok = True
    for k in range(m.NEV):
        inside = prop_lb[k] <= ref[k] <= prop_ub[k]
        ok &= inside
        print("%-4d %-14.6f %-14.6f %-14.6f %-9.2e %-8s"
              % (k + 1, prop_lb[k], ref[k], prop_ub[k],
                 prop_ub[k] - prop_lb[k], "OUI" if inside else "NON !"))
    print("\nBILAN CONTENANCE : %s   [total %.0fs]"
          % ("PASS" if ok else "FAIL", time.time() - t0))

    # --- 5. figure ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    gmid = 0.5 * (g_lo[i_opt] + g_hi[i_opt])
    ax1.plot(rs, gmid * 1e3, "b-", lw=1.2)
    ax1.axhline(DELTA * 1e3, color="r", ls="--", lw=0.9)
    ax1.axhline(-DELTA * 1e3, color="r", ls="--", lw=0.9)
    ax1.set_xlabel("r")
    ax1.set_ylabel(r"$(A_\varphi - \bar B r/2) \times 10^3$")
    ax1.set_title("Gauge deviation: loop vs uniform field\n"
                  r"$\bar B$ = %.4f, guaranteed $\delta$ = %.5f"
                  % (BBAR, DELTA))
    ax1.grid(alpha=0.3)
    for k in range(m.NEV):
        ax2.add_patch(plt.Rectangle((0.8, lbs[k]), 0.4, ubs[k] - lbs[k],
                                    color="tab:blue", alpha=0.6))
        ax2.add_patch(plt.Rectangle((1.8, prop_lb[k]), 0.4,
                                    prop_ub[k] - prop_lb[k],
                                    color="tab:orange", alpha=0.5))
        ax2.plot([1.8, 2.2], [ref[k]] * 2, "k-", lw=1)
        ax2.plot([2.8, 3.2], [ref[k]] * 2, "k-", lw=1.5)
    ax2.set_xticks([1.0, 2.0, 3.0])
    ax2.set_xticklabels(["bounds for\nuniform field", "bounds for\nloop field", "fine reference\n(loop field)"])
    ax2.set_ylabel("E")
    ax2.set_title("Current loop: guaranteed eigenvalue intervals")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2d_spire_chaine.png"), dpi=300)
    print("Figure : fig2d_spire_chaine.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
