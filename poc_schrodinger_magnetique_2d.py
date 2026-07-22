#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etape 2 : Schrodinger MAGNETIQUE 2D a bornes certifiees, avec propagation
de l'incertitude sur le potentiel vecteur (chaine classique -> quantique).

Operateur : H_A = (-i grad - A)^2 sur le dot carre Omega = [-w, w]^2,
Dirichlet (unites hbar^2/2m = 1, q = 1).

Champ "vrai" (realiste, non uniforme) : B(x,y) = B0 [1 + beta rho / w^2],
rho = x^2+y^2 (bombement quadratique ~ paire de bobines hors reglage
Helmholtz). Jauge symetrique exacte :
    A_true = (-y, x) [ B0/2 + B0 beta rho / (4 w^2) ]   (div A = 0)
Le certificateur ne connait que le champ uniforme A* = (-y, x) B0/2 et la
borne EXACTE  delta = sup_Omega |A_true - A*| = [B0 beta/(4w^2)] (sqrt2 w)^3
(dans le pipeline final, delta est fourni par le solveur magnetostatique
certifie DGM ; ici le champ est analytique donc delta est exact).

LEMME DE PROPAGATION (le pont classique->quantique) : pour tout k,
    | sqrt(E_k(A)) - sqrt(E_k(A')) | <= ||A - A'||_inf.
Preuve min-max + Young : ||(-i grad - A)u||^2 <= (1+eps)||(-i grad - A')u||^2
+ (1+1/eps) delta^2 ||u||^2, optimisation en eps => (sqrt(E') + delta)^2.
Donc  E_k(A_true) appartient a [ (sqrt(lb_k) - delta)^2, (sqrt(ub_k) + delta)^2 ]
ou [lb_k, ub_k] est l'encadrement certifie de E_k(A*).

CERTIFICATION 2D de H* = (-i grad - A*)^2 :
  * borne SUP : Rayleigh-Ritz Q1 conforme, quadrature de Gauss 5x5 par
    maille = EXACTE (integrandes polynomiaux de degre <= 8 par direction).
  * borne INF : encadrement de Liu  E_k >= lam_h/(1 + C_h^2 lam_h) avec
    constante C_h explicite par dualite. Ingredients, tous demontres
    elementairement sur maillage carre uniforme (interpolation Q1 tensorisee,
    Pi0 = projection L2 par element, Poincare/Wirtinger 1D de constante h/pi) :
       || grad(z - I_h z) ||      <= sqrt2 (h/pi)   || D2 z ||
       || z - I_h z ||            <= sqrt3 (h/pi)^2 || D2 z ||
       || D2 z || <= || Lap z ||   (domaine convexe, Dirichlet, Grisvard)
       E_1(H*) >= 2 pi^2 / L^2     (inegalite diamagnetique, L = 2w)
    Pour l'operateur decale de s >= 0 (spectre decale exactement) :
       ||z||    <= ||phi|| / (f+s),   ||Dz|| <= ||phi|| / sqrt(f+s),
       ||grad z|| <= ||phi|| (1/sqrt(f+s) + aoo/(f+s)),   aoo = ||A*||_inf
       ||Lap z|| <= K_s ||phi||,
       K_s = 1 + 2 aoo (1/sqrt(f+s) + aoo/(f+s)) + (aoo^2 + s)/(f+s)
       C_h(s) = [ sqrt2 h/pi + (aoo + sqrt(s)) sqrt3 (h/pi)^2 ] K_s
    et on prend le max des bornes sur une grille de s (chacune est valide).

Validation : reference numerique fine (Q1 + extrapolation de Richardson)
pour H* ET pour H_{A_true} ; verification de contenance partout.
Bonus physiques certifies : decalage diamagnetique de E1, levee de
degenerescence (paire (1,2)/(2,1)) certifiee.

Reserves (memes que l'etape 1) : algebre lineaire flottante (residus
rapportes), cloture par arithmetique d'intervalles = etape ulterieure.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

PI = np.pi
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Parametres physiques
# ----------------------------------------------------------------------
W = 0.5                 # demi-cote du dot ; Omega = [-w, w]^2, L = 1
L = 2 * W
B0 = 6.0                # champ au centre
BETA = 0.02             # bombement quadratique de 2 % (bobines reelles)
DCOEF = B0 * BETA / (4 * W * W)          # A_true = (-y,x)(B0/2 + DCOEF*rho)
DELTA = DCOEF * (np.sqrt(2.0) * W) ** 3  # ||A_true - A*||_inf (EXACT, au coin)
AOO = 0.5 * B0 * np.sqrt(2.0) * W        # ||A*||_inf
FLOOR = 2 * PI ** 2 / L ** 2             # plancher diamagnetique E1 >= 2pi^2/L^2
NEV = 8

S_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 5e3, 90)])
NG = 5                                    # Gauss 5 pts/direction (exact <= deg 9)


def A_star(x, y):
    return -0.5 * B0 * y, 0.5 * B0 * x


def A_true(x, y):
    f = 0.5 * B0 + DCOEF * (x * x + y * y)
    return -y * f, x * f


# ----------------------------------------------------------------------
# Assemblage Q1 (grille uniforme n x n sur [-w,w]^2, quadrature exacte)
# ----------------------------------------------------------------------

def assemble_2d(n, Afun):
    """Matrices A (forme magnetique) et M (masse), dofs interieurs, CSR."""
    h = L / n
    gp, gw = np.polynomial.legendre.leggauss(NG)
    gp = 0.5 * (gp + 1.0)                  # sur [0,1]
    gw = 0.5 * gw
    # points de quadrature tensorises sur la maille de reference
    XI, ETA = np.meshgrid(gp, gp, indexing="ij")
    WQ = np.outer(gw, gw).ravel() * h * h  # poids x jacobien
    XI, ETA = XI.ravel(), ETA.ravel()      # (G,)
    G = len(WQ)
    # fonctions de forme Q1 et gradients (physiques) aux points de Gauss
    N = np.stack([(1 - XI) * (1 - ETA), XI * (1 - ETA), XI * ETA,
                  (1 - XI) * ETA], axis=1)                    # (G,4)
    Gx = np.stack([-(1 - ETA), (1 - ETA), ETA, -ETA], axis=1) / h
    Gy = np.stack([-(1 - XI), -XI, XI, (1 - XI)], axis=1) / h
    # coordonnees des points de Gauss de toutes les mailles
    cx, cy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx, cy = cx.ravel(), cy.ravel()                            # (C,)
    x0, y0 = -W + cx * h, -W + cy * h
    XQ = x0[:, None] + XI[None, :] * h                         # (C,G)
    YQ = y0[:, None] + ETA[None, :] * h
    Ax, Ay = Afun(XQ, YQ)
    A2 = Ax * Ax + Ay * Ay
    # matrices locales
    S = np.einsum("g,gi,gj->ij", WQ, Gx, Gx) + np.einsum("g,gi,gj->ij", WQ, Gy, Gy)
    Mloc = np.einsum("g,gi,gj->ij", WQ, N, N)
    P = np.einsum("cg,gi,gj->cij", A2 * WQ[None, :], N, N)
    R = (np.einsum("cg,gi,gj->cij", Ax * WQ[None, :], N, Gx)
         + np.einsum("cg,gi,gj->cij", Ay * WQ[None, :], N, Gy))
    Aloc = S[None, :, :] + P + 1j * (R - np.transpose(R, (0, 2, 1)))
    # scatter global
    nn = n + 1
    node = (cy[:, None] + np.array([0, 0, 1, 1])[None, :]) * nn \
         + (cx[:, None] + np.array([0, 1, 1, 0])[None, :])
    rows = np.repeat(node, 4, axis=1).ravel()
    cols = np.tile(node, (1, 4)).ravel()
    Ag = sp.coo_matrix((Aloc.ravel(), (rows, cols)),
                       shape=(nn * nn, nn * nn)).tocsr()
    Mg = sp.coo_matrix((np.broadcast_to(Mloc[None, :, :],
                                        (len(cx), 4, 4)).ravel(),
                        (rows, cols)), shape=(nn * nn, nn * nn)).tocsr()
    # dofs interieurs
    ix, iy = np.meshgrid(np.arange(nn), np.arange(nn), indexing="ij")
    interior = ((ix > 0) & (ix < n) & (iy > 0) & (iy < n)).T.ravel()
    idx = np.where(interior)[0]
    return Ag[np.ix_(idx, idx)].tocsc(), Mg[np.ix_(idx, idx)].tocsc(), h


def eig_low(A, M, nev):
    vals, vecs = spla.eigsh(A, k=nev, M=M, sigma=0.0, which="LM")
    order = np.argsort(vals.real)
    vals, vecs = vals.real[order], vecs[:, order]
    res = 0.0
    for j in range(nev):
        v = vecs[:, j]
        res = max(res, np.linalg.norm(A @ v - vals[j] * (M @ v))
                  / np.linalg.norm(M @ v))
    return vals, vecs, res


# ----------------------------------------------------------------------
# Borne inferieure de Liu (constantes 2D explicites, decalage optimise)
# ----------------------------------------------------------------------

def liu_lower_2d(lam_h, h, aoo, floor, s_grid):
    best = -np.inf
    for s in s_grid:
        fs = floor + s
        Ks = 1.0 + 2.0 * aoo * (1.0 / np.sqrt(fs) + aoo / fs) + (aoo * aoo + s) / fs
        Ch = (np.sqrt(2.0) * h / PI
              + (aoo + np.sqrt(s)) * np.sqrt(3.0) * (h / PI) ** 2) * Ks
        lam_s = lam_h + s
        best = max(best, lam_s / (1.0 + Ch * Ch * lam_s) - s)
    return best


def certify(n, Afun, aoo):
    A, M, h = assemble_2d(n, Afun)
    vals, vecs, res = eig_low(A, M, NEV)
    lbs = np.array([liu_lower_2d(v, h, aoo, FLOOR, S_GRID) for v in vals])
    return lbs, vals, vecs, res, h


def richardson(e_coarse, e_fine, r=2.0):
    return (r * r * e_fine - e_coarse) / (r * r - 1.0)


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Etape 2 : Schrodinger magnetique 2D certifie + propagation delta(A)")
    print("=" * 78)
    print("Dot [-%.1f,%.1f]^2, B0 = %.1f, bombement beta = %.0f%%" % (W, W, B0, 100 * BETA))
    print("||A*||_inf = %.4f, delta = ||A_true - A*||_inf = %.6f (exact)"
          % (AOO, DELTA))
    print("Plancher diamagnetique E1 >= %.4f" % FLOOR)

    # --- certification de H* (champ uniforme) a plusieurs maillages ---
    ns_cert = [32, 64, 128, 256]
    cert = {}
    for n in ns_cert:
        t = time.time()
        lbs, ubs, vecs, res, h = certify(n, A_star, AOO)
        cert[n] = (lbs, ubs, vecs, res, h)
        print("H* certifie n=%4d (h=%.4g) : res alg %.1e, largeur E1 %.2e, E8 %.2e  [%.1fs]"
              % (n, h, res, ubs[0] - lbs[0], ubs[7] - lbs[7], time.time() - t))

    n_c = ns_cert[-1]
    lbs, ubs = cert[n_c][0], cert[n_c][1]

    # --- references fines (Richardson) pour H* et H_true ---
    refs = {}
    for tag, Afun in (("H*", A_star), ("H_true", A_true)):
        t = time.time()
        try:
            n2 = 512
            _, v2, _, _, _ = certify_vals_only(n2, Afun)
            n1 = 256
        except MemoryError:
            n2, n1 = 384, 192
            _, v2, _, _, _ = certify_vals_only(n2, Afun)
        _, v1, _, _, _ = certify_vals_only(n1, Afun)
        refs[tag] = richardson(v1, v2, r=n2 / n1)
        print("Reference %s : Richardson(n=%d, n=%d)  [%.1fs]" % (tag, n1, n2, time.time() - t))

    # --- contenance H* ---
    print("\nH* (champ uniforme B=%.0f), n=%d :" % (B0, n_c))
    print("%-4s %-14s %-14s %-14s %-9s %-8s"
          % ("k", "borne inf", "reference", "borne sup", "largeur", "contenu"))
    ok_all = True
    for k in range(NEV):
        inside = lbs[k] <= refs["H*"][k] <= ubs[k]
        ok_all &= inside
        print("%-4d %-14.6f %-14.6f %-14.6f %-9.2e %-8s"
              % (k + 1, lbs[k], refs["H*"][k], ubs[k], ubs[k] - lbs[k],
                 "OUI" if inside else "NON !"))

    # --- propagation vers le champ vrai ---
    prop_lb = np.maximum(np.sqrt(np.maximum(lbs, 0.0)) - DELTA, 0.0) ** 2
    prop_ub = (np.sqrt(ubs) + DELTA) ** 2
    print("\nPropagation |sqrt(E(A_true)) - sqrt(E(A*))| <= delta = %.5f :" % DELTA)
    print("%-4s %-14s %-14s %-14s %-9s %-8s"
          % ("k", "inf propage", "ref champ vrai", "sup propage", "largeur", "contenu"))
    for k in range(NEV):
        inside = prop_lb[k] <= refs["H_true"][k] <= prop_ub[k]
        ok_all &= inside
        print("%-4d %-14.6f %-14.6f %-14.6f %-9.2e %-8s"
              % (k + 1, prop_lb[k], refs["H_true"][k], prop_ub[k],
                 prop_ub[k] - prop_lb[k], "OUI" if inside else "NON !"))

    # --- bonus certifies : decalage diamagnetique + levee de degenerescence ---
    lbs0, ubs0, _, _, _ = certify(n_c, lambda x, y: (0.0 * x, 0.0 * y), 0.0)
    shift_lo = lbs[0] - ubs0[0]
    shift_hi = ubs[0] - lbs0[0]
    split_lo = lbs[2] - ubs[1]
    split_hi = ubs[2] - lbs[1]
    print("\nDecalage diamagnetique certifie de E1 (B=0 -> B=%.0f) : [%.4f, %.4f]"
          % (B0, shift_lo, shift_hi))
    print("Levee de degenerescence certifiee paire (1,2)/(2,1) : [%.4f, %.4f]"
          % (split_lo, split_hi))
    print("(sans champ, E2 = E3 = 5 pi^2 : splitting nul ; reference fine %.4f)"
          % (refs["H*"][2] - refs["H*"][1]))
    ok_all &= (split_lo > 0.0) and (shift_lo > 0.0)

    print("\nBILAN CONTENANCE GLOBALE : %s   [total %.0fs]"
          % ("PASS" if ok_all else "FAIL", time.time() - t0))

    # --- convergence des largeurs ---
    hs = np.array([cert[n][4] for n in ns_cert])
    widths = np.array([[cert[n][1][k] - cert[n][0][k] for k in (0, 3, 7)]
                       for n in ns_cert])
    slopes = [np.polyfit(np.log(hs), np.log(widths[:, j]), 1)[0] for j in range(3)]
    print("Pentes de convergence (largeurs, niveaux 1/4/8) : %s (attendu ~2)"
          % ", ".join("%.2f" % s for s in slopes))

    # ------------------------- figures -------------------------
    # Fig 1 : echelle des niveaux, 3 colonnes
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for k in range(NEV):
        ax.plot([0.8, 1.2], [refs["H*"][k]] * 2, "k-", lw=1)
        ax.add_patch(plt.Rectangle((0.8, lbs[k]), 0.4, ubs[k] - lbs[k],
                                   color="tab:blue", alpha=0.6))
        ax.add_patch(plt.Rectangle((1.8, prop_lb[k]), 0.4,
                                   prop_ub[k] - prop_lb[k],
                                   color="tab:orange", alpha=0.5))
        ax.plot([1.8, 2.2], [refs["H_true"][k]] * 2, "k-", lw=1)
        ax.plot([2.8, 3.2], [refs["H_true"][k]] * 2, "k-", lw=1.5)
    ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(["certified H*\n(uniform field)",
                        "propagated intervals\n(+/- delta on A)",
                        "fine reference\ntrue field"])
    ax.set_ylabel("E")
    ax.set_title("Certified classical-to-quantum chain (B0=%.0f, delta=%.3f)\n"
                 "black bars = fine numerical references" % (B0, DELTA))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2d_echelle_niveaux.png"), dpi=300)

    # Fig 2 : convergence
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    labs = ["level 1", "level 4", "level 8"]
    for j in range(3):
        ax.loglog(hs, widths[:, j], "o-", label="%s (slope %.2f)" % (labs[j], slopes[j]))
    ax.loglog(hs, widths[-1, 0] * (hs / hs[-1]) ** 2, "k--", lw=0.8, label=r"slope 2")
    ax.set_xlabel("h")
    ax.set_ylabel("certified width")
    ax.set_title("Convergence of the enclosures (H*)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2d_convergence.png"), dpi=300)

    # Fig 3 : densites |psi|^2 des 4 premiers etats (H*, n=128)
    n_v = 128
    _, vals_v, vecs_v, _, _ = certify(n_v, A_star, AOO)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for j in range(4):
        psi = np.zeros((n_v + 1) * (n_v + 1), dtype=complex)
        ix, iy = np.meshgrid(np.arange(n_v + 1), np.arange(n_v + 1), indexing="ij")
        interior = ((ix > 0) & (ix < n_v) & (iy > 0) & (iy < n_v)).T.ravel()
        psi[np.where(interior)[0]] = vecs_v[:, j]
        dens = np.abs(psi.reshape(n_v + 1, n_v + 1)) ** 2
        axs[j].imshow(dens, origin="lower", extent=[-W, W, -W, W], cmap="magma")
        axs[j].set_title("state %d: E in [%.3f, %.3f]"
                         % (j + 1, cert[n_c][0][j], cert[n_c][1][j]), fontsize=8)
    fig.suptitle("Densities |psi|^2 (H*, B0=%.0f); the 2-3 pair is the lifted degeneracy" % B0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2d_densites.png"), dpi=300)

    print("Figures : fig2d_echelle_niveaux.png, fig2d_convergence.png, fig2d_densites.png")
    return 0 if ok_all else 1


def certify_vals_only(n, Afun):
    A, M, h = assemble_2d(n, Afun)
    vals, vecs, res = eig_low(A, M, NEV)
    return None, vals, None, res, h


if __name__ == "__main__":
    raise SystemExit(main())
