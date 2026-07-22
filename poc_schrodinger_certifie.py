#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoC : encadrement CERTIFIE (bornes garanties) des niveaux d'energie de
l'equation de Schrodinger 1D stationnaire

        -psi'' + V(x) psi = E psi        (unites hbar^2/2m = 1)

par une paire encadrante primal/dual, l'analogue spectral de Prager-Synge :

  * borne SUPERIEURE : Rayleigh-Ritz P1 conforme. Par le principe min-max,
    chaque valeur propre discrete lam_h,k majore la valeur propre exacte E_k.
    Integration EXACTE (V constant par morceaux, maillage aligne sur les
    discontinuites) : aucune "crime variationnel".

  * borne INFERIEURE : encadrement de Liu (X. Liu, Appl. Math. Comput. 267,
    2015) :  E_k >= lam_h,k / (1 + C_h^2 lam_h,k), ou C_h est une constante
    EXPLICITE verifiant  ||u - P_h u||_L2 <= C_h ||u - P_h u||_a  pour la
    projection de Ritz P_h. C_h est obtenue par dualite (argument
    d'Aubin-Nitsche a constantes garanties) avec uniquement des inegalites
    de Poincare-Wirtinger elementaires de constante h/pi :

      C_h(s) = (h/pi) * sqrt(1 + Vmax_s (h/pi)^2) * (1 + sqrt(Vmax_s/floor_s))

    pour l'operateur decale de s >= 0 (Vs = V + s, spectre decale de s
    exactement, floor_s = borne inferieure a priori de la premiere valeur
    propre decalee : Poincare pour Dirichlet, s pour Bloch). Chaque s donne
    une borne inferieure VALIDE ; on prend le max sur une grille de s.

Cas 1 : puits fini tronque, Dirichlet sur [-X, X]  -> validation contre la
        solution analytique du probleme tronque (matrices de transfert).
Cas 2 : Kronig-Penney periodique, conditions de Bloch -> bandes certifiees
        E_n(k) et gaps certifies (bords de bande en k=0 et k=pi/L, theorie
        de Hill), validation contre la relation de dispersion analytique.

Reserve : l'algebre lineaire est en flottant (ARPACK/LAPACK) ; le residu
algebrique est rapporte a titre de diagnostic (~1e-12), la cloture par
arithmetique d'intervalles est une etape ulterieure connue (type verifyeig).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

PI = np.pi
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Assemblage P1 (integration exacte, V constant par element)
# ----------------------------------------------------------------------

def assemble_full(x, v_elem, dtype=float):
    """Matrices A (raideur + potentiel) et M (masse) P1 sur tous les noeuds."""
    h = np.diff(x)
    n = len(x)
    e = np.arange(n - 1)
    K11, K12 = 1.0 / h, -1.0 / h
    Mm11, Mm12 = h / 3.0, h / 6.0
    A11 = K11 + v_elem * Mm11
    A12 = K12 + v_elem * Mm12
    rows = np.concatenate([e, e + 1, e, e + 1])
    cols = np.concatenate([e, e + 1, e + 1, e])
    Ad = np.concatenate([A11, A11, A12, A12]).astype(dtype)
    Md = np.concatenate([Mm11, Mm11, Mm12, Mm12]).astype(dtype)
    A = sp.coo_matrix((Ad, (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix((Md, (rows, cols)), shape=(n, n)).tocsc()
    return A, M


def _algebraic_residual(A, M, vals, vecs):
    r = 0.0
    for j in range(len(vals)):
        v = vecs[:, j]
        r = max(r, np.linalg.norm(A @ v - vals[j] * (M @ v)) / np.linalg.norm(M @ v))
    return r


def eig_dirichlet(x, v_elem, nev):
    """Valeurs propres discretes (bornes sup garanties par min-max), Dirichlet."""
    A, M = assemble_full(x, v_elem)
    A, M = A[1:-1, 1:-1], M[1:-1, 1:-1]
    vals, vecs = spla.eigsh(A, k=nev, M=M, sigma=0.0, which="LM")
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    return vals, _algebraic_residual(A, M, vals, vecs)


def eig_bloch(x, v_elem, kL, nev):
    """Valeurs propres discretes, cellule periodique avec phase de Bloch e^{i kL}."""
    n = len(x)
    A, M = assemble_full(x, v_elem, dtype=complex)
    N = n - 1
    tau = np.exp(1j * kL)
    P = sp.lil_matrix((n, N), dtype=complex)
    for j in range(N):
        P[j, j] = 1.0
    P[N, 0] = tau                      # u_N = e^{ikL} u_0
    P = P.tocsc()
    Ar = (P.conj().T @ A @ P).tocsc()
    Mr = (P.conj().T @ M @ P).tocsc()
    Ar = (Ar + Ar.conj().T) * 0.5      # symetrisation numerique
    Mr = (Mr + Mr.conj().T) * 0.5
    vals, vecs = spla.eigsh(Ar, k=nev, M=Mr, sigma=0.0, which="LM")
    idx = np.argsort(vals.real)
    vals, vecs = vals.real[idx], vecs[:, idx]
    return vals, _algebraic_residual(Ar, Mr, vals, vecs)


# ----------------------------------------------------------------------
# Borne inferieure certifiee (Liu + constante explicite par dualite)
# ----------------------------------------------------------------------

def liu_lower(lam_h, h, vmax, floor0, s_grid):
    """Borne inferieure garantie de la valeur propre exacte.
    floor0 : borne inf a priori de lam_1 SANS decalage (Poincare pour
    Dirichlet, 0 pour Bloch). Chaque decalage s de la grille donne une borne
    valide ; on retourne la meilleure."""
    best = -np.inf
    for s in s_grid:
        fl = floor0 + s
        if fl <= 0.0:
            continue
        vm = vmax + s
        C = (h / PI) * np.sqrt(1.0 + vm * (h / PI) ** 2) * (1.0 + np.sqrt(vm / fl))
        lam_s = lam_h + s
        lb = lam_s / (1.0 + C * C * lam_s) - s
        best = max(best, lb)
    return best


S_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 2e3, 90)])

# ----------------------------------------------------------------------
# References analytiques (matrices de transfert, potentiel constant/morceaux)
# ----------------------------------------------------------------------

def transfer(E, v, ell):
    """Matrice de transfert de [psi, psi'] sur une longueur ell, potentiel v."""
    d = E - v
    if d > 1e-12:
        a = np.sqrt(d)
        c, s = np.cos(a * ell), np.sin(a * ell)
        return np.array([[c, s / a], [-a * s, c]])
    if d < -1e-12:
        b = np.sqrt(-d)
        c, s = np.cosh(b * ell), np.sinh(b * ell)
        return np.array([[c, s / b], [b * s, c]])
    return np.array([[1.0, ell], [0.0, 1.0]])


def well_exact_levels(a, X, V0, nmax, Emax):
    """Niveaux exacts du puits tronque [-X,X] (Dirichlet), par parite."""
    def shoot(E, parity):
        y = np.array([1.0, 0.0]) if parity == "even" else np.array([0.0, 1.0])
        y = transfer(E, 0.0, a) @ y
        y = transfer(E, V0, X - a) @ y
        return y[0]

    roots = []
    for parity in ("even", "odd"):
        Es = np.linspace(1e-9, Emax, 24001)
        vals = np.array([shoot(E, parity) for E in Es])
        sgn = np.sign(vals)
        for i in np.where(sgn[:-1] * sgn[1:] < 0)[0]:
            roots.append(brentq(shoot, Es[i], Es[i + 1], args=(parity,),
                                xtol=1e-13, rtol=8.9e-16))
    return np.sort(np.array(roots))[:nmax]


def kp_disc(E, aw, bb, V0):
    Mm = transfer(E, V0, bb) @ transfer(E, 0.0, aw)
    return 0.5 * (Mm[0, 0] + Mm[1, 1])


def kp_exact_levels(aw, bb, V0, kL, nmax, Emax):
    """Racines E de disc(E) = cos(kL) : niveaux de Bloch exacts."""
    f = lambda E: kp_disc(E, aw, bb, V0) - np.cos(kL)
    Es = np.linspace(1e-9, Emax, 16001)
    vals = np.array([f(E) for E in Es])
    sgn = np.sign(vals)
    roots = []
    for i in np.where(sgn[:-1] * sgn[1:] < 0)[0]:
        roots.append(brentq(f, Es[i], Es[i + 1], xtol=1e-13, rtol=8.9e-16))
    return np.sort(np.array(roots))[:nmax]


# ----------------------------------------------------------------------
# Cas 1 : puits fini tronque
# ----------------------------------------------------------------------
V0W, AW, XW = 20.0, 1.0, 6.0      # profondeur, demi-largeur, troncature
LAMP = (PI / (2 * XW)) ** 2       # Poincare-Friedrichs sur [-X, X]
NLEV_W = 3


def mesh_well(m):
    """h = 1/m ; noeuds sur -X..X, alignes sur +-a (a et X entiers)."""
    N = int(2 * XW * m)
    x = np.linspace(-XW, XW, N + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    v = np.where(np.abs(xm) < AW, 0.0, V0W)
    return x, v, 1.0 / m


def certify_well(m, nev=NLEV_W):
    x, v, h = mesh_well(m)
    vals, res = eig_dirichlet(x, v, nev)
    lbs = np.array([liu_lower(lv, h, V0W, LAMP, S_GRID) for lv in vals])
    return lbs, vals, res


# ----------------------------------------------------------------------
# Cas 2 : Kronig-Penney (cellule [0, L], puits [0, a], barriere [a, L])
# ----------------------------------------------------------------------
V0K, AK, LK = 50.0, 0.7, 1.0
BK = LK - AK
NLEV_K = 4


def mesh_kp(m):
    """h = L/(10 m) ; noeud aligne sur x = a = 0.7."""
    N = 10 * m
    x = np.linspace(0.0, LK, N + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    v = np.where(xm < AK, 0.0, V0K)
    return x, v, LK / N


def certify_kp(m, kL, nev=NLEV_K):
    x, v, h = mesh_kp(m)
    vals, res = eig_bloch(x, v, kL, nev)
    lbs = np.array([liu_lower(lv, h, V0K, 0.0, S_GRID) for lv in vals])
    return lbs, vals, res


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PoC Schrodinger 1D certifie : bornes garanties [inf, sup] sur les niveaux")
    print("=" * 78)

    # ---------------- Cas 1 : puits fini ----------------
    print("\nCAS 1 : puits fini (V0=%.0f, a=%.0f, domaine [-%.0f,%.0f] Dirichlet)"
          % (V0W, AW, XW, XW))
    E_ex_w = well_exact_levels(AW, XW, V0W, NLEV_W, Emax=60.0)

    m_fine = 200
    lbs, ubs, res = certify_well(m_fine)
    h_fine = 1.0 / m_fine
    print("Maillage fin : h = %.4g (N = %d elements), residu algebrique max = %.1e"
          % (h_fine, int(2 * XW * m_fine), res))
    print("%-4s %-16s %-16s %-16s %-10s %-8s"
          % ("k", "borne inf", "E exact", "borne sup", "largeur", "contenu"))
    ok_all = True
    for k in range(NLEV_W):
        inside = lbs[k] <= E_ex_w[k] <= ubs[k]
        ok_all &= inside
        print("%-4d %-16.10f %-16.10f %-16.10f %-10.2e %-8s"
              % (k + 1, lbs[k], E_ex_w[k], ubs[k], ubs[k] - lbs[k],
                 "OUI" if inside else "NON !"))

    # Convergence
    ms = [5, 10, 20, 40, 80, 160]
    widths = np.zeros((len(ms), NLEV_W))
    hs = []
    for i, m in enumerate(ms):
        lb, ub, _ = certify_well(m)
        widths[i] = ub - lb
        hs.append(1.0 / m)
    hs = np.array(hs)
    slopes = [np.polyfit(np.log(hs), np.log(widths[:, k]), 1)[0]
              for k in range(NLEV_W)]
    print("Pentes de convergence (largeur d'intervalle vs h) :",
          ", ".join("%.2f" % s for s in slopes), "(attendu : 2)")

    # ---------------- Cas 2 : Kronig-Penney ----------------
    print("\nCAS 2 : Kronig-Penney (L=%.0f, puits a=%.1f, barriere V0=%.0f)"
          % (LK, AK, V0K))
    kLs = np.linspace(0.0, PI, 9)
    m_kp_coarse, m_kp_fine = 2, 60

    bands_lb_c = np.zeros((len(kLs), NLEV_K))
    bands_ub_c = np.zeros((len(kLs), NLEV_K))
    bands_lb_f = np.zeros((len(kLs), NLEV_K))
    bands_ub_f = np.zeros((len(kLs), NLEV_K))
    bands_ex = np.zeros((len(kLs), NLEV_K))
    res_max = 0.0
    ok_kp = True
    for i, kL in enumerate(kLs):
        lb, ub, r1 = certify_kp(m_kp_coarse, kL)
        bands_lb_c[i], bands_ub_c[i] = lb, ub
        lb, ub, r2 = certify_kp(m_kp_fine, kL)
        bands_lb_f[i], bands_ub_f[i] = lb, ub
        res_max = max(res_max, r1, r2)
        ex = kp_exact_levels(AK, BK, V0K, kL, NLEV_K, Emax=600.0)
        bands_ex[i] = ex
        ok_kp &= np.all((bands_lb_f[i] <= ex) & (ex <= bands_ub_f[i]))
        ok_kp &= np.all((bands_lb_c[i] <= ex) & (ex <= bands_ub_c[i]))
    print("Maillage fin : h = %.4g, residu algebrique max = %.1e"
          % (LK / (10 * m_kp_fine), res_max))
    print("Contenance analytique dans [inf, sup] (9 k x %d bandes x 2 maillages) : %s"
          % (NLEV_K, "OUI" if ok_kp else "NON !"))
    ok_all &= ok_kp

    # Tableau au bord de zone k = pi/L
    i_pi = len(kLs) - 1
    print("Bord de zone kL = pi (maillage fin) :")
    print("%-6s %-16s %-16s %-16s %-10s"
          % ("bande", "borne inf", "E exact", "borne sup", "largeur"))
    for n in range(NLEV_K):
        print("%-6d %-16.8f %-16.8f %-16.8f %-10.2e"
              % (n + 1, bands_lb_f[i_pi, n], bands_ex[i_pi, n],
                 bands_ub_f[i_pi, n], bands_ub_f[i_pi, n] - bands_lb_f[i_pi, n]))

    # Gaps certifies (bords de bande en k=0 ou pi : theorie de Hill 1D)
    gap1_lb = bands_lb_f[i_pi, 1] - bands_ub_f[i_pi, 0]     # entre bandes 1 et 2 (k=pi)
    gap1_ex = bands_ex[i_pi, 1] - bands_ex[i_pi, 0]
    gap2_lb = bands_lb_f[0, 2] - bands_ub_f[0, 1]           # entre bandes 2 et 3 (k=0)
    gap2_ex = bands_ex[0, 2] - bands_ex[0, 1]
    print("GAP 1 certifie (k=pi) : >= %.6f  (exact %.6f)" % (gap1_lb, gap1_ex))
    print("GAP 2 certifie (k=0)  : >= %.6f  (exact %.6f)" % (gap2_lb, gap2_ex))

    print("\nBILAN CONTENANCE GLOBALE : %s" % ("PASS" if ok_all else "FAIL"))

    # ---------------- Figures ----------------
    # Fig 1 : puits + niveaux certifies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.linspace(-XW, XW, 1000)
    Vx = np.where(np.abs(xs) < AW, 0.0, V0W)
    ax1.plot(xs, Vx, "k-", lw=1.5)
    cols = ["tab:blue", "tab:orange", "tab:green"]
    for k in range(NLEV_W):
        ax1.axhspan(lbs[k], ubs[k], color=cols[k], alpha=0.35)
        ax1.axhline(E_ex_w[k], color=cols[k], lw=0.8, ls="--")
        ax1.text(XW * 0.65, ubs[k] + 0.35, "E%d" % (k + 1), color=cols[k])
    ax1.set_xlabel("x")
    ax1.set_ylabel("E")
    ax1.set_title("Finite well: certified levels\n(bands = [lower, upper], dashed = analytic)")
    for k in range(NLEV_W):
        ax2.plot([k + 1, k + 1], [(lbs[k] - E_ex_w[k]) * 1e3, (ubs[k] - E_ex_w[k]) * 1e3],
                 "-", color=cols[k], lw=6, solid_capstyle="butt", alpha=0.7)
        ax2.plot(k + 1, 0.0, "kx", ms=9)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("level k")
    ax2.set_ylabel(r"(bound $-$ analytic) $\times 10^{3}$")
    ax2.set_title("Enclosure around the analytic value (h = %.3g)" % h_fine)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_puits_certifie.png"), dpi=300)

    # Fig 2 : convergence
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for k in range(NLEV_W):
        ax.loglog(hs, widths[:, k], "o-", color=cols[k],
                  label="level %d (slope %.2f)" % (k + 1, slopes[k]))
    ax.loglog(hs, widths[-1, 0] * (hs / hs[-1]) ** 2, "k--", lw=0.8,
              label=r"slope 2 ($\propto h^2$)")
    ax.set_xlabel("h")
    ax.set_ylabel("certified interval width")
    ax.set_title("Convergence of the enclosures (finite well)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=300)

    # Fig 3 : bandes Kronig-Penney certifiees
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    kk = kLs / PI
    for n in range(NLEV_K):
        c = "C%d" % n
        ax1.fill_between(kk, bands_lb_c[:, n], bands_ub_c[:, n], color=c, alpha=0.35)
        ax1.plot(kk, bands_ex[:, n], "k.", ms=4)
        ax2.fill_between(kk, bands_lb_f[:, n], bands_ub_f[:, n], color=c, alpha=0.8)
        ax2.plot(kk, bands_ex[:, n], "k.", ms=4)
    ax1.set_title("Coarse mesh (h = %.3g):\ncertified ribbons visible" % (LK / (10 * m_kp_coarse)))
    ax2.set_title("Fine mesh (h = %.3g):\nwidth ~ %.0e" % (LK / (10 * m_kp_fine),
                  np.max(bands_ub_f - bands_lb_f)))
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$kL/\pi$")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("E")
    ax2.annotate("certified gap 1\n>= %.3f" % gap1_lb,
                 xy=(1.0, 0.5 * (bands_ub_f[i_pi, 0] + bands_lb_f[i_pi, 1])),
                 xytext=(0.55, 0.5 * (bands_ub_f[i_pi, 0] + bands_lb_f[i_pi, 1])),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle("Kronig-Penney: certified band structure (black dots = analytic dispersion)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_kp_bandes.png"), dpi=300)

    print("\nFigures ecrites dans %s :" % OUT)
    print("  fig_puits_certifie.png, fig_convergence.png, fig_kp_bandes.png")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
