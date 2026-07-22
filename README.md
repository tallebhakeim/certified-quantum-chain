# Certified chain from coil geometry to quantum spectra

Python implementation accompanying the manuscript:

> H. Talleb, *A certified chain from coil geometry to quantum spectra:
> guaranteed two-sided eigenvalue bounds for the magnetic Schrödinger
> operator*, submitted to Computer Physics Communications (COMPHY-D-26-00722).

Every table and figure of the paper is reproduced by the scripts below.
All bounds printed as `[lower, upper]` are mathematically guaranteed
enclosures (see the paper for the analysis); each script ends with a
containment self-test against analytic spectra or fine numerical references
and prints `PASS`/`FAIL`.

## Requirements

Python >= 3.9 with `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).
No other dependency; no compiled code. Tested on macOS with Python 3.9.

```
pip install -r requirements.txt
```

## Scripts

| Script | What it does | Paper items | Runtime* |
|--------|--------------|-------------|----------|
| `poc_schrodinger_certifie.py` | 1D certified bounds: finite well (vs analytic transfer matrices) + Kronig-Penney Bloch bands and certified gaps | Table 1, Figs. 2-4 | ~1 min |
| `poc_schrodinger_magnetique_2d.py` | 2D magnetic operator: Rayleigh-Ritz upper + Liu lower bounds with explicit tensorized constants; propagation lemma demo (2 % synthetic field bulge); certified diamagnetic shift and degeneracy lifting | Table 2, Figs. 5, 6, 9 | ~2 min |
| `fan_B_certifie.py` | Certified magneto-spectroscopic fan, B = 0..10 (imports the 2D module) | Fig. 7 | ~1 min |
| `brique_bobine_certifiee.py` | Real coplanar current loop: rigorous Biot-Savart enclosure, equioscillating reference field, end-to-end certified spectrum (imports the 2D module) | Table 3, Fig. 8 | ~1 min |
| `fig_bridge.py` | Conceptual schematic (classical-quantum bridge) | Fig. 1 | seconds |
| `gfx_abstract.py` | Graphical abstract | - | seconds |

*Indicative, on a laptop; the 2D scripts assemble complex Hermitian Q1
matrices up to 512 x 512 meshes and use `scipy.sparse.linalg.eigsh`
(shift-invert).

Run each script from its own directory, e.g.:

```
python3 poc_schrodinger_certifie.py
python3 poc_schrodinger_magnetique_2d.py
python3 fan_B_certifie.py
python3 brique_bobine_certifiee.py
```

Figures (PNG, 300 dpi) are written next to the scripts.

## Structure of the certification (pointers into the code)

* Upper bounds: conforming Rayleigh-Ritz with exact Gauss quadrature,
  `assemble_2d` / `eig_low` in `poc_schrodinger_magnetique_2d.py`.
* Lower bounds: Liu's framework with fully explicit constants,
  `liu_lower_2d` (2D, tensorized interpolation constants sqrt(2) h/pi and
  sqrt(3) (h/pi)^2, convexity, diamagnetic floor, optimized spectral shift)
  and `liu_lower` (1D) in `poc_schrodinger_certifie.py`.
* Propagation lemma |sqrt(E_k(A)) - sqrt(E_k(A'))| <= ||A - A'||_inf:
  applied in both 2D scripts (`prop_lb`, `prop_ub`).
* Rigorous Biot-Savart enclosure (composite trapezoid + explicit f''
  majorant + floating-point inflation): `aphi_enclose` in
  `brique_bobine_certifiee.py`.

## Reserves

Linear algebra runs in floating point; algebraic eigenresidua (~1e-10) are
printed with every result and are at least four orders of magnitude below
all certified widths. Closing this last gap with interval arithmetic and
verified eigensolvers is standard and modular (see the paper, Section 6).
