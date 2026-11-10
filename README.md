# Guaranteed eigenvalue bounds for the magnetic Schrodinger operator

Code for the manuscript

> H. Talleb, *Guaranteed two-sided eigenvalue bounds for the magnetic Schrodinger operator with an inexactly known vector potential*, Journal of Computational and Applied Mathematics (submitted).

The code computes intervals that contain the eigenvalues of (-i grad - A)^2 with
Dirichlet conditions on a square: upper bounds by the conforming Q1 Rayleigh-Ritz
method with exact quadrature, lower bounds by Liu's theorem with the explicit
constant of Theorem 2 of the manuscript, and transfer of the bounds to an
inexactly known vector potential (Lemma 1). It reproduces every table and figure.

## Requirements

Python 3.9 or later, numpy, scipy, matplotlib (`pip install -r requirements.txt`).

## Reproduce

```
bash run_all.sh
```

Each script ends with a check that the exact or reference eigenvalues lie in the
computed intervals and prints PASS or FAIL. Total run time: about 3 minutes on a laptop.

| Script | Manuscript |
|---|---|
| `exemple_a_la_main.py` | Section 4, Table 1 (worked example) |
| `poc_schrodinger_certifie.py` | Section 5.1, Table 2, Figs. 1-2 (finite well, Kronig-Penney) |
| `poc_schrodinger_magnetique_2d.py` | Section 5.2, Algorithm 1, Table 3, Fig. 3 |
| `fan_B_certifie.py` | Section 5.2, Fig. 4 (fields 0 to 10) |
| `brique_bobine_certifiee.py` | Section 6, Algorithm 2, Table 4, Fig. 5 (current loop) |

Main functions: `liu_lower_2d` (lower bound of Theorem 2 and Remark 2),
`liu_lower` (one-dimensional constant), `assemble_2d` (Q1 matrices with the exact
5 x 5 Gauss rule), `aphi_enclose` (enclosure of the Biot-Savart integral).

## Versions

- v2.0.0 (2026-11-05): version of the JCAM submission. Gradient interpolation constant
  h/pi (proved in Lemma 2), sharper bound of the second derivative in the
  Biot-Savart quadrature error, worked example added.
- Earlier unversioned upload: code of a previous submission.

## Citation

Archived on Zenodo: concept DOI https://doi.org/10.5281/zenodo.24117325 (all versions); this version: https://doi.org/10.5281/zenodo.24117326.

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
