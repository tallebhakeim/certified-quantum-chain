#!/usr/bin/env python3
"""Worked example of Section 4 (recomputable by hand or with a pocket calculator).
Lower bound for E_1 of (-i grad - A*)^2 on the unit square, B = 6, n = 256,
from the Rayleigh-Ritz value lam_h (upper bound) only."""
import numpy as np
PI = np.pi
lam_h = 20.311372          # Q1 Rayleigh-Ritz value, n = 256 (Table 3)
B, n = 6.0, 256
h = 1.0 / n
f = 2 * PI**2              # Dirichlet ground state of the unit square
aoo = B / 2 * np.sqrt(2) / 2   # max |A*| = (B/2) * corner radius
print(f"h = {h:.6e}, h/pi = {h/PI:.6e}, f = {f:.6f}, a_inf = {aoo:.6f}")
def bound(s):
    fs = f + s
    K = 1 + 2*aoo*(1/np.sqrt(fs) + aoo/fs) + (aoo**2 + s)/fs
    C = (h/PI + (aoo + np.sqrt(s))*np.sqrt(3)*(h/PI)**2) * K
    lb = (lam_h + s)/(1 + C**2*(lam_h + s)) - s
    return K, C, lb
for s in [0.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
    K, C, lb = bound(s)
    print(f"s = {s:7.1f}: K_s = {K:.6f}, C_h = {C:.6e}, lower bound = {lb:.6f}, width = {lam_h-lb:.3e}")
ss = np.concatenate([[0.0], np.geomspace(1e-2, 5e3, 90)])
lbs = [bound(s)[2] for s in ss]; i = int(np.argmax(lbs))
print(f"best s on the grid = {ss[i]:.3f}, lower bound = {lbs[i]:.6f}")
sf = np.linspace(0, 400, 400001); lf = [bound(s)[2] for s in sf]; j = int(np.argmax(lf))
print(f"best s (fine scan) = {sf[j]:.3f}, lower bound = {lf[j]:.6f}")
# transport with delta (loop field, Section 6)
delta = 0.004349
print("transport example: (sqrt(20.311372)+delta)^2 =", (np.sqrt(20.311372)+delta)**2)
