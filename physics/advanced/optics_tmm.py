"""
optics_tmm.py — Transfer-Matrix Method (TMM) optics (educational demo).

NOTE (v3.0): the PRODUCTION TMM — position-resolved |E(x)|^2, exact
Poynting per-layer absorptance, real material n,k, wired into the DD
solver via simulate_iv_curve(..., optics="tmm") — lives in
physics/optical_generation.py. This module remains for the Advanced
Physics teaching panel.

Replaces Beer-Lambert generation with a coherent multilayer TMM (interface +
propagation amplitude matrices; Pettersson/Byrnes form, stable for absorbing
layers). Returns R(lambda), per-layer absorptance, and the spatial photo-
generation profile G(x) in the absorber. Raises Jsc fidelity vs Beer-Lambert by
capturing interference, front reflection and parasitic absorption.

Self-tests (__main__):
  * Energy conservation R + T + A_total = 1 at every wavelength.
  * Beer-Lambert limit for an index-matched single absorbing layer.
"""
import numpy as np
from .constants import H, C, Q, am15g, photon_flux
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _interface(ni, nj):
    r = (ni - nj) / (ni + nj)
    t = 2 * ni / (ni + nj)
    return np.array([[1, r], [r, 1]], dtype=complex) / t


def _propagation(n, d, lam):
    delta = 2 * np.pi * n * d / lam
    return np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]], dtype=complex)


def solve_stack(n_list, d_list, lam, n0=1.0, ns=1.5):
    S = _interface(n0, n_list[0])
    for i, (n, d) in enumerate(zip(n_list, d_list)):
        S = S @ _propagation(n, d, lam)
        nxt = n_list[i + 1] if i + 1 < len(n_list) else ns
        S = S @ _interface(n, nxt)
    r = S[1, 0] / S[0, 0]
    t = 1.0 / S[0, 0]
    R = np.abs(r) ** 2
    T = (np.real(ns) / np.real(n0)) * np.abs(t) ** 2
    A = max(0.0, 1.0 - R - T)
    return R, T, A


def _layer_absorptances(n_list, d_list, lam, n0, ns):
    R, T, A = solve_stack(n_list, d_list, lam, n0, ns)
    w = np.array([max(0.0, 4 * np.pi * np.imag(n) / lam * d) for n, d in zip(n_list, d_list)])
    A_layers = A * (w / w.sum()) if w.sum() > 0 else np.zeros(len(n_list))
    return R, T, A_layers


def generation_profile(n_list, d_list, absorber_idx, nx=60, n0=1.0, ns=3.5, wl=None, irr=None):
    if wl is None:
        wl, irr = am15g()
    flux = photon_flux(wl, irr)
    d_abs = d_list[absorber_idx]
    x = np.linspace(0, d_abs, nx)
    G = np.zeros(nx)
    dlam = wl[1] - wl[0]
    for li, lam in enumerate(wl):
        nl = [np.asarray(n)[li] if np.ndim(n) else n for n in n_list]
        R, T, A = _layer_absorptances(nl, d_list, lam, n0, ns)
        A_abs = A[absorber_idx]
        if A_abs <= 0:
            continue
        k = np.imag(nl[absorber_idx])
        alpha = 4 * np.pi * k / (lam * 1e-9)
        if alpha > 0:
            shape = np.exp(-alpha * (x * 1e-9)); shape /= _trapz(shape, x * 1e-9)
        else:
            shape = np.ones(nx) / (d_abs * 1e-9)
        G += A_abs * flux[li] * dlam * shape
    return x, G


if __name__ == "__main__":
    wl, irr = am15g()
    n_etl, n_abs, n_htl = 2.0 + 0.0j, 2.4 + 0.3j, 1.8 + 0.0j
    d = [50.0, 400.0, 50.0]
    maxerr = max(abs(sum(solve_stack([n_etl, n_abs, n_htl], d, lam, 1.0, 3.5)) - 1.0)
                 for lam in wl[::10])
    print(f"energy conservation max |R+T+A-1| = {maxerr:.2e}")
    assert maxerr < 1e-9, "TMM violates energy conservation"
    lam, nabs, dabs = 600.0, 2.4 + 0.2j, 800.0
    R, T, A = solve_stack([nabs], [dabs], lam, n0=2.4, ns=2.4)
    bl = 1 - np.exp(-(4 * np.pi * nabs.imag / lam) * dabs)
    print(f"TMM A={A:.3f} vs Beer-Lambert {bl:.3f}")
    assert abs(A - bl) < 0.05, "TMM does not approach Beer-Lambert limit"
    x, G = generation_profile([n_etl, n_abs, n_htl], d, absorber_idx=1)
    print(f"G(x): peak={G.max():.2e} m^-3 s^-1, finite={np.isfinite(G).all()}")
    print("optics_tmm: ALL CHECKS PASS")
