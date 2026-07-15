"""
defects.py — beyond single-level SRH.

The base solver uses one discrete trap level. Real perovskites have multiple
levels and continuous band-tail / Gaussian defect distributions. This module
provides:
  * srh_rate            single-level Shockley-Read-Hall
  * multilevel_srh      sum over discrete levels
  * gaussian_dos / exp_tail  continuous distributions, integrated over energy

Self-tests (__main__):
  * multilevel with one level == srh_rate.
  * a Gaussian of vanishing width -> the discrete single-level rate.
"""
import numpy as np
from .constants import K_B, Q, T_REF


def srh_rate(n, p, ni, Et_rel_mid, Nt, sigma_n=1e-15, sigma_p=1e-15, vth=1e7, T=T_REF):
    """Single-level SRH (cm^-3 s^-1). Et_rel_mid: trap energy relative to mid-gap (eV).
    n,p,ni in cm^-3; Nt in cm^-3; sigma in cm^2; vth in cm/s."""
    kT = K_B * T / Q
    n1 = ni * np.exp(Et_rel_mid / kT)
    p1 = ni * np.exp(-Et_rel_mid / kT)
    cn = sigma_n * vth
    cp = sigma_p * vth
    return cn * cp * Nt * (n * p - ni ** 2) / (cn * (n + n1) + cp * (p + p1))


def multilevel_srh(n, p, ni, levels, T=T_REF):
    """levels: list of dicts with keys Et_rel_mid, Nt, [sigma_n, sigma_p, vth]."""
    total = 0.0
    for L in levels:
        total += srh_rate(n, p, ni, L["Et_rel_mid"], L["Nt"],
                          L.get("sigma_n", 1e-15), L.get("sigma_p", 1e-15),
                          L.get("vth", 1e7), T)
    return total


def gaussian_dos(n, p, ni, Et0, sigma_E, Nt_total, npts=121, span=5.0, **kw):
    """Gaussian defect band centred at Et0 (eV vs mid-gap), std sigma_E (eV),
    integrated SRH recombination."""
    E = np.linspace(Et0 - span * sigma_E, Et0 + span * sigma_E, npts)
    g = np.exp(-0.5 * ((E - Et0) / sigma_E) ** 2)
    g /= np.trapezoid(g, E) if hasattr(np, "trapezoid") else np.trapz(g, E)
    dens = Nt_total * g
    rates = np.array([srh_rate(n, p, ni, e, nt, **kw) for e, nt in zip(E, dens * (E[1] - E[0]))])
    return rates.sum()


def exp_tail(n, p, ni, E0, Nt0, E_char, npts=80, **kw):
    """Exponential band tail: Nt(E) = Nt0 exp(-(E0-E)/E_char) for E<E0 (eV)."""
    E = np.linspace(E0 - 8 * E_char, E0, npts)
    dens = Nt0 * np.exp(-(E0 - E) / E_char)
    rates = np.array([srh_rate(n, p, ni, e, nt, **kw) for e, nt in zip(E, dens * (E[1] - E[0]))])
    return rates.sum()


if __name__ == "__main__":
    n, p, ni = 1e16, 1e14, 1e8
    single = srh_rate(n, p, ni, 0.0, 1e15)
    multi = multilevel_srh(n, p, ni, [dict(Et_rel_mid=0.0, Nt=1e15)])
    print(f"single-level SRH = {single:.3e}")
    assert abs(multi - single) / single < 1e-12, "1-level multilevel must equal single"

    narrow = gaussian_dos(n, p, ni, 0.0, 1e-3, 1e15)
    print(f"narrow-Gaussian = {narrow:.3e} (should ~ single {single:.3e})")
    assert abs(narrow - single) / single < 0.05, "narrow Gaussian must approach discrete level"

    two = multilevel_srh(n, p, ni, [dict(Et_rel_mid=0.0, Nt=1e15),
                                     dict(Et_rel_mid=0.1, Nt=5e14)])
    assert two > single, "adding a level should add recombination"
    print(f"two-level = {two:.3e}; exp-tail = {exp_tail(n,p,ni,0.2,1e15,0.05):.3e}")
    print("defects: ALL CHECKS PASS")
