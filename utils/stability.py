"""
utils/stability.py — reduced-order hysteresis and lifetime projection.
=======================================================================
Two tools with disclosed model classes and Consensus-verified sources.

1.  SCAN-RATE HYSTERESIS INDEX (reduced-order).
    Full transient ionic-electronic drift-diffusion (the Richardson 2016
    treatment, DOI 10.1039/C5EE02740C) shows hysteresis vanishing at both
    the fast-scan limit (ions frozen — forward and reverse traces coincide)
    and the slow-scan limit (ions equilibrated at every bias — traces again
    coincide), with a maximum when the scan time is comparable to the ionic
    relaxation time. We ship that verified *shape* as a reduced-order model:

        HI(rate) = HI_max · 4x/(1+x)²,   x = t_scan/τ_ion

    which is exactly zero in both limits and peaks at t_scan = τ_ion.
    HI_max scales with mobile-ion density via saturating N/(N+N_ref).
    This is a design-guidance model, NOT a transient solver — the module
    docstring, the UI, and MAJOR_CONCERNS.md all say so. The full
    transient solver remains on the roadmap.

2.  ARRHENIUS T80 LIFETIME PROJECTION.
    T80 (time to 80% of initial performance) is the ISOS consensus
    stability metric (Khenkin et al. 2020, Nature Energy,
    DOI 10.1038/s41560-019-0529-5). Given measured T80 values at two or
    more stress temperatures, we fit  t80(T) = A·exp(Ea/kT)  and project
    to operating temperature — the standard accelerated-ageing analysis.
    The fit quality and the extrapolation distance are both reported,
    because an Arrhenius projection is only as good as the assumption of
    a single thermally activated mechanism.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

K_B_EV = 8.617333262e-5      # eV/K

HYSTERESIS_REF = "richardson_2016_ees"
LIFETIME_REF = "khenkin_2020_natenergy"


# ─────────────────────────────────────────────────────────────────────────
#  1. Reduced-order hysteresis index
# ─────────────────────────────────────────────────────────────────────────
def hysteresis_index(scan_rate_V_per_s: float,
                     tau_ion_s: float = 10.0,
                     N_ion_cm3: float = 1e17,
                     V_range_V: float = 1.2,
                     HI0: float = 0.30,
                     N_ref_cm3: float = 1e17) -> float:
    """Hysteresis index HI ∈ [0, ~HI0) for a given scan rate.

    HI is the normalised area between forward and reverse J-V traces.
    Shape from the transient ionic drift-diffusion literature
    (Richardson 2016): →0 for both very fast and very slow scans,
    maximum at t_scan ≈ τ_ion. τ_ion ~ 1-100 s and mobile-ion densities
    ~1e16-1e18 cm⁻³ span the values reported for MAPbI3-class devices.
    """
    if scan_rate_V_per_s <= 0:
        raise ValueError("scan rate must be positive")
    t_scan = V_range_V / scan_rate_V_per_s
    x = t_scan / tau_ion_s
    shape = 4.0 * x / (1.0 + x) ** 2          # 0 at both limits, 1 at x=1
    hi_max = HI0 * (N_ion_cm3 / (N_ion_cm3 + N_ref_cm3))
    return float(hi_max * shape)


def hysteresis_curve(rates: Sequence[float], **kw) -> Dict:
    rates = np.asarray(list(rates), float)
    hi = np.array([hysteresis_index(r, **kw) for r in rates])
    return {"rates_V_per_s": rates.tolist(), "HI": hi.tolist(),
            "reference": HYSTERESIS_REF,
            "model": "reduced-order (shape-calibrated); not a transient solver"}


# ─────────────────────────────────────────────────────────────────────────
#  2. Arrhenius T80 projection
# ─────────────────────────────────────────────────────────────────────────
def fit_arrhenius(temps_C: Sequence[float],
                  t80_hours: Sequence[float]) -> Tuple[float, float, float]:
    """Fit t80(T) = A·exp(Ea/kT). Returns (Ea_eV, lnA, R²).

    Requires ≥2 stress temperatures; ln(t80) is linear in 1/T.
    """
    T = np.asarray(list(temps_C), float) + 273.15
    t = np.asarray(list(t80_hours), float)
    if T.size < 2:
        raise ValueError("need t80 at ≥2 stress temperatures")
    if np.any(t <= 0):
        raise ValueError("t80 values must be positive")
    x = 1.0 / (K_B_EV * T)
    y = np.log(t)
    Ea, lnA = np.polyfit(x, y, 1)
    yhat = Ea * x + lnA
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(Ea), float(lnA), r2


def project_t80(Ea_eV: float, lnA: float, T_op_C: float) -> float:
    """Projected t80 (hours) at operating temperature."""
    T = T_op_C + 273.15
    return float(np.exp(lnA + Ea_eV / (K_B_EV * T)))


def t80_report(temps_C, t80_hours, T_op_C=35.0) -> Dict:
    Ea, lnA, r2 = fit_arrhenius(temps_C, t80_hours)
    t_op = project_t80(Ea, lnA, T_op_C)
    extrap = abs(min(temps_C) - T_op_C)
    warn = []
    if r2 < 0.95:
        warn.append(f"Arrhenius fit R²={r2:.3f} < 0.95 — a single "
                    "activated mechanism may not describe these data.")
    if extrap > 30:
        warn.append(f"projection extrapolates {extrap:.0f} °C below the "
                    "lowest stress temperature — treat as indicative only.")
    return {"Ea_eV": Ea, "R2": r2, "T_op_C": T_op_C,
            "t80_projected_h": t_op,
            "t80_projected_years": t_op / 8760.0,
            "reference": LIFETIME_REF,
            "metric": "T80 per ISOS consensus (Khenkin 2020)",
            "warnings": warn}
