"""
utils/energy_yield.py — annual energy yield for all four technologies.
=======================================================================
STC (25 °C, AM1.5G, 1000 W/m²) is a certification fiction; revenue is
kWh under real operating temperatures. This module estimates annual
specific yield (kWh per kWp) for any of the tool's technologies using:

1.  ENGINE-DERIVED temperature coefficients: γ = dP/dT is computed by
    running the actual physics engine at two temperatures (288 K, 318 K)
    — never a hardcoded datasheet number. For silicon this reproduces
    the published SHJ-class behaviour; for perovskite the drift-diffusion
    fast model's Voc(T) supplies the slope; the tandem inherits a
    power-weighted combination of its subcells.
2.  The NOCT cell-temperature model  T_cell = T_amb + (NOCT-20)/800 · G,
    the standard correlation class reviewed by Skoplaki & Palyvos 2009
    (DOI 10.1016/j.solener.2008.10.008, Consensus-verified).
3.  A linear power-temperature correlation  P = P_STC·(G/1000)·
    [1 + γ·(T_cell - 25)] — the working-equation class tabulated in the
    same review. DISCLOSED SIMPLIFICATIONS: linear-in-G response (no
    low-light Voc drop), clear-sky synthetic profiles rather than TMY
    weather files, no spectral (air-mass) correction. These make the
    output a comparative estimate between technologies and climates,
    not a bankability simulation — stated in the UI.

Note on organic cells: the organic engine's Voc model (E_g - E_loss) has
no explicit temperature dependence, so its engine-derived γ reflects only
the FF/transport terms and is smaller in magnitude than measured OPV
coefficients; this is disclosed wherever the number is shown.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

REF = "skoplaki_2009_solen"           # registry key for the model class

# ── representative climates: (label → season list of (T_amb_day °C,
#     peak irradiance W/m², daylight hours, weight in year)) ────────────
CLIMATES: Dict[str, list] = {
    "Hot desert (Riyadh-class)": [
        (38, 1000, 13, 0.30), (30, 950, 12, 0.25),
        (20, 850, 11, 0.20), (28, 950, 12, 0.25)],
    "Hot humid (Karachi-class)": [
        (34, 900, 13, 0.30), (30, 800, 12, 0.30),
        (22, 800, 11, 0.15), (28, 850, 12, 0.25)],
    "Temperate (Berlin-class)": [
        (22, 850, 16, 0.30), (12, 650, 12, 0.25),
        (3, 350, 8, 0.20), (12, 650, 12, 0.25)],
    "High-altitude cool (Andes-class)": [
        (15, 1050, 13, 0.30), (10, 1000, 12, 0.25),
        (5, 900, 11, 0.20), (10, 1000, 12, 0.25)],
}

NOCT_C = 45.0        # typical open-rack NOCT (Skoplaki & Palyvos 2009)


def _tech_power(tech: str, T_K: float) -> float:
    """STC-irradiance power (mW/cm²) of a representative device of the
    given technology at cell temperature T_K, from the live engines."""
    if tech == "Perovskite (MAPbI3 workbench)":
        from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
        from physics.device import fast_simulate
        r = fast_simulate(HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],
                          ETL_DB["SnO2"], 150, 500, 50, 1e14, T=T_K)
        return r["PCE"]
    if tech == "Silicon (SHJ-IBC record class)":
        from physics.silicon import SILICON_PRESETS, simulate_silicon
        r = simulate_silicon(SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"], T=T_K)
        return r["PCE"]
    if tech == "Organic (PM6:Y6)":
        from physics.organic import ORGANIC_PRESETS, simulate_organic
        r = simulate_organic(ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"],
                             T=T_K)
        return r["PCE"]
    if tech == "Tandem (perovskite/Si, 2020-class)":
        # power-weighted subcell combination (engines at T; interconnect
        # values from the certified 2020 grade)
        pk = _tech_power("Perovskite (MAPbI3 workbench)", T_K)
        si = _tech_power("Silicon (SHJ-IBC record class)", T_K)
        return 0.62 * pk / 19.10 * 29.16 * 0.62 ** 0 + 0.38 * si / 26.77 * 29.16 \
            if False else 29.16 * (0.62 * pk / 19.10 + 0.38 * si / 26.77)
    raise KeyError(tech)


TECHNOLOGIES = ("Perovskite (MAPbI3 workbench)",
                "Silicon (SHJ-IBC record class)",
                "Organic (PM6:Y6)",
                "Tandem (perovskite/Si, 2020-class)")


def engine_gamma(tech: str) -> Tuple[float, float]:
    """(P_STC [% PCE at 25°C], γ [1/K fractional]) from the live engine
    evaluated at 288 K and 318 K."""
    p_lo = _tech_power(tech, 288.0)
    p_hi = _tech_power(tech, 318.0)
    p_25 = _tech_power(tech, 298.15)
    gamma = (p_hi - p_lo) / 30.0 / p_25          # fractional per K
    return p_25, gamma


def annual_yield(tech: str, climate: str) -> Dict:
    """Annual specific yield (kWh/kWp), performance ratio, and the
    engine-derived temperature coefficient for a technology/climate pair."""
    p25, gamma = engine_gamma(tech)
    seasons = CLIMATES[climate]
    kwh_per_kwp = 0.0
    ref_kwh = 0.0                    # same irradiance at fixed 25 °C
    day_series = []
    for (Tamb, Gpk, hours, w) in seasons:
        t = np.linspace(0, hours, 48)
        G = Gpk * np.clip(np.sin(np.pi * t / hours), 0.0, None) ** 1.5
        Tcell = Tamb + (NOCT_C - 20.0) / 800.0 * G        # Skoplaki NOCT
        P = (G / 1000.0) * (1.0 + gamma * (Tcell - 25.0)) # per kWp
        e_day = np.trapezoid(P, t)                        # kWh/kWp/day
        e_ref = np.trapezoid(G / 1000.0, t)
        kwh_per_kwp += w * 365.0 * e_day
        ref_kwh += w * 365.0 * e_ref
        day_series.append({"season_weight": w, "t_h": t.tolist(),
                           "P_per_kWp": P.tolist(),
                           "Tcell_C": Tcell.tolist()})
    return {"tech": tech, "climate": climate,
            "P_STC_pct": p25,
            "gamma_pct_per_K": gamma * 100.0,
            "kWh_per_kWp_year": float(kwh_per_kwp),
            "performance_ratio": float(kwh_per_kwp / ref_kwh),
            "reference": REF,
            "day_series": day_series,
            "note": ("Engine-derived γ; NOCT cell-temperature model and "
                     "linear power correlation per Skoplaki & Palyvos 2009. "
                     "Clear-sky synthetic profiles — comparative estimate, "
                     "not a bankability simulation.")}
