"""
physics/energy_yield.py — energy yield under real operating conditions
(v3.1, Phase 4)
=======================================================================

The synopsis promises design "considering diverse operating conditions".
STC efficiency (AM1.5G, 25 C, 1000 W/m2) is a single point; what a deployed
cell delivers is the integral of P(G, T) over the irradiance/temperature it
actually sees. This module provides that integral.

Physics
-------
* Temperature dependence: the drift-diffusion / fast model is evaluated
  directly at each cell temperature (band gap, ni, kT effects included by
  the solver itself — no external temperature coefficient is assumed).
* Irradiance dependence (standard one-diode scaling relations, e.g.
  M. A. Green, "Solar Cells: Operating Principles, Technology and System
  Applications", Prentice-Hall (1982), ch. 5):
      Jsc(S)  = S * Jsc(1 sun)                       (linear generation)
      Voc(S)  = Voc(1 sun) + n_id * kT/q * ln(S)     (diode law)
      FF(S)   ~ FF(1 sun)  (weak dependence; stated approximation)
  with ideality n_id = 1.3 (typical perovskite/CdTe SRH-dominated value;
  configurable).
* Cell temperature from ambient + irradiance via the NOCT model
  (IEC 61215): T_cell = T_amb + (NOCT - 20 C) * G / 800 W/m2,
  NOCT = 45 C default.

Outputs: P(G, T) surface, daily energy yield [Wh/m2/day] for a clear-sky
profile or a user-supplied irradiance series, and the harvesting efficiency
(energy yield / plane-of-array insolation).
"""
from __future__ import annotations
import numpy as np

KB_EV = 8.617333262e-5   # eV/K


def clear_sky_day(peak_W_m2=1000.0, hours=None):
    """Simple clear-sky irradiance profile: half-sine between 6h and 18h.
    Returns (hours, G_W_m2). Peak at solar noon."""
    if hours is None:
        hours = np.linspace(0.0, 24.0, 97)
    G = np.where((hours >= 6.0) & (hours <= 18.0),
                 peak_W_m2 * np.sin(np.pi * (hours - 6.0) / 12.0), 0.0)
    return hours, np.maximum(G, 0.0)


def power_at(G_W_m2, T_amb_C, stc, n_id=1.3, NOCT_C=45.0, T_stc_K=298.15):
    """Power density [W/m2 = mW/cm2 *10] at irradiance G and ambient T,
    scaled from the solver's STC metrics dict `stc` = {Voc, Jsc, FF, PCE}
    evaluated at the CELL temperature (see energy_yield for the exact-T
    pathway). Returns (P_W_m2, T_cell_C)."""
    if G_W_m2 <= 1e-6:
        return 0.0, T_amb_C
    S = G_W_m2 / 1000.0
    T_cell_C = T_amb_C + (NOCT_C - 20.0) * G_W_m2 / 800.0
    T_cell_K = T_cell_C + 273.15
    jsc = stc["Jsc"] * S                                   # mA/cm2
    voc = stc["Voc"] + n_id * KB_EV * T_cell_K * np.log(max(S, 1e-6))
    voc = max(voc, 0.0)
    ff = stc["FF"]
    P_mW_cm2 = voc * jsc * ff                              # mW/cm2
    return float(P_mW_cm2 * 10.0), T_cell_C                # W/m2


def energy_yield(sim_fn, T_amb_C=25.0, profile=None, n_id=1.3,
                 NOCT_C=45.0, n_T_anchors=4):
    """Daily energy yield of a device.

    sim_fn(T_K) -> {Voc, Jsc, FF, PCE}: the tool's solver evaluated at 1 sun
    and cell temperature T_K (e.g. lambda T: fast_simulate(..., T=T)). The
    temperature dependence is taken from the SOLVER via interpolation over
    `n_T_anchors` anchor temperatures spanning the day's cell-temperature
    range; only the irradiance scaling uses the diode relations above.

    profile: (hours, G_W_m2) tuple; default clear-sky half-sine day.

    Returns dict with hours, G, P(t) [W/m2], T_cell(t), E_day [Wh/m2],
    insolation [Wh/m2], harvesting_efficiency [%].
    """
    if profile is None:
        profile = clear_sky_day()
    hours, G = profile
    # cell-temperature range for the day
    Tc = T_amb_C + (NOCT_C - 20.0) * G / 800.0
    T_lo, T_hi = float(Tc.min()), float(max(Tc.max(), Tc.min() + 1.0))
    T_anchors = np.linspace(T_lo, T_hi, max(2, n_T_anchors))
    stc_at = [sim_fn(T + 273.15) for T in T_anchors]

    def stc_interp(T_C):
        out = {}
        for k in ("Voc", "Jsc", "FF", "PCE"):
            vals = [s[k] for s in stc_at]
            out[k] = float(np.interp(T_C, T_anchors, vals))
        return out

    P = np.zeros_like(G)
    Tcell = np.zeros_like(G)
    for i, (g, ) in enumerate(zip(G)):
        t_c = T_amb_C + (NOCT_C - 20.0) * g / 800.0
        p, _ = power_at(g, T_amb_C, stc_interp(t_c), n_id=n_id, NOCT_C=NOCT_C)
        P[i] = p
        Tcell[i] = t_c
    dt = np.gradient(hours)                       # hours
    E_day = float(np.sum(P * dt))                 # Wh/m2
    insol = float(np.sum(G * dt))                 # Wh/m2
    return {"hours": hours, "G_W_m2": G, "P_W_m2": P, "T_cell_C": Tcell,
            "E_day_Wh_m2": E_day, "insolation_Wh_m2": insol,
            "harvesting_efficiency_pct": 100.0 * E_day / max(insol, 1e-9),
            "assumptions": f"one-diode scaling (Green 1982), n_id={n_id}, "
                           f"NOCT={NOCT_C} C (IEC 61215), FF(S)~const; "
                           "temperature dependence from the solver itself"}


def intensity_temperature_map(sim_fn, suns=(0.2, 0.5, 1.0), T_C=(15, 25, 45, 65),
                              n_id=1.3):
    """PCE(suns, T) map for reporting. Temperature axis: exact solver calls;
    intensity axis: diode scaling from each temperature's STC point."""
    out = np.zeros((len(T_C), len(suns)))
    for i, T in enumerate(T_C):
        stc = sim_fn(T + 273.15)
        for j, S in enumerate(suns):
            T_K = T + 273.15
            jsc = stc["Jsc"] * S
            voc = max(stc["Voc"] + n_id * KB_EV * T_K * np.log(S), 0.0)
            p_mw = voc * jsc * stc["FF"]
            out[i, j] = p_mw / S if S > 0 else 0.0   # mW/cm2 / (S*100 mW/cm2) * 100 = %
    return {"T_C": list(T_C), "suns": list(suns), "PCE_pct": out}
