"""
physics/tandem.py — Multi-technology tandem engine
=========================================================
Couples the perovskite fast simulator (physics/device.py) with the new
silicon (physics/silicon.py) and organic (physics/organic.py) engines
into 2-terminal (monolithic) and 4-terminal tandem devices.

Physics
-------
* The bottom cell is simulated under the spectrum transmitted by the
  top cell: phi_bot(lam) = phi_AM15G(lam) * T_top(lam), with
  T_top = (1 - R_int) * exp(-alpha_pk(lam) * d_top) and the corrected
  Tauc absorption alpha_pk(E) = alpha0 * sqrt(E - Eg)/E of the top
  absorber, plus a flat parasitic loss for the interconnect/TCO stack.
* 2T coupling: proper series J-V addition. Each subcell J-V curve is
  inverted to V(J) on its photocurrent branch; the tandem curve is
  V_2T(J) = V_top(J) + V_bot(J) on the common current range, and the
  MPP is extracted from the summed curve. This replaces min-Jsc
  heuristics and correctly captures current-mismatch FF gains
  (the mismatched subcell operates off its own MPP).
* 4T: independent MPPs, P = P_top + P_bot(filtered).

Validation targets (scripts/run_multi_tech_validation.py):
  * 29.15% monolithic perovskite(1.68 eV)/SHJ — Al-Ashouri et al.,
    Science 370, 1300 (2020), DOI 10.1126/science.abd4016
    (Voc 1.92 V, Jsc 19.26 mA/cm2, FF 79.5%)
  * 31.25% — Chin et al., Science 381, 59 (2023), DOI 10.1126/science.adg0091
  * 32.5%  — Mariotti et al., Science 381, 63 (2023), DOI 10.1126/science.adf5872
  * 34.58% record — Jia et al., Nature 644, 912 (2025),
    DOI 10.1038/s41586-025-09333-z
  The engine is expected to bracket this 29-34.6% range as interface
  quality (Nt, J0s, R_int) is varied between 2020-grade and 2025-grade
  presets; per-device agreement targets are <=10% on PCE.
"""
from __future__ import annotations

import numpy as np

Q = 1.602176634e-19
PIN = 100.0
HC_EV_NM = 1239.84


def _top_transmission(lams_nm, Eg_top, alpha0, d_top_nm, R_int=0.05,
                      parasitic=0.03):
    """Spectral transmission through the top absorber + interconnect."""
    E = HC_EV_NM / np.maximum(np.asarray(lams_nm, float), 1e-6)
    alpha = np.where(E > Eg_top,
                     alpha0 * np.sqrt(np.maximum(E - Eg_top, 0.0)) / E, 0.0)
    T = (1.0 - R_int) * (1.0 - parasitic) * np.exp(-alpha * d_top_nm * 1e-7)
    return T


def _v_of_j(voltages, currents):
    """Invert a J-V curve to V(J) on the photocurrent branch (J>=0)."""
    v = np.asarray(voltages, float)
    j = np.asarray(currents, float)
    mask = j > -0.5  # keep the generating quadrant + a little past Voc
    v, j = v[mask], j[mask]
    order = np.argsort(j)
    j_s, v_s = j[order], v[order]
    # deduplicate for interp
    j_u, idx = np.unique(j_s, return_index=True)
    return j_u, v_s[idx]


def simulate_tandem_2T(r_top, r_bot, Rs_int_ohm_cm2=1.5):
    """
    Series-connect two subcell results (dicts with voltages/currents).
    Returns tandem metrics from the summed V(J) curve.

    Rs_int_ohm_cm2: additional lumped series resistance of the monolithic
    interconnect (recombination junction / TCO / metallization), typical
    1-3 Ohm cm2 in published perovskite/Si tandems. This is what keeps
    2T tandem FF in the measured 77-84% range rather than the subcell-sum
    ideal.
    """
    jt, vt = _v_of_j(r_top["voltages"], r_top["currents"])
    jb, vb = _v_of_j(r_bot["voltages"], r_bot["currents"])
    J_max = min(jt.max(), jb.max())
    J = np.linspace(0.0, J_max * 0.9995, 400)
    V = (np.interp(J, jt, vt) + np.interp(J, jb, vb)
         - J * Rs_int_ohm_cm2 / 1000.0)
    P = V * J
    k = int(np.argmax(P))
    Pmax, Vmpp, Jmpp = float(P[k]), float(V[k]), float(J[k])
    Voc = float(np.interp(0.0, jt, vt) + np.interp(0.0, jb, vb))
    Jsc = float(J_max)  # limited by the lower-current subcell
    FF = Pmax / (Jsc * Voc) if Jsc > 0 and Voc > 0 else 0.0
    return {
        "terminal": "2T", "Voc": Voc, "Jsc": Jsc, "FF": FF,
        "Pmax": Pmax, "PCE": Pmax / PIN * 100.0,
        "Vmpp": Vmpp, "Jmpp": Jmpp,
        "J_axis": J, "V_axis": V,
    }


def simulate_tandem_4T(r_top, r_bot):
    Pmax = r_top["Pmax"] + r_bot["Pmax"]
    return {
        "terminal": "4T", "Pmax": Pmax, "PCE": Pmax / PIN * 100.0,
        "Voc": r_top["Voc"], "Jsc": r_top["Jsc"],
        "FF": r_top["FF"],
        "P_top": r_top["Pmax"], "P_bot": r_bot["Pmax"],
    }


def simulate_perovskite_silicon_tandem(
        top_htl, top_abs, top_etl, d_top_abs_nm,
        si_arch, Nt_top=1e14, terminal="2T",
        R_int=0.05, parasitic=0.03, T=300.0,
        d_top_htl=15, d_top_etl=25, Rs_int_ohm_cm2=1.5,
        lc_eta=0.0):
    """
    Monolithic (2T) or mechanically stacked (4T) perovskite/c-Si tandem.

    top_*   : perovskite subcell materials (Material objects) and thickness
    si_arch : SiliconArchitecture for the bottom cell
    """
    from physics.device import fast_simulate
    from physics.silicon import simulate_silicon
    from physics.spectrum import photon_flux, AM15G_WAVELENGTHS

    # --- top subcell under full AM1.5G ---
    # path_factor: optical path enhancement in the top absorber from the
    # textured front (conformal perovskite on pyramidal Si; Chin 2023).
    # The same effective optical thickness is used for the transmission
    # to the bottom cell so photon bookkeeping stays consistent.
    path_factor = kwargs_path = 1.6
    d_opt_nm = d_top_abs_nm * path_factor
    r_top = fast_simulate(top_htl, top_abs, top_etl,
                          d_top_htl, d_top_abs_nm, d_top_etl, Nt_top, T)
    # rescale the top photocurrent branch to the textured optical thickness
    _lam = AM15G_WAVELENGTHS.astype(float)
    _phi = photon_flux(_lam)
    _a0 = getattr(top_abs, "alpha_coeff", 1e5) or 1e5
    _E = HC_EV_NM / _lam
    _al = np.where(_E > top_abs.Eg,
                   _a0 * np.sqrt(np.maximum(_E - top_abs.Eg, 0)) / _E, 0.0)
    _m = _E > top_abs.Eg
    _dl = np.gradient(_lam)
    j_flat = float(np.sum(_phi[_m] * (1 - np.exp(-_al[_m] * d_top_abs_nm * 1e-7)) * _dl[_m]))
    j_text = float(np.sum(_phi[_m] * (1 - np.exp(-_al[_m] * d_opt_nm * 1e-7)) * _dl[_m]))
    scale = j_text / max(j_flat, 1e-9)
    r_top = dict(r_top)
    r_top["currents"] = r_top["currents"] * scale
    for k in ("Jsc", "Jmpp", "Pmax"):
        r_top[k] = r_top[k] * scale
    r_top["PCE"] = r_top["Pmax"] / PIN * 100.0

    # --- bottom subcell under the filtered spectrum ---
    lams = AM15G_WAVELENGTHS.astype(float)
    phi0 = photon_flux(lams)
    alpha0 = _a0
    T_top = _top_transmission(lams, top_abs.Eg, alpha0, d_opt_nm,
                              R_int, parasitic)
    r_bot = simulate_silicon(si_arch, T=T, incident=(lams, phi0 * T_top))

    out = (simulate_tandem_2T(r_top, r_bot, Rs_int_ohm_cm2) if terminal == "2T"
           else simulate_tandem_4T(r_top, r_bot))

    # --- luminescent coupling (optional, 2T): photons re-emitted by the
    # radiatively recombining top junction are absorbed by the bottom cell.
    # Reduced one-pass model per Steiner & Geisz 2012 (nonlinear LC ∝ the
    # luminescent junction's excess recombination J_gen − J; registry
    # steiner_2012_apl): at the operating point the top recombines at
    # (Jsc_top − Jmpp), a fraction lc_eta of which returns as bottom-cell
    # photocurrent. Magnitude anchor: >50% of excess pairs usable in
    # perovskite/Si tandems (Jaeger 2020, jager_2020_solrrl). NOT a
    # self-consistent emission calculation — disclosed in the UI.
    if lc_eta > 0.0 and terminal == "2T":
        dJ = lc_eta * max(r_top["Jsc"] - out["Jmpp"], 0.0)
        if dJ > 0 and r_bot["Jsc"] > 0:
            boost = 1.0 + dJ / r_bot["Jsc"]
            r_bot = simulate_silicon(si_arch, T=T,
                                     incident=(lams, phi0 * T_top * boost))
            out = simulate_tandem_2T(r_top, r_bot, Rs_int_ohm_cm2)
            out["lc_dJ_bot_mA_cm2"] = dJ
    out["lc_eta"] = lc_eta
    out.update({"top": r_top, "bottom": r_bot,
                "stack": f"perovskite({top_abs.Eg:.2f} eV)/Si-{si_arch.name}",
                "T_top_spectrum": (lams, T_top)})
    return out


def simulate_perovskite_organic_tandem(
        top_htl, top_abs, top_etl, d_top_abs_nm, blend,
        Nt_top=1e14, terminal="2T", R_int=0.05, parasitic=0.03, T=300.0):
    """Perovskite (wide-gap) / organic (narrow-gap) tandem."""
    from physics.device import fast_simulate
    from physics.organic import simulate_organic
    from physics.spectrum import photon_flux, AM15G_WAVELENGTHS

    r_top = fast_simulate(top_htl, top_abs, top_etl,
                          15, d_top_abs_nm, 25, Nt_top, T)
    lams = AM15G_WAVELENGTHS.astype(float)
    phi0 = photon_flux(lams)
    alpha0 = getattr(top_abs, "alpha_coeff", 1e5) or 1e5
    T_top = _top_transmission(lams, top_abs.Eg, alpha0, d_top_abs_nm,
                              R_int, parasitic)
    r_bot = simulate_organic(blend, T=T, incident=(lams, phi0 * T_top))
    out = (simulate_tandem_2T(r_top, r_bot) if terminal == "2T"
           else simulate_tandem_4T(r_top, r_bot))
    out.update({"top": r_top, "bottom": r_bot,
                "stack": f"perovskite({top_abs.Eg:.2f} eV)/{blend.name}"})
    return out


def current_matching_scan(top_htl, top_abs, top_etl, si_arch,
                          d_range_nm=(150, 800), n=14, Nt_top=1e14, **kw):
    """Scan top-absorber thickness to find the 2T current-matching point."""
    ds = np.linspace(d_range_nm[0], d_range_nm[1], n)
    rows = []
    for d in ds:
        r = simulate_perovskite_silicon_tandem(
            top_htl, top_abs, top_etl, float(d), si_arch,
            Nt_top=Nt_top, terminal="2T", **kw)
        rows.append({"d_top_nm": float(d), "PCE": r["PCE"],
                     "Jsc_top": r["top"]["Jsc"], "Jsc_bot": r["bottom"]["Jsc"],
                     "Voc": r["Voc"], "FF": r["FF"]})
    best = max(rows, key=lambda x: x["PCE"])
    return rows, best
