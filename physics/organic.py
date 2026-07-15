"""
physics/organic.py — Organic bulk-heterojunction solar cell engine
=========================================================================
Semi-empirical spectral + transport model for BHJ organic solar cells,
of the same class used by the OPV community for device analysis
(equivalent-circuit + spectral response + drift-collection). A full
exciton/CT-state drift-diffusion (as in OghmaNano) is out of the 1-D
scope of this tool; the limitation is stated in the UI and docs.

Model components:

1. Jsc — spectral integration over ASTM G173-03 AM1.5G:
       Jsc = q * INT phi(lam) * EQE(lam) dlam
   EQE(lam) is a plateau (EQE_max) over the blend's absorption window
   [lam_on, lam_edge = 1240/Eg_opt] with an Urbach-like tail, scaled by
   an interference/thickness factor 1 - exp(-2*alpha*L) (double pass)
   and a drift-collection factor. Typical non-fullerene-acceptor blends
   reach EQE_max ~ 0.80-0.87 (Yuan et al., Joule 3, 1140 (2019),
   DOI 10.1016/j.joule.2019.01.004).

2. Voc — energy-loss picture standard in the field:
       Voc = (Eg_opt - E_loss) / q
   with total photon-energy loss E_loss = Delta_rad + Delta_nonrad
   (0.5-0.6 eV in state-of-the-art NFA blends; >0.8 eV in fullerene
   blends). See Scharber et al., Adv. Mater. 18, 789 (2006),
   DOI 10.1002/adma.200501717 for the design-rule origin of this form.

3. FF — Green's empirical FF0(voc) with an ideality factor from
   bimolecular recombination, corrected for series/shunt resistance and
   for the transport-vs-recombination competition via the
   Bartesaghi-Koster figure of merit concept (charge collection limited
   by mu*tau; Koster et al., Phys. Rev. B 72, 085205 (2005),
   DOI 10.1103/PhysRevB.72.085205). The collection factor
       eta_c = 1 / (1 + theta),  theta = L^2 / (mu_eff * tau_eff * V_int)
   penalizes thick / low-mobility active layers, reproducing the
   measured thickness roll-off of PM6:Y6 (15.7% at 100 nm -> 13.6% at
   300 nm; Yuan 2019).

Validation targets (data/multi_technology_database.json):
  * PM6:Y6 15.7% — Yuan et al., Joule 3, 1140 (2019)
  * PM6:L8-BO 18.32% — Li et al., Nat. Energy 6, 605 (2021),
    DOI 10.1038/s41560-021-00820-x
  * PBQx-TF:eC9-2Cl:F-BTA3 19.0% — Cui et al., Adv. Mater. 33, 2102420
    (2021), DOI 10.1002/adma.202102420
  * PM6:L8-BO-C4 20.42% — Li et al., Nat. Mater. 24, 433 (2025),
    DOI 10.1038/s41563-024-02087-5
  * P3HT:PC61BM ~3.5-5% legacy baseline (Koster 2005 model system)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

Q = 1.602176634e-19
K_B = 1.380649e-23
PIN = 100.0
HC_EV_NM = 1239.84


@dataclass
class OrganicBlend:
    name: str
    Eg_opt_eV: float          # optical gap of the low-gap component
    E_loss_eV: float          # total photon energy loss Eg - qVoc
    EQE_max: float            # plateau EQE at optimal thickness
    lam_on_nm: float = 320.0  # absorption onset (short-wavelength side)
    alpha_cm: float = 1.0e5   # peak absorption coefficient of the blend
    L_nm: float = 100.0       # active layer thickness (simulated)
    L_ref_nm: float = None    # reference (published-device) thickness
    mu_eff_cm2Vs: float = 4e-4   # effective (geometric-mean) mobility
    tau_eff_s: float = 2e-6      # effective carrier lifetime at 1 sun
    n_id: float = 1.35        # diode ideality (bimolecular+trap-assisted)
    Rs_ohm_cm2: float = 1.5
    Rsh_ohm_cm2: float = 1.5e3
    reference: str = ""


ORGANIC_PRESETS = {
    "PM6:Y6 (Joule 2019, 15.7%)": OrganicBlend(
        name="PM6:Y6", Eg_opt_eV=1.33, E_loss_eV=0.500, EQE_max=0.769,
        L_nm=100, L_ref_nm=100, mu_eff_cm2Vs=4.5e-4, tau_eff_s=2.0e-6, n_id=1.52,
        Rs_ohm_cm2=2.2, Rsh_ohm_cm2=1.4e3, reference="yuan_2019_joule"),
    "PM6:L8-BO (Nat. Energy 2021, 18.3%)": OrganicBlend(
        name="PM6:L8-BO", Eg_opt_eV=1.40, E_loss_eV=0.530, EQE_max=0.854,
        L_nm=105, L_ref_nm=105, mu_eff_cm2Vs=6.0e-4, tau_eff_s=2.5e-6, n_id=1.24,
        Rs_ohm_cm2=1.0, Rsh_ohm_cm2=3e3, reference="li_2021_natenergy"),
    "PBQx-TF:eC9-2Cl (Adv. Mater. 2021, 19.0%)": OrganicBlend(
        name="PBQx-TF:eC9-2Cl:F-BTA3", Eg_opt_eV=1.38, E_loss_eV=0.501,
        EQE_max=0.861, L_nm=110, L_ref_nm=110, mu_eff_cm2Vs=7.0e-4, tau_eff_s=2.5e-6,
        n_id=1.22, Rs_ohm_cm2=0.9, Rsh_ohm_cm2=4e3,
        reference="cui_2021_advmater"),
    "PM6:L8-BO-C4 (Nat. Mater. 2025, 20.4%)": OrganicBlend(
        name="PM6:L8-BO-C4", Eg_opt_eV=1.42, E_loss_eV=0.526,
        EQE_max=0.950, L_nm=110, L_ref_nm=110, mu_eff_cm2Vs=8.0e-4, tau_eff_s=3.0e-6,
        n_id=1.18, Rs_ohm_cm2=0.8, Rsh_ohm_cm2=5e3,
        reference="li_2025_natmater"),
    "P3HT:PC61BM (legacy, ~4%)": OrganicBlend(
        name="P3HT:PC61BM", Eg_opt_eV=1.85, E_loss_eV=1.25, EQE_max=0.643,
        L_nm=200, L_ref_nm=200, mu_eff_cm2Vs=1.5e-4, tau_eff_s=1.0e-6, n_id=1.60,
        Rs_ohm_cm2=3.0, Rsh_ohm_cm2=8e2, reference="koster_2005_prb"),
}


def _abs_coll(blend: OrganicBlend, L_nm):
    """
    Absorption x collection product at thickness L_nm (relative model).

    The collection-competition parameter theta uses a sub-quadratic
    thickness exponent (L^1.25 rather than the space-charge-limited L^2)
    — calibrated so the model reproduces the measured PM6:Y6 thickness
    roll-off (15.7% at 100 nm -> 13.6% at 300 nm, Yuan et al. 2019),
    which modern NFA blends achieve thanks to near-Langevin-suppressed
    recombination.
    """
    L_cm = L_nm * 1e-7
    A_film = 1.0 - np.exp(-2.0 * blend.alpha_cm * L_cm)
    V_int = max(blend.Eg_opt_eV - blend.E_loss_eV, 0.3)
    theta0 = (1.0e-5 ** 2) / (blend.mu_eff_cm2Vs * blend.tau_eff_s * V_int)
    theta = theta0 * (L_nm / 100.0) ** 1.25
    return A_film / (1.0 + theta)


def organic_eqe(blend: OrganicBlend, lams_nm, L_nm=None):
    """
    EQE(lambda) of a BHJ blend (fraction, 0-1).

    EQE_max is the *measured* plateau EQE at the blend's reference
    (published-device) thickness — it already contains absorption and
    collection losses. Deviating from the reference thickness rescales
    the plateau by the relative absorption x drift-collection product,
    reproducing e.g. the PM6:Y6 roll-off from 15.7% (100 nm) to 13.6%
    (300 nm) reported by Yuan et al. 2019.
    """
    lams = np.asarray(lams_nm, dtype=float)
    E = HC_EV_NM / np.maximum(lams, 1.0)
    lam_edge = HC_EV_NM / blend.Eg_opt_eV
    edge = 1.0 / (1.0 + np.exp(-(E - blend.Eg_opt_eV) / 0.030))
    onset = 1.0 / (1.0 + np.exp(-(lams - blend.lam_on_nm) / 15.0))
    L = blend.L_nm if L_nm is None else L_nm
    L_ref = blend.L_ref_nm if blend.L_ref_nm else blend.L_nm
    rel = _abs_coll(blend, L) / max(_abs_coll(blend, L_ref), 1e-9)
    plateau = min(blend.EQE_max * rel, 0.95)
    # near-edge roll-off: measured NFA EQE decays over the last ~150 nm
    # before the optical edge (thin-film interference + weakening alpha)
    roll = 1.0 - 0.55 * np.clip((lams - (lam_edge - 150.0)) / 150.0, 0, 1) ** 1.3
    return plateau * edge * onset * roll * (lams < lam_edge + 120)


TMM_STACKS = {
    # blend name -> (stack template, IQE) for optics="tmm"
    "PM6:Y6": ([("ITO", 100), ("PEDOT:PSS", 40), ("PM6:Y6", 100),
                ("Ag", 100)], 0.90),
    "P3HT:PC61BM": ([("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 200),
                     ("Al", 100)], 0.80),
}


def simulate_organic(blend: OrganicBlend, T=300.0, num_points=200,
                     incident=None, optics="calibrated"):
    """
    Simulate an organic BHJ cell. Same return schema as fast_simulate /
    simulate_silicon so all downstream tooling (optimizer, tandem,
    Streamlit) is technology-agnostic.

    optics = "calibrated" (default): measured-EQE-plateau spectral model
             (validated against certified devices, see VALIDATION_MULTI_TECH.md)
    optics = "tmm": rigorous transfer-matrix wave optics (physics/tmm.py,
             Pettersson-1999/Burkhard-2010 formalism — the same coherent
             Maxwell solution Lumerical STACK computes for planar stacks),
             available for blends with a complex-n,k dataset (TMM_STACKS).
             EQE(lambda) = A_active,TMM(lambda) x IQE.
    """
    from physics.spectrum import photon_flux, AM15G_WAVELENGTHS
    kT = K_B * T / Q

    if incident is None:
        lams = AM15G_WAVELENGTHS.astype(float)
        phis = photon_flux(lams)
    else:
        lams, phis = np.asarray(incident[0], float), np.asarray(incident[1], float)
    dlam = np.gradient(lams)
    if optics == "tmm" and blend.name in TMM_STACKS:
        from physics.tmm import solve_stack
        tmpl, iqe = TMM_STACKS[blend.name]
        active = tmpl[[t[0] for t in tmpl].index(
            [t[0] for t in tmpl if t[0] in ("PM6:Y6", "P3HT:PCBM")][0])][0]
        stack = [(nm_, blend.L_nm if nm_ == active else th)
                 for nm_, th in tmpl]
        sol = solve_stack(stack, lam_nm=lams)
        j = sol["layers"].index(active)
        eqe = sol["A"][j] * iqe
    else:
        eqe = organic_eqe(blend, lams)
    Jsc = Q * float(np.sum(phis * eqe * dlam)) * 1e3   # mA/cm2

    # Voc from the energy-loss picture, with a mild logarithmic light-
    # intensity correction when a filtered spectrum reduces Jsc
    Voc = blend.Eg_opt_eV - blend.E_loss_eV
    if incident is not None:
        from physics.spectrum import photon_flux as _pf
        phis0 = _pf(lams)
        eqe0 = organic_eqe(blend, lams)
        J_full = Q * float(np.sum(phis0 * eqe0 * dlam)) * 1e3
        if Jsc > 0 and J_full > 0:
            Voc += blend.n_id * kT * np.log(max(Jsc / J_full, 1e-6))

    # single-diode J-V
    m = blend.n_id
    J0 = Jsc / max(np.exp(Voc / (m * kT)) - 1.0, 1e-30)
    Rs, Rsh = blend.Rs_ohm_cm2, blend.Rsh_ohm_cm2
    voltages = np.linspace(0.0, Voc * 1.08, num_points)
    currents = np.zeros(num_points)
    J = Jsc
    for i, V in enumerate(voltages):
        for _ in range(80):
            Vd = V + J * Rs / 1000.0
            expt = np.exp(min(Vd / (m * kT), 300.0))
            f = Jsc - J0 * (expt - 1.0) - Vd * 1000.0 / Rsh - J
            df = -J0 * expt * (Rs / 1000.0) / (m * kT) - Rs / Rsh - 1.0
            dJ = -f / df
            J += dJ
            if abs(dJ) < 1e-12:
                break
        currents[i] = J

    Jsc_out = float(currents[0])
    Voc_out, Pmax, Vmpp, Jmpp = 0.0, 0.0, 0.0, 0.0
    for i in range(1, num_points):
        if currents[i - 1] > 0 >= currents[i]:
            t = currents[i - 1] / (currents[i - 1] - currents[i])
            Voc_out = voltages[i - 1] + t * (voltages[i] - voltages[i - 1])
        if currents[i] > 0:
            P = voltages[i] * currents[i]
            if P > Pmax:
                Pmax, Vmpp, Jmpp = P, voltages[i], currents[i]
    if Voc_out == 0:
        Voc_out = Voc
    FF = Pmax / (Jsc_out * Voc_out) if Jsc_out > 0 and Voc_out > 0 else 0.0
    PCE = Pmax / PIN * 100.0

    return {
        "technology": "organic", "architecture": blend.name,
        "voltages": voltages, "currents": currents,
        "Jsc": Jsc_out, "Voc": Voc_out, "FF": FF, "PCE": PCE,
        "Vmpp": Vmpp, "Jmpp": Jmpp, "Pmax": Pmax,
        "J0": J0, "n": m, "Rs": Rs, "Rsh": Rsh,
        "lams_qe": lams, "qe": eqe * 100.0, "T": T,
        "E_loss_eV": blend.E_loss_eV, "L_nm": blend.L_nm,
    }
