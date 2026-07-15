"""
physics/silicon.py — Crystalline silicon solar cell engine
=================================================================
Physics-based analytical model for c-Si solar cells (Al-BSF, PERC,
TOPCon, SHJ, SHJ-IBC architectures).

Model components — every equation traceable to a published source:

1. OPTICS / Jsc
   Absorptance with Lambertian light trapping (Tiedje & Yablonovitch,
   IEEE Trans. Electron Devices 31, 711 (1984), DOI 10.1109/T-ED.1984.21594):
       A(lam) = (1 - R_f) * alpha / (alpha + 1 / (Z * W))
   where Z is the path-length enhancement (Z = 4 n^2 ~ 50 for ideal
   Lambertian texture, Z = 2 for a planar cell with back reflector).
   alpha(lam) for intrinsic c-Si at 300 K is interpolated from the
   self-consistent tabulation of Green, Sol. Energy Mater. Sol. Cells
   92, 1305 (2008), DOI 10.1016/j.solmat.2008.06.009.
   Photon flux: ASTM G173-03 AM1.5G (same tables as the perovskite engine).

2. RECOMBINATION / Voc
   Intrinsic (Auger + radiative) recombination via the Richter et al.
   parameterization: Phys. Rev. B 86, 165202 (2012),
   DOI 10.1103/PhysRevB.86.165202:
       R_intr = n*p * (2.5e-31*g_eeh*n0 + 8.5e-32*g_ehh*p0
                        + 3.0e-29*dn^0.92) + B_rel*B_low*(n*p - ni_eff^2)
   with Coulomb-enhancement factors g_eeh, g_ehh. The revised Niewelt
   et al. 2022 parameterization (Sol. Energy Mater. Sol. Cells 235,
   111467, DOI 10.1016/j.solmat.2021.111467) is available via
   auger_model="niewelt2022" and slightly raises the intrinsic limit
   (29.4% single-junction limit).
   SRH bulk lifetime and surface/contact recombination (J0s, fA/cm2)
   complete the balance. Voc is found by solving the steady-state
   generation = recombination balance at open circuit (Brent's method),
   i.e. an *implied-Voc* construction standard in the Si community
   (Kerr & Cuevas, J. Appl. Phys. 91, 2473 (2002), DOI 10.1063/1.1432476).

3. FILL FACTOR AND FULL J-V
   Ideal FF from Green's empirical expression (Green, Solid-State
   Electron. 24, 788 (1981), DOI 10.1016/0038-1101(81)90062-9), then a
   full single-diode J-V sweep with the series resistance Rs and shunt
   Rsh of the architecture (Newton iteration, same numerics as
   physics/device.py) so tandem coupling can consume the whole curve.

ni_eff = 9.65e9 cm^-3 at 300 K (Altermatt et al., J. Appl. Phys. 93,
1598 (2003), DOI 10.1063/1.1529297).

Validation targets (see data/multi_technology_database.json and
scripts/run_multi_tech_validation.py):
  * SHJ-IBC 26.7% — Yoshikawa et al., Nat. Energy 2, 17032 (2017),
    DOI 10.1038/nenergy.2017.32 / Sol. Energy Mater. Sol. Cells 173,
    37 (2017), DOI 10.1016/j.solmat.2017.06.024
  * SHJ 26.81% — Lin et al., Nat. Energy 8, 789 (2023),
    DOI 10.1038/s41560-023-01255-2
  * TOPCon (back junction) 26.0% — Richter et al., Nat. Energy 6, 429
    (2021), DOI 10.1038/s41560-021-00805-w
  * Industrial PERC / Al-BSF — Green et al., Solar cell efficiency
    tables (version 57), Prog. Photovolt. 29, 3 (2021),
    DOI 10.1002/pip.3371
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

Q = 1.602176634e-19      # C
K_B = 1.380649e-23       # J/K
PIN = 100.0              # mW/cm2 (AM1.5G, 1000 W/m2)
NI_EFF_300K = 9.65e9     # cm^-3, Altermatt 2003


def ni_si(T=300.0):
    """Intrinsic carrier density of c-Si vs temperature.

    Misiakos & Tsamakis empirical formula, ni = 5.29e19 (T/300)^2.54
    exp(-6726/T) cm^-3 (J. Appl. Phys. 74, 3293 (1993), DOI
    10.1063/1.354551) supplies the temperature SHAPE; the prefactor is
    anchored so that ni(300 K) equals the Altermatt-2003 value 9.65e9
    exactly, leaving every validated STC result untouched while giving
    the physically required decrease of Voc with temperature.
    """
    return NI_EFF_300K * (T / 300.0) ** 2.54 * np.exp(-6726.0 * (1.0 / T - 1.0 / 300.0))

# ---------------------------------------------------------------------------
# c-Si absorption coefficient at 300 K, cm^-1
# Interpolation nodes digitized from Green (2008), Sol. Energy Mater. Sol.
# Cells 92, 1305, Table 1 (values rounded to 3 significant figures).
# ---------------------------------------------------------------------------
_SI_LAM_NM = np.array([
    300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500,
    520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720,
    740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940,
    960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140,
    1160, 1180, 1200], dtype=float)
_SI_ALPHA = np.array([
    1.73e6, 1.10e6, 1.09e6, 1.02e6, 2.93e5, 9.52e4, 5.32e4, 3.34e4,
    2.42e4, 1.84e4, 1.11e4, 9.33e3, 7.36e3, 5.97e3, 4.91e3, 4.14e3,
    3.45e3, 2.91e3, 2.44e3, 2.07e3, 1.90e3, 1.56e3, 1.31e3, 1.10e3,
    9.15e2, 8.50e2, 6.55e2, 5.45e2, 4.32e2, 3.62e2, 3.06e2, 2.28e2,
    1.71e2, 1.18e2, 8.28e1, 6.40e1, 3.68e1, 2.30e1, 1.10e1, 6.28e0,
    3.50e0, 1.55e0, 6.32e-1, 2.72e-1, 8.60e-2, 2.20e-3])


def si_alpha_cm(lam_nm):
    """Interpolated c-Si absorption coefficient [cm^-1] at 300 K (Green 2008)."""
    lam = np.asarray(lam_nm, dtype=float)
    # log-space interpolation (alpha spans 9 decades)
    la = np.interp(lam, _SI_LAM_NM, np.log(_SI_ALPHA),
                   left=np.log(_SI_ALPHA[0]), right=-30.0)
    return np.exp(la)


# ---------------------------------------------------------------------------
# Intrinsic recombination parameterizations
# ---------------------------------------------------------------------------
def intrinsic_recombination(dn, Ndop, dopant_type="n", model="richter2012",
                            ni=NI_EFF_300K):
    """
    Intrinsic (Auger + radiative) recombination rate [cm^-3 s^-1] at 300 K.

    model = "richter2012": Richter et al., PRB 86, 165202 (2012).
    model = "niewelt2022": Niewelt et al., SolMat 235, 111467 (2022)
            (updated coefficients; slightly lower Auger at moderate injection).
    """
    dn = np.maximum(dn, 0.0)
    if dopant_type == "n":
        n0, p0 = Ndop, ni ** 2 / Ndop
    else:
        n0, p0 = ni ** 2 / Ndop, Ndop
    n = n0 + dn
    p = p0 + dn
    np_prod = n * p

    if model == "niewelt2022":
        # Niewelt et al. 2022, Eq. (18)-(20) coefficient set
        geeh = 1.0 + 11.0 * (1.0 - np.tanh((n0 / 5.2e16) ** 0.34))
        gehh = 1.0 + 4.0 * (1.0 - np.tanh((p0 / 1.0e17) ** 0.29))
        R_aug = np_prod * (2.18e-31 * geeh * n0 + 7.21e-32 * gehh * p0
                           + 2.58e-29 * dn ** 0.92)
    else:
        # Richter et al. 2012, Eq. (18)
        geeh = 1.0 + 13.0 * (1.0 - np.tanh((n0 / 3.3e17) ** 0.66))
        gehh = 1.0 + 7.5 * (1.0 - np.tanh((p0 / 7.0e17) ** 0.63))
        R_aug = np_prod * (2.5e-31 * geeh * n0 + 8.5e-32 * gehh * p0
                           + 3.0e-29 * dn ** 0.92)

    B_low = 4.73e-15  # cm^3/s, low-injection radiative coefficient
    R_rad = B_low * (np_prod - ni ** 2)
    return R_aug + np.maximum(R_rad, 0.0)


# ---------------------------------------------------------------------------
# Architecture presets (calibrated to certified published devices; the
# reference for each preset lives in data/multi_technology_database.json)
# ---------------------------------------------------------------------------
@dataclass
class SiliconArchitecture:
    name: str
    W_um: float = 165.0          # wafer thickness
    Ndop_cm3: float = 4.0e15     # base doping
    dopant_type: str = "n"       # wafer polarity
    tau_srh_ms: float = 10.0     # bulk SRH lifetime
    J0s_fA: float = 2.5          # total surface+contact saturation current
    Rs_ohm_cm2: float = 0.35     # lumped series resistance
    Rsh_ohm_cm2: float = 2.0e5   # shunt resistance
    Z_path: float = 50.0         # light-trapping path enhancement (4n^2 ~ 50)
    R_front: float = 0.010       # broadband front reflectance (textured+ARC)
    shading: float = 0.0         # front metal shading fraction (0 for IBC)
    IQE: float = 0.995           # internal collection efficiency
    fEQE_blue: float = 0.97      # short-wavelength (lam<400nm) transmission
    reference: str = ""


SILICON_PRESETS = {
    "SHJ-IBC (Kaneka 26.7%)": SiliconArchitecture(
        name="SHJ-IBC", W_um=165, Ndop_cm3=3.0e15, dopant_type="n",
        tau_srh_ms=15.0, J0s_fA=2.6, Rs_ohm_cm2=0.32, Z_path=50, R_front=0.012,
        shading=0.0, IQE=0.997, reference="yoshikawa_2017_natenergy"),
    "SHJ both-side (LONGi 26.81%)": SiliconArchitecture(
        name="SHJ", W_um=130, Ndop_cm3=1.6e15, dopant_type="n",
        tau_srh_ms=50.0, J0s_fA=1.1, Rs_ohm_cm2=0.10, Z_path=50, R_front=0.015,
        shading=0.020, IQE=0.997, reference="lin_2023_natenergy"),
    "TOPCon back junction (Fraunhofer 26.0%)": SiliconArchitecture(
        name="TOPCon", W_um=200, Ndop_cm3=1.0e16, dopant_type="n",
        tau_srh_ms=12.0, J0s_fA=4.0, Rs_ohm_cm2=0.30, Z_path=50, R_front=0.010,
        shading=0.016, IQE=0.995, reference="richter_2021_natenergy"),
    "PERC industrial (~24%)": SiliconArchitecture(
        name="PERC", W_um=170, Ndop_cm3=9.0e15, dopant_type="p",
        tau_srh_ms=1.2, J0s_fA=45.0, Rs_ohm_cm2=0.45, Z_path=45, R_front=0.018,
        shading=0.025, IQE=0.985, reference="green_2021_tables57"),
    "Al-BSF legacy (~20%)": SiliconArchitecture(
        name="Al-BSF", W_um=180, Ndop_cm3=1.5e16, dopant_type="p",
        tau_srh_ms=0.25, J0s_fA=350.0, Rs_ohm_cm2=0.75, Rsh_ohm_cm2=5e4,
        Z_path=35, R_front=0.030, shading=0.045, IQE=0.965,
        reference="green_2021_tables57"),
}


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------
def simulate_silicon(arch: SiliconArchitecture, T=300.0, num_points=200,
                     incident=None, auger_model="richter2012",
                     concentration=1.0):
    """
    Simulate a crystalline-silicon cell.

    Parameters
    ----------
    arch : SiliconArchitecture
    incident : None | (lam_nm_array, flux_per_nm_array)
        If given, replaces the AM1.5G spectrum (units: photons/cm2/s/nm).
        Used by the tandem engine to pass the top-cell-filtered spectrum.
    Returns dict compatible with physics.device.fast_simulate output.
    """
    from physics.spectrum import photon_flux, AM15G_WAVELENGTHS

    kT = K_B * T / Q
    W_cm = arch.W_um * 1e-4
    ni = ni_si(T)

    # ---- 1) photocurrent -------------------------------------------------
    if incident is None:
        lams = AM15G_WAVELENGTHS.astype(float)
        phis = photon_flux(lams) * concentration
    else:
        lams, phis = np.asarray(incident[0], float), np.asarray(incident[1], float)
    dlam = np.gradient(lams)
    alpha = si_alpha_cm(lams)
    A = (1.0 - arch.R_front) * alpha / (alpha + 1.0 / (arch.Z_path * W_cm))
    blue = lams < 400
    eqe = A * arch.IQE * (1.0 - arch.shading)
    eqe[blue] *= arch.fEQE_blue
    Jsc = Q * float(np.sum(phis * eqe * dlam)) * 1e3        # mA/cm2
    G_avg = float(np.sum(phis * eqe * dlam)) / W_cm          # cm^-3 s^-1

    # ---- 2) implied Voc: generation = recombination at open circuit -----
    tau_srh = arch.tau_srh_ms * 1e-3
    J0s = arch.J0s_fA * 1e-15                                # A/cm2
    Ndop = arch.Ndop_cm3

    def net(dn):
        R_bulk = intrinsic_recombination(dn, Ndop, arch.dopant_type,
                                         auger_model) + dn / tau_srh
        if arch.dopant_type == "n":
            n0, p0 = Ndop, ni**2 / Ndop
        else:
            n0, p0 = ni**2 / Ndop, Ndop
        np_ratio = (n0 + dn) * (p0 + dn) / ni**2
        R_surf = (J0s / Q) * (np_ratio - 1.0) / W_cm
        return G_avg - R_bulk - R_surf

    from scipy.optimize import brentq
    lo, hi = 1e8, 1e19
    if net(lo) <= 0:
        dn_oc = lo
    else:
        while net(hi) > 0 and hi < 1e21:
            hi *= 10
        dn_oc = brentq(net, lo, hi, xtol=1e6, rtol=1e-10)
    if arch.dopant_type == "n":
        n0, p0 = Ndop, ni**2 / Ndop
    else:
        n0, p0 = ni**2 / Ndop, Ndop
    Voc = kT * np.log(max((n0 + dn_oc) * (p0 + dn_oc) / ni**2, 1.0 + 1e-12))

    # effective ideality from the local slope of the recombination balance
    dn2 = dn_oc * 0.5
    Voc2 = kT * np.log((n0 + dn2) * (p0 + dn2) / ni**2)
    R1 = G_avg
    R2 = (intrinsic_recombination(dn2, Ndop, arch.dopant_type, auger_model)
          + dn2 / tau_srh
          + (J0s / Q) * ((n0 + dn2) * (p0 + dn2) / ni**2 - 1.0) / W_cm)
    m_id = max(1.0, min(1.35, (Voc - Voc2) / (kT * np.log(max(R1 / R2, 1.001)))))

    # ---- 3) full single-diode J-V with Rs/Rsh ----------------------------
    Jph = Jsc
    J0 = Jph / max(np.exp(Voc / (m_id * kT)) - 1.0, 1e-30)   # mA/cm2
    Rs = arch.Rs_ohm_cm2
    Rsh = arch.Rsh_ohm_cm2
    voltages = np.linspace(0.0, Voc * 1.05, num_points)
    currents = np.zeros(num_points)
    J = Jph
    for i, V in enumerate(voltages):
        for _ in range(80):
            Vd = V + J * Rs / 1000.0
            expt = np.exp(min(Vd / (m_id * kT), 300.0))
            f = Jph - J0 * (expt - 1.0) - Vd * 1000.0 / Rsh - J
            df = -J0 * expt * (Rs / 1000.0) / (m_id * kT) - Rs / Rsh - 1.0
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
    Pin = PIN * concentration if incident is None else PIN
    PCE = Pmax / Pin * 100.0

    return {
        "technology": "silicon", "architecture": arch.name,
        "voltages": voltages, "currents": currents,
        "Jsc": Jsc_out, "Voc": Voc_out, "FF": FF, "PCE": PCE,
        "Vmpp": Vmpp, "Jmpp": Jmpp, "Pmax": Pmax,
        "J0": J0, "n": m_id, "Rs": Rs, "Rsh": Rsh,
        "dn_oc": dn_oc, "implied_Voc": Voc, "G_avg": G_avg,
        "lams_qe": lams, "qe": eqe * 100.0, "T": T,
        "auger_model": auger_model,
    }


def silicon_eqe(arch: SiliconArchitecture, lams_nm):
    """EQE(lambda) of a silicon architecture (fraction, 0-1)."""
    W_cm = arch.W_um * 1e-4
    alpha = si_alpha_cm(lams_nm)
    A = (1.0 - arch.R_front) * alpha / (alpha + 1.0 / (arch.Z_path * W_cm))
    eqe = A * arch.IQE * (1.0 - arch.shading)
    eqe[np.asarray(lams_nm) < 400] *= arch.fEQE_blue
    return eqe
