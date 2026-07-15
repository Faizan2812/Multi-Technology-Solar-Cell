"""
physics/tmm.py — Transfer-matrix optical solver
============================================================
Full coherent wave-optics for planar multilayer stacks: complex-index
transfer-matrix method (TMM) at normal incidence, following the
formalism of Pettersson, Roman & Inganäs, J. Appl. Phys. 86, 487 (1999),
DOI 10.1063/1.370757, in the widely used implementation style of
Burkhard, Hoke & McGehee, Adv. Mater. 22, 3293 (2010),
DOI 10.1002/adma.201000883 (the Stanford/McGehee TMM script).

Why this exists — cross-tool validation
---------------------------------------
For *planar* thin-film stacks, commercial optical solvers (Lumerical
STACK, and Lumerical FDTD in the planar limit) solve the same coherent
Maxwell problem this module solves; published organic-solar-cell
optical simulations are overwhelmingly TMM-based (Pettersson 1999;
Sievers, Park & Yang, J. Appl. Phys. 100, 114509 (2006),
DOI 10.1063/1.2388854; Kotlarski et al., J. Appl. Phys. 103, 084502
(2008), DOI 10.1063/1.2905243). This module therefore provides the
Lumerical-class optical layer of the tool, and
scripts/run_cross_tool_validation.py verifies it against:
  * exact energy conservation R + T + sum_j A_j = 1 (machine precision)
  * the analytic Fresnel limit for a bare interface
  * the published interference oscillation of Jsc vs active-layer
    thickness in P3HT:PCBM (first optical maximum ~70-90 nm, second
    ~200-250 nm; Sievers 2006, Kotlarski 2008, Monestier 2007)
  * the ~100 nm optimum of NFA blends (Im et al., Molecules 28, 2985
    (2023), DOI 10.3390/molecules28072985)
What TMM (and Lumerical STACK) does NOT cover: textured / nanostructured
(2-D/3-D) light trapping — that needs full FDTD and is outside this
tool's 1-D scope; stated in docs/CROSS_TOOL_VALIDATION.md.

Complex refractive-index database
---------------------------------
Compact tabulations digitized from the literature (interpolated
log-linearly in k, linearly in n; confidence MEDIUM — ellipsometry of
organic blends is batch-dependent by ±5-10%):
  * P3HT:PCBM (1:1): Monestier et al., Sol. Energy Mater. Sol. Cells
    91, 405 (2007), DOI 10.1016/j.solmat.2006.10.019; Burkhard 2010
  * PM6:Y6: ellipsometry literature (e.g. Kerremans et al., Adv. Opt.
    Mater. 8, 2000319 (2020), DOI 10.1002/adom.202000319)
  * Al: Rakic, Appl. Opt. 34, 4755 (1995), DOI 10.1364/AO.34.004755
  * Ag: Johnson & Christy, Phys. Rev. B 6, 4370 (1972),
    DOI 10.1103/PhysRevB.6.4370
  * ITO / PEDOT:PSS / ZnO / glass: standard device-modeling values
    (Burkhard 2010 supplementary and refs therein)
"""
from __future__ import annotations

import numpy as np

Q_E = 1.602176634e-19
H = 6.62607015e-34
C0 = 2.99792458e8

# ---------------------------------------------------------------------------
# n,k tables: {material: (lam_nm[], n[], k[])}
# ---------------------------------------------------------------------------
_NK = {
    "glass": ([300, 1200], [1.52, 1.52], [0.0, 0.0]),
    "ITO": ([350, 400, 500, 600, 700, 800, 900, 1000, 1100],
            [2.10, 2.05, 1.95, 1.85, 1.78, 1.72, 1.65, 1.60, 1.55],
            [0.060, 0.030, 0.010, 0.008, 0.010, 0.020, 0.030, 0.050, 0.070]),
    "PEDOT:PSS": ([350, 500, 700, 1000, 1100],
                  [1.58, 1.52, 1.48, 1.42, 1.41],
                  [0.012, 0.008, 0.015, 0.040, 0.050]),
    "ZnO": ([350, 380, 400, 500, 700, 1100],
            [2.15, 2.05, 1.95, 1.92, 1.88, 1.85],
            [0.080, 0.020, 0.005, 0.001, 0.0005, 0.0003]),
    "P3HT:PCBM": ([350, 400, 450, 480, 500, 520, 550, 580, 600, 620,
                   650, 700, 800, 1000, 1100],
                  [1.70, 1.72, 1.80, 1.90, 1.95, 2.00, 2.08, 2.12, 2.12,
                   2.05, 1.98, 1.90, 1.85, 1.82, 1.82],
                  [0.080, 0.120, 0.220, 0.280, 0.310, 0.330, 0.300, 0.250,
                   0.170, 0.100, 0.035, 0.005, 0.001, 1e-4, 1e-4]),
    "PM6:Y6": ([350, 400, 450, 500, 550, 580, 600, 620, 650, 700, 750,
                800, 820, 850, 880, 900, 950, 1000, 1100],
               [1.65, 1.70, 1.75, 1.80, 1.95, 2.05, 2.10, 2.15, 2.05,
                1.95, 2.00, 2.10, 2.15, 2.20, 2.10, 2.00, 1.90, 1.85, 1.83],
               [0.170, 0.200, 0.170, 0.230, 0.360, 0.430, 0.450, 0.430,
                0.340, 0.400, 0.550, 0.650, 0.680, 0.570, 0.340, 0.170,
                0.035, 0.005, 1e-4]),
    "Al": ([350, 400, 500, 600, 700, 800, 850, 900, 1000, 1100],
           [0.41, 0.49, 0.77, 1.20, 1.83, 2.80, 2.75, 2.06, 1.35, 1.20],
           [4.20, 4.86, 6.08, 7.26, 8.31, 8.45, 8.31, 8.30, 9.58, 10.7]),
    "Ag": ([350, 400, 500, 600, 700, 800, 900, 1000, 1100],
           [0.07, 0.05, 0.05, 0.06, 0.14, 0.04, 0.04, 0.04, 0.04],
           [1.70, 2.10, 3.10, 4.00, 4.50, 5.40, 6.20, 6.90, 7.60]),
    "air": ([300, 1200], [1.0, 1.0], [0.0, 0.0]),
}


def nk(material, lam_nm):
    """Complex refractive index ñ = n + i k of a database material."""
    lam = np.asarray(lam_nm, dtype=float)
    tab_l, tab_n, tab_k = _NK[material]
    n = np.interp(lam, tab_l, tab_n)
    k = np.exp(np.interp(lam, tab_l, np.log(np.maximum(tab_k, 1e-9))))
    k = np.where(k < 2e-9, 0.0, k)
    return n + 1j * k


def available_nk_materials():
    return sorted(_NK.keys())


# ---------------------------------------------------------------------------
# Core TMM at one wavelength
# ---------------------------------------------------------------------------
def _interface(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return np.array([[1, r], [r, 1]], dtype=complex) / t


def _layer(n, d_nm, lam_nm):
    xi = 2 * np.pi * n / lam_nm  # per nm
    ph = 1j * xi * d_nm
    return np.array([[np.exp(-ph), 0], [0, np.exp(ph)]], dtype=complex)


def tmm_single(lam_nm, indices, thicknesses_nm, x_points=200):
    """
    Solve one wavelength for stack:
    [incident semi-infinite | layer_1 ... layer_m | exit semi-infinite].

    indices        : complex ñ for incident, layers..., exit (len m+2)
    thicknesses_nm : thicknesses of the m finite layers
    Returns dict: R, T, A (per finite layer), plus per-layer position grid
    and normalized dissipation q(x) [1/nm] for generation profiles.
    """
    m = len(thicknesses_nm)
    n0 = indices[0].real

    # full system matrix
    S = _interface(indices[0], indices[1])
    for j in range(1, m + 1):
        S = S @ _layer(indices[j], thicknesses_nm[j - 1], lam_nm)
        S = S @ _interface(indices[j], indices[j + 1])
    r = S[1, 0] / S[0, 0]
    t = 1.0 / S[0, 0]
    R = abs(r) ** 2
    T = abs(t) ** 2 * indices[-1].real / n0

    A, xs, qs = [], [], []
    for j in range(1, m + 1):
        # partial matrices: S' (before layer j), S'' (after layer j)
        Sp = _interface(indices[0], indices[1])
        for i in range(1, j):
            Sp = Sp @ _layer(indices[i], thicknesses_nm[i - 1], lam_nm)
            Sp = Sp @ _interface(indices[i], indices[i + 1])
        Spp = _interface(indices[j], indices[j + 1])
        for i in range(j + 1, m + 1):
            Spp = Spp @ _layer(indices[i], thicknesses_nm[i - 1], lam_nm)
            Spp = Spp @ _interface(indices[i], indices[i + 1])

        rpp = Spp[1, 0] / Spp[0, 0]
        d = thicknesses_nm[j - 1]
        xi = 2 * np.pi * indices[j] / lam_nm
        denom = (Sp[0, 0] * np.exp(-1j * xi * d)
                 + Sp[0, 1] * rpp * np.exp(1j * xi * d))
        tjp = 1.0 / denom  # forward amplitude at the layer's RIGHT boundary

        x = np.linspace(0.0, d, x_points)
        # field referenced to the right boundary: E(x) =
        #   t_j^+ [ e^{i xi (x-d)} + r'' e^{i xi (d-x)} ]
        E = tjp * (np.exp(1j * xi * (x - d))
                   + rpp * np.exp(1j * xi * (d - x)))
        nj, kj = indices[j].real, indices[j].imag
        # normalized dissipated power density per unit length [1/nm]
        q = (4 * np.pi * kj / lam_nm) * (nj / n0) * np.abs(E) ** 2
        A.append(float(np.trapezoid(q, x) if hasattr(np, "trapezoid")
                       else np.trapz(q, x)))
        xs.append(x)
        qs.append(q)

    return {"R": float(R), "T": float(T), "A": A, "x": xs, "q": qs}


# ---------------------------------------------------------------------------
# Spectral solver for a named stack
# ---------------------------------------------------------------------------
def solve_stack(stack, lam_nm=None, incident="glass", exit_medium="air",
                incoherent_glass_correction=True):
    """
    stack : list of (material_name, thickness_nm) finite layers,
            light-incidence side first, e.g.
            [("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 90), ("Al", 100)]
    Returns spectral R(λ), T(λ), A_j(λ) and layer bookkeeping.

    The thick glass substrate is treated as the incident semi-infinite
    medium with an incoherent air→glass intensity correction
    T_ag = 1 - ((n_g-1)/(n_g+1))^2 (standard Burkhard-2010 treatment).
    """
    if lam_nm is None:
        lam_nm = np.arange(350, 1005, 5.0)
    lam_nm = np.asarray(lam_nm, dtype=float)
    names = [s[0] for s in stack]
    ds = [float(s[1]) for s in stack]

    T_ag = 1.0
    if incoherent_glass_correction and incident == "glass":
        ng = 1.52
        T_ag = 1.0 - ((ng - 1) / (ng + 1)) ** 2

    R = np.zeros_like(lam_nm)
    T = np.zeros_like(lam_nm)
    A = np.zeros((len(stack), lam_nm.size))
    profiles = []
    for i, lam in enumerate(lam_nm):
        idx = ([nk(incident, lam)]
               + [nk(nm_, lam) for nm_ in names]
               + [nk(exit_medium, lam)])
        idx = [np.complex128(v) for v in idx]
        out = tmm_single(lam, idx, ds)
        R[i], T[i] = out["R"], out["T"]
        A[:, i] = out["A"]
        profiles.append(out)

    return {
        "lam_nm": lam_nm, "layers": names, "thicknesses_nm": ds,
        "R": R * T_ag + (1 - T_ag),      # air-side reflectance incl. glass face
        "R_internal": R, "T": T * T_ag,
        "A": A * T_ag,                    # absorptance per layer, air-referenced
        "A_internal": A, "T_ag": T_ag,
        "_profiles": profiles,
    }


def jsc_from_stack(stack, active_layer, IQE=1.0, lam_nm=None, incident_flux=None):
    """
    Short-circuit current from TMM absorption in `active_layer`
    (name or index), under AM1.5G (or a supplied flux) with a flat IQE.
    Returns (Jsc_mA_cm2, sol) where sol is the solve_stack output.
    """
    from physics.spectrum import photon_flux
    sol = solve_stack(stack, lam_nm=lam_nm)
    lam = sol["lam_nm"]
    phi = photon_flux(lam) if incident_flux is None else incident_flux
    if isinstance(active_layer, str):
        j = sol["layers"].index(active_layer)
    else:
        j = int(active_layer)
    dlam = np.gradient(lam)
    Jsc = Q_E * float(np.sum(phi * sol["A"][j] * IQE * dlam)) * 1e3
    return Jsc, sol


def jsc_vs_thickness(stack_template, active_layer, thicknesses_nm,
                     IQE=1.0, lam_nm=None):
    """Scan active-layer thickness; returns (thicknesses, Jsc array)."""
    out = []
    for d in thicknesses_nm:
        stack = [(nm_, (float(d) if nm_ == active_layer else th))
                 for nm_, th in stack_template]
        Jsc, _ = jsc_from_stack(stack, active_layer, IQE=IQE, lam_nm=lam_nm)
        out.append(Jsc)
    return np.asarray(thicknesses_nm, float), np.asarray(out)


def energy_conservation_error(stack, lam_nm=None):
    """max |1 - (R + T + sum_j A_j)| over the spectrum (internal, coherent)."""
    sol = solve_stack(stack, lam_nm=lam_nm, incoherent_glass_correction=False)
    total = sol["R_internal"] + sol["T"] / sol["T_ag"] + sol["A_internal"].sum(axis=0)
    return float(np.max(np.abs(1.0 - total)))
