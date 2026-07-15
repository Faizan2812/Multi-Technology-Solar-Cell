"""
dd_ion.py — Mobile ions coupled into the production drift-diffusion solver
===========================================================================

Upgrades the previous *standalone* leading-order Voc-shift heuristic
(`physics.device.simulate_hysteresis`, now deprecated) to a genuine coupling
of mobile ionic charge into the Poisson equation of the Scharfetter-Gummel
solver (`physics.dd_solver`). J-V hysteresis is then an *output of the
solver*, not a bolted-on correction.

Model (quasi-static / adiabatic mobile-ion approximation)
---------------------------------------------------------
Halide perovskites contain mobile ionic defects — predominantly iodide
vacancies, with diffusivity D_I ~ 1e-12 cm^2/s and activation ~0.6 eV
[Eames et al., Nat. Commun. 6, 7497 (2015), DOI 10.1038/ncomms8497]. Ion
motion is 6-9 orders of magnitude slower than electronic transport, which
justifies the asymptotic treatment used by the dedicated perovskite DD codes
IonMonger [Courtier et al., J. Comput. Electron. 18, 1435 (2019),
DOI 10.1007/s10825-019-01396-2] and Driftfusion [Calado et al.,
J. Comput. Electron. 21, 960 (2022), DOI 10.1007/s10825-021-01827-z]:

  * Electrons/holes are always in steady state for the instantaneous
    ionic charge distribution (electronic time scales << scan time).
  * The mobile cation (iodide-vacancy) density P(x) in the absorber relaxes
    toward its Boltzmann equilibrium in the instantaneous electrostatic
    potential,
        P_eq(x) = N0 * exp(-psi(x)/V_T) / <exp(-psi/V_T)>_absorber ,
    conserving the total ion number N0 * d_abs (ions are confined to the
    perovskite layer; blocking boundaries).
  * On the time scale of a J-V scan the actual distribution is interpolated
    between the preconditioning distribution and the instantaneous
    equilibrium with a single relaxation fraction
        f = 1 - exp(-t_scan / tau_ion),   tau_ion = d_abs^2 / (pi^2 D_ion),
    the diffusive relaxation time of the lowest spatial mode.

  Limits (both unit-tested):
    N_ion -> 0        : identical J-V to the ion-free solver (HI -> 0).
    scan_rate -> 0    : f -> 1, forward == reverse (equilibrium, HI -> 0).
    fast scan         : ions frozen at their preconditioning distribution;
                        forward (from short-circuit precondition) and reverse
                        (from forward-bias precondition) scans differ -> HI > 0.

The self-consistent (psi, n, p, P) fixed point at each bias is found by damped
Picard iteration between the electronic solve (with rho_extra = P - N0) and
the ionic Boltzmann update.

What this is NOT: a full transient Poisson-Nernst-Planck integration with
ion dynamics resolved inside each voltage step (IonMonger). The quasi-static
model captures the leading-order, experimentally dominant physics — field
screening by slow ions and its scan-rate dependence — at ~2x the cost of the
ion-free solve. The scope is stated in the app and docs.
"""
from __future__ import annotations
import numpy as np

from physics.dd_solver import (build_mesh, solve_dd, extract_device_metrics,
                               K_B, Q)

# Iodide-vacancy diffusivity at 300 K (Eames 2015, DFT + kinetic model)
D_ION_DEFAULT = 1e-12         # cm^2 / s
Z_ION = +1                    # monovalent cation-like defect (V_I^+)


def _ion_equilibrium(mesh, psi, N0_cm3, T=300.0, P_max_factor=50.0):
    """Boltzmann distribution of mobile cations in the absorber, conserving
    the total areal ion number  N0 * d_abs, with a STERIC SATURATION cap
    P <= P_max = P_max_factor * N0.

    Unregularized Boltzmann statistics predict unbounded ion pile-up in the
    interfacial Debye layers (concentration factors of e^{|psi|/V_T} >> lattice
    site density), which is unphysical and destroys the electronic solve. All
    dedicated perovskite ion-DD codes regularize this — via finite-volume
    Fermi-like (Bikerman) statistics in IonMonger [Courtier 2019] or a
    saturation limit in Driftfusion [Calado 2022]. We use the simple
    saturation cap with iterative renormalization so the total ion number is
    still conserved exactly."""
    V_T = K_B * T / Q
    m = (mesh.layer == 1)
    P = np.zeros(mesh.N)
    if not np.any(m):
        return P
    x = mesh.x[m]
    d_abs = x[-1] - x[0]
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    u = -Z_ION * (psi[m] - psi[m].mean()) / V_T
    u = np.clip(u, -60.0, 60.0)
    w = np.exp(u - u.max())                       # overflow-safe
    P_max = P_max_factor * N0_cm3
    target = N0_cm3 * d_abs                       # conserved areal number
    # iterative renormalization under the cap
    Pm = np.zeros_like(w)
    scale = target / max(trapz(w, x), 1e-300)
    for _ in range(30):
        Pm = np.minimum(scale * w, P_max)
        tot = trapz(Pm, x)
        if tot >= 0.999 * target or np.all(Pm >= P_max * 0.999999):
            break
        free = Pm < P_max * 0.999999
        deficit = target - trapz(np.where(free, 0.0, Pm), x)
        denom = trapz(np.where(free, w, 0.0), x)
        if denom <= 0 or deficit <= 0:
            break
        scale = deficit / denom
    P[m] = Pm
    return P


def solve_dd_ions(mesh, G, V, mat_left, mat_right, T=300.0,
                  N_ion=1e17, P_init=None, P_start=None, relax=1.0,
                  max_outer=25, tol=0.01, damping=0.35):
    """Self-consistent electronic + quasi-static ionic solve at one bias.

    P_init : the physical ANCHOR distribution (preconditioning state). The
        fixed point solved for is  P* = P_init + relax*(P_eq[psi(P*)] - P_init).
    P_start: numerical starting guess for the Picard iteration (e.g. the
        converged P from the previous voltage point of a scan). Defaults to
        P_init. Changes convergence speed only, never the fixed point.
    relax in [0, 1]: fraction of the way the ion distribution moves from
        P_init toward the instantaneous Boltzmann equilibrium (scan-rate
        physics). relax=1 -> fully equilibrated ions; relax=0 -> frozen at
        P_init.
    Returns (DDResult, P) — the electronic solution and the ion profile used.
    """
    m = (mesh.layer == 1)
    if P_init is None:
        P_init = np.zeros(mesh.N)
        P_init[m] = N_ion
    P = (P_start if P_start is not None else P_init).copy()
    r = None
    ws = None                      # warm start (psi, n, p) across ion updates
    for k in range(max_outer):
        rho_extra = Z_ION * (P - np.where(m, N_ion, 0.0))   # net ionic charge
        r = solve_dd(mesh, G, V, mat_left, mat_right, T=T,
                     rho_extra=rho_extra, warm_start=ws,
                     max_gummel=50 if k == 0 else 30)
        if np.all(np.isfinite(r.psi)):
            ws = (r.psi, r.n, r.p)
        P_eq = _ion_equilibrium(mesh, r.psi, N_ion, T)
        P_target = P_init + relax * (P_eq - P_init)
        dP = np.max(np.abs(P_target - P)) / max(N_ion, 1e-30)
        P = P + damping * (P_target - P)
        if (dP < tol and r.converged) or relax == 0.0:
            break
    return r, P


def hysteresis_jv(htl_mat, abs_mat, etl_mat, d_htl_nm, d_abs_nm, d_etl_nm,
                  Nt_abs=None, T=300.0, N_ion=1e17, scan_rate=0.1,
                  D_ion=D_ION_DEFAULT, N_V=17, V_max=None, G_profile=None):
    """Forward and reverse J-V scans with quasi-static mobile ions.

    scan_rate : V/s. Determines the ion relaxation fraction
        f = 1 - exp(-t_scan / tau_ion),  tau_ion = d_abs^2 / (pi^2 D_ion).
    Forward scan: preconditioned at short circuit (V=0, equilibrated ions).
    Reverse scan: preconditioned at forward bias ~Voc (equilibrated ions).
    During each scan the ion profile relaxes by the fraction f toward the
    instantaneous equilibrium — the standard quasi-static protocol.

    Returns dict with both curves, metrics, and the hysteresis index
        HI = (PCE_rev - PCE_fwd) / PCE_rev.
    """
    mesh = build_mesh([htl_mat, abs_mat, etl_mat],
                      [d_htl_nm, d_abs_nm, d_etl_nm],
                      Nt_override=[None, Nt_abs, None], T=T)
    if G_profile is None:
        # consistent with device.simulate_iv_curve DD path (Beer-Lambert)
        from physics.device import _dd_beer_lambert_generation
        G_profile = _dd_beer_lambert_generation(mesh, abs_mat)
    if V_max is None:
        V_max = min(abs_mat.Eg * 0.80, 1.30)
    voltages = np.linspace(0.0, V_max, N_V)

    d_abs_cm = d_abs_nm * 1e-7
    tau_ion = d_abs_cm ** 2 / (np.pi ** 2 * max(D_ion, 1e-30))
    t_scan = V_max / max(scan_rate, 1e-6)
    f_relax = 1.0 - np.exp(-t_scan / tau_ion)

    def _scan(V_arr, V_precondition):
        # equilibrate ions at the preconditioning bias
        _, P_pre = solve_dd_ions(mesh, G_profile, V_precondition,
                                 htl_mat, etl_mat, T, N_ion, relax=1.0)
        J = np.zeros(len(V_arr)); conv = np.zeros(len(V_arr), dtype=bool)
        P = P_pre
        for i, V in enumerate(V_arr):
            r, P = solve_dd_ions(mesh, G_profile, V, htl_mat, etl_mat, T,
                                 N_ion, P_init=P_pre, P_start=P,
                                 relax=f_relax)
            J[i] = r.J_total; conv[i] = r.converged
        return J, conv

    if N_ion <= 0:
        # exact ion-free limit: single sweep, both directions identical
        from physics.dd_solver import jv_sweep
        V_arr, J, conv = jv_sweep(mesh, G_profile, htl_mat, etl_mat,
                                  V_min=0.0, V_max=V_max, N_V=N_V, T=T)
        J_fwd = J_rev = J; conv_f = conv_r = conv
        voltages = V_arr
    elif f_relax > 0.99:
        # Fully-equilibrated (slow-scan) limit: in the quasi-static model the
        # ion distribution at each bias is the unique equilibrium P_eq(V) —
        # the J-V is path-independent BY CONSTRUCTION. A single equilibrated
        # sweep therefore serves both scan directions (HI -> 0 exactly),
        # avoiding spurious numerical path dependence.
        J = np.zeros(N_V); conv = np.zeros(N_V, dtype=bool)
        P = None
        for i, V in enumerate(voltages):
            r, P = solve_dd_ions(mesh, G_profile, V, htl_mat, etl_mat, T,
                                 N_ion, P_start=P, relax=1.0)
            J[i] = r.J_total; conv[i] = r.converged
        J_fwd = J_rev = J; conv_f = conv_r = conv
    else:
        J_fwd, conv_f = _scan(voltages, 0.0)
        J_rev, conv_r = _scan(voltages[::-1], V_max)
        J_rev, conv_r = J_rev[::-1], conv_r[::-1]

    m_fwd = extract_device_metrics(voltages, J_fwd, converged_flags=conv_f)
    m_rev = extract_device_metrics(voltages, J_rev, converged_flags=conv_r)
    pce_r = max(m_rev["PCE"], 1e-9)
    HI = (m_rev["PCE"] - m_fwd["PCE"]) / pce_r
    return {
        "voltages": voltages,
        "J_forward_mA": np.where(np.isfinite(J_fwd), J_fwd * 1000.0, np.nan),
        "J_reverse_mA": np.where(np.isfinite(J_rev), J_rev * 1000.0, np.nan),
        "metrics_forward": m_fwd, "metrics_reverse": m_rev,
        "hysteresis_index": float(HI),
        "f_ion_relaxation": float(f_relax),
        "tau_ion_s": float(tau_ion), "t_scan_s": float(t_scan),
        "N_ion": float(N_ion), "scan_rate_V_s": float(scan_rate),
        "model": ("quasi-static mobile-ion DD coupling "
                  "(Eames 2015 D_ion; Courtier 2019 / Calado 2022 asymptotics)"),
    }
