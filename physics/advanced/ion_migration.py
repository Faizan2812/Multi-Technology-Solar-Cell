"""
NOTE (v3.0): superseded by physics/dd_ion.py (ions coupled into the
production Poisson equation). Kept as the fast teaching model.

ion_migration.py — mobile-ion dynamics and J-V hysteresis.

The base solver has no mobile ions and therefore cannot reproduce the J-V
hysteresis that defines perovskite operation. This module adds a reduced,
physically-grounded mobile-ion model: a layer of mobile ionic charge whose
interfacial accumulation Q(t) relaxes toward its bias-dependent steady state
with a migration time constant tau_ion. The lagging ionic charge screens the
internal field and modulates non-radiative recombination, so a finite-rate
voltage scan yields different forward and reverse J-V curves.

This is the same physical mechanism as IonMonger/Driftfusion (mobile-ion field
screening) in a reduced dynamical form suitable for fast screening; it is NOT a
full coupled ion-electron PDE solve (see ROADMAP). It is validated on the limits
that any correct hysteresis model must satisfy.

Self-tests (__main__):
  * Hysteresis index -> 0 as scan rate -> 0 (quasi-static).
  * Hysteresis index -> 0 as mobile-ion density -> 0.
  * Hysteresis index > 0 at finite scan rate and finite ion density.
"""
import numpy as np
from .constants import K_B, Q, T_REF


def _diode_jv(V, Jph, J0, n_ideal, Rs, Rsh, T, recomb_boost):
    """Single-diode J(V) (mA/cm^2) with an ion-screening recombination boost.
    recomb_boost (>=0) inflates J0 to represent extra interfacial recombination
    when the ionic charge has not screened the field for the present bias."""
    kT = K_B * T / Q
    J0e = J0 * (1.0 + recomb_boost)
    J = Jph - J0e * (np.exp(V / (n_ideal * kT)) - 1.0) - V / Rsh
    return J


def simulate_scan(scan_rate_V_s=0.1, ion_density=1e17, tau_ion=1.0,
                  Jph=22.0, J0=1e-12, n_ideal=1.5, Rs=2.0, Rsh=2000.0,
                  T=T_REF, Vmax=1.2, npts=241):
    """Simulate forward (0->Vmax) then reverse (Vmax->0) J-V at a finite scan
    rate. Returns dict with V, J_fwd, J_rev, PCE_fwd, PCE_rev, hysteresis_index.

    The ionic state q in [0,1] is the normalised interfacial screening; its
    target q_ss tracks bias, and dq/dt = (q_ss - q)/tau_ion. The screening
    deficit (q_ss - q) drives recombination_boost, scaled by ion_density."""
    V_up = np.linspace(0, Vmax, npts)
    V_dn = V_up[::-1]
    dt = (V_up[1] - V_up[0]) / max(scan_rate_V_s, 1e-12)
    ion_scale = ion_density / 1e17                      # 1 at the reference density

    def run(Vseq, q0):
        q = q0; Js = []
        for V in Vseq:
            q_ss = 1.0 / (1.0 + np.exp(-(V - 0.6) / 0.1))   # sigmoidal bias target
            q += (q_ss - q) * (1 - np.exp(-dt / tau_ion))    # exact relaxation step
            deficit = abs(q_ss - q)
            boost = 6.0 * ion_scale * deficit                # screening deficit -> recombination
            Js.append(_diode_jv(V, Jph, J0, n_ideal, Rs, Rsh, T, boost))
        return np.array(Js), q

    # start reverse-poled (high screening), typical for stabilised cells
    J_fwd, qend = run(V_up, q0=0.0)
    J_rev, _ = run(V_dn, q0=qend)
    J_rev = J_rev[::-1]                                   # align to V_up ordering

    def metrics(V, J):
        P = V * J                                        # mW/cm^2
        Pmax = max(P.max(), 1e-9)
        return Pmax / 100.0 * 100.0                      # PCE% (100 mW/cm^2 input)
    PCE_fwd = metrics(V_up, J_fwd)
    PCE_rev = metrics(V_up, J_rev)
    HI = abs(PCE_rev - PCE_fwd) / max((PCE_rev + PCE_fwd) / 2, 1e-9)
    return dict(V=V_up, J_fwd=J_fwd, J_rev=J_rev,
                PCE_fwd=PCE_fwd, PCE_rev=PCE_rev, hysteresis_index=HI)


if __name__ == "__main__":
    fast = simulate_scan(scan_rate_V_s=1.0, ion_density=1e17)
    slow = simulate_scan(scan_rate_V_s=1e-4, ion_density=1e17)
    noion = simulate_scan(scan_rate_V_s=1.0, ion_density=0.0)
    print(f"HI  fast scan          = {fast['hysteresis_index']:.4f}")
    print(f"HI  slow (quasi-static)= {slow['hysteresis_index']:.4f}")
    print(f"HI  no ions            = {noion['hysteresis_index']:.4f}")
    print(f"PCE fwd/rev (fast)     = {fast['PCE_fwd']:.2f} / {fast['PCE_rev']:.2f} %")
    assert fast["hysteresis_index"] > 1e-3, "finite scan + ions must show hysteresis"
    assert slow["hysteresis_index"] < fast["hysteresis_index"], "slow scan must reduce HI"
    assert noion["hysteresis_index"] < 1e-6, "no ions must remove hysteresis"
    print("ion_migration: ALL CHECKS PASS")
