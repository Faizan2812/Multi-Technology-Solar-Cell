"""Measurement-fitting verification: the forward model must be exact, blind
recovery of a known synthetic device must land inside the reported intervals,
the J0-n degeneracy must be detected, and CSV ingestion must be robust."""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.measurement_fit import (diode_jv, fit_measured_jv, parse_jv_csv,
                                   demo_measurement, PARAM_NAMES)


def test_lambertw_forward_model_is_exact():
    """The closed-form J(V) must satisfy the implicit diode equation to
    numerical precision at every bias point."""
    Vt = 1.380649e-23 * 300 / 1.602176634e-19
    p = dict(Jph=21.8, J0=3e-10, n=1.45, Rs=1.8, Rsh=2500.0)
    V = np.linspace(0, 1.15, 40)
    J = diode_jv(V, p["Jph"], p["J0"], p["n"], p["Rs"], p["Rsh"])
    lhs = J
    rhs = (p["Jph"]
           - p["J0"] * (np.exp((V + J * p["Rs"] * 1e-3) / (p["n"] * Vt)) - 1)
           - (V + J * p["Rs"] * 1e-3) / (p["Rsh"] * 1e-3))
    assert float(np.max(np.abs(lhs - rhs))) < 1e-8


def test_forward_model_limits():
    """Rsh→∞, Rs→0 must reduce toward the ideal diode; Jsc≈Jph; Voc matches
    the analytic ideal-diode value."""
    Vt = 1.380649e-23 * 300 / 1.602176634e-19
    J = diode_jv(np.array([0.0]), 22.0, 1e-10, 1.0, 1e-3, 1e8)
    assert abs(J[0] - 22.0) < 0.01
    Voc_analytic = 1.0 * Vt * np.log(22.0 / 1e-10 + 1)
    V = np.linspace(0, 1.2, 2000)
    Jv = diode_jv(V, 22.0, 1e-10, 1.0, 1e-3, 1e8)
    Voc_num = float(np.interp(0.0, -Jv, V))
    assert abs(Voc_num - Voc_analytic) < 2e-3


def test_blind_recovery_within_confidence_intervals():
    """The tool's own standard: hide a device, fit its noisy curve, and the
    truth must sit inside the reported 95% intervals for every parameter."""
    V, J, truth = demo_measurement(noise_mA=0.15, seed=7)
    fr = fit_measured_jv(V, J, n_bootstrap=120, seed=1)
    key_map = dict(zip(("Jph_mA_cm2", "J0_mA_cm2", "n",
                        "Rs_ohm_cm2", "Rsh_ohm_cm2"),
                       ("Jph_mA_cm2", "J0_mA_cm2", "n",
                        "Rs_ohm_cm2", "Rsh_ohm_cm2")))
    for k in PARAM_NAMES:
        t = truth[key_map[k]]
        assert fr.ci_low[k] <= t <= fr.ci_high[k], (
            f"{k}: truth {t} outside [{fr.ci_low[k]}, {fr.ci_high[k]}]")
    assert fr.rmse_mA_cm2 < 0.35
    # fitted observables close to the noiseless truth
    J_true = diode_jv(V, truth["Jph_mA_cm2"], truth["J0_mA_cm2"], truth["n"],
                      truth["Rs_ohm_cm2"], truth["Rsh_ohm_cm2"])
    assert float(np.max(np.abs(fr.J_fit - J_true))) < 0.5


def test_j0_n_degeneracy_is_flagged():
    """J0 and n compensate along a J-V curve — the non-uniqueness report must
    surface that correlation instead of hiding it."""
    V, J, _ = demo_measurement(noise_mA=0.15, seed=3)
    fr = fit_measured_jv(V, J, n_bootstrap=120, seed=2)
    pairs = {frozenset((a, b)) for a, b, _ in fr.degenerate_pairs}
    assert frozenset(("J0_mA_cm2", "n")) in pairs
    assert any("NOT independently identifiable" in w for w in fr.warnings)


def test_csv_parsing_variants_and_sign_autocorrect():
    body = "V,J\n" + "\n".join(
        f"{v:.3f};{-j:.3f}" for v, j in zip(
            np.linspace(0, 1.1, 20),
            diode_jv(np.linspace(0, 1.1, 20), 21, 1e-9, 1.5, 1.0, 3000)))
    V, J = parse_jv_csv(body.encode())
    assert V.size == 20
    assert np.interp(0.0, V, J) > 20            # sign auto-corrected
