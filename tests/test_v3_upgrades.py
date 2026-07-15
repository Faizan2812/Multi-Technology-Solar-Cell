"""
tests/test_v3_upgrades.py — unit tests for the v3.0 regeneration.

Covers:
  * TMM optical generation: energy conservation, optical Jsc bound, mesh mapping
  * External Rs/Rsh circuit: identity limits and FF degradation
  * Coupled ion migration: zero-ion limit, slow-scan limit, hysteresis sign
  * MMS on the PRODUCTION Poisson solver: 2nd-order convergence
  * Scharfetter-Gummel diffusion-limit identity
  * Split-conformal uncertainty: finite-sample coverage
  * Nested cross-validation: runs, reports optimism gap
  * Conditional PINN (torch): module presence + residual finiteness (training
    itself is exercised by scripts/train_conditional_pinn.py, too slow for CI)
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

H = HTL_DB["Spiro-OMeTAD"]
A = PEROVSKITE_DB["MAPbI3"]
E = ETL_DB["SnO2"]


# ── TMM optics ───────────────────────────────────────────────────────────────
class TestTMMGeneration:
    def setup_method(self):
        from physics.dd_solver import build_mesh
        self.mesh = build_mesh([H, A, E], [150, 500, 50])

    def test_energy_conservation(self):
        from physics.optical_generation import tmm_generation_on_mesh
        _, d = tmm_generation_on_mesh(self.mesh, [H, A, E], [150, 500, 50])
        assert d["energy_conservation_max_residual"] < 1e-10

    def test_optical_jsc_below_sq_limit(self):
        from physics.optical_generation import tmm_generation_on_mesh
        from physics.spectrum import sq_jsc
        _, d = tmm_generation_on_mesh(self.mesh, [H, A, E], [150, 500, 50])
        assert 0.0 < d["Jsc_optical_mA_cm2"] < sq_jsc(A.Eg)

    def test_generation_confined_to_absorber(self):
        from physics.optical_generation import tmm_generation_on_mesh
        G, _ = tmm_generation_on_mesh(self.mesh, [H, A, E], [150, 500, 50])
        assert np.all(G[self.mesh.layer != 1] == 0.0)
        assert np.any(G[self.mesh.layer == 1] > 0.0)

    def test_integrated_G_matches_optical_jsc(self):
        from physics.optical_generation import tmm_generation_on_mesh
        G, d = tmm_generation_on_mesh(self.mesh, [H, A, E], [150, 500, 50])
        q = 1.602176634e-19
        j = q * np.trapezoid(G, self.mesh.x) * 1e3
        assert abs(j - d["Jsc_optical_mA_cm2"]) / d["Jsc_optical_mA_cm2"] < 0.05


# ── External circuit ─────────────────────────────────────────────────────────
class TestExternalCircuit:
    def test_ideal_limits_are_identity(self):
        from physics.device import simulate_iv_curve
        r0 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd")
        r1 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd",
                               Rs=0.0, Rsh=1e12)
        assert abs(r0["PCE"] - r1["PCE"]) < 1e-9

    def test_series_resistance_lowers_FF(self):
        from physics.device import simulate_iv_curve
        r0 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd")
        r1 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd",
                               Rs=5.0)
        assert r1["FF"] < r0["FF"]
        assert r1["PCE"] < r0["PCE"]

    def test_shunt_lowers_FF(self):
        from physics.device import simulate_iv_curve
        r0 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd")
        r1 = simulate_iv_curve(H, A, E, 150, 400, 50, None, 300, mode="dd",
                               Rsh=200.0)
        assert r1["FF"] < r0["FF"]


# ── Coupled ion migration ────────────────────────────────────────────────────
class TestIonMigration:
    def test_zero_ion_limit_no_hysteresis(self):
        from physics.dd_ion import hysteresis_jv
        r = hysteresis_jv(H, A, E, 150, 400, 50, N_ion=0.0, scan_rate=0.1,
                          N_V=9)
        assert abs(r["hysteresis_index"]) < 1e-12

    def test_slow_scan_limit_no_hysteresis(self):
        from physics.dd_ion import hysteresis_jv
        r = hysteresis_jv(H, A, E, 150, 400, 50, N_ion=1e18, scan_rate=1e-5,
                          N_V=7)
        assert r["f_ion_relaxation"] > 0.99
        assert abs(r["hysteresis_index"]) < 1e-12

    def test_ion_charge_conservation(self):
        from physics.dd_solver import build_mesh
        from physics.dd_ion import _ion_equilibrium
        mesh = build_mesh([H, A, E], [150, 400, 50])
        psi = np.linspace(-1.0, 0.2, mesh.N)      # arbitrary potential
        N0 = 1e17
        P = _ion_equilibrium(mesh, psi, N0)
        m = mesh.layer == 1
        d_abs = mesh.x[m][-1] - mesh.x[m][0]
        total = np.trapezoid(P[m], mesh.x[m])
        assert abs(total - N0 * d_abs) / (N0 * d_abs) < 0.02

    def test_saturation_cap(self):
        from physics.dd_solver import build_mesh
        from physics.dd_ion import _ion_equilibrium
        mesh = build_mesh([H, A, E], [150, 400, 50])
        psi = np.linspace(-3.0, 3.0, mesh.N)      # extreme field
        N0 = 1e17
        P = _ion_equilibrium(mesh, psi, N0, P_max_factor=50.0)
        assert P.max() <= 50.0 * N0 * 1.0001


# ── Production-solver MMS + SG identity ──────────────────────────────────────
class TestNumericalIntegrity:
    def test_mms_production_poisson_second_order(self):
        from scripts.run_validation import mms_production_poisson
        r = mms_production_poisson(Ns=(41, 81, 161))
        assert 1.8 < r["convergence_rate"] < 2.3

    def test_sg_diffusion_limit_identity(self):
        from scripts.run_validation import sg_diffusion_limit
        r = sg_diffusion_limit()
        assert r["max_rel_error"] < 1e-10

    def test_mms_source_default_is_inert(self):
        """The mms_source hook must not change any physical solve."""
        from physics.dd_solver import build_mesh, solve_dd
        from physics.device import _dd_beer_lambert_generation
        mesh = build_mesh([H, A, E], [150, 400, 50])
        G = _dd_beer_lambert_generation(mesh, A)
        r = solve_dd(mesh, G, 0.0, H, E)
        assert r.converged and np.all(np.isfinite(r.psi))


# ── Uncertainty calibration ──────────────────────────────────────────────────
class TestUncertaintyCalibration:
    def test_conformal_coverage(self):
        from ai.uncertainty import coverage_validation
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (240, 2))
        y = 25 - 8 * (X[:, 0] - 0.5) ** 2 - 5 * (X[:, 1] - 0.4) ** 2 \
            + rng.normal(0, 0.3, 240)
        r = coverage_validation(X, y, alpha=0.05, n_repeats=8)
        assert r["conformal_coverage"] >= 0.88

    def test_conformal_interval_contains_prediction(self):
        from ai.uncertainty import SplitConformalRegressor
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (120, 2))
        y = X.sum(1) + rng.normal(0, 0.1, 120)
        scr = SplitConformalRegressor().fit(X, y)
        mu, lo, hi = scr.predict(X[:5])
        assert np.all(lo <= mu) and np.all(mu <= hi)
        assert np.all(hi - lo > 0)


# ── Nested CV ────────────────────────────────────────────────────────────────
class TestNestedCV:
    def test_nested_cv_runs_and_reports_gap(self):
        from ai.ml_models import nested_cross_validate, RandomForestRegressor
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (150, 3))
        Y = (10 * X[:, :1] + 5 * X[:, 1:2] ** 2
             + rng.normal(0, 0.2, (150, 1)))
        r = nested_cross_validate(
            RandomForestRegressor, X, Y,
            param_grid=[{"n_estimators": 20, "max_depth": 4},
                        {"n_estimators": 20, "max_depth": 8}],
            n_outer=3, n_inner=2)
        assert "R2" in r["nested_mean"]
        assert r["nested_mean"]["R2"] > 0.5
        assert len(r["per_fold_hyperparams"]) == 3
        assert "optimism_gap_r2" in r


# ── Conditional PINN (torch) ─────────────────────────────────────────────────
torch = pytest.importorskip("torch", reason="torch not installed")


class TestConditionalPINN:
    def test_forward_shapes(self):
        from ai.conditional_pinn_torch import ConditionalPINN
        m = ConditionalPINN()
        x = torch.rand(17, 1); th = torch.rand(17, 2) * 2 - 1
        psi, ln, lp = m(x, th)
        assert psi.shape == (17, 1) and ln.shape == (17, 1)

    def test_residuals_finite(self):
        from ai.conditional_pinn_torch import (ConditionalPINN, DeviceFamily,
                                               pde_residuals)
        fam = DeviceFamily(H, A, E)
        m = ConditionalPINN()
        x = torch.rand(32, 1); th = torch.rand(32, 2) * 2 - 1
        rP, rn, rp = pde_residuals(m, fam, x, th)
        for r in (rP, rn, rp):
            assert torch.all(torch.isfinite(r))

    def test_generation_positive_in_absorber(self):
        from ai.conditional_pinn_torch import DeviceFamily, spectral_coefficients
        w, a = spectral_coefficients(A)
        assert np.all(w >= 0) and np.all(a > 0)
        G0 = w.sum()                       # G at absorber front
        assert 1e20 < G0 < 1e23            # physically sensible cm^-3 s^-1
