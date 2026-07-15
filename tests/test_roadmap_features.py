"""Roadmap-feature verification: engine-derived temperature coefficients in
published bands, luminescent-coupling physics and limits, hysteresis-index
limits, Arrhenius blind recovery, and the joint light+dark fit tightening."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── energy yield ────────────────────────────────────────────────────────
def test_engine_temperature_coefficients_in_published_bands():
    from utils.energy_yield import engine_gamma
    _, g_si = engine_gamma("Silicon (SHJ-IBC record class)")
    _, g_pk = engine_gamma("Perovskite (MAPbI3 workbench)")
    # silicon high-Voc heterojunction class: about -0.2 to -0.35 %/K
    assert -0.0040 < g_si < -0.0015, g_si
    # perovskite fast model: negative, sub-silicon-Al-BSF magnitude
    assert -0.0045 < g_pk < -0.0008, g_pk


def test_hot_climate_yields_less_than_cool_at_same_irradiance_class():
    from utils.energy_yield import annual_yield
    hot = annual_yield("Silicon (SHJ-IBC record class)",
                       "Hot desert (Riyadh-class)")
    cool = annual_yield("Silicon (SHJ-IBC record class)",
                        "High-altitude cool (Andes-class)")
    # performance ratio (temperature losses only here) must be worse hot
    assert hot["performance_ratio"] < cool["performance_ratio"]
    assert 0.80 < hot["performance_ratio"] < 1.0
    assert hot["kWh_per_kWp_year"] > 500          # sanity scale


def test_yield_runs_for_all_technologies():
    from utils.energy_yield import annual_yield, TECHNOLOGIES
    for tech in TECHNOLOGIES:
        r = annual_yield(tech, "Temperate (Berlin-class)")
        assert r["kWh_per_kWp_year"] > 0
        assert "not a bankability simulation" in r["note"]


# ── luminescent coupling ────────────────────────────────────────────────
def _tandem(lc_eta=0.0, d_top=780):
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.silicon import SILICON_PRESETS
    from physics.tandem import simulate_perovskite_silicon_tandem as sim
    # thick top -> bottom-limited: the regime where LC matters (Jäger 2020)
    return sim(HTL_DB["2PACz"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"],
               ETL_DB.get("C60"), d_top,
               SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"],
               Nt_top=2e14, R_int=0.07, parasitic=0.05,
               Rs_int_ohm_cm2=4.5, lc_eta=lc_eta)


def test_lc_zero_changes_nothing():
    r0, r1 = _tandem(0.0), _tandem(0.0)
    assert r0["PCE"] == pytest.approx(r1["PCE"], rel=1e-12)
    assert r0["lc_eta"] == 0.0 and "lc_dJ_bot_mA_cm2" not in r0


def test_lc_raises_pce_and_bounded_by_top_excess():
    r0 = _tandem(0.0)
    r1 = _tandem(0.5)
    assert r1["PCE"] > r0["PCE"]
    dJ = r1["lc_dJ_bot_mA_cm2"]
    excess = r0["top"]["Jsc"] - r0["Jmpp"]
    assert 0 < dJ <= 0.5 * excess + 1e-9
    # monotone in eta
    r2 = _tandem(0.9)
    assert r2["PCE"] >= r1["PCE"]


# ── hysteresis reduced-order model ──────────────────────────────────────
def test_hysteresis_index_limits_and_peak():
    from utils.stability import hysteresis_index
    hi_fast = hysteresis_index(1e4)      # ions frozen
    hi_slow = hysteresis_index(1e-5)     # ions equilibrated
    hi_peak = hysteresis_index(1.2 / 10.0)   # t_scan = tau
    assert hi_fast < 0.01 and hi_slow < 0.01
    assert hi_peak > 0.10
    # peak is the maximum over a rate sweep
    rates = np.logspace(-5, 4, 60)
    his = [hysteresis_index(r) for r in rates]
    assert 0.95 * hi_peak <= max(his) <= hi_peak + 1e-12


def test_hysteresis_scales_with_ion_density():
    from utils.stability import hysteresis_index
    lo = hysteresis_index(0.12, N_ion_cm3=1e15)
    hi = hysteresis_index(0.12, N_ion_cm3=1e18)
    assert hi > lo


# ── Arrhenius T80 ───────────────────────────────────────────────────────
def test_arrhenius_blind_recovery():
    from utils.stability import fit_arrhenius, project_t80, t80_report
    Ea_true, lnA_true = 0.85, -20.0
    kB = 8.617333262e-5
    temps = [85.0, 65.0, 45.0]
    t80 = [np.exp(lnA_true + Ea_true / (kB * (T + 273.15))) for T in temps]
    Ea, lnA, r2 = fit_arrhenius(temps, t80)
    assert Ea == pytest.approx(Ea_true, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    rep = t80_report(temps, t80, T_op_C=35.0)
    assert rep["t80_projected_h"] == pytest.approx(
        project_t80(Ea_true, lnA_true, 35.0), rel=1e-6)
    assert rep["metric"].startswith("T80 per ISOS")


# ── joint light+dark fitting tightens identifiability ──────────────────
def test_joint_dark_fit_tightens_intervals_and_recovers_truth():
    from utils.measurement_fit import (demo_measurement, fit_measured_jv,
                                       diode_jv)
    V, J, truth = demo_measurement()
    Vd = np.linspace(0.2, 1.1, 25)
    Jd = np.abs(diode_jv(Vd, 0.0, truth["J0_mA_cm2"], truth["n"],
                         truth["Rs_ohm_cm2"], truth["Rsh_ohm_cm2"]))
    Jd *= (1 + np.random.default_rng(1).normal(0, 0.05, Vd.size))
    f0 = fit_measured_jv(V, J, n_bootstrap=80, seed=2)
    f1 = fit_measured_jv(V, J, n_bootstrap=80, seed=2, V_dark=Vd, J_dark=Jd)
    w0 = f0.ci_high["n"] - f0.ci_low["n"]
    w1 = f1.ci_high["n"] - f1.ci_low["n"]
    assert w1 < w0                              # tighter n interval
    assert f1.ci_low["n"] <= truth["n"] <= f1.ci_high["n"]
    assert abs(f1.params["n"] - truth["n"]) < 0.05
    assert any("dark" in w for w in f1.warnings)
