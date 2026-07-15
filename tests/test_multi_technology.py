"""tests/test_multi_technology.py — v4.0 engines: silicon, organic, tandem, interop."""
import dataclasses
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ---------------------------------------------------------------- silicon
def test_silicon_presets_match_certified_devices():
    from physics.silicon import SILICON_PRESETS, simulate_silicon
    db = json.load(open(os.path.join(ROOT, "data",
                                     "multi_technology_database.json")))
    for b in db["benchmarks"]["silicon"]:
        r = simulate_silicon(SILICON_PRESETS[b["preset"]])
        err = abs(r["PCE"] - b["target"]["PCE"]) / b["target"]["PCE"] * 100
        assert err <= b["tolerance_PCE_pct"], (b["id"], r["PCE"])


def test_silicon_physics_trends():
    from physics.silicon import (SILICON_PRESETS, simulate_silicon,
                                 intrinsic_recombination)
    base = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]
    r0 = simulate_silicon(base)
    # worse surface passivation -> lower Voc
    worse = dataclasses.replace(base, J0s_fA=base.J0s_fA * 30)
    assert simulate_silicon(worse)["Voc"] < r0["Voc"]
    # thinner wafer -> lower Jsc (less IR absorption)
    thin = dataclasses.replace(base, W_um=50)
    assert simulate_silicon(thin)["Jsc"] < r0["Jsc"]
    # Auger rate increases with injection
    assert (intrinsic_recombination(1e16, 5e15) >
            intrinsic_recombination(1e14, 5e15))
    # Niewelt 2022 model runs and stays within 3% of Richter on a record cell
    r_n = simulate_silicon(base, auger_model="niewelt2022")
    assert abs(r_n["PCE"] - r0["PCE"]) / r0["PCE"] < 0.03


def test_silicon_below_thermodynamic_limit():
    """No silicon configuration may exceed the 29.4% single-junction limit
    (Niewelt et al. 2022, DOI 10.1016/j.solmat.2021.111467)."""
    from physics.silicon import SiliconArchitecture, simulate_silicon
    utopia = SiliconArchitecture(name="utopia", W_um=110, Ndop_cm3=1e15,
                                 tau_srh_ms=1000, J0s_fA=0.01,
                                 Rs_ohm_cm2=0.0, R_front=0.0, shading=0.0,
                                 IQE=1.0, fEQE_blue=1.0)
    assert simulate_silicon(utopia)["PCE"] < 29.4 + 0.3


# ---------------------------------------------------------------- organic
def test_organic_presets_match_certified_devices():
    from physics.organic import ORGANIC_PRESETS, simulate_organic
    db = json.load(open(os.path.join(ROOT, "data",
                                     "multi_technology_database.json")))
    for b in db["benchmarks"]["organic"]:
        blend = ORGANIC_PRESETS[b["preset"]]
        if "override" in b:
            blend = dataclasses.replace(blend, **b["override"])
        r = simulate_organic(blend)
        err = abs(r["PCE"] - b["target"]["PCE"]) / b["target"]["PCE"] * 100
        assert err <= b["tolerance_PCE_pct"], (b["id"], r["PCE"])


def test_organic_thickness_rolloff_direction():
    from physics.organic import ORGANIC_PRESETS, simulate_organic
    b = ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"]
    p100 = simulate_organic(b)["PCE"]
    p300 = simulate_organic(dataclasses.replace(b, L_nm=300))["PCE"]
    assert p300 < p100
    assert p300 > 0.7 * p100   # NFA blends stay functional at 300 nm


# ----------------------------------------------------------------- tandem
def test_tandem_record_lineage():
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.silicon import SILICON_PRESETS
    from physics.tandem import simulate_perovskite_silicon_tandem
    htl, etl = HTL_DB["2PACz"], ETL_DB.get("C60", ETL_DB.get("PCBM"))
    wg = PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"]
    si = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]
    r = simulate_perovskite_silicon_tandem(
        htl, wg, etl, 650, si, Nt_top=2e14, R_int=0.07,
        parasitic=0.05, Rs_int_ohm_cm2=4.5)
    assert abs(r["PCE"] - 29.15) / 29.15 < 0.05      # Al-Ashouri 2020
    assert 1.85 < r["Voc"] < 2.0
    # tandem must beat both subcells
    assert r["PCE"] > r["top"]["PCE"]
    assert r["PCE"] > 26.81 * 0.95                    # ~best single-junction Si


def test_tandem_2T_vs_4T_and_filtering():
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.silicon import SILICON_PRESETS, simulate_silicon
    from physics.tandem import simulate_perovskite_silicon_tandem
    htl, etl = HTL_DB["2PACz"], ETL_DB.get("C60", ETL_DB.get("PCBM"))
    wg = PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"]
    si = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]
    r2 = simulate_perovskite_silicon_tandem(htl, wg, etl, 600, si)
    r4 = simulate_perovskite_silicon_tandem(htl, wg, etl, 600, si,
                                            terminal="4T")
    # filtered bottom Jsc must be well below the full-sun value
    full = simulate_silicon(si)["Jsc"]
    assert r2["bottom"]["Jsc"] < 0.65 * full
    # 4T is free of current-matching, so >= 2T at fixed thickness
    assert r4["PCE"] >= r2["PCE"] - 0.3


# ---------------------------------------------------------------- optimizer
def test_multi_tech_optimizer_improves_silicon():
    from ai.multi_tech_optimizer import optimize
    from physics.silicon import SILICON_PRESETS, simulate_silicon
    base = SILICON_PRESETS["PERC industrial (~24%)"]
    res = optimize("silicon", {"tau_srh_ms": (0.5, 30), "J0s_fA": (2, 60)},
                   base, maxiter=6, popsize=6, sensitivity=False)
    assert res["best_value"] >= simulate_silicon(base)["PCE"] - 1e-6


# ------------------------------------------------------------------ interop
def test_interop_roundtrip(tmp_path):
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.device import fast_simulate
    from utils.interop import (export_scaps_def, export_device_json,
                               import_device_json, export_jv_csv,
                               export_eqe_csv)
    htl, ab, etl = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"]
    stack = [{"name": "ETL", "thickness_nm": 50, "material": etl},
             {"name": "absorber", "thickness_nm": 500, "material": ab},
             {"name": "HTL", "thickness_nm": 150, "material": htl}]
    txt = export_scaps_def(stack)
    assert "layers 3" in txt and "Eg" in txt
    # SCAPS export must re-import through our own SCAPS parser
    from utils.scaps_import import parse_def
    parsed = parse_def(txt)
    assert len(parsed["layers"]) == 3
    assert abs(parsed["layers"][1].Eg - ab.Eg) < 1e-6

    r = fast_simulate(htl, ab, etl, 150, 500, 50)
    js = export_device_json("perovskite", stack=stack, result=r)
    doc = import_device_json(js)
    assert doc["technology"] == "perovskite"
    assert abs(doc["result"]["PCE"] - r["PCE"]) < 1e-6
    assert export_jv_csv(r).startswith("V_V")
    assert export_eqe_csv(r).startswith("lambda_nm")


def test_reference_registry_integrity():
    db = json.load(open(os.path.join(ROOT, "data",
                                     "multi_technology_database.json")))
    for key, ref in db["references"].items():
        assert ref["doi"].startswith("10."), key
        assert len(ref["citation"]) > 30, key
        assert ref["confidence"] in ("HIGH", "MEDIUM", "LOW"), key
    # every benchmark points at a registered reference
    for tech in db["benchmarks"].values():
        for b in tech:
            assert b["reference"] in db["references"], b["id"]


# --------------------------------------------------------------------- tmm
def test_tmm_energy_conservation_and_fresnel():
    from physics.tmm import energy_conservation_error, tmm_single
    stk = [("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 90), ("Al", 100)]
    assert energy_conservation_error(stk) < 5e-4
    out = tmm_single(550.0, [np.complex128(1.52), np.complex128(1.0)], [])
    assert abs(out["R"] - ((1.52 - 1) / (1.52 + 1)) ** 2) < 1e-12
    assert abs(out["R"] + out["T"] - 1.0) < 1e-12


def test_tmm_p3ht_interference_structure():
    from physics.tmm import jsc_vs_thickness
    from scipy.signal import argrelextrema
    stk = [("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 90), ("Al", 100)]
    ds, js = jsc_vs_thickness(stk, "P3HT:PCBM", np.arange(30, 305, 5), IQE=0.8)
    mx = argrelextrema(js, np.greater, order=3)[0]
    assert len(mx) >= 2
    assert 60 <= ds[mx[0]] <= 95        # Sievers 2006 / Monestier 2007
    assert 190 <= ds[mx[1]] <= 260      # Kotlarski 2008
    assert 8.0 <= js[mx[1]] <= 12.0


def test_tmm_pm6y6_matches_certified_and_calibrated_paths():
    from physics.tmm import jsc_from_stack
    from physics.organic import ORGANIC_PRESETS, simulate_organic
    J, _ = jsc_from_stack([("ITO", 100), ("PEDOT:PSS", 40),
                           ("PM6:Y6", 100), ("Ag", 100)], "PM6:Y6", IQE=0.90)
    assert abs(J - 25.3) / 25.3 < 0.08   # Yuan 2019 certified device
    b = ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"]
    p_t = simulate_organic(b, optics="tmm")["PCE"]
    p_c = simulate_organic(b)["PCE"]
    assert abs(p_t - p_c) / p_c < 0.08   # independent code paths agree
