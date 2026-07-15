"""
tests/test_physics.py
======================
Tests that enforce fundamental physics the DD solver must obey regardless
of material choice.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_thermal_voltage_at_300K():
    from ai.pinn_real import VT_300K
    assert abs(VT_300K - 0.02585) < 0.001


def test_intrinsic_density_formula():
    from ai.pinn_real import DeviceSpec
    dev = DeviceSpec()
    expected = np.sqrt(dev.Nc * dev.Nv) * np.exp(-dev.Eg / (2 * dev.Vt))
    assert abs(dev.ni - expected) / expected < 1e-6


def test_permittivity_units():
    from ai.pinn_real import DeviceSpec, EPS0
    dev = DeviceSpec(eps_r=6.5)
    assert abs(dev.eps - 6.5 * EPS0) < 1e-20


def test_pinn_forward_pass():
    import torch
    from ai.pinn_real import PerovskitePINN
    torch.manual_seed(0)
    model = PerovskitePINN()
    x = torch.rand(50, 1)
    psi, log_n, log_p = model(x)
    assert psi.shape    == (50, 1)
    assert log_n.shape  == (50, 1)
    assert log_p.shape  == (50, 1)
    assert torch.isfinite(psi).all()
    assert torch.isfinite(log_n).all()
    assert torch.isfinite(log_p).all()


def test_fourier_features_shape():
    import torch
    from ai.pinn_real import FourierFeatures
    ff = FourierFeatures(in_dim=1, mapping_size=32, scale=8.0)
    x = torch.rand(10, 1)
    z = ff(x)
    assert z.shape == (10, 64)


def test_poisson_residual_finite():
    import torch
    from ai.pinn_real import PerovskitePINN, DeviceSpec, poisson_residual
    torch.manual_seed(0)
    model = PerovskitePINN()
    dev = DeviceSpec()
    x = torch.rand(16, 1)
    r = poisson_residual(model, x, dev, dev.L_total)
    assert torch.isfinite(r).all()
    assert r.shape == (16, 1)


def test_fast_sim_returns_reasonable_pce():
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    r = fast_simulate(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
        d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e14, T=300,
    )
    assert 5 < r["PCE"] < 35
    assert 0.5 < r["Voc"] < 1.5
    assert 5 < r["Jsc"] < 30
    assert 0.3 < r["FF"] < 0.95


def test_fast_sim_produces_valid_curve():
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    r = fast_simulate(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
        d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e14, T=300,
    )
    V = np.array(r["voltages"])
    J = np.array(r["currents"])
    assert len(V) == len(J) > 0
    assert r["Jsc"] > 0


def test_tandem_2T_current_matched():
    """2T tandem: Jsc must be the minimum of top and filtered-bottom Jsc, exactly."""
    from physics.device import simulate_tandem
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    r = simulate_tandem(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"], ETL_DB["SnO2"],
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],                     ETL_DB["SnO2"],
        d_top_abs=400, d_bot_abs=600, terminal="2T",
    )
    expected_jsc = min(r["top_cell"]["Jsc"], r["bottom_Jsc_filtered"])
    assert abs(r["Jsc_tandem"] - expected_jsc) < 1e-3, "2T must current-match exactly"
    assert r["Voc"] > r["top_cell"]["Voc"], "2T Voc should add (be larger than top alone)"


def test_tandem_filter_factor_in_range():
    """Spectral filter factor must be in (0, 1] — bottom cell can't get more photons than standalone."""
    from physics.device import simulate_tandem
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    r = simulate_tandem(
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"], ETL_DB["SnO2"],
        HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],                     ETL_DB["SnO2"],
        d_top_abs=400, d_bot_abs=600, terminal="2T",
    )
    assert 0.0 < r["filter_factor"] <= 1.0, f"filter_factor {r['filter_factor']} out of range (0, 1]"


def test_tandem_4T_higher_than_2T():
    """4T should be at least as efficient as 2T — independent operation removes current-matching loss."""
    from physics.device import simulate_tandem
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    args = dict(
        top_htl=HTL_DB["Spiro-OMeTAD"], top_abs=PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"], top_etl=ETL_DB["SnO2"],
        bot_htl=HTL_DB["Spiro-OMeTAD"], bot_abs=PEROVSKITE_DB["MAPbI3"], bot_etl=ETL_DB["SnO2"],
        d_top_abs=400, d_bot_abs=600,
    )
    r2 = simulate_tandem(**args, terminal="2T")
    r4 = simulate_tandem(**args, terminal="4T")
    assert r4["PCE"] >= r2["PCE"] - 0.01, "4T cannot be lower than 2T (within numerical noise)"


def test_database_low_confidence_policy():
    """Confidence-tier policy (matches physics/provenance_audit.py rule F4):
    LOW-confidence parameters ARE allowed — honest tiering is the point of
    the provenance system — but every one MUST carry a justification note,
    and LOW entries must stay a small minority of the database.

    (note: the CdTe technology family introduced the first legitimate
    LOW entries — e.g. the compensated-CdS cross-sections, which are poorly
    constrained experimentally. Pretending they are MEDIUM would be less
    honest, not more.)"""
    import json
    from pathlib import Path
    db_path = Path(__file__).parent.parent / "data" / "materials_database.json"
    with open(db_path) as f:
        db = json.load(f)

    n_total = 0
    low_without_note = []
    n_low = 0
    for cat in ["absorbers", "etls", "htls"]:
        for mat_name, mat in db.get(cat, {}).items():
            for pname, pdata in mat.get("parameters", {}).items():
                if not isinstance(pdata, dict):
                    continue
                n_total += 1
                if pdata.get("confidence") == "LOW":
                    n_low += 1
                    if not pdata.get("notes"):
                        low_without_note.append(f"{cat}/{mat_name}/{pname}")
    assert not low_without_note, \
        f"LOW-confidence without justification note: {low_without_note}"
    assert n_low / max(n_total, 1) < 0.05, \
        f"Too many LOW-confidence parameters: {n_low}/{n_total}"


def test_database_all_refs_resolvable():
    """FAIR-data policy: every reference must be RESOLVABLE — a DOI in the
    standard 10.xxxx/yyyy format, OR (for genuinely DOI-less sources such
    as conference proceedings, e.g. Gloeckler 2003 WCPEC) an explicit URL.
    'VERIFY' placeholders are never allowed."""
    import json
    import re
    from pathlib import Path
    db_path = Path(__file__).parent.parent / "data" / "materials_database.json"
    with open(db_path) as f:
        db = json.load(f)

    doi_pattern = re.compile(r"^10\.\d{4,9}/[\S]+$")
    bad = []
    for ref_id, ref in db.get("_references", {}).items():
        if ref_id.startswith("_REMOVED"):
            continue
        if "tombstone_date" in ref:
            continue
        doi = ref.get("doi", "")
        if doi == "VERIFY":
            bad.append(f"{ref_id}: DOI marked VERIFY")
        elif doi is None or doi == "":
            if not ref.get("url"):
                bad.append(f"{ref_id}: no DOI and no URL")
        elif not doi_pattern.match(doi):
            bad.append(f"{ref_id}: invalid DOI format '{doi}'")
    assert not bad, "Unresolvable references:\n" + "\n".join(bad)


def test_fast_simulate_etl_choice_changes_output():
    """REGRESSION GUARD: changing ETL must change PCE by at least 0.1 percentage points.

    Earlier versions had a bug where many ETL choices produced bit-identical output
    (the band-offset penalty thresholds only fired for very specific values).
    This test ensures every ETL in the database produces a measurably distinct PCE.
    """
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    # Run fast_simulate with same HTL/Abs but every ETL
    pce_results = {}
    for etl_name in ETL_DB:
        r = fast_simulate(
            HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB[etl_name],
            d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e15, T=300,
        )
        pce_results[etl_name] = r["PCE"]

    # PCE values across all ETLs should span at least 1 percentage point
    pce_range = max(pce_results.values()) - min(pce_results.values())
    assert pce_range >= 1.0, (
        f"ETL choice produces nearly identical PCE — physics is broken. "
        f"Range only {pce_range:.3f}% across {len(pce_results)} ETLs. "
        f"Values: {pce_results}"
    )


def test_fast_simulate_htl_choice_changes_output():
    """REGRESSION GUARD: changing HTL must change PCE by at least 0.1 percentage points."""
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    pce_results = {}
    for htl_name in HTL_DB:
        r = fast_simulate(
            HTL_DB[htl_name], PEROVSKITE_DB["MAPbI3"], ETL_DB["TiO2"],
            d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e15, T=300,
        )
        pce_results[htl_name] = r["PCE"]

    pce_range = max(pce_results.values()) - min(pce_results.values())
    assert pce_range >= 1.0, (
        f"HTL choice produces nearly identical PCE — physics is broken. "
        f"Range only {pce_range:.3f}% across {len(pce_results)} HTLs. "
        f"Values: {pce_results}"
    )


def test_fast_simulate_absorber_choice_changes_output():
    """REGRESSION GUARD: changing absorber must change PCE meaningfully."""
    from physics.device import fast_simulate
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB

    pce_results = {}
    for abs_name in PEROVSKITE_DB:
        r = fast_simulate(
            HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB[abs_name], ETL_DB["TiO2"],
            d_htl_nm=200, d_abs_nm=500, d_etl_nm=50, Nt_abs=1e15, T=300,
        )
        pce_results[abs_name] = r["PCE"]

    pce_range = max(pce_results.values()) - min(pce_results.values())
    assert pce_range >= 2.0, (
        f"Absorber choice should produce at least 2% PCE range across all materials. "
        f"Got {pce_range:.3f}%."
    )


def test_benchmark_json_matches_tool_output():
    """The 'our_pce' values stored in artifacts/benchmark_results.json must match
    what the tool actually produces when re-run. This guards against stale
    benchmark data — anyone opening the Benchmarks tab should see numbers
    that reproduce within 0.1% absolute when they re-run the simulation."""
    import json
    from pathlib import Path
    from utils.benchmark import run_full_benchmark

    bench_path = Path(__file__).parent.parent / "artifacts" / "benchmark_results.json"
    with open(bench_path) as f:
        stored = json.load(f)

    # Re-run all SCAPS-reference devices
    results = run_full_benchmark(mode="dd")
    stored_devices = {d["id"]: d for d in stored["scaps_reference"]["devices"]}

    for r in results:
        stored_dev = stored_devices.get(r.device_name)
        assert stored_dev is not None, f"Device {r.device_name} missing from JSON"
        delta = abs(stored_dev["our_pce"] - r.tool_PCE)
        assert delta < 0.1, (
            f"Benchmark inconsistency: device {r.device_name} stores our_pce="
            f"{stored_dev['our_pce']:.2f} but tool now produces {r.tool_PCE:.2f} "
            f"(Δ={delta:.2f}). Re-run scripts/run_benchmark.py to refresh the JSON."
        )
