"""
tests/test_v31_phases.py — tests for the prior four-phase upgrade.

CdTe technology (materials, benchmark, light-side regression gate)
         + PINN inverse parameter prediction (incl. identifiability finding)
Phase 2: optimization head-to-head machinery + hybrid optimizer
Phase 3: dataset integration (seed dataset, PDP loader schema, training)
Phase 4: energy yield, SCAPS .def importer, packaging metadata
"""
import os
import sys
import json

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────── Phase 1a: CdTe ─────────────────────────────
class TestCdTe:
    def test_materials_loaded_with_provenance(self):
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        assert "CdTe" in PEROVSKITE_DB and "CdS" in ETL_DB
        assert "CdTe-BSF" in HTL_DB
        a = PEROVSKITE_DB["CdTe"]
        assert a.Eg == pytest.approx(1.5) and a.eps == pytest.approx(9.4)
        # sigma wired from JSON (Gloeckler capture cross-section)
        assert a.sigma_e == pytest.approx(1e-12)

    def test_perovskite_regression_gate(self):
        """light_side/window_filter params must NOT change the validated
        perovskite benchmark path."""
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from physics.device import simulate_iv_curve
        h, a, e = (HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],
                   ETL_DB["SnO2"])
        r = simulate_iv_curve(h, a, e, 150, 500, 50, None, 300, mode="dd")
        assert r["PCE"] == pytest.approx(18.33, abs=0.01)

    def test_light_side_matters_for_thick_absorber(self):
        """For a 4-um CdTe absorber, illuminating the junction (ETL) side
        must give materially different (higher) current than back-side."""
        from physics.cdte import cdte_materials
        from physics.dd_solver import build_mesh
        from physics.device import _dd_beer_lambert_generation
        h, a, e = cdte_materials()
        mesh = build_mesh([h, a, e], [100, 4000, 25],
                          N_per_layer=[10, 80, 10])
        G_etl = _dd_beer_lambert_generation(mesh, a, light_side="etl")
        G_htl = _dd_beer_lambert_generation(mesh, a, light_side="htl")
        m = mesh.layer == 1
        x = mesh.x[m]
        # centroid of generation must sit near the CdS side for 'etl'
        c_etl = np.sum(x * G_etl[m]) / np.sum(G_etl[m])
        c_htl = np.sum(x * G_htl[m]) / np.sum(G_htl[m])
        assert c_etl > c_htl

    @pytest.mark.slow
    def test_cdte_benchmark_against_published(self):
        from physics.cdte import run_cdte_benchmark
        out = run_cdte_benchmark(verbose=False)
        assert out["errors_pct"]["PCE"] < 1.0     # 0.0% at calibration
        assert out["errors_pct"]["Voc"] < 3.0
        assert out["errors_pct"]["Jsc"] < 4.0
        assert out["errors_pct"]["FF"] < 3.0
        assert out["C2_pass"]                      # out-of-sample band


# ─────────────────────────── Phase 1b: PINN inversion ───────────────────────
class TestPinnInverse:
    @pytest.fixture(scope="class")
    def setup(self):
        torch = pytest.importorskip("torch")
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from ai.conditional_pinn_torch import ConditionalPINN, DeviceFamily
        wts = os.path.join(ROOT, "artifacts", "conditional_pinn.pt")
        if not os.path.exists(wts):
            pytest.skip("trained conditional PINN weights not present")
        fam = DeviceFamily(HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],
                           ETL_DB["SnO2"])
        m = ConditionalPINN()
        m.load_state_dict(torch.load(wts, map_location="cpu"))
        m.eval()
        return m, fam

    @pytest.mark.slow
    def test_thickness_recovery_from_profiles(self, setup):
        from ai.pinn_inverse import invert_from_dd_device
        m, fam = setup
        r = invert_from_dd_device(m, fam, 520.0, 3e14)
        assert r["err_d_abs_pct"] < 8.0

    def test_nt_unidentifiable_from_v0_profiles(self, setup):
        """The identifiability finding itself: ground-truth DD profiles are
        insensitive to Nt at V=0 (change far below PINN noise floor)."""
        _, fam = setup
        _, r1 = fam.dd_solution(500.0, 1e13)
        _, r2 = fam.dd_solution(500.0, 1e15)
        dln = (np.linalg.norm(np.log(r1.n.clip(1e5)) - np.log(r2.n.clip(1e5)))
               / np.linalg.norm(np.log(r2.n.clip(1e5))))
        assert dln < 0.01     # << 0.07 PINN interpolation floor

    @pytest.mark.slow
    def test_two_stage_recovers_both_parameters(self, setup):
        from ai.pinn_inverse import invert_with_voc
        from physics.device import fast_simulate
        m, fam = setup
        d_true, Nt_true = 520.0, 3e14
        mesh, rsol = fam.dd_solution(d_true, Nt_true)
        xn = mesh.x / fam.L_total_cm(d_true)
        Voc = fast_simulate(fam.htl, fam.abs, fam.etl, 150, d_true, 50,
                            Nt_true, 300)["Voc"]
        out = invert_with_voc(m, fam, xn, rsol.psi - rsol.psi.mean(),
                              np.log(rsol.n.clip(1e5)), Voc)
        assert abs(out["d_abs_nm"] - d_true) / d_true < 0.08
        assert abs(np.log10(out["Nt_cm3"]) - np.log10(Nt_true)) < 0.3


# ───────────────────────── Phase 2: optimization study ──────────────────────
class TestOptBenchmark:
    def test_objective_counting_and_bounds_clip(self):
        from utils.opt_benchmark import make_objective, BOUNDS
        f = make_objective()
        v = f([1e9, -50, 1e9, 99])       # wildly out of bounds -> clipped
        assert np.isfinite(v) and f.state["n"] == 1

    def test_hybrid_optimizer_analytic(self):
        from ai.optimizer import hybrid_optimize
        def obj(x):
            return -((x[0] - 2) ** 2) * 3.0 - (x[1] - 5.0) ** 2
        x, v, h = hybrid_optimize(obj, [(0, 4 - 1e-9), (0, 10)], budget=90,
                                  integer_dims=(0,), seed=3)
        assert int(x[0]) == 2 and abs(x[1] - 5.0) < 0.3
        assert h["n"] <= 90

    def test_artifacts_exist_with_expected_shape(self):
        for prob in ("A", "B"):
            p = os.path.join(ROOT, "artifacts", f"opt_benchmark_{prob}.json")
            assert os.path.exists(p), f"run python -m utils.opt_benchmark ({prob})"
            d = json.load(open(p))
            assert "Hybrid DE+NM, integer-aware (this tool)" in d["results"]
            for r in d["results"].values():
                assert len(r["mean_trace"]) == d["budget"]

    def test_hybrid_matches_local_methods_on_smooth_problem(self):
        d = json.load(open(os.path.join(ROOT, "artifacts",
                                        "opt_benchmark_A.json")))
        hyb = d["results"]["Hybrid DE+NM, integer-aware (this tool)"]
        nm = d["results"]["Nelder-Mead (multi-start)"]
        assert hyb["best_mean"] > nm["best_mean"] - 0.05

    def test_quasi_newton_degrades_on_categorical_axes(self):
        """The synopsis' criticism, quantified: L-BFGS on Problem B is worse
        and higher-variance than the tool's hybrid."""
        d = json.load(open(os.path.join(ROOT, "artifacts",
                                        "opt_benchmark_B.json")))
        lb = d["results"]["L-BFGS-B (quasi-Newton, Silvaco-style)"]
        hyb = d["results"]["Hybrid DE+NM, integer-aware (this tool)"]
        assert hyb["best_mean"] > lb["best_mean"] + 1.0
        assert hyb["best_std"] < lb["best_std"]


# ─────────────────────────── Phase 3: dataset IO ────────────────────────────
class TestDatasetIO:
    def test_seed_dataset_fully_cited(self):
        from utils.dataset_io import load_seed_dataset
        rows = load_seed_dataset()
        assert len(rows) >= 14
        for r in rows:
            assert r["doi"] and str(r["doi"]).strip() not in ("", "NA")
            assert 0 < r["PCE"] < 30 and 0 < r["FF"] <= 1.0

    def test_feature_extraction(self):
        from utils.dataset_io import load_seed_dataset, dataset_to_features
        X, y, names, kept = dataset_to_features(load_seed_dataset())
        assert X.shape[1] == 5 and len(y) == X.shape[0] >= 14

    def test_training_pathway(self):
        from utils.dataset_io import load_seed_dataset, train_surrogate_on_dataset
        r = train_surrogate_on_dataset(load_seed_dataset())
        assert np.isfinite(r["test_metrics"]["R2"])
        assert r["n_train"] + r["n_test"] >= 14

    def test_pdp_loader_rejects_wrong_schema(self, tmp_path):
        from utils.dataset_io import load_perovskite_database
        p = tmp_path / "bad.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError):
            load_perovskite_database(str(p))

    def test_pdp_loader_parses_valid_schema(self, tmp_path):
        from utils.dataset_io import load_perovskite_database, PDP_COLUMNS
        cols = [PDP_COLUMNS[k] for k in
                ("bandgap", "abs_thick", "etl", "htl", "Voc", "Jsc", "FF",
                 "PCE", "doi")]
        p = tmp_path / "pdp.csv"
        p.write_text(",".join(cols) + "\n"
                     "1.55,500,SnO2,Spiro,1.10,23.0,79.5,20.1,10.1000/x\n"
                     "1.60,450,TiO2,PTAA,1.05,22.0,0.75,17.3,10.1000/y\n")
        rows = load_perovskite_database(str(p))
        assert len(rows) == 2
        assert rows[0]["FF"] == pytest.approx(0.795)   # % normalized
        assert rows[1]["FF"] == pytest.approx(0.75)


# ─────────────────────── Phase 4: yield, SCAPS, packaging ───────────────────
class TestPhase4:
    def _sim(self):
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from physics.device import fast_simulate
        h, a, e = (HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],
                   ETL_DB["SnO2"])
        return lambda T_K: fast_simulate(h, a, e, 150, 500, 50, 1e14, T_K)

    def test_energy_yield_physics(self):
        from physics.energy_yield import energy_yield
        r = energy_yield(self._sim())
        assert 0 < r["E_day_Wh_m2"] < r["insolation_Wh_m2"]
        assert 5 < r["harvesting_efficiency_pct"] < 30
        assert r["P_W_m2"][0] == 0.0           # night

    def test_pce_falls_with_temperature(self):
        from physics.energy_yield import intensity_temperature_map
        m = intensity_temperature_map(self._sim())
        pce = np.asarray(m["PCE_pct"])
        assert pce[0, -1] > pce[-1, -1]        # 15 C beats 65 C at 1 sun
        assert 10 < pce[1, -1] < 26            # sane magnitude at 25 C

    def test_scaps_def_parser(self):
        from utils.scaps_import import parse_def
        p = parse_def(os.path.join(ROOT, "examples", "example_cdte.def"))
        assert len(p["layers"]) == 2
        cds, cdte = p["layers"]
        assert cds.Eg == pytest.approx(2.4) and cds.doping_type == "n"
        assert cdte.doping == pytest.approx(2e14) and cdte.doping_type == "p"
        assert cdte.sigma_e == pytest.approx(5e-14)
        assert cdte.thickness_um == pytest.approx(4.0)

    @pytest.mark.slow
    def test_scaps_import_simulates(self):
        from utils.scaps_import import parse_def, simulate_imported
        p = parse_def(os.path.join(ROOT, "examples", "example_cdte.def"))
        r = simulate_imported(p)
        assert 5 < r["metrics"]["PCE"] < 30
        assert any("2-layer" in w for w in r["warnings"])

    def test_packaging_metadata_present(self):
        assert os.path.exists(os.path.join(ROOT, "pyproject.toml"))
        assert os.path.exists(os.path.join(ROOT, "CITATION.cff"))
        txt = open(os.path.join(ROOT, "pyproject.toml")).read()
        assert 'version = "1.1.0"' in txt
