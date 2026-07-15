"""Blend builder: certified pairs must equal benchmarked presets exactly;
Scharber estimates must be flagged, physical, and refuse unsourced inputs;
the AI optimizer must accept built blends."""
import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.layer_library import (build_organic_blend, ORGANIC_DONORS,
                                   ORGANIC_ACCEPTORS,
                                   _ORGANIC_CERTIFIED_BLENDS)
from physics.organic import ORGANIC_PRESETS, simulate_organic


@pytest.mark.parametrize("pair,preset", list(_ORGANIC_CERTIFIED_BLENDS.items()))
def test_certified_pairs_equal_presets_exactly(pair, preset):
    b, status, _ = build_organic_blend(*pair)
    assert status == "CERTIFIED"
    assert simulate_organic(b)["PCE"] == pytest.approx(
        simulate_organic(ORGANIC_PRESETS[preset])["PCE"], rel=1e-12)


def test_scharber_estimate_flagged_and_physical():
    b, status, msg = build_organic_blend("P3HT", "Y6")
    assert status == "EXTRAPOLATED" and "Scharber" in msg
    r = simulate_organic(b)
    assert 0 < r["Voc"] < b.Eg_opt_eV            # below the optical gap
    assert 0 < r["PCE"] < 21.0                   # below certified record
    assert b.reference == "scharber_2006_advmater"
    # Scharber Voc rule reproduced: (|HOMO_D|-|LUMO_A|) - 0.3 = 0.60 V
    assert r["Voc"] == pytest.approx(0.60, abs=0.03)


def test_unsourced_energetics_refused():
    with pytest.raises(ValueError, match="unsourced"):
        build_organic_blend("PBQx-TF", "Y6")


def test_bad_level_alignment_refused():
    # temporarily craft a donor with LUMO below Y6's
    ORGANIC_DONORS["_bad"] = {"HOMO": -5.4, "LUMO": -4.5, "Eg_opt": 1.5,
                              "refs": (), "note": "test"}
    try:
        with pytest.raises(ValueError, match="alignment"):
            build_organic_blend("_bad", "Y6")
    finally:
        ORGANIC_DONORS.pop("_bad")


def test_optimizer_accepts_built_blend():
    from ai.multi_tech_optimizer import optimize
    b, _, _ = build_organic_blend("P3HT", "Y6")
    base = simulate_organic(b)["PCE"]
    res = optimize("organic", {"L_nm": (60, 250), "E_loss_eV": (0.45, 0.9)},
                   b, maxiter=4, popsize=6, sensitivity=False)
    assert res["best_value"] >= base - 1e-6
