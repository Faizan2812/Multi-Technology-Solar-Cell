"""Layer-library integrity: layer-built stacks must reconstruct the certified
devices, every cited reference must be registered, and interconnect choices
must match the validated tandem grades."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.layer_library import (
    SILICON_FRONT_STACKS, SILICON_REAR_STACKS, SILICON_TEXTURES,
    TANDEM_INTERCONNECTS, ORGANIC_INTERLAYERS, ORGANIC_METALS,
    build_silicon_from_layers, silicon_combination_status,
    build_organic_stack, organic_combination_status,
    interconnect_params, all_layer_reference_keys, _CERTIFIED_PAIRS)
from physics.silicon import SILICON_PRESETS, simulate_silicon

ROOT = os.path.join(os.path.dirname(__file__), "..")
DB = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))


# ── provenance ──────────────────────────────────────────────────────────
def test_every_layer_reference_is_registered():
    missing = [k for k in all_layer_reference_keys()
               if k not in DB["references"]]
    assert not missing, f"unregistered layer references: {missing}"


def test_layer_reference_dois_wellformed():
    for k in all_layer_reference_keys():
        doi = DB["references"][k].get("doi") or ""
        assert doi.startswith("10."), f"{k} lacks a valid DOI"


# ── silicon reconstruction: the authenticity guarantee ─────────────────
@pytest.mark.parametrize("pair,preset", list(_CERTIFIED_PAIRS.items()))
def test_layer_built_silicon_reconstructs_certified_preset(pair, preset):
    front, rear = pair
    arch = build_silicon_from_layers(front, rear)
    ref = SILICON_PRESETS[preset]
    # electrical totals must match the certified preset exactly
    assert abs(arch.J0s_fA - ref.J0s_fA) < 1e-9
    assert abs(arch.Rs_ohm_cm2 - ref.Rs_ohm_cm2) < 1e-9
    # and the simulated device must match the certified-preset result
    r_layer = simulate_silicon(arch)
    r_ref = simulate_silicon(ref)
    assert abs(r_layer["PCE"] - r_ref["PCE"]) < 0.05, (
        f"{preset}: layer-built {r_layer['PCE']:.2f}% vs preset "
        f"{r_ref['PCE']:.2f}%")
    status, hit = silicon_combination_status(front, rear)
    assert status == "CERTIFIED" and hit == preset


def test_extrapolated_combination_is_flagged_and_physical():
    front = "Boron emitter + AlOx/SiNx (TOPCon front)"
    rear = "a-Si:H(i/p) rear heterocontact (SHJ)"      # hybrid: no record
    status, _ = silicon_combination_status(front, rear)
    assert status == "EXTRAPOLATED"
    arch = build_silicon_from_layers(front, rear)
    assert abs(arch.J0s_fA - (2.5 + 0.6)) < 1e-9
    r = simulate_silicon(arch)
    # physical sanity: between the parent devices, under the 29.4% guard
    assert 20.0 < r["PCE"] < 29.4
    # better rear passivation than TOPCon rear -> Voc at least TOPCon's
    r_topcon = simulate_silicon(SILICON_PRESETS[
        "TOPCon back junction (Fraunhofer 26.0%)"])
    assert r["Voc"] >= r_topcon["Voc"] - 1e-3


def test_texture_ordering_is_physical():
    """Weaker light trapping must not increase Jsc."""
    front = "a-Si:H(i/n) heterocontact + ARC (SHJ front)"
    rear = "IBC a-Si:H(i/n+p) rear contacts (SHJ-IBC)"
    js = []
    for tex in ["Random pyramids + ARC (record-class)",
                "Industrial pyramids", "Planar / legacy"]:
        js.append(simulate_silicon(
            build_silicon_from_layers(front, rear, texture_key=tex))["Jsc"])
    assert js[0] >= js[1] >= js[2]


# ── organic stack builder ───────────────────────────────────────────────
def test_organic_default_stack_matches_certified_template():
    from physics.organic import TMM_STACKS
    tmpl, _ = TMM_STACKS["PM6:Y6"]
    built = build_organic_stack("PM6:Y6", L_nm=100)
    assert built == [(m, float(d)) for m, d in tmpl]
    status, _ = organic_combination_status("PM6:Y6", 
        "PEDOT:PSS (40 nm, hole side)", "Ag (100 nm)")
    assert status == "CERTIFIED-STACK"


def test_organic_stack_energy_conservation_and_metal_swap():
    from physics.tmm import solve_stack
    import numpy as np
    for metal in ("Ag (100 nm)", "Al (100 nm)"):
        stack = build_organic_stack("PM6:Y6", 100, metal_key=metal)
        sol = solve_stack(stack)
        budget = sol["R"] + sol["T"] + np.sum(sol["A"], axis=0)
        assert float(np.max(np.abs(budget - 1))) < 5e-4
    st1, _ = organic_combination_status("PM6:Y6",
        "PEDOT:PSS (40 nm, hole side)", "Al (100 nm)")
    assert st1 == "EXTRAPOLATED"


# ── tandem interconnects vs validated grades ────────────────────────────
def test_interconnect_2020_class_reproduces_certified_record():
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.tandem import simulate_perovskite_silicon_tandem
    p = interconnect_params("ITO recombination junction (2020-class)")
    r = simulate_perovskite_silicon_tandem(
        HTL_DB["2PACz"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"],
        ETL_DB.get("C60", ETL_DB.get("PCBM")), 650,
        SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"], Nt_top=2e14, **p)
    assert abs(r["PCE"] - 29.15) / 29.15 < 0.05


def test_interconnect_ordering_is_physical():
    """Better junctions (lower Rs/parasitic) must not lower tandem PCE."""
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.tandem import simulate_perovskite_silicon_tandem
    pces = []
    for key in ["ITO recombination junction (2020-class)",
                "nc-Si:H recombination junction (textured)",
                "Advanced passivated interconnect (2025-class)"]:
        p = interconnect_params(key)
        pces.append(simulate_perovskite_silicon_tandem(
            HTL_DB["2PACz"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"],
            ETL_DB.get("C60", ETL_DB.get("PCBM")), 650,
            SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"],
            Nt_top=2e14, **p)["PCE"])
    assert pces[0] < pces[1] < pces[2]
