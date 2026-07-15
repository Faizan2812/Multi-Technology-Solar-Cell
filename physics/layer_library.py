"""
physics/layer_library.py — layer-by-layer material selection for silicon,
organic, and tandem devices, in the same spirit as the perovskite workbench.

DESIGN CONTRACT (what makes combinations "authentic")
------------------------------------------------------
1.  Every layer entry carries literature references (registry keys with DOIs).
2.  Component values (per-layer J0, Rs, optical parameters) are chosen INSIDE
    published ranges such that the STACK TOTALS reconstruct the certified
    record devices exactly. The validated quantity is therefore the total:
    `tests/test_layer_library.py` asserts that layer-built stacks reproduce
    the certified-preset results, and the certified presets themselves pass
    the 14-device experimental suite.
3.  Combinations that no published device has demonstrated are permitted but
    are flagged by `combination_status()` as EXTRAPOLATED — the UI prints the
    flag. The tool never silently presents an unvalidated combination as if
    it were a benchmarked one.

Silicon per-layer J0 splits are consistent with the published device totals:
SHJ passivating contacts at the 1-3 fA/cm2 level (Yoshikawa 2017, Lin 2023),
TOPCon poly-Si rear + boron emitter front at the few-fA level (Richter 2021),
PERC and Al-BSF diffused-contact levels from the efficiency-tables lineage
(Green 2021).  Interconnect values follow the certified tandem lineage:
ITO recombination junction (Al-Ashouri 2020), nanocrystalline-Si:H
recombination junction on texture (Sahli 2018, Nature Materials — verified
via the Consensus academic index for this edition), and the 2025-class
passivated interconnect (Jia 2025).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from physics.silicon import SiliconArchitecture, SILICON_PRESETS


# ════════════════════════════════════════════════════════════════════════
#  SILICON — front-surface and rear-surface stacks
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SiliconLayerStack:
    """One side (front or rear) of a c-Si cell: passivation + contact scheme."""
    name: str
    J0_fA: float            # saturation-current contribution of this side
    Rs_ohm_cm2: float       # series-resistance contribution of this side
    R_front: float | None   # front reflectance (front stacks only)
    fEQE_blue: float | None # blue response factor (front stacks only)
    refs: Tuple[str, ...]   # registry keys (multi_technology_database.json)
    note: str


SILICON_FRONT_STACKS: Dict[str, SiliconLayerStack] = {
    "a-Si:H(i/n) heterocontact + ARC (SHJ front)": SiliconLayerStack(
        "a-Si:H(i/n) heterocontact + ARC", J0_fA=1.0, Rs_ohm_cm2=0.05,
        R_front=0.012, fEQE_blue=0.97,
        refs=("yoshikawa_2017_natenergy", "yoshikawa_2017_solmat"),
        note="Intrinsic/doped amorphous-Si passivating front of the IBC "
             "record device; component J0 consistent with the 2.6 fA/cm2 "
             "device total (Yoshikawa 2017)."),
    "a-Si:H(i/n) bifacial front (LONGi SHJ)": SiliconLayerStack(
        "a-Si:H(i/n) bifacial front", J0_fA=0.5, Rs_ohm_cm2=0.05,
        R_front=0.015, fEQE_blue=0.96,
        refs=("lin_2023_natenergy",),
        note="Front side of the 26.81% both-side-contacted SHJ; split of the "
             "1.1 fA/cm2 device total (Lin 2023)."),
    "Boron emitter + AlOx/SiNx (TOPCon front)": SiliconLayerStack(
        "Boron emitter + AlOx/SiNx", J0_fA=2.5, Rs_ohm_cm2=0.15,
        R_front=0.010, fEQE_blue=0.95,
        refs=("richter_2021_natenergy",),
        note="Diffused boron front with AlOx/SiNx passivation of the 26.0% "
             "back-junction TOPCon (Richter 2021)."),
    "Phosphorus emitter + SiNx (PERC front)": SiliconLayerStack(
        "Phosphorus emitter + SiNx", J0_fA=30.0, Rs_ohm_cm2=0.25,
        R_front=0.018, fEQE_blue=0.97,
        refs=("green_2021_tables57",),
        note="Industrial diffused emitter with SiNx ARC; PERC-class level "
             "(efficiency-tables lineage, Green 2021)."),
    "Phosphorus emitter, legacy (Al-BSF front)": SiliconLayerStack(
        "Phosphorus emitter, legacy", J0_fA=150.0, Rs_ohm_cm2=0.35,
        R_front=0.030, fEQE_blue=0.97,
        refs=("green_2021_tables57",),
        note="Heavily doped legacy emitter; Al-BSF-era level."),
}

SILICON_REAR_STACKS: Dict[str, SiliconLayerStack] = {
    "IBC a-Si:H(i/n+p) rear contacts (SHJ-IBC)": SiliconLayerStack(
        "IBC a-Si:H rear contacts", J0_fA=1.6, Rs_ohm_cm2=0.27,
        R_front=None, fEQE_blue=None,
        refs=("yoshikawa_2017_natenergy",),
        note="Interdigitated rear heterocontacts of the Kaneka record."),
    "a-Si:H(i/p) rear heterocontact (SHJ)": SiliconLayerStack(
        "a-Si:H(i/p) rear heterocontact", J0_fA=0.6, Rs_ohm_cm2=0.05,
        R_front=None, fEQE_blue=None,
        refs=("lin_2023_natenergy",),
        note="Rear hole contact of the 26.81% SHJ (Lin 2023)."),
    "n-poly-Si/SiOx passivating rear (TOPCon)": SiliconLayerStack(
        "n-poly-Si/SiOx passivating rear", J0_fA=1.5, Rs_ohm_cm2=0.15,
        R_front=None, fEQE_blue=None,
        refs=("richter_2021_natenergy",),
        note="Tunnel-oxide passivated contact rear (Richter 2021)."),
    "AlOx/SiNx + local Al-BSF rear (PERC)": SiliconLayerStack(
        "AlOx/SiNx + local Al-BSF rear", J0_fA=15.0, Rs_ohm_cm2=0.20,
        R_front=None, fEQE_blue=None,
        refs=("green_2021_tables57",),
        note="Dielectric rear passivation with local contacts."),
    "Full-area Al-BSF rear (legacy)": SiliconLayerStack(
        "Full-area Al-BSF rear", J0_fA=200.0, Rs_ohm_cm2=0.40,
        R_front=None, fEQE_blue=None,
        refs=("green_2021_tables57",),
        note="Legacy full-area aluminium back-surface field."),
}

SILICON_TEXTURES: Dict[str, Tuple[int, str]] = {
    # name -> (Lambertian path factor Z, note). 4n^2 ~ 50 is the
    # Tiedje-Yablonovitch bound; textured record cells operate near it.
    "Random pyramids + ARC (record-class)": (50, "Near-Lambertian light "
        "trapping (Tiedje 1984 bound 4n²≈50; Green 2008 optics)."),
    "Industrial pyramids": (45, "Production-line texture."),
    "Planar / legacy": (35, "Weak trapping; legacy processing."),
}
_TEXTURE_REFS = ("tiedje_1984_ted", "green_2008_solmat")

# Which (front, rear) pairs correspond to certified devices
_CERTIFIED_PAIRS = {
    ("a-Si:H(i/n) heterocontact + ARC (SHJ front)",
     "IBC a-Si:H(i/n+p) rear contacts (SHJ-IBC)"): "SHJ-IBC (Kaneka 26.7%)",
    ("a-Si:H(i/n) bifacial front (LONGi SHJ)",
     "a-Si:H(i/p) rear heterocontact (SHJ)"): "SHJ both-side (LONGi 26.81%)",
    ("Boron emitter + AlOx/SiNx (TOPCon front)",
     "n-poly-Si/SiOx passivating rear (TOPCon)"): "TOPCon back junction (Fraunhofer 26.0%)",
    ("Phosphorus emitter + SiNx (PERC front)",
     "AlOx/SiNx + local Al-BSF rear (PERC)"): "PERC industrial (~24%)",
    ("Phosphorus emitter, legacy (Al-BSF front)",
     "Full-area Al-BSF rear (legacy)"): "Al-BSF legacy (~20%)",
}


def build_silicon_from_layers(front_key: str, rear_key: str,
                              texture_key: str | None = None,
                              W_um: float | None = None,
                              Ndop_cm3: float | None = None,
                              tau_srh_ms: float | None = None,
                              dopant_type: str | None = None,
                              name: str | None = None) -> SiliconArchitecture:
    """Assemble a SiliconArchitecture from cited layer stacks.

    J0s and Rs are the SUMS of the side contributions; optics come from the
    front stack and the texture choice. If `texture_key` is None and the
    (front, rear) pair matches a certified device, the certified device's own
    light-trapping factor is inherited (so reconstruction is exact); for
    non-certified pairs the record-class texture is the default. Wafer
    parameters default to the certified device when the pair matches one.
    """
    f, r = SILICON_FRONT_STACKS[front_key], SILICON_REAR_STACKS[rear_key]
    base_name = _CERTIFIED_PAIRS.get((front_key, rear_key))
    base = SILICON_PRESETS[base_name] if base_name else SILICON_PRESETS[
        "TOPCon back junction (Fraunhofer 26.0%)"]
    if texture_key is None:
        Z = base.Z_path if base_name else \
            SILICON_TEXTURES["Random pyramids + ARC (record-class)"][0]
    else:
        Z, _ = SILICON_TEXTURES[texture_key]
    return replace(
        base,
        name=name or (base_name or f"custom: {f.name} / {r.name}"),
        J0s_fA=f.J0_fA + r.J0_fA,
        Rs_ohm_cm2=round(f.Rs_ohm_cm2 + r.Rs_ohm_cm2, 4),
        R_front=f.R_front, fEQE_blue=f.fEQE_blue, Z_path=Z,
        W_um=W_um if W_um is not None else base.W_um,
        Ndop_cm3=Ndop_cm3 if Ndop_cm3 is not None else base.Ndop_cm3,
        tau_srh_ms=tau_srh_ms if tau_srh_ms is not None else base.tau_srh_ms,
        dopant_type=dopant_type or base.dopant_type,
    )


def silicon_combination_status(front_key: str, rear_key: str) -> Tuple[str, str]:
    """('CERTIFIED', preset) if the pair is a benchmarked device,
    else ('EXTRAPOLATED', advice)."""
    hit = _CERTIFIED_PAIRS.get((front_key, rear_key))
    if hit:
        return "CERTIFIED", hit
    return ("EXTRAPOLATED",
            "No certified record device uses this exact pair; results follow "
            "the same validated physics but are not benchmark-anchored — "
            "treat as design guidance, verify against your own baseline.")


def silicon_budget_rows(front_key: str, rear_key: str) -> List[dict]:
    """Per-layer J0/Rs budget with references, for UI tables."""
    rows = []
    for side, st in (("Front", SILICON_FRONT_STACKS[front_key]),
                     ("Rear", SILICON_REAR_STACKS[rear_key])):
        rows.append({"Side": side, "Stack": st.name,
                     "J0 (fA/cm²)": st.J0_fA, "Rs (Ω·cm²)": st.Rs_ohm_cm2,
                     "References": ", ".join(st.refs), "Note": st.note})
    return rows


# ════════════════════════════════════════════════════════════════════════
#  ORGANIC — full-stack builder (electrode / interlayer / blend / metal)
# ════════════════════════════════════════════════════════════════════════
# n,k provenance lives in physics/tmm.py (_NK): ITO & PEDOT:PSS (Burkhard
# 2010), PM6:Y6 (Kerremans 2020, ±5–10% batch spread), P3HT:PCBM (Monestier
# 2007; Burkhard 2010), Al (Rakic 1995), Ag (Johnson & Christy 1972),
# ZnO (standard device-modeling values, disclosed as such).
ORGANIC_INTERLAYERS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "PEDOT:PSS (40 nm, hole side)": ("PEDOT:PSS", ("burkhard_2010_advmater",)),
    "ZnO (30 nm, electron side)":   ("ZnO", ()),
}
ORGANIC_METALS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "Ag (100 nm)": ("Ag", ("johnson_1972_prb",)),
    "Al (100 nm)": ("Al", ("rakic_1995_ao",)),
}
_ORGANIC_CERTIFIED_STACKS = {
    ("PM6:Y6", "PEDOT:PSS", "Ag"),
    ("P3HT:PCBM", "PEDOT:PSS", "Al"),
}


def build_organic_stack(blend_nk_name: str, L_nm: float,
                        interlayer_key: str = "PEDOT:PSS (40 nm, hole side)",
                        metal_key: str = "Ag (100 nm)",
                        ito_nm: float = 100.0,
                        il_nm: float = 40.0) -> List[Tuple[str, float]]:
    il, _ = ORGANIC_INTERLAYERS[interlayer_key]
    metal, _ = ORGANIC_METALS[metal_key]
    return [("ITO", float(ito_nm)), (il, float(il_nm)),
            (blend_nk_name, float(L_nm)), (metal, 100.0)]


def organic_combination_status(blend_nk_name: str, interlayer_key: str,
                               metal_key: str) -> Tuple[str, str]:
    il, _ = ORGANIC_INTERLAYERS[interlayer_key]
    metal, _ = ORGANIC_METALS[metal_key]
    if (blend_nk_name, il, metal) in _ORGANIC_CERTIFIED_STACKS:
        return "CERTIFIED-STACK", ("Matches the stack used for the certified "
                                   "benchmark of this blend.")
    return "EXTRAPOLATED", ("Physically valid TMM solve with cited n,k data, "
                            "but this exact electrode/interlayer combination "
                            "was not the certified-benchmark stack.")


# ════════════════════════════════════════════════════════════════════════
#  TANDEM — interconnect (recombination-junction) material selection
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Interconnect:
    name: str
    Rs_int_ohm_cm2: float   # series resistance of the junction
    parasitic: float        # parasitic absorption fraction (bottom-cell path)
    R_int: float            # internal reflection loss at the junction
    refs: Tuple[str, ...]
    note: str


TANDEM_INTERCONNECTS: Dict[str, Interconnect] = {
    "ITO recombination junction (2020-class)": Interconnect(
        "ITO recombination junction", 4.5, 0.05, 0.07,
        ("alashouri_2020_science",),
        "TCO-based junction of the 29.15% certified device "
        "(Al-Ashouri 2020); grade values reproduce that record."),
    "nc-Si:H recombination junction (textured)": Interconnect(
        "nc-Si:H recombination junction", 2.5, 0.035, 0.05,
        ("sahli_2018_natmater",),
        "Nanocrystalline-silicon junction demonstrated on fully textured "
        "silicon (Sahli 2018, Nat. Mater., certified 25.2%); lower parasitic "
        "absorption and shunt-resilient vs TCO — the >1 mA/cm² bottom-cell "
        "optical gain reported in the companion study motivates the lower "
        "parasitic/reflection values."),
    "Advanced passivated interconnect (2025-class)": Interconnect(
        "Advanced passivated interconnect", 1.5, 0.02, 0.03,
        ("jia_2025_nature",),
        "Interface-quality level of the 34.58% record lineage (Jia 2025); "
        "matches the validated 2025 grade preset."),
}


def interconnect_params(key: str) -> dict:
    ic = TANDEM_INTERCONNECTS[key]
    return {"Rs_int_ohm_cm2": ic.Rs_int_ohm_cm2, "parasitic": ic.parasitic,
            "R_int": ic.R_int}


def all_layer_reference_keys() -> List[str]:
    """Every registry key cited anywhere in this library (for tests)."""
    keys = set(_TEXTURE_REFS)
    for db in (SILICON_FRONT_STACKS, SILICON_REAR_STACKS):
        for st in db.values():
            keys.update(st.refs)
    for _, refs in list(ORGANIC_INTERLAYERS.values()) + list(ORGANIC_METALS.values()):
        keys.update(refs)
    for ic in TANDEM_INTERCONNECTS.values():
        keys.update(ic.refs)
    return sorted(keys)


# ════════════════════════════════════════════════════════════════════════
#  ORGANIC — donor:acceptor blend builder (certified pairs + Scharber
#  estimation for novel pairs)
# ════════════════════════════════════════════════════════════════════════
# Component energetics (eV, vs vacuum) as reported in the certified-preset
# papers already in the registry. Materials whose energetics we have NOT
# stored with confidence carry energetics=None and are offered ONLY in
# their certified pairing — the builder refuses to estimate with values it
# cannot source. Optical gaps Eg_opt are the values the certified presets
# carry; the blend optical gap is min(donor, acceptor).
ORGANIC_DONORS: Dict[str, dict] = {
    "PM6": {"HOMO": -5.45, "LUMO": -3.65, "Eg_opt": 1.80,
            "refs": ("yuan_2019_joule",),
            "note": "Benchmark donor of the Y6/L8-BO record lineage."},
    "P3HT": {"HOMO": -5.00, "LUMO": -3.00, "Eg_opt": 1.85,
             "refs": ("koster_2005_prb",),
             "note": "Legacy workhorse polymer donor."},
    "PBQx-TF": {"HOMO": None, "LUMO": None, "Eg_opt": None,
                "refs": ("cui_2021_advmater",),
                "note": "Energetics not stored — offered only in its "
                        "certified pairing (eC9-2Cl)."},
}
ORGANIC_ACCEPTORS: Dict[str, dict] = {
    "Y6": {"HOMO": -5.65, "LUMO": -4.10, "Eg_opt": 1.33,
           "refs": ("yuan_2019_joule",),
           "note": "The A-DA'D-A NFA that transformed the field."},
    "L8-BO": {"HOMO": -5.68, "LUMO": -3.90, "Eg_opt": 1.40,
              "refs": ("li_2021_natenergy",),
              "note": "Branched-side-chain Y6 derivative."},
    "PC61BM": {"HOMO": -6.10, "LUMO": -3.91, "Eg_opt": 2.00,
               "refs": ("koster_2005_prb",),
               "note": "Fullerene reference acceptor."},
    "eC9-2Cl": {"HOMO": None, "LUMO": None, "Eg_opt": None,
                "refs": ("cui_2021_advmater",),
                "note": "Energetics not stored — certified pairing only."},
    "L8-BO-C4": {"HOMO": None, "LUMO": None, "Eg_opt": None,
                 "refs": ("li_2025_natmater",),
                 "note": "Energetics not stored — certified pairing only."},
}
_ORGANIC_CERTIFIED_BLENDS = {
    ("PM6", "Y6"): "PM6:Y6 (Joule 2019, 15.7%)",
    ("PM6", "L8-BO"): "PM6:L8-BO (Nat. Energy 2021, 18.3%)",
    ("PBQx-TF", "eC9-2Cl"): "PBQx-TF:eC9-2Cl (Adv. Mater. 2021, 19.0%)",
    ("PM6", "L8-BO-C4"): "PM6:L8-BO-C4 (Nat. Mater. 2025, 20.4%)",
    ("P3HT", "PC61BM"): "P3HT:PC61BM (legacy, ~4%)",
}


def build_organic_blend(donor_key: str, acceptor_key: str):
    """(blend, status, message) for a donor:acceptor choice.

    CERTIFIED  : the pair is a benchmarked preset — returned verbatim, so
                 results equal the certified-suite numbers exactly.
    EXTRAPOLATED: both components carry sourced energetics — a Scharber
                 design-rule estimate (scharber_2006_advmater):
                 Voc = (|HOMO_D| - |LUMO_A|)/e - 0.3 V, conservative
                 EQE plateau 0.65, remaining transport parameters from the
                 nearest certified class. A physics-consistent ESTIMATE for
                 screening — never presented as a benchmarked device.
    Raises ValueError when energetics are unavailable or the level
    alignment cannot separate charge (no donor→acceptor offsets).
    """
    from dataclasses import replace as _replace
    from physics.organic import ORGANIC_PRESETS
    pair = (donor_key, acceptor_key)
    if pair in _ORGANIC_CERTIFIED_BLENDS:
        name = _ORGANIC_CERTIFIED_BLENDS[pair]
        return (ORGANIC_PRESETS[name], "CERTIFIED",
                f"Certified pair — results equal the benchmarked preset "
                f"'{name}' (validated in the 14-device suite).")
    D = ORGANIC_DONORS[donor_key]
    A = ORGANIC_ACCEPTORS[acceptor_key]
    if D["HOMO"] is None or A["LUMO"] is None:
        raise ValueError(
            f"No certified benchmark exists for {donor_key}:{acceptor_key} "
            "and sourced energetics are not stored for at least one "
            "component — the builder does not estimate with unsourced "
            "values. Choose a certified pairing for that material.")
    # charge-separation gates: both level offsets must be favourable
    if not (D["LUMO"] > A["LUMO"] and D["HOMO"] > A["HOMO"]):
        raise ValueError(
            f"{donor_key}:{acceptor_key}: energy-level alignment does not "
            "provide donor→acceptor electron transfer (LUMO_D must lie "
            "above LUMO_A and HOMO_D above HOMO_A).")
    Eg = min(D["Eg_opt"], A["Eg_opt"])
    Voc_est = (abs(D["HOMO"]) - abs(A["LUMO"])) - 0.30   # Scharber 2006
    E_loss = float(np.clip(Eg - Voc_est, 0.45, 1.30))
    # transport/parasitic class from the nearest certified preset
    donor_class = ("P3HT:PC61BM (legacy, ~4%)" if acceptor_key == "PC61BM"
                   else "PM6:Y6 (Joule 2019, 15.7%)")
    base = ORGANIC_PRESETS[donor_class]
    blend = _replace(
        base,
        name=f"{donor_key}:{acceptor_key} (Scharber estimate)",
        Eg_opt_eV=float(Eg), E_loss_eV=E_loss, EQE_max=0.65,
        reference="scharber_2006_advmater")
    return (blend, "EXTRAPOLATED",
            "No certified device for this pair — Scharber design-rule "
            f"estimate (Voc ≈ {max(Eg - E_loss, 0):.2f} V, EQE plateau "
            "0.65, transport from the nearest certified class; registry: "
            "scharber_2006_advmater). Screening guidance only.")


import numpy as np  # noqa: E402  (used by build_organic_blend)
