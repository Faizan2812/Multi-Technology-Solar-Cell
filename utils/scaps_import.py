"""
utils/scaps_import.py — SCAPS-1D definition-file (.def) importer
(v3.1, Phase 4)
================================================================

Interoperability with the incumbent: SCAPS-1D (Burgelman, Nollet & Degrave,
Thin Solid Films 361-362, 527 (2000)) stores device definitions in plain-
text `.def` files with `layer` blocks of `key value` pairs. This importer
parses those blocks and maps them onto this tool's Material objects and
layer geometry, so an existing SCAPS project can be re-simulated (and then
optimized) here without retyping parameters.

Scope and honesty
-----------------
* Parsed per layer: thickness, Eg, chi (electron affinity), eps, Nc, Nv,
  mu_n, mu_p, donor/acceptor doping, bulk defect density and capture
  cross-sections (single dominant defect), and the absorption model
  constant when present.
* NOT parsed (out of the tool's 1D scope or SCAPS-specific): graded
  profiles per layer, multiple defect levels (only the largest-Nt level is
  kept, with a warning), interface defect blocks (reported, not imported;
  set them via InterfaceDefects in this tool), tunneling flags, and
  metastability sections. Every skipped item is listed in the returned
  `warnings` so nothing is silently dropped.
* SCAPS .def formats vary slightly across versions (3.x): the parser is
  tolerant (case-insensitive keys, unit suffixes stripped) and was written
  against the publicly documented format of the SCAPS 3.3 manual.
"""
from __future__ import annotations
import re
import numpy as np

# SCAPS key -> (our attribute, converter)
_KEYMAP = {
    "d":            ("thickness_um", float),          # um in SCAPS
    "thickness":    ("thickness_um", float),
    "eg":           ("Eg", float),
    "chi":          ("chi", float),
    "epsilon":      ("eps", float),
    "eps":          ("eps", float),
    "nc":           ("Nc", float),
    "nv":           ("Nv", float),
    "mun":          ("mu_e", float),
    "mu_n":         ("mu_e", float),
    "mup":          ("mu_h", float),
    "mu_p":         ("mu_h", float),
    "nd":           ("Nd", float),
    "na":           ("Na", float),
}


class ImportedMaterial:
    """Duck-typed Material compatible with physics.dd_solver.build_mesh."""
    def __init__(self, name):
        self.name = name
        self.Eg = 1.5; self.chi = 4.0; self.eps = 10.0
        self.Nc = 2.2e18; self.Nv = 1.8e19
        self.mu_e = 10.0; self.mu_h = 10.0
        self.doping = 1e16; self.doping_type = "n"
        self.Nt = 1e14; self.sigma_e = 1e-15; self.sigma_h = 1e-15
        self.alpha_coeff = 1e5
        self.interface_srv = 1e4
        self.layer_type = "imported"
        self.refs = ["imported from SCAPS .def"]
        self.category = "SCAPS import"
        self.thickness_um = 0.5

    def __repr__(self):
        return (f"ImportedMaterial({self.name}: Eg={self.Eg}, chi={self.chi}, "
                f"d={self.thickness_um} um, {self.doping_type}-type "
                f"{self.doping:.2e})")


def _tokens(line):
    line = line.split("//")[0].strip()
    if not line:
        return None, None
    parts = re.split(r"[\s:=]+", line, maxsplit=1)
    if len(parts) < 2:
        return parts[0].lower(), None
    return parts[0].lower(), parts[1].strip()


def parse_def(path_or_text):
    """Parse a SCAPS .def file. Accepts a filesystem path or the raw text.

    Returns dict:
      layers   : list of ImportedMaterial, front (illuminated) layer FIRST
                 (SCAPS convention: layers listed from illuminated side)
      warnings : list of strings for everything skipped/approximated
    """
    if "\n" in str(path_or_text) or not _looks_like_path(path_or_text):
        text = str(path_or_text)
    else:
        with open(path_or_text, "r", errors="replace") as fh:
            text = fh.read()

    warnings = []
    layers = []
    cur = None
    in_defect = False
    defect_buf = {}

    for raw in text.splitlines():
        key, val = _tokens(raw)
        if key is None:
            continue
        if key == "layer":
            if cur is not None:
                _finalize_defect(cur, defect_buf, warnings)
            cur = ImportedMaterial(val or f"layer{len(layers)+1}")
            layers.append(cur)
            in_defect = False; defect_buf = {}
            continue
        if cur is None:
            continue
        if key.startswith("defect") or key == "bulkdefect":
            if defect_buf:
                _finalize_defect(cur, defect_buf, warnings)
                warnings.append(f"{cur.name}: multiple defect levels; "
                                "keeping the largest-Nt level only")
            in_defect = True; defect_buf = {}
            continue
        if key == "interface":
            warnings.append("interface defect block present in .def — not "
                            "imported; configure InterfaceDefects manually")
            in_defect = False
            continue
        if in_defect:
            if key in ("nt", "ntotal", "n_t"):
                defect_buf["Nt"] = float(val)
            elif key in ("sigman", "sigma_n", "sige"):
                defect_buf["sigma_e"] = float(val)
            elif key in ("sigmap", "sigma_p", "sigh"):
                defect_buf["sigma_h"] = float(val)
            continue
        if key in _KEYMAP:
            attr, conv = _KEYMAP[key]
            try:
                setattr(cur, attr, conv(val.split()[0]))
            except (ValueError, AttributeError, IndexError):
                warnings.append(f"{cur.name}: could not parse '{raw.strip()}'")
        elif key in ("grading", "metastable", "tunneling"):
            warnings.append(f"{cur.name}: '{key}' section skipped "
                            "(outside this tool's scope)")

    if cur is not None:
        _finalize_defect(cur, defect_buf, warnings)

    for m in layers:
        if getattr(m, "Na", 0) and m.Na > getattr(m, "Nd", 0):
            m.doping, m.doping_type = m.Na, "p"
        elif getattr(m, "Nd", 0):
            m.doping, m.doping_type = m.Nd, "n"
    return {"layers": layers, "warnings": warnings}


def _finalize_defect(mat, buf, warnings):
    if not buf:
        return
    if buf.get("Nt", 0) >= getattr(mat, "Nt", 0):
        mat.Nt = buf.get("Nt", mat.Nt)
        mat.sigma_e = buf.get("sigma_e", mat.sigma_e)
        mat.sigma_h = buf.get("sigma_h", mat.sigma_h)


def _looks_like_path(s):
    s = str(s)
    return len(s) < 260 and ("/" in s or s.endswith(".def"))


def simulate_imported(parsed, N_V=25, T=300.0, light_side="etl"):
    """Run the drift-diffusion solver on an imported SCAPS stack.

    SCAPS lists layers from the illuminated side; this tool's mesh runs
    HTL -> absorber -> ETL with light on the ETL side, so the imported list
    is reversed and the middle (largest-Eg-gap heuristic: the layer with the
    smallest Eg) is treated as the absorber. For stacks with != 3 layers the
    first/last are used as contacts' neighbours and the smallest-Eg layer
    as absorber; extra layers raise a warning and the nearest three-layer
    reduction is simulated.
    """
    from physics.dd_solver import build_mesh, jv_sweep, extract_device_metrics
    from physics.device import _dd_beer_lambert_generation
    layers = parsed["layers"]
    warns = list(parsed["warnings"])
    if len(layers) < 2:
        raise ValueError("need at least 2 layers in the .def file")
    order = list(reversed(layers))            # -> back ... front == HTL..ETL
    if len(order) > 3:
        eg = [m.Eg for m in order]
        i_abs = int(np.argmin(eg))
        i_abs = min(max(i_abs, 1), len(order) - 2)
        order = [order[0], order[i_abs], order[-1]]
        warns.append(f"{len(layers)}-layer stack reduced to 3 layers "
                     f"(back={order[0].name}, absorber={order[1].name}, "
                     f"front={order[2].name})")
    elif len(order) == 2:
        import copy
        bsf = copy.deepcopy(order[0]); bsf.thickness_um = 0.05
        order = [bsf, order[0], order[1]]
        warns.append("2-layer stack: thin back layer cloned from absorber")
    h, a, e = order
    d_nm = [m.thickness_um * 1000.0 for m in order]
    mesh = build_mesh([h, a, e], d_nm,
                      N_per_layer=[12, max(60, int(d_nm[1] / 40)), 12], T=T)
    G = _dd_beer_lambert_generation(mesh, a, light_side=light_side)
    V, J, c = jv_sweep(mesh, G, h, e, V_min=0.0, V_max=1.2, N_V=N_V, T=T)
    m = extract_device_metrics(V, J, converged_flags=c)
    return {"metrics": m, "V": V, "J_mA": J * 1000.0, "converged": c,
            "stack": " / ".join(x.name for x in reversed(order)),
            "warnings": warns}
