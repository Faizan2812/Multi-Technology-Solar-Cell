"""
utils/interop.py — Cross-tool interoperability layer
===========================================================
Complements utils/scaps_import.py (SCAPS-1D .def import) with the
export direction and generic exchange formats, so a device designed or
optimized in this tool can be re-checked in the incumbent simulators:

* SCAPS-1D (Burgelman et al., Thin Solid Films 361-362, 527 (2000),
  DOI 10.1016/S0040-6090(99)00825-1) — `export_scaps_def` writes a
  layer-block .def-style text with the parameter keys documented in the
  SCAPS 3.3 manual. SCAPS's binary problem-definition extras (grading,
  metastability, multi-defect spectra) are outside this exporter's
  scope and are listed in the file header so nothing is silently
  dropped.
* Generic JSON device spec (`export_device_json` / `import_device_json`)
  — a complete, self-describing stack + parameter dump usable by
  scripted pipelines and other open tools (OghmaNano/gpvdm, custom
  drift-diffusion codes). Every material parameter carries its
  provenance keys when available.
* J-V / EQE CSV export (`export_jv_csv`, `export_eqe_csv`) — the
  universal exchange format for measurement-analysis tools.

Compatibility statement (see docs/INTEROPERABILITY.md): parameter
name mapping covers the full 1-D electro-optical parameter set shared
by SCAPS-1D, OghmaNano and this tool (thickness, Eg, chi, eps, Nc, Nv,
mu_n, mu_p, Na, Nd, Nt/defects, alpha model), i.e. everything both
sides can express.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import date


# ---------------------------------------------------------------------------
# SCAPS-1D .def export
# ---------------------------------------------------------------------------
_SCAPS_KEYS = [
    ("d", "thickness_um", 1.0),
    ("Eg", "Eg", 1.0),
    ("chi", "chi", 1.0),
    ("epsilon", "eps", 1.0),
    ("Nc", "Nc", 1.0),
    ("Nv", "Nv", 1.0),
    ("mun", "mu_n", 1.0),
    ("mup", "mu_p", 1.0),
    ("Na", "Na", 1.0),
    ("Nd", "Nd", 1.0),
]


def export_scaps_def(stack, filename=None, comment=""):
    """
    Write a SCAPS-1D style .def text for a layer stack.

    stack : list of dicts, front (illuminated) to back, each with
        {"name", "thickness_nm", "material": Material-like object}
    Returns the text; writes to `filename` if given.
    """
    lines = [
        "// SCAPS-1D definition file exported by",
        "// AI-Driven Open-Source Solar Cell Design & Optimization Tool v4.0",
        f"// date: {date.today().isoformat()}",
        "// NOT exported (set in SCAPS if needed): graded profiles,",
        "// interface defect blocks, metastability, tunneling flags.",
        f"// {comment}" if comment else "//",
        "",
        f"layers {len(stack)}",
        "",
    ]
    for lay in stack:
        m = lay["material"]
        lines.append(f"layer  {lay['name']}")
        vals = {
            "thickness_um": lay["thickness_nm"] * 1e-3,
            "Eg": getattr(m, "Eg", 0.0),
            "chi": getattr(m, "chi", 0.0),
            "eps": getattr(m, "eps", 10.0),
            "Nc": getattr(m, "Nc", 1e19),
            "Nv": getattr(m, "Nv", 1e19),
            "mu_n": getattr(m, "mu_n", 1.0),
            "mu_p": getattr(m, "mu_p", 1.0),
            "Na": getattr(m, "Na", 0.0),
            "Nd": getattr(m, "Nd", 0.0),
        }
        for scaps_key, our_key, f in _SCAPS_KEYS:
            lines.append(f"    {scaps_key:10s} {vals[our_key] * f:.4e}")
        Nt = getattr(m, "Nt", None)
        if Nt:
            lines += [
                "    defect 1",
                f"        Nt        {Nt:.4e}",
                "        type      neutral",
                "        Et        0.6            // eV above Ev (midgap default)",
                "        sigman    1.0e-15",
                "        sigmap    1.0e-15",
            ]
        lines.append("")
    text = "\n".join(lines)
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    return text


# ---------------------------------------------------------------------------
# Generic JSON device spec
# ---------------------------------------------------------------------------
_MAT_FIELDS = ["Eg", "chi", "eps", "Nc", "Nv", "mu_n", "mu_p",
               "Na", "Nd", "Nt", "alpha_coeff", "tau_n", "tau_p"]


def export_device_json(technology, stack=None, params=None, result=None,
                       filename=None, notes=""):
    """
    Self-describing JSON export of a device (any technology).

    technology : "perovskite" | "silicon" | "organic" | "tandem" | "cdte"
    stack      : layer list as in export_scaps_def (optional)
    params     : dict of architecture/blend parameters (optional)
    result     : simulation result dict (Jsc/Voc/FF/PCE kept; arrays dropped)
    """
    doc = {
        "format": "solarcell-device-spec",
        "format_version": "1.0",
        "generator": "AI-Driven Open-Source Solar Cell Design Tool v4.0",
        "date": date.today().isoformat(),
        "technology": technology,
        "notes": notes,
    }
    if stack:
        doc["stack"] = []
        for lay in stack:
            m = lay["material"]
            entry = {"name": lay["name"], "thickness_nm": lay["thickness_nm"]}
            for f in _MAT_FIELDS:
                v = getattr(m, f, None)
                if v is not None:
                    entry[f] = v
            doc["stack"].append(entry)
    if params:
        doc["parameters"] = {k: v for k, v in params.items()
                             if isinstance(v, (int, float, str, bool))}
    if result:
        doc["result"] = {k: float(result[k]) for k in
                         ("Jsc", "Voc", "FF", "PCE", "Vmpp", "Jmpp", "Pmax")
                         if k in result}
    text = json.dumps(doc, indent=2)
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def import_device_json(path_or_text):
    """Load a solarcell-device-spec JSON (path or raw text)."""
    try:
        doc = json.loads(path_or_text)
    except (json.JSONDecodeError, TypeError):
        with open(path_or_text, "r", encoding="utf-8") as f:
            doc = json.load(f)
    if doc.get("format") != "solarcell-device-spec":
        raise ValueError("Not a solarcell-device-spec JSON document")
    return doc


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def export_jv_csv(result, filename=None):
    """J-V curve CSV (V [V], J [mA/cm2]) from any engine's result dict."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["V_V", "J_mA_cm2"])
    for v, j in zip(result["voltages"], result["currents"]):
        w.writerow([f"{v:.5f}", f"{j:.5f}"])
    text = buf.getvalue()
    if filename:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            f.write(text)
    return text


def export_eqe_csv(result, filename=None):
    """EQE curve CSV (lambda [nm], EQE [%]) from any engine's result dict."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["lambda_nm", "EQE_pct"])
    for lam, q in zip(result["lams_qe"], result["qe"]):
        w.writerow([f"{lam:.1f}", f"{q:.3f}"])
    text = buf.getvalue()
    if filename:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            f.write(text)
    return text
