"""
scripts/run_multi_tech_validation.py — final-release validation suite
============================================================
Runs every silicon, organic and tandem benchmark defined in
data/multi_technology_database.json against the corresponding engine
and writes artifacts/validation_multi_tech.json plus a console summary.

Every target is a certified, peer-reviewed device (references with DOIs
in the same JSON file). Pass criterion: |PCE_model - PCE_published| /
PCE_published <= tolerance_PCE_pct.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DB = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))


def _entry(bench, result, target_pce, tol):
    err = abs(result["PCE"] - target_pce) / target_pce * 100.0
    return {
        "id": bench["id"],
        "reference": bench["reference"],
        "doi": DB["references"][bench["reference"]]["doi"],
        "target_PCE": target_pce,
        "model": {k: round(float(result[k]), 4)
                  for k in ("PCE", "Voc", "Jsc", "FF")},
        "error_PCE_pct": round(err, 2),
        "tolerance_PCE_pct": tol,
        "pass": bool(err <= tol),
    }


def run():
    rows = []

    # ---------- silicon ----------
    from physics.silicon import SILICON_PRESETS, simulate_silicon
    for b in DB["benchmarks"]["silicon"]:
        arch = SILICON_PRESETS[b["preset"]]
        r = simulate_silicon(arch)
        rows.append(_entry(b, r, b["target"]["PCE"], b["tolerance_PCE_pct"]))

    # ---------- organic ----------
    from physics.organic import ORGANIC_PRESETS, simulate_organic
    for b in DB["benchmarks"]["organic"]:
        blend = ORGANIC_PRESETS[b["preset"]]
        if "override" in b:
            blend = dataclasses.replace(blend, **b["override"])
        r = simulate_organic(blend)
        rows.append(_entry(b, r, b["target"]["PCE"], b["tolerance_PCE_pct"]))

    # ---------- tandem ----------
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.silicon import SILICON_PRESETS as SP
    from physics.tandem import simulate_perovskite_silicon_tandem
    htl = HTL_DB["2PACz"]
    etl = ETL_DB.get("C60", ETL_DB.get("PCBM"))
    wg = PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"]
    si = SP["SHJ-IBC (Kaneka 26.7%)"]
    for b in DB["benchmarks"]["tandem"]:
        p = b["params"]
        if "d_top_nm" in p:
            r = simulate_perovskite_silicon_tandem(
                htl, wg, etl, p["d_top_nm"], si, Nt_top=p["Nt_top"],
                R_int=p["R_int"], parasitic=p["parasitic"],
                Rs_int_ohm_cm2=p["Rs_int_ohm_cm2"])
        else:
            best = None
            for d in np.linspace(400, 1100, 8):
                r_ = simulate_perovskite_silicon_tandem(
                    htl, wg, etl, float(d), si, Nt_top=p["Nt_top"],
                    R_int=p["R_int"], parasitic=p["parasitic"],
                    Rs_int_ohm_cm2=p["Rs_int_ohm_cm2"])
                if best is None or r_["PCE"] > best["PCE"]:
                    best = r_
            r = best
        if "PCE_range" in b["target"]:
            lo, hi = b["target"]["PCE_range"]
            mid = 0.5 * (lo + hi)
            err = (0.0 if lo <= r["PCE"] <= hi
                   else min(abs(r["PCE"] - lo), abs(r["PCE"] - hi)) / mid * 100)
            rows.append({
                "id": b["id"], "reference": b["reference"],
                "doi": DB["references"][b["reference"]]["doi"],
                "target_PCE_range": [lo, hi],
                "model": {k: round(float(r[k]), 4)
                          for k in ("PCE", "Voc", "Jsc", "FF")},
                "error_PCE_pct": round(err, 2),
                "tolerance_PCE_pct": b["tolerance_PCE_pct"],
                "pass": bool(err <= b["tolerance_PCE_pct"]),
            })
        else:
            rows.append(_entry(b, r, b["target"]["PCE"],
                               b["tolerance_PCE_pct"]))

    n_pass = sum(1 for r in rows if r["pass"])
    errs = [r["error_PCE_pct"] for r in rows]
    summary = {
        "suite": "multi-technology validation v4.0",
        "n_benchmarks": len(rows),
        "n_pass": n_pass,
        "mean_PCE_error_pct": round(float(np.mean(errs)), 2),
        "median_PCE_error_pct": round(float(np.median(errs)), 2),
        "max_PCE_error_pct": round(float(np.max(errs)), 2),
        "results": rows,
    }
    out = os.path.join(ROOT, "artifacts", "validation_multi_tech.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Multi-technology validation: {n_pass}/{len(rows)} pass ===")
    print(f"mean |PCE error| = {summary['mean_PCE_error_pct']}%, "
          f"median = {summary['median_PCE_error_pct']}%, "
          f"max = {summary['max_PCE_error_pct']}%\n")
    for r in rows:
        tgt = r.get("target_PCE", r.get("target_PCE_range"))
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['id']:28s} "
              f"target={tgt} model={r['model']['PCE']:.2f}% "
              f"err={r['error_PCE_pct']}%")
    return summary


if __name__ == "__main__":
    run()
