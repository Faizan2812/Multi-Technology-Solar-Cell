"""
scripts/run_cross_tool_validation.py — Final Release cross-tool suite
==================================================================
Verifies this tool against the three classes of external simulators a
reviewer will ask about, writing artifacts/validation_cross_tool.json:

C1  Lumerical-class optical validation (physics/tmm.py).
    For planar stacks, Lumerical STACK / planar-limit FDTD solve the
    same coherent Maxwell problem as the transfer-matrix method; the
    published OPV optical literature is TMM-based (Pettersson 1999;
    Sievers 2006; Kotlarski 2008). Checks:
      C1a  Energy conservation R + T + sum A_j = 1 (numerical exactness)
      C1b  Analytic Fresnel limit for a bare glass/air interface
      C1c  P3HT:PCBM Jsc(L) interference: first max 60-95 nm, minimum
           110-155 nm, second max 190-260 nm (Sievers 2006 / Kotlarski
           2008 / Monestier 2007), amplitude 8-12 mA/cm2 at IQE 0.8
      C1d  NFA blend optimum near ~100 nm (Im 2023)
      C1e  PM6:Y6 absolute TMM Jsc vs certified device (Yuan 2019)
           within 8% at IQE 0.90 (published TMM-vs-experiment band is
           ~16% on average, Rosa 2021)
      C1f  Independent-path consistency: TMM-optics PCE vs
           calibrated-optics PCE for PM6:Y6 within 8%

C2  SCAPS-1D interoperability + numerical cross-check.
    C2a  .def export -> re-import round trip (parameter-exact)
    C2b  the existing 10-reference published-SCAPS benchmark suite
         (artifacts/benchmark_results.json, v3): summary re-asserted

C3  Internal cross-engine consistency: the same physical device family
    simulated through independent code paths must agree (TMM organic
    vs calibrated organic; silicon Richter vs Niewelt models).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
checks = []


def check(cid, desc, passed, detail):
    checks.append({"id": cid, "description": desc,
                   "pass": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {cid:6s} {desc} — {detail}")


def run():
    print("\n=== C1: Lumerical-class optical validation (TMM) ===")
    from physics.tmm import (energy_conservation_error, tmm_single,
                             jsc_vs_thickness, jsc_from_stack)
    from scipy.signal import argrelextrema

    stk_p3ht = [("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 90), ("Al", 100)]
    stk_y6 = [("ITO", 100), ("PEDOT:PSS", 40), ("PM6:Y6", 100), ("Ag", 100)]

    e1 = energy_conservation_error(stk_p3ht)
    e2 = energy_conservation_error(stk_y6)
    check("C1a", "energy conservation R+T+ΣA=1",
          max(e1, e2) < 5e-4, f"max err {max(e1, e2):.2e}")

    out = tmm_single(550.0, [np.complex128(1.52), np.complex128(1.0)], [])
    fres = ((1.52 - 1) / (1.52 + 1)) ** 2
    check("C1b", "Fresnel limit glass/air",
          abs(out["R"] - fres) < 1e-12,
          f"R={out['R']:.8f} vs analytic {fres:.8f}")

    ds, js = jsc_vs_thickness(stk_p3ht, "P3HT:PCBM",
                              np.arange(30, 305, 5), IQE=0.8)
    mx = argrelextrema(js, np.greater, order=3)[0]
    mn = argrelextrema(js, np.less, order=3)[0]
    ok = (len(mx) >= 2 and 60 <= ds[mx[0]] <= 95
          and 190 <= ds[mx[1]] <= 260
          and len(mn) >= 1 and 110 <= ds[mn[0]] <= 155
          and 8.0 <= js[mx[1]] <= 12.0)
    check("C1c", "P3HT:PCBM Jsc(L) interference vs published TMM", ok,
          f"maxima {ds[mx[:2]].tolist()} nm at "
          f"{np.round(js[mx[:2]], 2).tolist()} mA/cm², min {ds[mn[0]]} nm "
          "(Sievers 2006 / Kotlarski 2008 / Monestier 2007)")

    ds2, js2 = jsc_vs_thickness(stk_y6, "PM6:Y6",
                                np.arange(60, 305, 5), IQE=0.9)
    d_opt = float(ds2[int(np.argmax(js2[:20]))])
    check("C1d", "NFA blend first optical optimum near ~100 nm",
          80 <= d_opt <= 135, f"first-region optimum {d_opt:.0f} nm (Im 2023)")

    J_y6, _ = jsc_from_stack(stk_y6, "PM6:Y6", IQE=0.90)
    dev = abs(J_y6 - 25.3) / 25.3 * 100
    check("C1e", "PM6:Y6 TMM Jsc vs certified device (Yuan 2019)",
          dev <= 8.0, f"{J_y6:.2f} vs 25.3 mA/cm² ({dev:.1f}% dev; "
          "published TMM band ~16%, Rosa 2021)")

    from physics.organic import ORGANIC_PRESETS, simulate_organic
    b = ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"]
    p_cal = simulate_organic(b)["PCE"]
    p_tmm = simulate_organic(b, optics="tmm")["PCE"]
    dev2 = abs(p_tmm - p_cal) / p_cal * 100
    check("C1f", "independent-path PCE: TMM optics vs calibrated optics",
          dev2 <= 8.0, f"{p_tmm:.2f}% vs {p_cal:.2f}% ({dev2:.1f}% dev)")

    print("\n=== C2: SCAPS-1D interoperability ===")
    from physics.materials import PEROVSKITE_DB, ETL_DB, HTL_DB
    from utils.interop import export_scaps_def
    from utils.scaps_import import parse_def
    stack = [{"name": "ETL", "thickness_nm": 50, "material": ETL_DB["TiO2"]},
             {"name": "absorber", "thickness_nm": 500,
              "material": PEROVSKITE_DB["MAPbI3"]},
             {"name": "HTL", "thickness_nm": 150,
              "material": HTL_DB["Spiro-OMeTAD"]}]
    parsed = parse_def(export_scaps_def(stack))
    ok = (len(parsed["layers"]) == 3
          and abs(parsed["layers"][1].Eg - PEROVSKITE_DB["MAPbI3"].Eg) < 1e-6
          and abs(parsed["layers"][1].chi
                  - PEROVSKITE_DB["MAPbI3"].chi) < 1e-6)
    check("C2a", ".def export → re-import round trip", ok,
          "layer count + Eg + χ parameter-exact")

    bench_path = os.path.join(ROOT, "artifacts", "benchmark_results.json")
    detail = "artifact missing"
    ok = False
    if os.path.exists(bench_path):
        bres = json.load(open(bench_path))
        sr = bres.get("scaps_reference", {}).get("summary", {})
        mean_err = sr.get("mean_pce_error_pct")
        if mean_err is not None:
            ok = float(mean_err) <= 15.0
            detail = (f"published-SCAPS suite: {sr.get('n_devices', '?')} devices, "
                      f"mean PCE error {float(mean_err):.1f}%, worst "
                      f"{sr.get('worst_pce_error_pct', '?')}% (v3 artifact re-asserted)")
    check("C2b", "published-SCAPS numerical cross-check (v3 suite)", ok, detail)

    print("\n=== C3: internal cross-engine consistency ===")
    from physics.silicon import SILICON_PRESETS, simulate_silicon
    arch = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]
    p_r = simulate_silicon(arch, auger_model="richter2012")["PCE"]
    p_n = simulate_silicon(arch, auger_model="niewelt2022")["PCE"]
    dev3 = abs(p_r - p_n) / p_r * 100
    check("C3a", "silicon: Richter-2012 vs Niewelt-2022 intrinsic models",
          dev3 <= 3.0, f"{p_r:.2f}% vs {p_n:.2f}% ({dev3:.2f}% dev)")

    n_pass = sum(1 for c in checks if c["pass"])
    summary = {"suite": "cross-tool validation final",
               "n_checks": len(checks), "n_pass": n_pass, "checks": checks}
    out_path = os.path.join(ROOT, "artifacts", "validation_cross_tool.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Cross-tool validation: {n_pass}/{len(checks)} pass ===")
    return summary


if __name__ == "__main__":
    run()
