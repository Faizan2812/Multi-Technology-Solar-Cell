"""
physics/cdte.py — CdTe technology family (v3.1, Phase 1)
=========================================================

Extends the tool beyond perovskites with the SECOND validated technology,
closing the title-scope gap ("Solar Cells", not "Perovskite Solar Cells")
identified against the PhD synopsis. The drift-diffusion core is unchanged —
this module supplies the CdTe device recipe, its optical front-stack model,
and the published-reference benchmark.

Device: superstrate  glass / SnO2(500 nm, optical) / CdS(25 nm) /
        CdTe(4000 nm) / p+ CdTe back-surface layer(100 nm) / metal
Parameter lineage: Gloeckler, Fahrenbruch & Sites, Proc. 3rd WCPEC 1,
491-494 (2003) — the community-baseline set (see materials_database.json
entries CdTe / CdS / CdTe-BSF for per-parameter provenance).

Calibration (documented, within published ranges):
  * absorber/BSF capture cross-section sigma = 5e-14 cm^2 (published spread
    1e-15..1e-12; with the baseline Nt = 2e14 cm^-3 this gives
    tau_SRH = 10 ns, consistent with Voc ~ 0.86-0.88 V devices)
  * CdS Tauc prefactor 4e5 cm^-1-eV^0.5 so that above-gap alpha ~ 1e5 cm^-1
    (literature CdS absorption; the sqrt-Tauc form otherwise underestimates
    the blue loss of the 25 nm window)
  * front optical budget R_front = 0.21 — total glass reflection + TCO
    free-carrier/UV absorption + interface reflections (typical CdTe front
    stacks lose 15-20% of above-gap photons; SnO2 UV filtering is modeled
    explicitly on top via the window chain)
  * Rs = 2 Ohm*cm^2 — standard TCO sheet + contact resistance assumption

Benchmark anchors (published, independently sourced):
  C1  in-sample : standard 4000-nm baseline replication —
      PCE 16.41%, Voc 0.87 V, Jsc 24.72 mA/cm2, FF 76.21%
      [reported in the literature replicating the Gloeckler baseline;
       see docs/REGENERATION_REPORT_v3.md Phase-1 notes]
  C2  out-of-sample sanity band : independent peer-reviewed SnO2-class/CdS/
      CdTe SCAPS-1D reference at 17.43% with CdS buffer (Zyoud et al.,
      Crystals 11, 1454 (2021), DOI 10.3390/cryst11121454). Our baseline
      must fall in [15.5, 18.0]%.
"""
from __future__ import annotations
import copy
import numpy as np


class _SnO2Optical:
    """Optical-only front TCO (electrically it is the contact)."""
    Eg = 3.6
    alpha_coeff = 1e5


# calibrated technology constants (see module docstring for justification)
SIGMA_CDTE = 5e-14        # cm^2
CDS_ALPHA0 = 4e5          # Tauc prefactor
R_FRONT = 0.21            # total front optical loss budget
RS_OHM_CM2 = 2.0          # series resistance
D_SNO2_NM = 500.0
D_CDS_NM = 25.0
D_CDTE_NM = 4000.0
D_BSF_NM = 100.0

BENCHMARK_C1 = {"PCE": 16.41, "Voc": 0.870, "Jsc": 24.72, "FF": 0.7621,
                "source": "Gloeckler-2003 baseline replication (4000 nm CdTe)"}
BENCHMARK_C2_BAND = (15.5, 18.0)   # % PCE, Zyoud et al. 2021 (Crystals, DOI 10.3390/cryst11121454)


def cdte_materials():
    """Return (bsf, cdte, cds) Material objects with the calibrated
    cross-sections/absorption applied (base values from the provenanced
    database; calibration documented above)."""
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    h = copy.deepcopy(HTL_DB["CdTe-BSF"])
    a = copy.deepcopy(PEROVSKITE_DB["CdTe"])
    e = copy.deepcopy(ETL_DB["CdS"])
    a.sigma_e = a.sigma_h = SIGMA_CDTE
    h.sigma_e = h.sigma_h = SIGMA_CDTE
    e.alpha_coeff = CDS_ALPHA0
    return h, a, e


def simulate_cdte(d_cdte_nm=D_CDTE_NM, d_cds_nm=D_CDS_NM, Nt_abs=None,
                  T=300.0, N_V=25, Rs=RS_OHM_CM2, Rsh=1e12,
                  sigma_abs=None):
    """Full drift-diffusion J-V of the CdTe stack.

    Returns the same result dict shape as device.simulate_iv_curve (dd mode).
    """
    from physics.dd_solver import build_mesh, jv_sweep, extract_device_metrics
    from physics.device import _dd_beer_lambert_generation
    h, a, e = cdte_materials()
    if Nt_abs is not None:
        a.Nt = Nt_abs
    if sigma_abs is not None:
        a.sigma_e = a.sigma_h = sigma_abs
    npl = [15, max(60, int(d_cdte_nm / 33)), 12]
    mesh = build_mesh([h, a, e], [D_BSF_NM, d_cdte_nm, d_cds_nm],
                      N_per_layer=npl, T=T)
    G = _dd_beer_lambert_generation(
        mesh, a, light_side="etl",
        window_filter=[(_SnO2Optical(), D_SNO2_NM), (e, d_cds_nm)],
        front_reflectance=R_FRONT)
    V_arr, J_arr, conv = jv_sweep(mesh, G, h, e, V_min=0.0, V_max=0.98,
                                  N_V=N_V, T=T, Rs=Rs, Rsh=Rsh)
    m = extract_device_metrics(V_arr, J_arr, converged_flags=conv)
    return {
        "voltages": V_arr, "currents": J_arr * 1000.0, "converged": conv,
        "PCE": m["PCE"], "Voc": m["Voc"], "Jsc": m["Jsc"], "FF": m["FF"],
        "technology": "CdTe",
        "stack": "glass/SnO2/CdS/CdTe/p+CdTe/metal",
        "solver": "drift-diffusion (Scharfetter-Gummel), superstrate optics, "
                  f"R_front={R_FRONT}, Rs={Rs} Ohm.cm2",
    }


def run_cdte_benchmark(verbose=True):
    """Benchmark the calibrated baseline against the published anchors."""
    r = simulate_cdte()
    c1 = BENCHMARK_C1
    errs = {
        "PCE": abs(r["PCE"] - c1["PCE"]) / c1["PCE"] * 100,
        "Voc": abs(r["Voc"] - c1["Voc"]) / c1["Voc"] * 100,
        "Jsc": abs(r["Jsc"] - c1["Jsc"]) / c1["Jsc"] * 100,
        "FF":  abs(r["FF"] - c1["FF"]) / c1["FF"] * 100,
    }
    in_band = BENCHMARK_C2_BAND[0] <= r["PCE"] <= BENCHMARK_C2_BAND[1]
    out = {"ours": {k: r[k] for k in ("PCE", "Voc", "Jsc", "FF")},
           "C1_published": c1, "errors_pct": errs,
           "C2_band": BENCHMARK_C2_BAND, "C2_pass": bool(in_band),
           "mean_error_pct": float(np.mean(list(errs.values())))}
    if verbose:
        print("CdTe benchmark (C1 baseline replication):")
        for k in ("PCE", "Voc", "Jsc", "FF"):
            print(f"  {k}: ours {r[k]:.3f}  vs published {c1[k]:.3f}"
                  f"  ({errs[k]:.1f}%)")
        print(f"  C2 out-of-sample band {BENCHMARK_C2_BAND}: "
              f"{'PASS' if in_band else 'FAIL'} (ours {r['PCE']:.2f}%)")
    return out


if __name__ == "__main__":
    run_cdte_benchmark()
