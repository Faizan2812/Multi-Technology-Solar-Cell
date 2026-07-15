"""
run_validation.py — numerical-integrity validation .

The v2.1 version verified second-order convergence of a TOY finite-difference
Poisson solver — a stand-in that never touched the production code. That was
the single biggest numerical-integrity flag. v3.0 validates the PRODUCTION
solver directly:

  1. Provenance audit of the material database (unchanged).
  2. MMS on the PRODUCTION Poisson solver: a manufactured solution
     psi_m(x) = A sin(pi x / L) is injected as an analytic source into
     `physics.dd_solver.solve_poisson_newton` (mms_source hook, zero in all
     physical runs) and the observed L2 convergence order is measured on the
     actual Newton/tridiagonal code path.
  3. Scharfetter-Gummel current check: in the zero-field, linear-density
     limit the SG edge current must reduce to the exact diffusion current
     J = q D dn/dx (machine-precision identity).
  4. Full-device mesh-convergence: PCE of benchmark device stacks at
     successively refined meshes; Richardson extrapolation gives the observed
     order and a discretization-error estimate for the tabulated PCE numbers.
  5. Uncertainty-calibration audit: empirical coverage of the split-conformal
     95% intervals (ai/uncertainty.py) on a physics-generated dataset.

Writes artifacts/validation_v3_0.json.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.provenance_audit import audit
from physics.dd_solver import (build_mesh, solve_poisson_newton, sg_currents,
                               DeviceMesh, K_B, Q)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MMS on the PRODUCTION Poisson solver
# ─────────────────────────────────────────────────────────────────────────────
def mms_production_poisson(Ns=(41, 81, 161, 321, 641)):
    """Manufactured solution on the production Newton-Poisson code.

    We construct a uniform-material mesh, freeze the carrier terms at
    negligible density (so the operator is the pure Poisson operator the
    Newton loop assembles), and inject the analytic source
        S(x) = -eps * psi_m''(x)
    via the mms_source hook. The solver then must reproduce psi_m to the
    discretization order of its OWN assembly (2nd order central)."""
    from physics.materials import PEROVSKITE_DB
    ab = PEROVSKITE_DB["MAPbI3"]
    L_nm = 600.0
    A = 0.5                                   # V
    errs = []
    for N in Ns:
        # uniform single-layer mesh via build_mesh, then override the grid to
        # be exactly uniform with N nodes (keeps the production arrays)
        mesh = build_mesh([ab, ab, ab], [L_nm / 3] * 3)
        x = np.linspace(0.0, L_nm * 1e-7, N)
        arrs = {}
        for f in ("eps", "chi", "Eg", "Nc", "Nv", "mu_n", "mu_p",
                  "Nd", "Na", "tau_n", "tau_p", "B_rad", "Cn", "Cp"):
            arrs[f] = np.full(N, getattr(mesh, f)[mesh.N // 2])
        arrs["Nd"][:] = 0.0; arrs["Na"][:] = 0.0     # no space charge
        m2 = DeviceMesh(x=x, layer=np.ones(N, int), interfaces=[], **arrs)
        L = x[-1]
        psi_m = A * np.sin(np.pi * x / L)
        eps = arrs["eps"][0]
        S = eps * A * (np.pi / L) ** 2 * np.sin(np.pi * x / L)   # = -eps psi_m''
        tiny = np.full(N, 1.0)                 # 1 cm^-3: carrier terms ~ 0
        bc = {"psi_left": 0.0, "psi_right": 0.0}
        psi, _, _ = solve_poisson_newton(m2, np.zeros(N), tiny, tiny, bc,
                                         max_iter=200, tol=1e-13,
                                         mms_source=S)
        errs.append(float(np.sqrt(np.trapezoid((psi - psi_m) ** 2, x) / L)))
    rate = float(-np.polyfit(np.log(Ns), np.log(errs), 1)[0])
    return {"N": list(Ns), "L2": errs, "convergence_rate": rate,
            "relL2_finest": errs[-1] / (A / np.sqrt(2.0)),
            "solver": "physics.dd_solver.solve_poisson_newton (production)"}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SG current diffusion-limit identity
# ─────────────────────────────────────────────────────────────────────────────
def sg_diffusion_limit():
    """Zero field, linear n(x): SG current must equal q*D*dn/dx exactly."""
    from physics.materials import PEROVSKITE_DB
    ab = PEROVSKITE_DB["MAPbI3"]
    mesh = build_mesh([ab, ab, ab], [200.0] * 3)
    N = mesh.N
    T = 300.0
    Vt = K_B * T / Q
    psi = np.zeros(N)
    n = 1e12 + (1e14 - 1e12) * (mesh.x - mesh.x[0]) / (mesh.x[-1] - mesh.x[0])
    p = np.full(N, 1e10)
    Jn, Jp = sg_currents(mesh, psi, n, p, T)
    dx = np.diff(mesh.x)
    D = mesh.mu_n[:-1] * Vt
    J_exact = Q * D * np.diff(n) / dx
    rel = float(np.max(np.abs(Jn - J_exact) / np.max(np.abs(J_exact))))
    return {"max_rel_error": rel, "pass": rel < 1e-10}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full-device mesh convergence (Richardson)
# ─────────────────────────────────────────────────────────────────────────────
def device_mesh_convergence(refinements=(1.0, 2.0, 4.0)):
    """PCE of a benchmark stack at refined meshes; observed order + error."""
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.dd_solver import jv_sweep, extract_device_metrics
    from physics.device import _dd_beer_lambert_generation
    h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
    pces, Ntot = [], []
    for rf in refinements:
        npl = [int(30 * rf), int(90 * rf), int(20 * rf)]
        mesh = build_mesh([h, a, e], [150, 500, 50], N_per_layer=npl)
        G = _dd_beer_lambert_generation(mesh, a)
        V, J, c = jv_sweep(mesh, G, h, e, V_min=0.0,
                           V_max=min(a.Eg * 0.80, 1.30), N_V=26)
        m = extract_device_metrics(V, J, converged_flags=c)
        pces.append(m["PCE"]); Ntot.append(mesh.N)
    out = {"N": Ntot, "PCE": pces}
    if len(pces) >= 3:
        e1, e2 = abs(pces[0] - pces[-1]), abs(pces[1] - pces[-1])
        if e2 > 0 and e1 > e2:
            r = (Ntot[1] - 1) / (Ntot[0] - 1)
            out["observed_order"] = float(np.log(e1 / e2) / np.log(r))
        out["discretization_error_estimate_abs_PCE"] = float(abs(pces[1] - pces[2]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Uncertainty-calibration audit
# ─────────────────────────────────────────────────────────────────────────────
def uq_calibration():
    from ai.uncertainty import coverage_validation
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import fast_simulate
    h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(180):
        d = rng.uniform(300, 800)
        lg = rng.uniform(13, 16)
        r = fast_simulate(h, a, e, 150, d, 50, 10 ** lg, 300)
        X.append([d / 800.0, (lg - 13) / 3.0]); y.append(r["PCE"])
    res = coverage_validation(np.array(X), np.array(y), alpha=0.05, n_repeats=10)
    res["pass"] = res["conformal_coverage"] >= 0.88
    return res


def main():
    out = {"version": "3.0"}
    db, findings = audit()
    errs = [f for f in findings if f["severity"] == "ERROR"]
    out["provenance"] = dict(
        refs=len(db["_references"]),
        params=sum(len(md.get("parameters", {})) for c in ("absorbers", "etls", "htls")
                   for md in db[c].values()),
        errors=len(errs), status="PASS" if not errs else "FAIL")

    print("VALIDATION v3.0 (production solver)")
    print(f"  [1] provenance: {out['provenance']['status']} "
          f"({out['provenance']['refs']} refs, {out['provenance']['params']} params)")

    out["mms_production"] = mms_production_poisson()
    print(f"  [2] MMS on production Poisson: order = "
          f"{out['mms_production']['convergence_rate']:.3f} (expect ~2), "
          f"rel L2 finest = {out['mms_production']['relL2_finest']:.2e}")

    out["sg_diffusion_limit"] = sg_diffusion_limit()
    print(f"  [3] SG diffusion-limit identity: max rel err = "
          f"{out['sg_diffusion_limit']['max_rel_error']:.2e} "
          f"({'PASS' if out['sg_diffusion_limit']['pass'] else 'FAIL'})")

    out["device_mesh_convergence"] = device_mesh_convergence()
    dmc = out["device_mesh_convergence"]
    print(f"  [4] device mesh convergence: N={dmc['N']} PCE={['%.3f' % p for p in dmc['PCE']]}"
          + (f", est. disc. error = {dmc.get('discretization_error_estimate_abs_PCE', 0):.3f} %abs"))

    out["uq_calibration"] = uq_calibration()
    u = out["uq_calibration"]
    print(f"  [5] conformal 95% coverage = {u['conformal_coverage']:.3f} "
          f"(naive bootstrap = {u['bootstrap_coverage']:.3f}) "
          f"({'PASS' if u['pass'] else 'FAIL'})")

    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open("artifacts/validation_v3_0.json", "w"), indent=2,
              default=float)
    print("  -> artifacts/validation_v3_0.json")
    return out


if __name__ == "__main__":
    main()
