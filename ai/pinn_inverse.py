"""
ai/pinn_inverse.py — inverse parameter prediction THROUGH the conditional
PINN (v3.1, Phase 1).

The PhD synopsis (Fig. 3) specifies a physics-informed neural network used
for INVERSE parameter prediction: desired/observed device behaviour in,
device parameters out. Until v3.0 the tool's inverse design ran differential
evolution over a surrogate — a metaheuristic, not the PINN route the synopsis
describes. This module closes that gap.

Method
------
The trained conditional PINN (ai/conditional_pinn_torch.py) is a smooth,
differentiable map  f: (x, theta) -> (psi, log n, log p)  with
theta = (d_abs, Nt) normalized to [-1, 1]^2. Given an OBSERVED solution
profile y*(x) — e.g. the electrostatic potential and carrier density of a
device whose parameters are unknown — the parameters are recovered by
gradient descent ON THE INPUTS through the frozen network:

    theta* = argmin_theta  || f(x, theta) - y*(x) ||^2
                            + barrier(theta outside [-1,1]^2)

using autograd for d L / d theta and multi-start Adam to avoid local minima.
This is classical PINN-based parameter identification (the inverse-problem
mode of Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 378, 686 (2019),
Sec. 4), applied to the drift-diffusion system.

Validation protocol (unit-tested): solve an UNSEEN device with the full
drift-diffusion solver, hand only the profiles to the inverter, and check
the recovered (d_abs, Nt) against the ground truth.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    _TORCH = True
except Exception:                                   # pragma: no cover
    _TORCH = False


def invert_parameters(model, fam, x_norm, psi_obs, logn_obs,
                      n_starts=6, n_steps=400, lr=0.05, seed=0,
                      w_logn=0.02, verbose=False):
    """Recover theta = (d_abs, Nt) from observed psi(x) and log n(x).

    Parameters
    ----------
    model : trained ConditionalPINN (weights frozen here)
    fam : DeviceFamily (for theta de-normalization)
    x_norm : (N,) positions normalized to [0, 1]
    psi_obs : (N,) observed potential, mean-subtracted [V]
    logn_obs : (N,) observed ln n [ln cm^-3]

    Returns dict with the best theta, the de-normalized (d_abs_nm, Nt),
    per-start results, and the final data misfit.
    """
    if not _TORCH:
        raise ImportError("PyTorch required for PINN inversion")
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    X = torch.tensor(np.asarray(x_norm).reshape(-1, 1), dtype=torch.float32)
    Yp = torch.tensor(np.asarray(psi_obs, dtype=np.float32))
    Yn = torch.tensor(np.asarray(logn_obs, dtype=np.float32))

    starts = torch.rand(n_starts, 2) * 1.6 - 0.8      # interior starts
    results = []
    for s in range(n_starts):
        theta = starts[s].clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            th = theta.unsqueeze(0).expand(X.shape[0], 2)
            psi, ln, _ = model(X, th)
            psi = psi.squeeze() - psi.mean()          # match mean-free obs
            loss = ((psi - Yp) ** 2).mean() + w_logn * ((ln.squeeze() - Yn) ** 2).mean()
            # smooth barrier keeping theta in the trained box
            loss = loss + 10.0 * (torch.relu(theta.abs() - 1.0) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            th_f = theta.clamp(-1.0, 1.0)
            results.append((float(loss.detach()), th_f.numpy().copy()))
    results.sort(key=lambda t: t[0])
    best_loss, best_theta = results[0]
    d_abs, Nt = fam.untheta(best_theta)
    return {
        "theta": best_theta.tolist(),
        "d_abs_nm": float(d_abs),
        "Nt_cm3": float(Nt),
        "misfit": float(best_loss),
        "all_starts": [{"loss": l, "theta": t.tolist()} for l, t in results],
        "method": "gradient descent on PINN inputs (Raissi 2019 inverse mode), "
                  f"{n_starts} starts x {n_steps} Adam steps",
    }


def invert_from_dd_device(model, fam, d_abs_true_nm, Nt_true, **kw):
    """End-to-end validation helper: run the FULL drift-diffusion solver on
    a device with known parameters, hand only the resulting profiles to the
    inverter, and report recovered-vs-true parameters."""
    mesh, r = fam.dd_solution(d_abs_true_nm, Nt_true)
    L = fam.L_total_cm(d_abs_true_nm)
    xn = mesh.x / L
    psi = r.psi - r.psi.mean()
    logn = np.log(np.clip(r.n, 1e5, None))
    out = invert_parameters(model, fam, xn, psi, logn, **kw)
    out["true_d_abs_nm"] = float(d_abs_true_nm)
    out["true_Nt_cm3"] = float(Nt_true)
    out["err_d_abs_pct"] = abs(out["d_abs_nm"] - d_abs_true_nm) / d_abs_true_nm * 100
    out["err_logNt_abs"] = abs(np.log10(out["Nt_cm3"]) - np.log10(Nt_true))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# IDENTIFIABILITY FINDING (quantified; see tests/test_v31_phases.py):
# The short-circuit (V = 0) potential and carrier profiles are INSENSITIVE to
# the trap density — the ground-truth drift-diffusion profiles change by
# < 1e-4 rel-L2 across Nt = 1e13..1e15 cm^-3, i.e. ~1600x below the PINN's
# own 7% interpolation floor, because carrier extraction dominates SRH
# recombination at short circuit for these lifetimes. Consequently the
# profile-only inversion above recovers GEOMETRY (thickness, 3-6% error)
# but cannot recover recombination parameters — no algorithm could, from
# that observable. Recombination is identifiable from J-V observables (Voc),
# which is what the two-stage inversion below uses. This mirrors standard
# practice in device characterization: geometry from profiling techniques,
# lifetimes from Voc / transient measurements.
# ─────────────────────────────────────────────────────────────────────────────
def invert_with_voc(model, fam, x_norm, psi_obs, logn_obs, Voc_obs,
                    d_htl_nm=150.0, d_etl_nm=50.0, **kw):
    """Two-stage inverse parameter prediction.

    Stage 1: thickness d_abs by gradient descent through the frozen PINN on
             the observed profiles (this observable identifies geometry).
    Stage 2: trap density Nt by a 1-D golden-section match of the observed
             open-circuit voltage using the physics model at the recovered
             thickness (Voc identifies recombination).

    Returns dict with both recovered parameters.
    """
    from physics.device import fast_simulate
    st1 = invert_parameters(model, fam, x_norm, psi_obs, logn_obs, **kw)
    d_rec = st1["d_abs_nm"]

    def voc_err(logNt):
        r = fast_simulate(fam.htl, fam.abs, fam.etl,
                          d_htl_nm, d_rec, d_etl_nm, 10.0 ** logNt, 300)
        return abs(r["Voc"] - Voc_obs)

    lo, hi = fam.n_lo, fam.n_hi
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = lo, hi
    c_ = b_ - gr * (b_ - a_); d_ = a_ + gr * (b_ - a_)
    fc, fd = voc_err(c_), voc_err(d_)
    for _ in range(40):
        if fc < fd:
            b_, d_, fd = d_, c_, fc
            c_ = b_ - gr * (b_ - a_); fc = voc_err(c_)
        else:
            a_, c_, fc = c_, d_, fd
            d_ = a_ + gr * (b_ - a_); fd = voc_err(d_)
    logNt_rec = (a_ + b_) / 2.0
    return {"d_abs_nm": d_rec, "Nt_cm3": float(10 ** logNt_rec),
            "stage1": st1, "voc_residual_V": float(voc_err(logNt_rec)),
            "method": "two-stage: PINN-gradient geometry + Voc golden-section "
                      "for recombination (identifiability-driven split)"}
