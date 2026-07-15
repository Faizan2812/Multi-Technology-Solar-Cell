"""
ai/conditional_pinn_torch.py — PRODUCTION conditional PINN 
==================================================================

Replaces the demonstrative numpy/finite-difference `ai/conditional_pinn.py`
(kept for the record) with a real conditional physics-informed neural network
in the Raissi 2019 sense, closing the "Conditional PINN: not implemented"
flag of the honest audit AND the "uniform G0 generation" flag:

    Network:  (x_norm, theta) -> (psi, log n, log p)
        theta = (d_abs_norm, logNt_norm) — device parameters as CONDITIONAL
        inputs, so ONE network generalizes across a device family instead of
        one network per device.
    Loss    = L_data (production drift-diffusion solutions at anchor devices)
            + L_poisson + L_continuity  (autograd residuals, evaluated at
              RANDOM (x, theta) collocation — physics constrains the
              interpolation BETWEEN devices, which is the entire point)
            + L_bc (contact Dirichlet values from the same DD solutions)

    Generation is the POSITION-DEPENDENT Beer-Lambert AM1.5G profile
        G(x) = sum_j  Phi_j alpha_j exp(-alpha_j x) dlambda,
    evaluated in torch from precomputed spectral coefficients — identical to
    the profile used by the drift-diffusion training data (single source of
    truth), not a uniform G0.

Anchor devices are solved with `physics.dd_solver` (the validated production
solver), NOT the fast analytical surrogate: the PINN learns and is judged
against first-principles solutions.

Validation protocol (scripts/train_conditional_pinn.py):
    train on the 4 corner devices of the (d_abs, Nt) box, evaluate on the
    UNSEEN center device — a genuine interpolation test with relative-L2
    metrics on psi(x) and log n(x).
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                    # pragma: no cover
    _TORCH = False

Q_E = 1.602176634e-19
K_B = 1.380649e-23
EPS0 = 8.854187817e-14


def _require_torch():
    if not _TORCH:
        raise ImportError(
            "PyTorch is required for the production conditional PINN "
            "(pip install torch). The numpy demo lives in ai/conditional_pinn.py.")


# ─────────────────────────────────────────────────────────────────────────────
# Spectral coefficients for the differentiable Beer-Lambert G(x)
# ─────────────────────────────────────────────────────────────────────────────
def spectral_coefficients(abs_mat, lam_min=280.0):
    """Precompute (w_j, alpha_j) with  G(x) = sum_j w_j exp(-alpha_j x)."""
    from physics.spectrum import photon_flux, AM15G_WAVELENGTHS, HC_EV_NM
    Eg = float(abs_mat.Eg)
    alpha0 = float(getattr(abs_mat, "alpha_coeff", 1e5) or 1e5)
    lam_edge = HC_EV_NM / Eg
    m = (AM15G_WAVELENGTHS <= lam_edge) & (AM15G_WAVELENGTHS >= lam_min)
    lams = AM15G_WAVELENGTHS[m]
    if lams.size < 2:
        return np.zeros(1), np.zeros(1)
    dlam = float(lams[1] - lams[0])
    phi = photon_flux(lams)
    E = HC_EV_NM / lams
    alphas = alpha0 * np.sqrt(np.maximum(E - Eg, 0.0)) / E
    good = alphas > 0
    return (phi[good] * alphas[good] * dlam).astype(np.float64), alphas[good].astype(np.float64)


class ConditionalPINN(nn.Module if _TORCH else object):
    """(x_norm, theta) -> (psi [V], log n, log p [ln cm^-3])."""

    N_FOURIER = 24
    FOURIER_SCALE = 6.0

    def __init__(self, n_theta=2, hidden=64, depth=4, seed=0):
        _require_torch()
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(1, self.N_FOURIER, generator=g) * self.FOURIER_SCALE
        self.register_buffer("B", B)
        in_dim = 2 * self.N_FOURIER + 1 + n_theta      # fourier(x), x, theta
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, theta):
        z = 2.0 * np.pi * x @ self.B
        feats = torch.cat([torch.sin(z), torch.cos(z), x, theta], dim=1)
        out = self.net(feats)
        psi = out[:, 0:1]
        log_n = 24.0 + 22.0 * torch.tanh(out[:, 1:2])   # ln n in [2, 46] ~ 7..1e20 cm^-3
        log_p = 24.0 + 22.0 * torch.tanh(out[:, 2:3])
        return psi, log_n, log_p


class DeviceFamily:
    """Parameter box (d_abs [nm], Nt [cm^-3]) around a base material stack.
    Handles theta normalization and DD anchor-solution generation."""

    def __init__(self, htl_mat, abs_mat, etl_mat, d_htl_nm=150.0, d_etl_nm=50.0,
                 d_abs_range=(300.0, 700.0), logNt_range=(13.0, 16.0)):
        self.htl, self.abs, self.etl = htl_mat, abs_mat, etl_mat
        self.d_htl, self.d_etl = d_htl_nm, d_etl_nm
        self.d_lo, self.d_hi = d_abs_range
        self.n_lo, self.n_hi = logNt_range
        self.w_spec, self.a_spec = spectral_coefficients(abs_mat)

    def theta(self, d_abs_nm, Nt):
        return np.array([
            2.0 * (d_abs_nm - self.d_lo) / (self.d_hi - self.d_lo) - 1.0,
            2.0 * (np.log10(Nt) - self.n_lo) / (self.n_hi - self.n_lo) - 1.0,
        ])

    def untheta(self, th):
        d = self.d_lo + (th[0] + 1.0) / 2.0 * (self.d_hi - self.d_lo)
        lg = self.n_lo + (th[1] + 1.0) / 2.0 * (self.n_hi - self.n_lo)
        return d, 10.0 ** lg

    def L_total_cm(self, d_abs_nm):
        return (self.d_htl + d_abs_nm + self.d_etl) * 1e-7

    def dd_solution(self, d_abs_nm, Nt, V=0.0, T=300.0):
        """Production drift-diffusion anchor solution."""
        from physics.dd_solver import build_mesh, solve_dd
        from physics.device import _dd_beer_lambert_generation
        mesh = build_mesh([self.htl, self.abs, self.etl],
                          [self.d_htl, d_abs_nm, self.d_etl],
                          Nt_override=[None, Nt, None], T=T)
        G = _dd_beer_lambert_generation(mesh, self.abs)
        r = solve_dd(mesh, G, V, self.htl, self.etl, T=T)
        return mesh, r


# ─────────────────────────────────────────────────────────────────────────────
# Physics residuals at (x, theta) collocation — autograd
# ─────────────────────────────────────────────────────────────────────────────
def pde_residuals(model, fam: DeviceFamily, x_norm, theta, T=300.0):
    """Poisson + electron/hole continuity residuals inside the ABSORBER of
    the theta-conditioned device, with position-dependent Beer-Lambert G(x).

    x_norm is the position normalized by the theta-dependent total length; the
    residual masks itself to the absorber region of each sample's device."""
    _require_torch()
    Vt = K_B * T / Q_E
    x_norm = x_norm.detach().requires_grad_(True)
    psi, log_n, log_p = model(x_norm, theta)
    n = torch.exp(torch.clamp(log_n, max=45.0))
    p = torch.exp(torch.clamp(log_p, max=45.0))

    grad = lambda y, x: torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    dpsi = grad(psi, x_norm)
    d2psi = grad(dpsi, x_norm)
    dlogn = grad(log_n, x_norm)
    dlogp = grad(log_p, x_norm)

    # per-sample device geometry
    d_abs_nm = fam.d_lo + (theta[:, 0:1] + 1.0) / 2.0 * (fam.d_hi - fam.d_lo)
    logNt = fam.n_lo + (theta[:, 1:2] + 1.0) / 2.0 * (fam.n_hi - fam.n_lo)
    L_cm = (fam.d_htl + d_abs_nm + fam.d_etl) * 1e-7
    x_cm = x_norm * L_cm
    x0_abs = fam.d_htl * 1e-7
    x1_abs = (fam.d_htl + d_abs_nm) * 1e-7
    in_abs = ((x_cm > x0_abs) & (x_cm < x1_abs)).float()

    ab = fam.abs
    eps = float(getattr(ab, "eps", 6.5)) * EPS0
    Na = float(getattr(ab, "doping", 1e13))
    ni = float(np.sqrt(ab.Nc * ab.Nv) * np.exp(-ab.Eg / (2 * Vt)))
    mu_n = float(getattr(ab, "mu_e", 2.0)); mu_p = float(getattr(ab, "mu_h", 2.0))

    # SRH lifetimes from the sample's Nt (sigma*v_th consistent with dd_solver)
    sigma_vth = 1e-15 * 1e7
    tau = 1.0 / (sigma_vth * torch.pow(10.0, logNt))

    # Position-dependent Beer-Lambert generation (depth from absorber front)
    w = torch.tensor(fam.w_spec, dtype=x_cm.dtype)
    a = torch.tensor(fam.a_spec, dtype=x_cm.dtype)
    depth = torch.clamp(x_cm - x0_abs, min=0.0)
    Gx = (w[None, :] * torch.exp(-a[None, :] * depth)).sum(dim=1, keepdim=True)
    Gx = Gx * in_abs

    # Poisson: eps psi'' + q (p - n - Na) = 0  (x-derivs via 1/L per sample)
    res_P = (eps * d2psi / L_cm ** 2 + Q_E * (p - n - Na)) / (Q_E * 1e15)

    # Continuity with SRH + G(x)
    dpsi_dx = dpsi / L_cm
    dn_dx = n * dlogn / L_cm
    dp_dx = p * dlogp / L_cm
    Jn = Q_E * mu_n * (n * dpsi_dx + Vt * dn_dx)
    Jp = Q_E * mu_p * (p * (-dpsi_dx) - Vt * dp_dx)
    dJn_dx = grad(Jn, x_norm) / L_cm
    dJp_dx = grad(Jp, x_norm) / L_cm
    R = (n * p - ni ** 2) / (tau * (n + ni) + tau * (p + ni) + 1e-30)
    sG = Q_E * 2.5e21
    res_n = (dJn_dx - Q_E * (R - Gx)) / sG
    res_p = (-dJp_dx - Q_E * (R - Gx)) / sG

    return res_P * in_abs, res_n * in_abs, res_p * in_abs


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_conditional_pinn(fam: DeviceFamily, anchors=None,
                           epochs_data=400, epochs_pde=1200,
                           n_colloc=192, lr=2e-3, seed=0, verbose=True):
    """Two-stage training on production DD anchor solutions.

    anchors: list of (d_abs_nm, Nt). Default = 4 corners of the theta box.
    Returns (model, history dict)."""
    _require_torch()
    torch.manual_seed(seed)
    if anchors is None:
        anchors = [(fam.d_lo, 10 ** fam.n_lo), (fam.d_lo, 10 ** fam.n_hi),
                   (fam.d_hi, 10 ** fam.n_lo), (fam.d_hi, 10 ** fam.n_hi)]

    # DD anchor data
    X, TH, Y = [], [], []
    for (d, Nt) in anchors:
        mesh, r = fam.dd_solution(d, Nt)
        L = fam.L_total_cm(d)
        xn = (mesh.x / L).reshape(-1, 1)
        th = np.tile(fam.theta(d, Nt), (len(xn), 1))
        y = np.stack([r.psi - r.psi.mean(),
                      np.log(np.clip(r.n, 1e5, None)),
                      np.log(np.clip(r.p, 1e5, None))], axis=1)
        X.append(xn); TH.append(th); Y.append(y)
    X = torch.tensor(np.vstack(X), dtype=torch.float32)
    TH = torch.tensor(np.vstack(TH), dtype=torch.float32)
    Y = torch.tensor(np.vstack(Y), dtype=torch.float32)

    model = ConditionalPINN(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss_data": [], "loss_pde": []}

    def data_loss():
        psi, ln, lp = model(X, TH)
        return (((psi.squeeze() - Y[:, 0]) ** 2).mean()
                + 0.02 * ((ln.squeeze() - Y[:, 1]) ** 2).mean()
                + 0.02 * ((lp.squeeze() - Y[:, 2]) ** 2).mean())

    # Stage A: data-only
    for ep in range(epochs_data):
        opt.zero_grad(); L = data_loss(); L.backward(); opt.step()
        hist["loss_data"].append(float(L.detach()))
    # Stage B: data + physics at random (x, theta) collocation.
    # Adaptive weighting (gradient-magnitude balancing, as in ai/pinn_real.py):
    # the physics term is capped to contribute at most ~10% of the data-loss
    # magnitude, so the PDE regularizes the theta-interpolation without
    # destroying the anchor fit.
    for ep in range(epochs_pde):
        opt.zero_grad()
        Ld = data_loss()
        xc = torch.rand(n_colloc, 1)
        thc = torch.rand(n_colloc, 2) * 2.0 - 1.0
        rP, rn, rp = pde_residuals(model, fam, xc, thc)
        Lp = (rP ** 2).mean() + (rn ** 2).mean() + (rp ** 2).mean()
        lam_pde = 0.1 * float(Ld.detach()) / max(float(Lp.detach()), 1e-30)
        (Ld + lam_pde * Lp).backward()
        opt.step()
        hist["loss_data"].append(float(Ld.detach()))
        hist["loss_pde"].append(float(Lp.detach()))
        if verbose and (ep + 1) % 300 == 0:
            print(f"  [PDE stage] epoch {ep+1}/{epochs_pde}  "
                  f"L_data={float(Ld.detach()):.3e}  L_pde={float(Lp.detach()):.3e}")
    return model, hist


def evaluate_interpolation(model, fam: DeviceFamily, d_abs_nm, Nt):
    """Relative-L2 errors of the PINN vs an UNSEEN production DD solution."""
    _require_torch()
    mesh, r = fam.dd_solution(d_abs_nm, Nt)
    L = fam.L_total_cm(d_abs_nm)
    xn = torch.tensor((mesh.x / L).reshape(-1, 1), dtype=torch.float32)
    th = torch.tensor(np.tile(fam.theta(d_abs_nm, Nt), (len(mesh.x), 1)),
                      dtype=torch.float32)
    with torch.no_grad():
        psi, ln, lp = model(xn, th)
    psi = psi.numpy().ravel(); ln = ln.numpy().ravel()
    psi_ref = r.psi - r.psi.mean()
    ln_ref = np.log(np.clip(r.n, 1e5, None))
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    return {"relL2_psi": rel(psi, psi_ref), "relL2_logn": rel(ln, ln_ref),
            "x_um": mesh.x * 1e4, "psi_pinn": psi, "psi_dd": psi_ref,
            "logn_pinn": ln, "logn_dd": ln_ref}
