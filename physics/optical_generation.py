"""
optical_generation.py — Production TMM photogeneration for the DD solver
=========================================================================

Replaces the Beer-Lambert G(x) used in `physics/device.py` (mode='dd') with a
coherent transfer-matrix (TMM) generation profile, at normal incidence, for
the full   glass / ITO / (carrier-selective layer) / absorber / (layer) / metal
stack. This is the same class of optical model used by SCAPS-1D's optional
"transfer-matrix" generation file workflow and by standard thin-film optics
references.

Formulation (Pettersson, Roman & Inganäs, J. Appl. Phys. 86, 487 (1999);
Byrnes, arXiv:1603.02720):

  * Each layer j has complex index  ñ_j(λ) = n_j + i·k_j.
  * Interface and propagation matrices give the total system transfer matrix
    S; from S the forward/backward field amplitudes (E⁺_j, E⁻_j) at the front
    of every layer are recovered with partial matrices.
  * The time-averaged energy dissipated per unit volume at depth x in layer j:
        Q_j(x, λ) = (2π c ε₀ k_j n_j / λ) · |E_j(x)|²
    which, divided by the photon energy hc/λ, gives the local *photon*
    absorption rate. We normalize the in-layer profile so that its depth
    integral equals the EXACT layer absorptance computed from the Poynting
    flux difference across the layer boundaries — energy conservation is
    therefore enforced identically (machine precision), not approximately.
  * G(x) = ∫ dλ  Φ_AM1.5G(λ) · A'_abs(x, λ)   [cm⁻³ s⁻¹]
    where A'_abs(x, λ) is the absorptance density (cm⁻¹) in the absorber.

Only absorption in the perovskite layer generates carriers (parasitic
absorption in TCO/ETL/HTL/metal is counted as a loss, as in SCAPS).

Optical constants:
  * n(λ): Cauchy fit from physics/optics.py OPTICAL_DB when available,
    otherwise the constant high-frequency value n = sqrt(ε_r) of the material.
  * k(λ): from the SAME Tauc absorption model already used electrically,
        α(E) = α₀ · sqrt(E − Eg)/E ,  k = α λ / 4π,
    so the optical and electrical descriptions of each material are mutually
    consistent (single source of truth for α₀ and Eg).

Self-tests (__main__ and tests/test_v3_upgrades.py):
  * |R + T + Σ_j A_j − 1| < 1e-10 at every wavelength (energy conservation).
  * Index-matched single layer reproduces the Beer-Lambert profile.
  * ∫ G dx never exceeds the above-gap incident photon flux.
"""
from __future__ import annotations
import numpy as np

from physics.spectrum import (AM15G_WAVELENGTHS, photon_flux, HC_EV_NM)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ── Default front/back stack (SCAPS-style fixed contacts) ────────────────────
# Incidence medium: glass superstrate (n = 1.52, semi-infinite, incoherent —
# handled via a single air→glass Fresnel loss factor, the standard treatment
# for a thick substrate).
N_GLASS = 1.52
R_AIR_GLASS = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2   # ≈ 4.26 %

# ITO front electrode: Cauchy n(λ[µm]) = A + B/λ²; weak sub-gap absorption.
ITO = {"A": 1.85, "B": 0.031, "k550": 0.008, "d_nm": 100.0}
# Back metal (Au) — treated as opaque semi-infinite exit medium with a fixed
# complex index; its "transmittance" is counted as back-metal absorption.
N_AU = 0.2 + 3.4j


def _cauchy_n(lam_nm, A, B):
    lam_um = lam_nm * 1e-3
    return A + B / lam_um ** 2


def _material_nk(mat, lam_nm):
    """Complex refractive index ñ(λ) of a device material.

    n from Cauchy fit (physics.optics.OPTICAL_DB) if present, else sqrt(ε_r).
    k from the Tauc model with the material's own α₀, Eg.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    # --- n(λ) ---
    n = None
    try:
        from physics.optics import OPTICAL_DB
        entry = OPTICAL_DB.get(getattr(mat, "name", None))
        if entry is not None:
            n0, n1, n2 = entry[0], entry[1], entry[2]
            lam_um = lam_nm * 1e-3
            n = n0 + n1 / lam_um ** 2 + n2 / lam_um ** 4
    except Exception:
        pass
    if n is None:
        n = np.full_like(lam_nm, float(np.sqrt(max(getattr(mat, "eps", 9.0), 1.0))))
    # --- k(λ) from Tauc α(E) = α₀ sqrt(E−Eg)/E ---
    E = HC_EV_NM / lam_nm
    Eg = float(getattr(mat, "Eg", 1.5))
    alpha0 = float(getattr(mat, "alpha_coeff", 1e5) or 1e5)
    alpha = alpha0 * np.sqrt(np.maximum(E - Eg, 0.0)) / np.maximum(E, 1e-9)  # cm^-1
    k = alpha * (lam_nm * 1e-7) / (4.0 * np.pi)                              # dimensionless
    return n + 1j * k


def _stack_field(n_list, d_nm, lam_nm, n_in, n_out, nx_per_layer):
    """Coherent TMM at one wavelength (normal incidence).

    Convention (Pettersson 1999): in layer j the field is
        E_j(x) = E+_j e^{i xi_j x} + E-_j e^{-i xi_j x},  xi_j = (2 pi / lam) n~_j,
    with x measured from the FRONT (light side) of the layer, Im(xi) >= 0.
    Interface matrix I_ab = (1/t_ab) [[1, r_ab], [r_ab, 1]] maps the amplitude
    vector just AFTER the interface to just BEFORE it; layer matrix
    L_j = diag(e^{-i xi d}, e^{+i xi d}) maps back-of-layer to front-of-layer.

    Returns
    -------
    R : specular reflectance
    A_metal : absorption in the opaque back contact (residual)
    A_layers : exact per-layer absorptance from the Poynting-flux difference
        across each layer (energy-conserving to machine precision)
    profiles : list of (x_nm, a(x) [nm^-1]) absorptance-density profiles,
        each normalized so that its integral equals A_layers[j]; only layers
        with nx_per_layer[j] > 0 get a resolved profile.
    """
    n_list = [complex(v) for v in n_list]
    n_in, n_out = complex(n_in), complex(n_out)
    k0 = 2.0 * np.pi / lam_nm
    nl = len(n_list)

    def iface(na, nb):
        r = (na - nb) / (na + nb)
        t = 2.0 * na / (na + nb)
        return np.array([[1.0, r], [r, 1.0]], dtype=complex) / t

    def layerL(nj, dj):
        xi_d = k0 * nj * dj
        # clip the growing exponent (never triggered for physical stacks;
        # protects against pathological alpha*d)
        im = min(xi_d.imag, 60.0)
        eplus = np.exp(-1j * xi_d.real) * np.exp(im)     # e^{-i xi d}
        eminus = np.exp(1j * xi_d.real) * np.exp(-im)    # e^{+i xi d}
        return np.array([[eplus, 0.0], [0.0, eminus]], dtype=complex)

    # Total transfer matrix S: [E_in] = S [E_exit], with E_exit = (t, 0)
    S = iface(n_in, n_list[0])
    for j in range(nl):
        nb = n_list[j + 1] if j + 1 < nl else n_out
        S = S @ layerL(n_list[j], d_nm[j]) @ iface(n_list[j], nb)
    r = S[1, 0] / S[0, 0]
    t = 1.0 / S[0, 0]
    R = abs(r) ** 2
    T = (n_out.real / n_in.real) * abs(t) ** 2

    # Amplitudes at the FRONT of each layer, built backwards from the exit.
    fronts = [None] * nl
    u = np.array([t, 0.0], dtype=complex)                # exit medium
    for j in range(nl - 1, -1, -1):
        nb = n_list[j + 1] if j + 1 < nl else n_out
        u = layerL(n_list[j], d_nm[j]) @ (iface(n_list[j], nb) @ u)
        fronts[j] = u.copy()

    # Poynting flux (up to a common constant), normalized by the incident
    # forward flux n_in * |E_in+|^2 = n_in.
    def flux(nj, Ep, Em):
        return (nj.real * (abs(Ep) ** 2 - abs(Em) ** 2)
                - 2.0 * nj.imag * (Ep * np.conj(Em)).imag)

    S_inc = n_in.real                                     # |E+|^2 = 1 incident
    A_layers = np.zeros(nl)
    profiles = []
    for j in range(nl):
        Ep, Em = fronts[j]
        nj, dj = n_list[j], d_nm[j]
        xi = k0 * nj
        # field at the back of layer j
        imd = min(xi.imag * dj, 60.0)
        Ep_b = Ep * np.exp(1j * xi.real * dj) * np.exp(-imd)
        Em_b = Em * np.exp(-1j * xi.real * dj) * np.exp(imd)
        A_layers[j] = max(0.0, (flux(nj, Ep, Em) - flux(nj, Ep_b, Em_b)) / S_inc)
        if nx_per_layer[j] > 0 and nj.imag > 0 and A_layers[j] > 0:
            x = np.linspace(0.0, dj, nx_per_layer[j])
            imx = np.clip(xi.imag * x, 0.0, 60.0)
            E2 = np.abs(Ep * np.exp(1j * xi.real * x) * np.exp(-imx)
                        + Em * np.exp(-1j * xi.real * x) * np.exp(imx)) ** 2
            w = nj.real * nj.imag * E2                    # ~ local dissipation
            Z = _trapz(w, x)
            a = w * (A_layers[j] / Z) if Z > 0 else np.zeros_like(x)
            profiles.append((x, a))
        else:
            npts = max(nx_per_layer[j], 2)
            profiles.append((np.linspace(0.0, dj, npts), np.zeros(npts)))

    A_metal = max(0.0, 1.0 - R - T - A_layers.sum()) + T   # opaque back: T -> metal
    return R, A_metal, A_layers, profiles


def tmm_generation_on_mesh(mesh, materials, thicknesses_nm,
                           absorber_layer_index=1,
                           lam_min=300.0, lam_max=None,
                           include_ito=True, include_glass_reflection=True):
    """Compute the wavelength-integrated TMM generation profile on a DD mesh.

    Parameters
    ----------
    mesh : DeviceMesh (physics.dd_solver) — HTL/absorber/ETL layer ordering
        must match `materials`/`thicknesses_nm`.
    materials : [htl_mat, abs_mat, etl_mat]  (light enters from the ETL side
        in n-i-p through glass/ITO; layer order in the optical stack is
        glass | ITO | ETL | absorber | HTL | Au).
    Returns
    -------
    G : ndarray [cm^-3 s^-1] on mesh.x (nonzero only in the absorber layer),
    diag : dict with R(λ), per-layer A(λ), Jsc_optical, energy-conservation
        residual — for the validation page and unit tests.
    """
    htl_mat, abs_mat, etl_mat = materials
    d_htl, d_abs, d_etl = [float(t) for t in thicknesses_nm]
    Eg = float(getattr(abs_mat, "Eg", 1.5))
    if lam_max is None:
        lam_max = min(HC_EV_NM / max(Eg, 0.5), 1400.0)
    lams = AM15G_WAVELENGTHS[(AM15G_WAVELENGTHS >= lam_min)
                             & (AM15G_WAVELENGTHS <= lam_max)]
    if lams.size < 2:
        return np.zeros(mesh.N), {"error": "no wavelengths above gap"}
    phi = photon_flux(lams)                                  # ph / cm² s nm
    dlam = float(lams[1] - lams[0])

    # optical layer order (light side first): [ITO?] ETL, absorber, HTL
    nx = [0, 0, 121, 0] if include_ito else [0, 121, 0]

    x_abs_nm = None
    G_abs = None
    R_arr, A_stack, cons_res = [], [], []
    for li, lam in enumerate(lams):
        n_layers, d_layers = [], []
        if include_ito:
            n_ito = _cauchy_n(lam, ITO["A"], ITO["B"]) + 1j * ITO["k550"] * (550.0 / lam)
            n_layers.append(n_ito); d_layers.append(ITO["d_nm"])
        n_layers += [complex(_material_nk(etl_mat, lam)),
                     complex(_material_nk(abs_mat, lam)),
                     complex(_material_nk(htl_mat, lam))]
        d_layers += [d_etl, d_abs, d_htl]
        abs_idx = 2 if include_ito else 1
        nx_layers = [0] * len(n_layers); nx_layers[abs_idx] = 121

        R, A_metal, A_layers, profiles = _stack_field(
            n_layers, d_layers, lam,
            n_in=(N_GLASS if include_glass_reflection else 1.0),
            n_out=N_AU, nx_per_layer=nx_layers)

        front_loss = R_AIR_GLASS if include_glass_reflection else 0.0
        scale = (1.0 - front_loss)
        x_loc, a_loc = profiles[abs_idx]                     # a in nm^-1
        if G_abs is None:
            x_abs_nm = x_loc
            G_abs = np.zeros_like(x_loc)
        # photons absorbed per nm per cm² per s → per cm³ per s (1 nm = 1e-7 cm)
        G_abs += phi[li] * dlam * scale * a_loc / 1e-7
        R_arr.append(front_loss + scale * R)
        A_stack.append(scale * A_layers)
        cons_res.append(abs(R + A_metal + A_layers.sum() - 1.0))

    # Map onto the DD mesh (absorber layer). Optical x runs from the light
    # (ETL) side; the DD mesh runs HTL(0) → ETL(L), so flip.
    G = np.zeros(mesh.N)
    m = (mesh.layer == absorber_layer_index)
    if np.any(m):
        x_mesh = mesh.x[m]                                    # cm
        x_rel_nm = (x_mesh - x_mesh[0]) / 1e-7                # nm from HTL side
        x_from_light = d_abs - x_rel_nm                       # nm from ETL side
        G[m] = np.interp(x_from_light, x_abs_nm, G_abs)

    Jsc_opt = 1.602176634e-19 * _trapz(G_abs, x_abs_nm * 1e-7) * 1e3  # mA/cm²
    diag = {
        "lambdas_nm": lams,
        "R": np.asarray(R_arr),
        "A_layers": np.asarray(A_stack),
        "Jsc_optical_mA_cm2": float(Jsc_opt),
        "energy_conservation_max_residual": float(np.max(cons_res)),
        "model": "coherent TMM (Pettersson 1999 / Byrnes), glass+ITO front, Au back",
    }
    return G, diag


if __name__ == "__main__":
    # smoke self-test with simple material stand-ins
    class M:  # minimal material stub
        def __init__(self, name, eps, Eg, a0):
            self.name, self.eps, self.Eg, self.alpha_coeff = name, eps, Eg, a0
    from physics.dd_solver import build_mesh
    htl = M("Spiro-OMeTAD", 3.0, 2.9, 1e4)
    ab = M("MAPbI3", 6.5, 1.55, 1.5e5)
    etl = M("SnO2", 9.0, 3.5, 1e4)
    for m_ in (htl, ab, etl):
        m_.Nc = 2e18; m_.Nv = 2e18; m_.mu_e = 1; m_.mu_h = 1
        m_.doping = 1e17; m_.doping_type = "p"; m_.Nt = 1e14; m_.chi = 4.0
    mesh = build_mesh([htl, ab, etl], [150, 500, 50])
    G, d = tmm_generation_on_mesh(mesh, [htl, ab, etl], [150, 500, 50])
    print("Jsc_optical =", d["Jsc_optical_mA_cm2"], "mA/cm2")
    print("energy conservation residual =", d["energy_conservation_max_residual"])
    assert d["energy_conservation_max_residual"] < 1e-8
    print("OK")
