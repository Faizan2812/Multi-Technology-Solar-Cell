"""
utils/measurement_fit.py — fit measured J-V data with honest uncertainty.
===========================================================================
The daily task of the 1-D simulation community is fitting measured
current-voltage curves to extract device parameters — and the literature
shows it is routinely done without uncertainty or identifiability analysis
(the root cause of the "unrealistic results" problem this tool answers;
Saidarsan 2025, DOI 10.1016/j.solmat.2025.113632).

This module provides:
  1.  An EXACT single-diode forward model via the Lambert-W closed form
      (Jain & Kapoor 2004, DOI 10.1016/j.solmat.2003.11.018) — the
      technology-agnostic workhorse of J-V analysis.
  2.  A multi-start bounded least-squares fit of the five parameters
      (J_ph, J_0, n, R_s, R_sh).
  3.  BOOTSTRAP confidence intervals (residual resampling): every reported
      parameter carries a [2.5%, 97.5%] interval, never a bare point value.
  4.  A NON-UNIQUENESS report: the bootstrap parameter correlation matrix,
      with |r| > 0.9 pairs flagged — making the classic J0↔n degeneracy
      visible instead of silently absorbed.

Verified by tests/test_measurement_fit.py: blind recovery of known synthetic
devices within the reported intervals, forward-model exactness against a
numerical solve, and the degeneracy flag firing on the J0-n pair.

Conventions: photocurrent positive (generator convention), J in mA/cm²,
V in volts, areas normalised out. T defaults to 300 K.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import lambertw

K_B = 1.380649e-23
Q_E = 1.602176634e-19


# ─────────────────────────────────────────────────────────────────────────
#  Forward model: exact Lambert-W single diode
# ─────────────────────────────────────────────────────────────────────────
def _lambertw_real(y: np.ndarray) -> np.ndarray:
    """W0(exp(y)) computed overflow-safely.

    For y <= 690 use scipy on exp(y) directly; beyond that use the
    asymptotic iteration W(e^y) = y - log(W(e^y)) which converges fast.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y <= 690.0
    out[small] = np.real(lambertw(np.exp(y[small])))
    if np.any(~small):
        w = y[~small] - np.log(np.maximum(y[~small], 1e-12))
        for _ in range(6):                       # Newton-style refinement
            w = y[~small] - np.log(w)
        out[~small] = w
    return out


def diode_jv(V, Jph, J0, n, Rs, Rsh, T=300.0):
    """Exact J(V) of the single-diode model, generator convention (J>0).

    Implicit equation  J = Jph - J0[exp((V+J·Rs)/(n·Vt)) - 1] - (V+J·Rs)/Rsh
    solved in closed form with the Lambert-W function.
    Units: J, Jph, J0 in mA/cm²; Rs, Rsh in Ω·cm² (consistent with mA via
    the 1e-3 factor handled internally).
    """
    V = np.asarray(V, dtype=float)
    Vt = K_B * T / Q_E
    nVt = n * Vt
    rs = Rs * 1e-3          # Ω·cm² acting on mA/cm² → volts
    rsh = Rsh * 1e-3
    # log-space argument of W to avoid overflow:
    #   x = (rs*J0*rsh)/(nVt*(rs+rsh)) * exp( rsh*(V + rs*(Jph+J0)) / (nVt*(rs+rsh)) )
    denom = nVt * (rs + rsh)
    log_x = (np.log(rs * J0 * rsh / denom)
             + rsh * (V + rs * (Jph + J0)) / denom)
    W = _lambertw_real(log_x)
    J = (rsh * (Jph + J0) - V) / (rs + rsh) - (nVt / rs) * W
    return J


PARAM_NAMES = ("Jph_mA_cm2", "J0_mA_cm2", "n", "Rs_ohm_cm2", "Rsh_ohm_cm2")


@dataclass
class FitResult:
    params: Dict[str, float]
    ci_low: Dict[str, float]
    ci_high: Dict[str, float]
    rmse_mA_cm2: float
    correlation: np.ndarray                # 5x5 bootstrap correlation
    degenerate_pairs: List[Tuple[str, str, float]]
    n_bootstrap: int
    V: np.ndarray
    J_meas: np.ndarray
    J_fit: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary_rows(self) -> List[dict]:
        rows = []
        for k in PARAM_NAMES:
            rows.append({"Parameter": k, "Estimate": self.params[k],
                         "CI 2.5%": self.ci_low[k], "CI 97.5%": self.ci_high[k]})
        return rows


def _theta_to_params(theta):
    Jph, logJ0, n, Rs, logRsh = theta
    return dict(zip(PARAM_NAMES, (Jph, 10 ** logJ0, n, Rs, 10 ** logRsh)))


def _fit_once(V, J, theta0, bounds, V_dark=None, J_dark=None):
    def resid(theta):
        p = _theta_to_params(theta)
        r = diode_jv(V, p["Jph_mA_cm2"], p["J0_mA_cm2"], p["n"],
                     p["Rs_ohm_cm2"], p["Rsh_ohm_cm2"]) - J
        if V_dark is not None:
            # dark branch: same diode, Jph = 0. Log-magnitude residuals so
            # the exponentially small low-bias currents carry weight — this
            # is what pins (J0, n) independently of the light curve.
            Jd = diode_jv(V_dark, 0.0, p["J0_mA_cm2"], p["n"],
                          p["Rs_ohm_cm2"], p["Rsh_ohm_cm2"])
            rd = (np.log10(np.clip(np.abs(Jd), 1e-12, None))
                  - np.log10(np.clip(np.abs(J_dark), 1e-12, None)))
            r = np.concatenate([r, 2.0 * rd])
        return r
    return least_squares(resid, theta0, bounds=bounds, method="trf",
                         x_scale="jac", max_nfev=400)


def fit_measured_jv(V, J, T=300.0, n_bootstrap=200,
                    seed: int | None = 0,
                    V_dark=None, J_dark=None) -> FitResult:
    """Fit the single-diode model to measured light J-V with bootstrap CIs.

    Multi-start in (Jph, log10 J0, n, Rs, log10 Rsh); residual-resampling
    bootstrap for intervals; correlation matrix for identifiability.
    """
    V = np.asarray(V, float)
    J = np.asarray(J, float)
    order = np.argsort(V)
    V, J = V[order], J[order]
    if V.size < 8:
        raise ValueError("need at least 8 J-V points to fit 5 parameters")

    Jsc0 = float(np.interp(0.0, V, J))
    Voc0 = float(np.interp(0.0, -J, V)) if np.any(J < 0) else float(V.max())
    Vt = K_B * T / Q_E

    bounds = (np.array([0.5 * Jsc0, -18.0, 0.8, 1e-3, 1.0]),
              np.array([1.3 * Jsc0, -1.0, 3.5, 20.0, 8.0]))
    # physics-guided multi-start: ideality guesses spanning thin-film range
    starts = []
    for n_guess in (1.1, 1.6, 2.2):
        logJ0_guess = np.log10(max(Jsc0, 1e-6)) - Voc0 / (n_guess * Vt) / np.log(10)
        logJ0_guess = float(np.clip(logJ0_guess, -17.5, -2.0))
        starts.append(np.array([Jsc0, logJ0_guess, n_guess, 0.5, 4.0]))

    if V_dark is not None:
        V_dark = np.asarray(V_dark, float)
        J_dark = np.asarray(J_dark, float)
    best = None
    for th0 in starts:
        try:
            res = _fit_once(V, J, th0, bounds, V_dark, J_dark)
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue
    if best is None:
        raise RuntimeError("all fit starts failed — check units/sign of data")

    p_hat = _theta_to_params(best.x)
    J_fit = diode_jv(V, *[p_hat[k] for k in PARAM_NAMES], T=T)
    resid = J_fit - J
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    # ── residual-resampling bootstrap ────────────────────────────────
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        J_boot = J_fit - rng.choice(resid, size=resid.size, replace=True)
        try:
            rb = _fit_once(V, J_boot, best.x, bounds, V_dark, J_dark)
            samples.append([_theta_to_params(rb.x)[k] for k in PARAM_NAMES])
        except Exception:
            continue
    S = np.array(samples)
    warnings = []
    if S.shape[0] < 0.8 * n_bootstrap:
        warnings.append(f"only {S.shape[0]}/{n_bootstrap} bootstrap refits "
                        "converged — intervals may be optimistic")

    # correlations on log-scaled positive params for meaningful r
    L = S.copy()
    for j, k in enumerate(PARAM_NAMES):
        if k in ("J0_mA_cm2", "Rsh_ohm_cm2"):
            L[:, j] = np.log10(np.maximum(L[:, j], 1e-30))
    corr = np.corrcoef(L, rowvar=False)
    degen = []
    for i in range(5):
        for j in range(i + 1, 5):
            if abs(corr[i, j]) > 0.9:
                degen.append((PARAM_NAMES[i], PARAM_NAMES[j],
                              float(corr[i, j])))
    if degen:
        warnings.append(
            "strong parameter degeneracy detected: "
            + "; ".join(f"{a}↔{b} (r={r:+.2f})" for a, b, r in degen)
            + " — these parameters are NOT independently identifiable from "
              "this J-V alone; report the interval, or add EQE / dark J-V / "
              "intensity-dependent data to break the degeneracy.")

    lo = {k: float(np.percentile(S[:, j], 2.5)) for j, k in enumerate(PARAM_NAMES)}
    hi = {k: float(np.percentile(S[:, j], 97.5)) for j, k in enumerate(PARAM_NAMES)}

    # device metrics from the fitted curve
    Vg = np.linspace(0, max(V.max(), Voc0), 400)
    Jg = diode_jv(Vg, *[p_hat[k] for k in PARAM_NAMES], T=T)
    P = Vg * Jg
    i_mp = int(np.argmax(P))
    Jsc = float(np.interp(0.0, Vg, Jg))
    Voc = float(np.interp(0.0, -Jg, Vg))
    metrics = {"Jsc": Jsc, "Voc": Voc,
               "FF": float(P[i_mp] / (Jsc * Voc)) if Jsc * Voc > 0 else 0.0,
               "PCE_at_100mW": float(P[i_mp] / 100.0)}

    if V_dark is not None:
        warnings.append("joint light + dark J-V fit: the dark branch pins "
                        "(J0, n) independently of the light curve.")
    return FitResult(params=p_hat, ci_low=lo, ci_high=hi, rmse_mA_cm2=rmse,
                     correlation=corr, degenerate_pairs=degen,
                     n_bootstrap=S.shape[0], V=V, J_meas=J, J_fit=J_fit,
                     metrics=metrics, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────
#  CSV ingestion
# ─────────────────────────────────────────────────────────────────────────
def parse_jv_csv(text_or_bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a two-column J-V file (V [V], J [mA/cm²]); header optional;
    comma/semicolon/tab/whitespace delimited. Sign is auto-corrected to the
    generator convention (Jsc > 0)."""
    if isinstance(text_or_bytes, bytes):
        text_or_bytes = text_or_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in io.StringIO(text_or_bytes):
        line = line.strip().replace(";", ",").replace("\t", ",")
        if not line:
            continue
        parts = [p for p in (line.split(",") if "," in line else line.split())
                 if p != ""]
        try:
            v, j = float(parts[0]), float(parts[1])
            rows.append((v, j))
        except (ValueError, IndexError):
            continue                                    # header / junk line
    if len(rows) < 8:
        raise ValueError("could not parse ≥8 numeric (V, J) rows")
    arr = np.array(rows, float)
    V, J = arr[:, 0], arr[:, 1]
    if np.interp(0.0, *_sorted(V, J)) < 0:              # absorber convention
        J = -J
    return V, J


def _sorted(V, J):
    o = np.argsort(V)
    return V[o], J[o]


def demo_measurement(noise_mA=0.15, seed=7):
    """Synthetic 'measured' curve with known ground truth, for the UI demo
    and for the blind-recovery test."""
    truth = dict(Jph_mA_cm2=21.8, J0_mA_cm2=3e-10, n=1.45,
                 Rs_ohm_cm2=1.8, Rsh_ohm_cm2=2500.0)
    V = np.linspace(0, 1.12, 60)
    rng = np.random.default_rng(seed)
    J = diode_jv(V, truth["Jph_mA_cm2"], truth["J0_mA_cm2"], truth["n"],
                 truth["Rs_ohm_cm2"], truth["Rsh_ohm_cm2"]) \
        + rng.normal(0, noise_mA, V.size)
    return V, J, truth
