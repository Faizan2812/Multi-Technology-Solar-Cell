"""
ai/multi_tech_optimizer.py — Technology-agnostic optimizer
=================================================================
One optimization interface for all four technologies. The existing
ai/optimizer.py (GA, DE, NSGA-II, Bayesian, inverse design) remains the
perovskite-specialized engine; this module generalizes the workflow so
silicon, organic and tandem devices get the same treatment:

  * differential evolution (scipy) global search over a named-parameter
    box — the same algorithm class validated in utils/opt_benchmark.py
  * optional random-forest surrogate fitted on the evaluated population
    for permutation-style sensitivity ranking (which knob matters most)

Usage:
    from ai.multi_tech_optimizer import optimize
    res = optimize("silicon", {"W_um": (100, 250), "Ndop_cm3": (1e15, 5e16),
                               "J0s_fA": (1, 50)}, base_preset="SHJ-IBC (Kaneka 26.7%)")
"""
from __future__ import annotations

import dataclasses
import numpy as np


def _make_objective(technology, param_names, base_obj, metric="PCE"):
    if technology == "silicon":
        from physics.silicon import simulate_silicon

        def f(x):
            arch = dataclasses.replace(base_obj,
                                       **dict(zip(param_names, x)))
            try:
                return -simulate_silicon(arch)[metric]
            except Exception:
                return 0.0
        return f

    if technology == "organic":
        from physics.organic import simulate_organic

        def f(x):
            blend = dataclasses.replace(base_obj,
                                        **dict(zip(param_names, x)))
            try:
                return -simulate_organic(blend)[metric]
            except Exception:
                return 0.0
        return f

    if technology == "perovskite":
        from physics.device import fast_simulate
        htl, ab, etl = base_obj  # tuple of Materials

        def f(x):
            kw = dict(zip(param_names, x))
            try:
                r = fast_simulate(htl, ab, etl,
                                  kw.get("d_htl_nm", 100),
                                  kw.get("d_abs_nm", 500),
                                  kw.get("d_etl_nm", 50),
                                  kw.get("Nt_abs", None))
                return -r[metric]
            except Exception:
                return 0.0
        return f

    if technology == "tandem":
        from physics.tandem import simulate_perovskite_silicon_tandem
        htl, ab, etl, si_arch = base_obj

        def f(x):
            kw = dict(zip(param_names, x))
            try:
                r = simulate_perovskite_silicon_tandem(
                    htl, ab, etl, kw.get("d_top_nm", 500), si_arch,
                    Nt_top=kw.get("Nt_top", 1e14),
                    R_int=kw.get("R_int", 0.05),
                    parasitic=kw.get("parasitic", 0.03),
                    Rs_int_ohm_cm2=kw.get("Rs_int_ohm_cm2", 1.5))
                return -r[metric]
            except Exception:
                return 0.0
        return f

    raise ValueError(f"unknown technology {technology!r}")


def optimize(technology, bounds, base_obj, metric="PCE",
             maxiter=25, popsize=12, seed=0, polish=True,
             sensitivity=True):
    """
    Global optimization of `metric` for a technology.

    bounds   : {param_name: (lo, hi)} — dataclass fields of the preset
               (silicon/organic), or thickness/Nt keys (perovskite/tandem)
    base_obj : the preset dataclass (silicon/organic), (htl, abs, etl)
               tuple (perovskite), or (htl, abs, etl, si_arch) (tandem)
    Returns dict with best params, best metric, history, and (optionally)
    a random-forest sensitivity ranking of the parameters.
    """
    from scipy.optimize import differential_evolution

    names = list(bounds.keys())
    box = [bounds[n] for n in names]
    f = _make_objective(technology, names, base_obj, metric)

    X_hist, y_hist = [], []

    def wrapped(x):
        val = f(x)
        X_hist.append(np.array(x, float))
        y_hist.append(-val)
        return val

    res = differential_evolution(wrapped, box, maxiter=maxiter,
                                 popsize=popsize, seed=seed, tol=1e-6,
                                 polish=polish, updating="deferred")
    out = {
        "technology": technology, "metric": metric,
        "best_params": dict(zip(names, [float(v) for v in res.x])),
        "best_value": float(-res.fun),
        "n_evaluations": len(y_hist),
        "history_best": list(np.maximum.accumulate(y_hist)),
    }

    if sensitivity and len(X_hist) >= 30:
        try:
            from sklearn.ensemble import RandomForestRegressor
            X = np.vstack(X_hist)
            y = np.array(y_hist)
            rf = RandomForestRegressor(n_estimators=120, random_state=0)
            rf.fit(X, y)
            imp = rf.feature_importances_
            out["sensitivity"] = dict(sorted(
                zip(names, [float(v) for v in imp]),
                key=lambda kv: -kv[1]))
        except Exception:
            out["sensitivity"] = None
    return out
