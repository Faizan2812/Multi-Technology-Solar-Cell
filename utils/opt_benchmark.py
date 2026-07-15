"""
utils/opt_benchmark.py — head-to-head optimization benchmark (v3.1, Phase 2)
=============================================================================

The synopsis' Expected Outcome 2 requires DEMONSTRATING that the tool's AI
optimization outperforms the traditional methods used by commercial
simulators — not asserting it. This module runs that controlled experiment.

Design problem (fixed for all methods):
    maximize PCE(d_HTL, d_abs, d_ETL, log10 Nt) of the reference
    Spiro-OMeTAD / MAPbI3 / SnO2 stack, physics objective = fast_simulate.
    Bounds: d_HTL in [50, 300] nm, d_abs in [200, 900] nm,
            d_ETL in [20, 100] nm, log10 Nt in [13, 16].

Contenders (each restricted to the SAME evaluation budget, repeated seeds):
    * Random search                — the null baseline
    * Nelder-Mead (multi-start)   — classic derivative-free local search
    * L-BFGS-B (multi-start, FD gradients) — the quasi-Newton family the
      synopsis attributes to Silvaco TCAD
    * Least-squares TRF           — the Levenberg-Marquardt family the
      synopsis attributes to Sentaurus TCAD (TRF used because scipy's 'lm'
      requires m >= n residuals)
    * Particle Swarm Optimization — canonical PSO (Kennedy & Eberhart 1995),
      the method the synopsis attributes to Lumerical
    * Differential evolution      — this tool (inverse-design engine)
    * Bayesian optimization (GP+EI) — this tool (ai/optimizer.py)

Metrics per method: best PCE found (mean +/- std over seeds), mean
evaluations to reach 99% of the global best, and success rate.

References: Nocedal & Wright, Numerical Optimization (2006) [L-BFGS];
Moré 1978 [LM]; Kennedy & Eberhart, Proc. ICNN (1995) [PSO];
Storn & Price, J. Glob. Optim. 11, 341 (1997) [DE];
Jones, Schonlau & Welch, J. Glob. Optim. 13, 455 (1998) [BO/EI].
"""
from __future__ import annotations
import json
import numpy as np

BOUNDS = [(50.0, 300.0), (200.0, 900.0), (20.0, 100.0), (13.0, 16.0)]

# Problem B: realistic design task — SELECT the HTL and ETL materials
# (categorical, encoded as continuous indices and rounded) AND optimize the
# geometry/quality. The discrete material axes make the landscape multimodal
# and discontinuous, which is where the synopsis' criticism of local
# quasi-Newton/LM methods actually applies.
def bounds_b():
    from physics.materials import HTL_DB, ETL_DB
    return [(0.0, len(HTL_DB) - 1e-9), (0.0, len(ETL_DB) - 1e-9),
            (50.0, 300.0), (200.0, 900.0), (20.0, 100.0), (13.0, 16.0)]


def make_objective_b(counter=None):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import fast_simulate
    htls = list(HTL_DB.values()); etls = list(ETL_DB.values())
    a = PEROVSKITE_DB["MAPbI3"]
    B = bounds_b()
    state = counter if counter is not None else {"n": 0, "best": -np.inf, "trace": []}

    def f(x):
        x = np.clip(np.asarray(x, float), [b[0] for b in B], [b[1] for b in B])
        h = htls[int(x[0])]; e = etls[int(x[1])]
        r = fast_simulate(h, a, e, x[2], x[3], x[4], 10.0 ** x[5], 300)
        pce = float(r["PCE"])
        state["n"] += 1
        state["best"] = max(state["best"], pce)
        state["trace"].append(state["best"])
        return pce
    f.state = state
    return f


def make_objective(counter=None):
    """PCE objective with evaluation counting and trace recording."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import fast_simulate
    h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
    state = counter if counter is not None else {"n": 0, "best": -np.inf, "trace": []}

    def f(x):
        x = np.clip(np.asarray(x, float),
                    [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        r = fast_simulate(h, a, e, x[0], x[1], x[2], 10.0 ** x[3], 300)
        pce = float(r["PCE"])
        state["n"] += 1
        state["best"] = max(state["best"], pce)
        state["trace"].append(state["best"])
        return pce
    f.state = state
    return f


def _rand_x(rng, B=None):
    B = B or BOUNDS
    return np.array([rng.uniform(lo, hi) for lo, hi in B])


# ── contenders (parametric in objective factory + bounds) ───────────────────
def run_random(budget, seed, mk=None, B=None):
    mk = mk or make_objective; B = B or BOUNDS
    rng = np.random.default_rng(seed)
    f = mk()
    for _ in range(budget):
        f(_rand_x(rng, B))
    return f.state


def run_nelder_mead(budget, seed, mk=None, B=None):
    from scipy.optimize import minimize
    mk = mk or make_objective; B = B or BOUNDS
    rng = np.random.default_rng(seed)
    f = mk()
    while f.state["n"] < budget:
        rem = budget - f.state["n"]
        minimize(lambda x: -f(x), _rand_x(rng, B), method="Nelder-Mead",
                 options={"maxfev": rem, "xatol": 1e-3, "fatol": 1e-4})
    return f.state


def run_lbfgsb(budget, seed, mk=None, B=None):
    from scipy.optimize import minimize
    mk = mk or make_objective; B = B or BOUNDS
    rng = np.random.default_rng(seed)
    f = mk()
    while f.state["n"] < budget:
        rem = budget - f.state["n"]
        minimize(lambda x: -f(x), _rand_x(rng, B), method="L-BFGS-B",
                 bounds=B, options={"maxfun": rem, "eps": 1e-3})
    return f.state


def run_least_squares_trf(budget, seed, mk=None, B=None):
    """Levenberg-Marquardt family (TRF): minimize residual (PCE_UB - PCE)."""
    from scipy.optimize import least_squares
    mk = mk or make_objective; B = B or BOUNDS
    rng = np.random.default_rng(seed)
    f = mk()
    PCE_UB = 35.0
    while f.state["n"] < budget:
        rem = budget - f.state["n"]
        if rem < 10:
            for _ in range(rem):
                f(_rand_x(rng, B))
            break
        least_squares(lambda x: np.array([PCE_UB - f(x)]), _rand_x(rng, B),
                      bounds=([b[0] for b in B], [b[1] for b in B]),
                      method="trf", max_nfev=rem, diff_step=1e-3)
    return f.state


def run_pso(budget, seed, mk=None, B=None, n_particles=12, w=0.7, c1=1.5, c2=1.5):
    """Canonical PSO (Kennedy & Eberhart 1995) with clipped bounds."""
    mk = mk or make_objective; B = B or BOUNDS
    rng = np.random.default_rng(seed)
    f = mk()
    lo = np.array([b[0] for b in B]); hi = np.array([b[1] for b in B])
    X = np.array([_rand_x(rng, B) for _ in range(n_particles)])
    V = np.zeros_like(X)
    pbest = X.copy(); pval = np.array([f(x) for x in X])
    g = int(np.argmax(pval))
    while f.state["n"] < budget:
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (pbest[g] - X)
        X = np.clip(X + V, lo, hi)
        for i in range(n_particles):
            if f.state["n"] >= budget:
                break
            v = f(X[i])
            if v > pval[i]:
                pval[i] = v; pbest[i] = X[i].copy()
        g = int(np.argmax(pval))
    return f.state


def run_de(budget, seed, mk=None, B=None):
    from scipy.optimize import differential_evolution
    mk = mk or make_objective; B = B or BOUNDS
    f = mk()
    pop = 10
    maxiter = max(1, budget // (pop * len(B)) - 1)
    differential_evolution(lambda x: -f(x), B, seed=seed, tol=1e-8,
                           popsize=pop, maxiter=maxiter, polish=False, init="sobol")
    rng = np.random.default_rng(seed)
    while f.state["n"] < budget:
        f(_rand_x(rng, B))
    return f.state


def run_bo(budget, seed, mk=None, B=None):
    from ai.optimizer import bayesian_optimization
    mk = mk or make_objective; B = B or BOUNDS
    f = mk()
    n_init = min(12, budget // 4)
    np.random.seed(seed)                    # ai/optimizer uses global RNG
    bayesian_optimization(f, B, n_initial=n_init,
                          n_iterations=budget - n_init)
    f.state["trace"] = f.state["trace"][:budget]
    return f.state


def run_hybrid_tool(budget, seed, mk=None, B=None):
    """This tool's purpose-built hybrid: integer-aware DE + Nelder-Mead
    polish on the continuous sub-space (ai.optimizer.hybrid_optimize)."""
    from ai.optimizer import hybrid_optimize
    mk = mk or make_objective; B = B or BOUNDS
    f = mk()
    int_dims = (0, 1) if len(B) == 6 else ()
    hybrid_optimize(f, B, budget=budget, integer_dims=int_dims, seed=seed)
    f.state["trace"] = f.state["trace"][:budget]
    return f.state


CONTENDERS = {
    "Random search": run_random,
    "Nelder-Mead (multi-start)": run_nelder_mead,
    "L-BFGS-B (quasi-Newton, Silvaco-style)": run_lbfgsb,
    "Least-squares TRF (LM family, Sentaurus-style)": run_least_squares_trf,
    "PSO (Lumerical-style)": run_pso,
    "Differential evolution (this tool)": run_de,
    "Bayesian optimization GP+EI (this tool)": run_bo,
    "Hybrid DE+NM, integer-aware (this tool)": run_hybrid_tool,
}


def run_study(budget=120, n_seeds=8, verbose=True, problem="A"):
    """Full controlled comparison on problem A (smooth 4-D geometry) or
    B (mixed categorical material selection + geometry, multimodal).
    Returns the results dict and writes artifacts/opt_benchmark_<problem>.json."""
    if problem == "B":
        mk, B = make_objective_b, bounds_b()
    else:
        mk, B = make_objective, BOUNDS
    results = {}
    # establish the global best across everything for evals-to-99% metric
    all_traces = {}
    for name, fn in CONTENDERS.items():
        finals, traces = [], []
        for s in range(n_seeds):
            st = fn(budget, seed=1000 + s, mk=mk, B=B)
            tr = (st["trace"] + [st["trace"][-1]] * budget)[:budget]
            finals.append(st["best"]); traces.append(tr)
        results[name] = {"best_mean": float(np.mean(finals)),
                         "best_std": float(np.std(finals)),
                         "finals": [float(v) for v in finals]}
        all_traces[name] = np.array(traces)
        if verbose:
            print(f"  {name:48s} best = {np.mean(finals):6.3f} +/- {np.std(finals):.3f} %PCE")
    global_best = max(np.max(t) for t in all_traces.values())
    thr = 0.99 * global_best
    for name in CONTENDERS:
        T = all_traces[name]
        hits = [(np.argmax(t >= thr) + 1) if np.any(t >= thr) else None for t in T]
        ok = [h for h in hits if h is not None]
        results[name]["evals_to_99pct"] = float(np.mean(ok)) if ok else None
        results[name]["success_rate"] = len(ok) / len(hits)
        results[name]["mean_trace"] = [float(v) for v in T.mean(axis=0)]
    out = {"budget": budget, "n_seeds": n_seeds, "bounds": B,
           "global_best_PCE": float(global_best), "threshold_99pct": float(thr),
           "results": results,
           "problem": ("A: smooth geometry-only (unimodal)" if problem == "A" else
                       "B: HTL/ETL material selection + geometry (multimodal, "
                       "discontinuous categorical axes)")}
    import os
    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open(f"artifacts/opt_benchmark_{problem}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Problem A — smooth 4-D geometry (unimodal), 120 evals x 8 seeds:")
    run_study(problem="A")
    print("Problem B — material selection + geometry (multimodal), 150 evals x 8 seeds:")
    run_study(budget=150, problem="B")
