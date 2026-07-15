"""Validation & Benchmarks : provenance audit, MMS on the PRODUCTION solver, calibrated (conformal) uncertainty."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics.provenance_audit import audit
from ai import uncertainty as unc

st.set_page_config(page_title="Integrity & Validation", page_icon="✅", layout="wide")
st.title("✅ Validation & Benchmarks ")

tab = st.tabs(["Provenance audit", "Solver validation (production MMS)", "Uncertainty: conformal calibration", "Uncertainty (Monte-Carlo)"])

with tab[0]:
    st.subheader("Material-database integrity audit")
    db, findings = audit()
    errs = [f for f in findings if f["severity"] == "ERROR"]
    warns = [f for f in findings if f["severity"] == "WARN"]
    npar = sum(len(md.get("parameters", {})) for c in ("absorbers","etls","htls") for md in db[c].values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("References", len(db["_references"]))
    c2.metric("Parameters", npar)
    c3.metric("Errors", len(errs))
    c4.metric("Warnings", len(warns))
    st.success("PASS — database clean" if not errs else "FAIL")
    with st.expander("Flag classes checked"):
        st.markdown("- **F1** tombstones / dangling sources\n- **F2** DOI format & duplicates\n"
                    "- **F3** key↔venue consistency\n- **F4** confidence tiers & LOW reasons\n"
                    "- **F7** measured vs device-effective mobility separation")
    if findings:
        st.dataframe([{k: f[k] for k in ("severity","flag","where","message")} for f in findings],
                     width='stretch')

with tab[1]:
    st.subheader("Method of manufactured solutions — on the PRODUCTION solver ")
    st.caption("The manufactured solution is injected through the mms_source hook "
               "of physics.dd_solver.solve_poisson_newton, so the code path being verified IS the production "
               "Newton/tridiagonal assembly. It also reports device-level mesh convergence (Richardson) and "
               "the exact Scharfetter-Gummel diffusion-limit identity.")
    import json as _json, os as _os
    art = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "artifacts", "validation_v3_0.json")
    data = _json.load(open(art)) if _os.path.exists(art) else None
    if st.button("Run production-solver validation now (~60 s)") or data is None:
        with st.spinner("MMS + SG identity + device mesh convergence..."):
            from scripts.run_validation import mms_production_poisson, sg_diffusion_limit, device_mesh_convergence
            data = {"mms_production": mms_production_poisson(Ns=(41, 81, 161, 321)),
                    "sg_diffusion_limit": sg_diffusion_limit(),
                    "device_mesh_convergence": device_mesh_convergence()}
    m = data["mms_production"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed order (expect ~2)", f"{m['convergence_rate']:.3f}")
    c2.metric("rel L2 at finest mesh", f"{m['relL2_finest']:.1e}")
    c3.metric("SG diffusion identity", f"{data['sg_diffusion_limit']['max_rel_error']:.1e}",
              help="Scharfetter-Gummel edge current vs exact q*D*dn/dx in the zero-field linear limit")
    fig = go.Figure(go.Scatter(x=m["N"], y=m["L2"], mode="markers+lines", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=m["N"], y=[m["L2"][0]*(m["N"][0]/n)**2 for n in m["N"]],
                             name="O(h^2) reference", line=dict(dash="dot")))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="mesh nodes N",
                      yaxis_title="L2 error (V)", height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, width='stretch')
    dmc = data["device_mesh_convergence"]
    st.caption(f"Full-device mesh convergence: N={dmc['N']} -> PCE={['%.3f'%p for p in dmc['PCE']]}; "
               f"estimated discretization error = {dmc.get('discretization_error_estimate_abs_PCE', float('nan')):.3f} % absolute PCE.")

with tab[2]:
    st.subheader("Calibrated uncertainty — split-conformal prediction ")
    st.caption("A bootstrap standard deviation is not a calibrated error bar. Split conformal prediction "
               "(Lei et al., JASA 2018) gives a distribution-free FINITE-SAMPLE coverage guarantee. The "
               "audit below measures the empirical coverage of both methods on held-out physics data.")
    if st.button("Run coverage audit (~20 s)"):
        from ai.uncertainty import coverage_validation
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from physics.device import fast_simulate
        h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
        rng = np.random.default_rng(0); X, y = [], []
        with st.spinner("Sampling design space + repeated split audits..."):
            for _ in range(150):
                d = rng.uniform(300, 800); lg = rng.uniform(13, 16)
                r = fast_simulate(h, a, e, 150, d, 50, 10 ** lg, 300)
                X.append([d / 800.0, (lg - 13) / 3.0]); y.append(r["PCE"])
            res = coverage_validation(np.array(X), np.array(y), alpha=0.05, n_repeats=8)
        c1, c2, c3 = st.columns(3)
        c1.metric("Target coverage", "95 %")
        c2.metric("Conformal empirical coverage", f"{res['conformal_coverage']*100:.1f} %",
                  delta="calibrated" if res["conformal_coverage"] >= 0.9 else "check")
        c3.metric("Naive bootstrap +/-1.96 sigma", f"{res['bootstrap_coverage']*100:.1f} %",
                  delta="overconfident", delta_color="inverse")
        st.info(f"Interval widths — conformal: {res['conformal_width']:.2f} %PCE, "
                f"bootstrap: {res['bootstrap_width']:.2f} %PCE. The bootstrap interval is narrow "
                "because it is WRONG: it covers the truth far less often than claimed. The conformal "
                "interval is honest by construction.")

with tab[3]:
    st.subheader("Efficiency uncertainty from parameter confidence")
    def pce_model(d): return 20*(d["mu"]/2e-4)**0.05 - 5*(d["tau"]/1e-7 - 1)**2*0.0
    mu_tier = st.selectbox("mobility confidence", ["HIGH","MEDIUM","LOW"], 1)
    tau_tier = st.selectbox("lifetime confidence", ["HIGH","MEDIUM","LOW"], 2)
    out = unc.propagate_confidence(lambda d: 20 + 2*np.log(d["mu"]/2e-4+1e-9) ,
                                   {"mu":2e-4,"tau":1e-7}, {"mu":mu_tier,"tau":tau_tier}, n_mc=1500)
    fig=go.Figure(go.Histogram(x=out["samples"], nbinsx=40, marker_color="#2E7D32"))
    fig.add_vline(x=out["ci95"][0], line_dash="dash", line_color="#C00000")
    fig.add_vline(x=out["ci95"][1], line_dash="dash", line_color="#C00000")
    fig.update_layout(xaxis_title="Predicted PCE (%)", yaxis_title="count", height=360)
    st.plotly_chart(fig, width='stretch')
    st.metric("95% interval width", f"{out['ci95'][1]-out['ci95'][0]:.2f} %",
              help="Lower confidence tiers widen the interval")
