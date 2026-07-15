"""Datasets, SCAPS interoperability and energy yield."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Datasets & Interop", page_icon="🗂️", layout="wide")
st.title("🗂️ Datasets, SCAPS Import & Energy Yield ")

tab_data, tab_scaps, tab_yield, tab_fit, tab_ay, tab_stab = st.tabs(
    ["📚 Experimental datasets", "🔄 SCAPS .def import", "☀️ Energy yield", "📈 Fit measured J-V", "☀️ Energy yield (all tech)", "⏳ Stability tools"])

# ── datasets ────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("External / experimental dataset pathway")
    st.caption(
        "Objective 2 of the synopsis calls for AI over datasets of REAL device "
        "performance. The tool ships a fully-cited seed dataset (every row has a "
        "DOI: the 10 SCAPS reference devices + 4 certified experimental cells used "
        "by the validation suite) and a loader for the Perovskite Database Project "
        "export (~43,000 devices; Jacobsson et al., Nature Energy 7, 107 (2022), "
        "DOI 10.1038/s41560-021-00941-3 — download the CSV from "
        "perovskitedatabase.com, it is not redistributed here)."
    )
    from utils.dataset_io import (load_seed_dataset, dataset_to_features,
                                  train_surrogate_on_dataset)
    rows = load_seed_dataset()
    st.write(f"**Bundled seed dataset:** {len(rows)} cited device records")
    st.dataframe([{k: r[k] for k in ("device_id", "source_type", "absorber",
                                     "bandgap", "abs_thick_nm", "Voc", "Jsc",
                                     "FF", "PCE", "doi")} for r in rows],
                 width='stretch', hide_index=True)

    up = st.file_uploader("Or load a Perovskite-Database-Project CSV export",
                          type=["csv"], key="pdp")
    if up is not None:
        import tempfile
        from utils.dataset_io import load_perovskite_database
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tf.write(up.read()); tmp = tf.name
        try:
            pdp = load_perovskite_database(tmp, max_rows=20000)
            st.success(f"Loaded {len(pdp)} cleaned device records")
            rows = pdp
        except ValueError as ex:
            st.error(str(ex))

    if st.button("Train Gradient-Boosting surrogate on this dataset", key="tds"):
        try:
            with st.spinner("Training from-scratch GB surrogate..."):
                r = train_surrogate_on_dataset(rows)
            m = r["test_metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("held-out R²", f"{m['R2']:.3f}")
            c2.metric("RMSE (%PCE)", f"{m['RMSE']:.2f}")
            c3.metric("train / test", f"{r['n_train']} / {r['n_test']}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r["y_test"], y=r["pred_test"],
                                     mode="markers", marker=dict(size=9,
                                     color="#1C7293"), name="test devices"))
            lim = [min(r["y_test"].min(), r["pred_test"].min()) - 1,
                   max(r["y_test"].max(), r["pred_test"].max()) + 1]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(dash="dash", color="grey"),
                                     showlegend=False))
            fig.update_layout(xaxis_title="reported PCE (%)",
                              yaxis_title="predicted PCE (%)", height=380,
                              title="Parity on held-out devices")
            st.plotly_chart(fig, width='stretch')
            if r["n_test"] < 10:
                st.info("Small-N caveat: with the 15-row seed dataset this is a "
                        "pipeline demonstration, not a performance claim — load "
                        "the full Perovskite Database export for meaningful "
                        "literature-trained surrogates.")
        except ValueError as ex:
            st.error(str(ex))

# ── SCAPS import ────────────────────────────────────────────────────────────
with tab_scaps:
    st.subheader("Import a SCAPS-1D .def device definition")
    st.caption(
        "Interoperability with the incumbent (Burgelman et al., Thin Solid Films "
        "361-362, 527 (2000)): drop in an existing SCAPS project and re-simulate it "
        "on this tool's solver — then optimize it, which SCAPS cannot do. Skipped "
        "or approximated items are reported explicitly, never silently dropped."
    )
    from utils.scaps_import import parse_def, simulate_imported
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(ROOT, "examples", "example_cdte.def")
    upd = st.file_uploader("Upload .def file", type=["def", "txt"], key="def")
    use_example = st.checkbox("...or use the bundled CdTe example", value=upd is None)
    src = None
    if upd is not None:
        src = upd.read().decode("utf-8", errors="replace")
    elif use_example and os.path.exists(example):
        src = open(example).read()
    if src and st.button("Parse & simulate", key="defsim"):
        p = parse_def(src)
        st.write("**Parsed layers (from illuminated side):**")
        for L in p["layers"]:
            st.code(repr(L), language=None)
        for w in p["warnings"]:
            st.warning(w)
        with st.spinner("Drift-diffusion solve of the imported stack..."):
            r = simulate_imported(p)
        for w in r["warnings"]:
            st.info(w)
        m = r["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PCE (%)", f"{m['PCE']:.2f}")
        c2.metric("Voc (V)", f"{m['Voc']:.3f}")
        c3.metric("Jsc (mA/cm²)", f"{m['Jsc']:.2f}")
        c4.metric("FF", f"{m['FF']:.3f}")
        fig = go.Figure(go.Scatter(x=r["V"], y=-np.array(r["J_mA"]),
                                   mode="lines+markers",
                                   line=dict(color="#1E2761", width=3)))
        fig.add_hline(y=0, line=dict(color="grey", width=1))
        fig.update_layout(xaxis_title="Voltage (V)", yaxis_title="J (mA/cm²)",
                          height=380, title=f"Imported: {r['stack']}")
        st.plotly_chart(fig, width='stretch')
        st.caption("Note: a raw .def carries no optical file, so the import path "
                   "applies no front-stack optical losses — expect higher Jsc than "
                   "the calibrated technology baselines.")

# ── energy yield ────────────────────────────────────────────────────────────
with tab_yield:
    st.subheader("Energy yield under real operating conditions")
    st.caption(
        "STC efficiency is one point; deployment delivers ∫P(G,T)dt. Temperature "
        "dependence comes from the SOLVER itself (evaluated at anchor temperatures); "
        "irradiance scaling uses the standard one-diode relations (Green, *Solar "
        "Cells*, 1982) with the NOCT cell-temperature model (IEC 61215). All "
        "assumptions are printed with the result."
    )
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import fast_simulate
    from physics.energy_yield import energy_yield, clear_sky_day, intensity_temperature_map

    c1, c2, c3 = st.columns(3)
    absorber = c1.selectbox("Absorber", [k for k in PEROVSKITE_DB
                                         if k != "CdTe"], index=0)
    T_amb = c2.slider("Ambient temperature (°C)", -5, 45, 25, 1)
    peak = c3.slider("Peak irradiance (W/m²)", 400, 1100, 1000, 50)

    if st.button("Compute daily energy yield", key="ey"):
        h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB[absorber], ETL_DB["SnO2"]
        sim = lambda T_K: fast_simulate(h, a, e, 150, 500, 50, 1e14, T_K)
        with st.spinner("Evaluating solver at temperature anchors..."):
            r = energy_yield(sim, T_amb_C=float(T_amb),
                             profile=clear_sky_day(peak_W_m2=float(peak)))
            mp = intensity_temperature_map(sim)
        c1, c2, c3 = st.columns(3)
        c1.metric("Daily yield", f"{r['E_day_Wh_m2']:.0f} Wh/m²")
        c2.metric("Insolation", f"{r['insolation_Wh_m2']:.0f} Wh/m²")
        c3.metric("Harvesting efficiency", f"{r['harvesting_efficiency_pct']:.2f}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["hours"], y=r["G_W_m2"], name="irradiance (W/m²)",
                                 line=dict(color="#E8A33D", width=2.5)))
        fig.add_trace(go.Scatter(x=r["hours"], y=r["P_W_m2"], name="output (W/m²)",
                                 line=dict(color="#1E2761", width=2.5)))
        fig.add_trace(go.Scatter(x=r["hours"], y=r["T_cell_C"], name="T_cell (°C)",
                                 yaxis="y2", line=dict(color="#C84B31", dash="dot")))
        fig.update_layout(xaxis_title="hour of day", yaxis_title="W/m²",
                          yaxis2=dict(title="°C", overlaying="y", side="right"),
                          height=420, title="Clear-sky day profile")
        st.plotly_chart(fig, width='stretch')
        st.caption("Assumptions: " + r["assumptions"])
        fig2 = go.Figure(go.Heatmap(z=mp["PCE_pct"], x=[f"{s} sun" for s in mp["suns"]],
                                    y=[f"{t} °C" for t in mp["T_C"]],
                                    colorscale="Viridis",
                                    colorbar=dict(title="PCE %")))
        fig2.update_layout(height=340, title="PCE(intensity, temperature) map")
        st.plotly_chart(fig2, width='stretch')


# ── 📈 Fit measured J-V: parameters with honest uncertainty ────────────────
with tab_fit:
    st.subheader("Fit your measured J-V — with confidence intervals and a "
                 "non-uniqueness report")
    st.caption("The single-diode model (exact Lambert-W solution, Jain & "
               "Kapoor 2004) fitted by multi-start least squares; intervals "
               "by residual bootstrap. Degenerate parameter pairs (the "
               "classic J₀↔n) are flagged, not hidden — the discipline the "
               "SCAPS fitting literature lacks (Saidarsan 2025).")
    from utils.measurement_fit import (fit_measured_jv, parse_jv_csv,
                                       demo_measurement)
    up = st.file_uploader("Upload J-V (two columns: V [V], J [mA/cm²]; "
                          "header optional)", type=["csv", "txt", "dat"])
    use_demo = st.checkbox("No file? Use the built-in demo measurement "
                           "(synthetic device with known ground truth)",
                           value=up is None)
    nboot = st.slider("Bootstrap resamples", 50, 400, 200, 50)
    V = J = truth = None
    if up is not None:
        try:
            V, J = parse_jv_csv(up.getvalue())
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    elif use_demo:
        V, J, truth = demo_measurement()
    if V is not None and st.button("Fit", type="primary"):
        with st.spinner("Fitting + bootstrapping…"):
            fr = fit_measured_jv(V, J, n_bootstrap=int(nboot))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jsc (fit)", f"{fr.metrics['Jsc']:.2f} mA/cm²")
        c2.metric("Voc (fit)", f"{fr.metrics['Voc']*1000:.0f} mV")
        c3.metric("FF (fit)", f"{fr.metrics['FF']*100:.1f} %")
        c4.metric("RMSE", f"{fr.rmse_mA_cm2:.3f} mA/cm²")
        import pandas as _pd
        df = _pd.DataFrame(fr.summary_rows())
        if truth is not None:
            df["Ground truth (demo)"] = [truth[k] for k in
                ("Jph_mA_cm2", "J0_mA_cm2", "n", "Rs_ohm_cm2",
                 "Rsh_ohm_cm2")]
        st.dataframe(df, use_container_width=True, hide_index=True)
        for w in fr.warnings:
            st.warning("⚠️ " + w)
        if not fr.degenerate_pairs:
            st.success("No strong parameter degeneracies detected at this "
                       "noise level.")
        import plotly.graph_objects as _go
        fg = _go.Figure()
        fg.add_scatter(x=fr.V, y=fr.J_meas, mode="markers", name="measured",
                       marker=dict(size=5))
        fg.add_scatter(x=fr.V, y=fr.J_fit, mode="lines", name="diode fit")
        fg.update_layout(height=340, xaxis_title="Voltage (V)",
                         yaxis_title="J (mA/cm²)")
        st.plotly_chart(fg, use_container_width=True)
        import json as _json
        st.download_button("Download fit report (JSON)", _json.dumps({
            "params": fr.params, "ci_low": fr.ci_low, "ci_high": fr.ci_high,
            "rmse_mA_cm2": fr.rmse_mA_cm2, "metrics": fr.metrics,
            "degenerate_pairs": fr.degenerate_pairs,
            "n_bootstrap": fr.n_bootstrap, "warnings": fr.warnings},
            indent=1), "jv_fit_report.json")


# ── ☀️ Energy yield across all four technologies ───────────────────────────
with tab_ay:
    st.subheader("Annual energy yield — all four technologies")
    st.caption("Temperature coefficients are ENGINE-DERIVED (each physics "
               "engine run at 288/318 K), cell temperature via the NOCT "
               "model, linear power correlation per Skoplaki & Palyvos "
               "2009 (registry: skoplaki_2009_solen). Clear-sky synthetic "
               "profiles — a comparative estimate, not a bankability "
               "simulation. Note: the organic engine's Voc model has no "
               "explicit T-dependence, so its coefficient reflects "
               "transport terms only.")
    from utils.energy_yield import annual_yield, TECHNOLOGIES, CLIMATES
    cA, cB = st.columns(2)
    with cA:
        ay_tech = st.selectbox("Technology", list(TECHNOLOGIES))
    with cB:
        ay_clim = st.selectbox("Climate", list(CLIMATES))
    if st.button("Compute annual yield", type="primary"):
        with st.spinner("Running engines at multiple temperatures…"):
            ay = annual_yield(ay_tech, ay_clim)
        k1, k2, k3 = st.columns(3)
        k1.metric("Specific yield", f"{ay['kWh_per_kWp_year']:.0f} kWh/kWp·yr")
        k2.metric("Performance ratio (T losses)",
                  f"{ay['performance_ratio']*100:.1f} %")
        k3.metric("Engine-derived γ", f"{ay['gamma_pct_per_K']:.2f} %/K")
        import plotly.graph_objects as _go
        ds = ay["day_series"][0]
        figy = _go.Figure()
        figy.add_scatter(x=ds["t_h"], y=ds["P_per_kWp"], name="P per kWp",
                         mode="lines")
        figy.add_scatter(x=ds["t_h"], y=[c/100 for c in ds["Tcell_C"]],
                         name="T_cell/100 (°C)", mode="lines",
                         line=dict(dash="dot"))
        figy.update_layout(height=320, xaxis_title="hour of day",
                           title="Representative summer day (season 1)")
        st.plotly_chart(figy, use_container_width=True)
        st.caption(ay["note"])

# ── ⏳ Stability tools ──────────────────────────────────────────────────────
with tab_stab:
    st.subheader("Stability tools")
    st.markdown("**1 · Scan-rate hysteresis index** — reduced-order model "
                "with the transient-solver shape of Richardson 2016 "
                "(registry: richardson_2016_ees): HI vanishes at fast and "
                "slow limits and peaks when the scan time matches the ionic "
                "relaxation time. Design guidance, not a transient solver.")
    from utils.stability import hysteresis_curve, t80_report
    import numpy as _np
    h1c, h2c = st.columns(2)
    with h1c:
        tau_ui = st.slider("Ionic relaxation time τ_ion (s)", 0.5, 100.0, 10.0)
    with h2c:
        nion_ui = 10 ** st.slider("Mobile-ion density log₁₀(N, cm⁻³)",
                                  15.0, 18.5, 17.0)
    rates = _np.logspace(-4, 3, 80)
    hc = hysteresis_curve(rates, tau_ion_s=float(tau_ui),
                          N_ion_cm3=float(nion_ui))
    import plotly.graph_objects as _go
    fh = _go.Figure()
    fh.add_scatter(x=hc["rates_V_per_s"], y=hc["HI"], mode="lines")
    fh.update_layout(height=300, xaxis_type="log",
                     xaxis_title="scan rate (V/s)",
                     yaxis_title="hysteresis index")
    st.plotly_chart(fh, use_container_width=True)

    st.markdown("**2 · Arrhenius T80 lifetime projection** — T80 per the "
                "ISOS consensus (Khenkin 2020, registry: "
                "khenkin_2020_natenergy). Enter measured T80 at ≥2 stress "
                "temperatures; the tool fits E_a and projects to operating "
                "temperature, reporting fit quality and extrapolation "
                "distance honestly.")
    d1, d2, d3 = st.columns(3)
    with d1:
        t85 = st.number_input("T80 @ 85 °C (h)", 1.0, 1e6, 500.0)
    with d2:
        t65 = st.number_input("T80 @ 65 °C (h)", 1.0, 1e7, 2200.0)
    with d3:
        top_c = st.number_input("Operating temperature (°C)", 15.0, 60.0, 35.0)
    rep = t80_report([85.0, 65.0], [float(t85), float(t65)],
                     T_op_C=float(top_c))
    e1, e2, e3 = st.columns(3)
    e1.metric("Activation energy", f"{rep['Ea_eV']:.2f} eV")
    e2.metric("Projected T80", f"{rep['t80_projected_h']:.0f} h")
    e3.metric("≈ years", f"{rep['t80_projected_years']:.1f}")
    for w in rep["warnings"]:
        st.warning("⚠️ " + w)
