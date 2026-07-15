"""Organic (BHJ) technology family: fullerene + NFA blends."""
import sys, os, json, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Organic Technology", page_icon="🟢", layout="wide")
st.title("🟢 Organic Solar Cell Technology")
st.caption(
    "Semi-empirical BHJ engine: AM1.5G spectral EQE integration, Scharber-style "
    "Voc = (Eg − E_loss)/q, Koster-type drift-collection competition and full "
    "single-diode J-V. Calibrated against certified devices from PM6:Y6 15.7% "
    "(Yuan 2019) through PM6:L8-BO-C4 20.4% (Li 2025) with ≤1.3% PCE error, and "
    "reproduces the measured PM6:Y6 thickness roll-off. Limitation stated "
    "honestly: no transfer-matrix interference or exciton/CT drift-diffusion — "
    "for those, export and cross-check in OghmaNano (Datasets & Interop page)."
)

from physics.organic import ORGANIC_PRESETS, OrganicBlend, simulate_organic

tab_bench, tab_explore, tab_tmm, tab_build, tab_opt = st.tabs(
    ["📊 Certified-device benchmarks", "🔧 Design explorer",
     "🌊 Wave optics (TMM)", "🤖 AI optimization", "🧱 Blend builder"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))

with tab_bench:
    st.subheader("Model vs certified published devices")
    rows = []
    for b in DB["benchmarks"]["organic"]:
        blend = ORGANIC_PRESETS[b["preset"]]
        if "override" in b:
            blend = dataclasses.replace(blend, **b["override"])
        r = simulate_organic(blend)
        ref = DB["references"][b["reference"]]
        err = abs(r["PCE"] - b["target"]["PCE"]) / b["target"]["PCE"] * 100
        rows.append({
            "Benchmark": b["id"], "Published PCE (%)": b["target"]["PCE"],
            "Model PCE (%)": round(r["PCE"], 2), "Error (%)": round(err, 2),
            "Voc (V)": round(r["Voc"], 3), "Jsc (mA/cm²)": round(r["Jsc"], 2),
            "FF (%)": round(r["FF"] * 100, 1), "DOI": ref["doi"],
        })
    st.dataframe(rows, use_container_width=True)

with tab_explore:
    st.subheader("Blend explorer")
    c0, c1 = st.columns([1, 2])
    with c0:
        preset_name = st.selectbox("Blend preset", list(ORGANIC_PRESETS))
        base = ORGANIC_PRESETS[preset_name]
        L = st.slider("Active layer thickness (nm)", 60, 350, int(base.L_nm))
        Eloss = st.slider("Energy loss Eg − qVoc (eV)", 0.40, 1.30,
                          float(base.E_loss_eV), 0.005)
        mu = 10 ** st.slider("log₁₀ μ_eff (cm²/Vs)", -5.0, -2.5,
                             float(np.log10(base.mu_eff_cm2Vs)), 0.05)
        blend = dataclasses.replace(base, L_nm=L, E_loss_eV=Eloss,
                                    mu_eff_cm2Vs=mu)
        r = simulate_organic(blend)
        m1, m2 = st.columns(2)
        m1.metric("PCE", f"{r['PCE']:.2f}%")
        m2.metric("Voc", f"{r['Voc']:.3f} V")
        m3, m4 = st.columns(2)
        m3.metric("Jsc", f"{r['Jsc']:.2f} mA/cm²")
        m4.metric("FF", f"{r['FF']*100:.1f}%")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=r["voltages"], y=r["currents"], name="J-V",
                        line=dict(width=3))
        fig.update_layout(title="J-V curve", xaxis_title="Voltage (V)",
                          yaxis_title="J (mA/cm²)", height=320)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_scatter(x=r["lams_qe"], y=r["qe"], name="EQE")
        fig2.update_layout(title="EQE", xaxis_title="Wavelength (nm)",
                           yaxis_title="EQE (%)", height=290)
        st.plotly_chart(fig2, use_container_width=True)

    # thickness roll-off curve
    Ls = np.linspace(60, 350, 30)
    pces = [simulate_organic(dataclasses.replace(base, L_nm=float(l)))["PCE"]
            for l in Ls]
    fig3 = go.Figure(go.Scatter(x=Ls, y=pces, mode="lines"))
    fig3.update_layout(title="Thickness roll-off (validated vs Yuan 2019 for PM6:Y6)",
                       xaxis_title="Active layer thickness (nm)",
                       yaxis_title="PCE (%)", height=280)
    st.plotly_chart(fig3, use_container_width=True)

    from utils.interop import export_jv_csv, export_device_json
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ J-V CSV", export_jv_csv(r), "organic_jv.csv")
    d2.download_button("⬇️ Device JSON (interop)",
                       export_device_json("organic",
                                          params=dataclasses.asdict(blend),
                                          result=r),
                       "organic_device.json")

with tab_tmm:
    st.subheader("Transfer-matrix wave optics — the Lumerical-class check")
    st.caption(
        "Rigorous coherent optics (Pettersson 1999 / Burkhard 2010 formalism): "
        "for planar stacks this is the same Maxwell solution Lumerical STACK "
        "computes. Validated 9/9 in docs/CROSS_TOOL_VALIDATION.md — energy "
        "conservation <1e-4, Fresnel limit exact, published P3HT:PCBM "
        "interference structure and certified PM6:Y6 Jsc within 5%."
    )
    from physics.tmm import solve_stack, jsc_vs_thickness, available_nk_materials
    from physics.organic import TMM_STACKS
    blend_key = st.selectbox(
        "Blend with n,k dataset",
        [k for k in ORGANIC_PRESETS if ORGANIC_PRESETS[k].name in TMM_STACKS])
    bsel = ORGANIC_PRESETS[blend_key]
    tmpl, iqe0 = TMM_STACKS[bsel.name]
    # ── layer-material selection (cited n,k for every choice) ──────────
    from physics.layer_library import (ORGANIC_INTERLAYERS, ORGANIC_METALS,
                                       build_organic_stack,
                                       organic_combination_status)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        il_key = st.selectbox("Interlayer", list(ORGANIC_INTERLAYERS))
    with lc2:
        metal_key = st.selectbox("Top electrode", list(ORGANIC_METALS))
    with lc3:
        ito_nm = st.slider("ITO thickness (nm)", 60, 180, 100, 5)
    L_tmm = st.slider("Active thickness (nm)", 40, 320, int(bsel.L_nm), key="Ltmm")
    iqe = st.slider("Internal quantum efficiency", 0.5, 1.0, float(iqe0), 0.01)
    nk_name = bsel.name.replace("PC61BM", "PCBM") if "P3HT" in bsel.name else bsel.name
    # ── measured n,k upload (ellipsometry CSV): replace the library data
    # for the active layer with YOUR batch's optical constants ──────────
    with st.expander("📤 Use my measured n,k for the active layer "
                     "(ellipsometry CSV: wavelength_nm, n, k)"):
        st.caption("Answers the ±5-10% batch spread of organic blends "
                   "(Kerremans 2020): load your own measurement instead of "
                   "the library value. Validated on upload (400-800 nm "
                   "coverage, n>0, k≥0); solved with the same "
                   "energy-conservation guarantee as built-ins. "
                   "Session-scoped — the cited database is never modified.")
        nk_up = st.file_uploader("n,k CSV", type=["csv", "txt", "dat"],
                                 key="nk_upload")
        if nk_up is not None:
            from physics.tmm_custom import load_nk_csv
            try:
                ck = load_nk_csv(nk_up.name.rsplit(".", 1)[0],
                                 nk_up.getvalue())
                st.session_state["custom_nk_key"] = ck
                st.success(f"Registered `{ck}` — now used as the active "
                           "layer in this tab.")
            except ValueError as e:
                st.error(f"Rejected: {e}")
        if st.session_state.get("custom_nk_key") and st.button(
                "Revert to library n,k"):
            st.session_state.pop("custom_nk_key")
    if st.session_state.get("custom_nk_key"):
        nk_name = st.session_state["custom_nk_key"]
        st.info(f"Active layer optics: **{nk_name}** (user-measured; "
                "library value bypassed).")
    stack = build_organic_stack(nk_name, L_tmm, interlayer_key=il_key,
                                metal_key=metal_key, ito_nm=ito_nm)
    stat, sdetail = organic_combination_status(nk_name, il_key, metal_key)
    (st.success if stat == "CERTIFIED-STACK" else st.warning)(
        ("✅ " if stat == "CERTIFIED-STACK" else "⚠️ ") + stat + " — " + sdetail)
    sol = solve_stack(stack)
    active_idx = next(i for i, (nm_, _) in enumerate(stack)
                      if nm_ in ("PM6:Y6", "P3HT:PCBM"))
    import plotly.graph_objects as go2
    figA = go2.Figure()
    for i, name in enumerate(sol["layers"]):
        figA.add_scatter(x=sol["lam_nm"], y=sol["A"][i] * 100, name=f"A: {name}",
                         stackgroup="one")
    figA.add_scatter(x=sol["lam_nm"], y=sol["R"] * 100, name="R (incl. glass face)",
                     line=dict(dash="dot", color="black"))
    figA.update_layout(title="Per-layer absorptance (parasitic losses visible)",
                       xaxis_title="Wavelength (nm)", yaxis_title="%", height=340)
    st.plotly_chart(figA, use_container_width=True)

    ds = list(range(40, 325, 5))
    _, js = jsc_vs_thickness(stack, sol["layers"][active_idx], ds, IQE=iqe)
    figB = go2.Figure(go2.Scatter(x=ds, y=js, mode="lines"))
    figB.update_layout(title="Jsc vs active thickness — interference oscillation "
                             "(the classic TMM/Lumerical signature)",
                       xaxis_title="Active thickness (nm)",
                       yaxis_title="Jsc (mA/cm²)", height=300)
    st.plotly_chart(figB, use_container_width=True)

    r_tmm = simulate_organic(bsel, optics="tmm")
    r_cal = simulate_organic(bsel)
    c1, c2, c3 = st.columns(3)
    c1.metric("PCE (TMM optics)", f"{r_tmm['PCE']:.2f}%")
    c2.metric("PCE (calibrated optics)", f"{r_cal['PCE']:.2f}%")
    dev = abs(r_tmm["PCE"] - r_cal["PCE"]) / r_cal["PCE"] * 100
    c3.metric("Cross-path deviation", f"{dev:.1f}%")

with tab_opt:
    st.subheader("Differential-evolution optimization")
    preset_name2 = st.selectbox("Baseline blend", list(ORGANIC_PRESETS),
                                key="opt_blend")
    iters = st.slider("DE iterations", 5, 40, 12, key="opt_it")
    if st.button("🚀 Optimize", type="primary"):
        from ai.multi_tech_optimizer import optimize
        with st.spinner("Optimizing..."):
            res = optimize("organic",
                           {"L_nm": (60, 300), "E_loss_eV": (0.45, 0.9),
                            "mu_eff_cm2Vs": (1e-4, 3e-3)},
                           ORGANIC_PRESETS[preset_name2], maxiter=iters)
        st.success(f"Best PCE: {res['best_value']:.2f}% "
                   f"({res['n_evaluations']} evaluations)")
        st.json(res["best_params"])
        if res.get("sensitivity"):
            sens = res["sensitivity"]
            fig = go.Figure(go.Bar(x=list(sens.values()), y=list(sens.keys()),
                                   orientation="h"))
            fig.update_layout(title="Parameter sensitivity", height=240)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: E_loss below ~0.5 eV is at the radiative frontier of "
                   "current materials — treat optimizer excursions there as "
                   "targets for material discovery, not free parameters.")


# ── 🧱 Blend builder: donor:acceptor selection with published anchoring ────
with tab_build:
    from physics.layer_library import (build_organic_blend, ORGANIC_DONORS,
                                       ORGANIC_ACCEPTORS)
    st.subheader("Build an organic cell from donor + acceptor")
    st.caption("Certified pairs return the benchmarked preset EXACTLY "
               "(validated against the certified record). Novel pairs use "
               "the Scharber design rules (Adv. Mater. 2006, registry: "
               "scharber_2006_advmater) with sourced component energetics "
               "— always flagged EXTRAPOLATED. Components without sourced "
               "energetics are offered only in their certified pairing.")
    b1, b2 = st.columns(2)
    with b1:
        dk = st.selectbox("Donor", list(ORGANIC_DONORS))
        st.caption(ORGANIC_DONORS[dk]["note"] + "  [refs: "
                   + ", ".join(ORGANIC_DONORS[dk]["refs"]) + "]")
    with b2:
        ak = st.selectbox("Acceptor", list(ORGANIC_ACCEPTORS))
        st.caption(ORGANIC_ACCEPTORS[ak]["note"] + "  [refs: "
                   + ", ".join(ORGANIC_ACCEPTORS[ak]["refs"]) + "]")
    try:
        blend_b, status_b, msg_b = build_organic_blend(dk, ak)
    except ValueError as e:
        st.error(str(e))
        blend_b = None
    if blend_b is not None:
        (st.success if status_b == "CERTIFIED" else st.warning)(
            ("✅ " if status_b == "CERTIFIED" else "⚠️ ") + status_b
            + " — " + msg_b)
        rb = simulate_organic(blend_b)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PCE", f"{rb['PCE']:.2f} %")
        c2.metric("Voc", f"{rb['Voc']*1000:.0f} mV")
        c3.metric("Jsc", f"{rb['Jsc']:.2f} mA/cm²")
        c4.metric("FF", f"{rb['FF']*100:.1f} %")
        st.caption(f"Eg,opt = {blend_b.Eg_opt_eV:.2f} eV · "
                   f"E_loss = {blend_b.E_loss_eV:.2f} eV · "
                   f"EQE plateau = {blend_b.EQE_max:.2f} · "
                   f"reference: {blend_b.reference}")
        it_b = st.slider("DE iterations", 4, 30, 10, key="build_opt_it")
        if st.button("🚀 Run AI optimization on this blend", type="primary"):
            from ai.multi_tech_optimizer import optimize
            with st.spinner("Optimizing thickness / energy loss / "
                            "mobility…"):
                res = optimize("organic",
                               {"L_nm": (60, 300),
                                "E_loss_eV": (0.45, 0.9),
                                "mu_eff_cm2Vs": (1e-4, 3e-3)},
                               blend_b, maxiter=it_b)
            st.success(f"Best PCE: {res['best_value']:.2f}% "
                       f"({res['n_evaluations']} evaluations) — baseline "
                       f"{rb['PCE']:.2f}%")
            st.json(res["best_params"])
            if status_b == "EXTRAPOLATED":
                st.info("Optimum of an ESTIMATED blend: treat as a "
                        "material-screening target, and verify any "
                        "promising pair with a fabricated device or a "
                        "certified benchmark before relying on it.")
