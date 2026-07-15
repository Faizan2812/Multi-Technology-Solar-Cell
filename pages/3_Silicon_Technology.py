"""Silicon technology family: Al-BSF / PERC / TOPCon / SHJ / SHJ-IBC."""
import sys, os, json, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Silicon Technology", page_icon="🔷", layout="wide")
st.title("🔷 Crystalline Silicon Technology")
st.caption(
    "Physics-based c-Si engine: Lambertian light trapping (Tiedje & Yablonovitch 1984), "
    "Green (2008) α(λ) table, Richter (2012) / Niewelt (2022) intrinsic recombination, "
    "implied-Voc balance (Kerr & Cuevas 2002), Green (1981) FF and a full single-diode "
    "J-V. All five architecture presets are validated against certified devices — "
    "SHJ-IBC 26.7% (Yoshikawa 2017), SHJ 26.81% (Lin 2023), TOPCon 26.0% (Richter 2021) — "
    "with ≤2% PCE error. DOIs on the 📚 References page."
)

from physics.silicon import (SILICON_PRESETS, SiliconArchitecture,
                             simulate_silicon)

tab_bench, tab_explore, tab_layers, tab_opt = st.tabs(
    ["📊 Certified-device benchmarks", "🔧 Design explorer",
     "🧱 Layer builder", "🤖 AI optimization"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))

with tab_bench:
    st.subheader("Model vs certified published devices")
    rows = []
    for b in DB["benchmarks"]["silicon"]:
        arch = SILICON_PRESETS[b["preset"]]
        r = simulate_silicon(arch)
        ref = DB["references"][b["reference"]]
        err = abs(r["PCE"] - b["target"]["PCE"]) / b["target"]["PCE"] * 100
        rows.append({
            "Preset": b["preset"], "Published PCE (%)": b["target"]["PCE"],
            "Model PCE (%)": round(r["PCE"], 2), "Error (%)": round(err, 2),
            "Voc (mV)": round(r["Voc"] * 1000, 1), "Jsc (mA/cm²)": round(r["Jsc"], 2),
            "FF (%)": round(r["FF"] * 100, 2), "DOI": ref["doi"],
        })
    st.dataframe(rows, use_container_width=True)
    st.info("Every row is a certified, fabricated record device from the peer-reviewed "
            "literature — not another group's simulation. Full citations with "
            "verification status: 📚 References page.")

with tab_explore:
    st.subheader("Architecture explorer")
    c0, c1 = st.columns([1, 2])
    with c0:
        preset_name = st.selectbox("Start from preset", list(SILICON_PRESETS))
        base = SILICON_PRESETS[preset_name]
        W = st.slider("Wafer thickness (µm)", 40, 300, int(base.W_um))
        Ndop = 10 ** st.slider("Base doping log₁₀(N) [cm⁻³]", 14.0, 17.0,
                               float(np.log10(base.Ndop_cm3)), 0.1)
        tau = st.slider("Bulk SRH lifetime (ms)", 0.05, 60.0,
                        float(base.tau_srh_ms))
        J0s = st.slider("Total surface J₀ₛ (fA/cm²)", 0.5, 400.0,
                        float(base.J0s_fA))
        Rs = st.slider("Series resistance (Ω·cm²)", 0.05, 1.5,
                       float(base.Rs_ohm_cm2))
        auger = st.radio("Intrinsic recombination model",
                         ["richter2012", "niewelt2022"], horizontal=True)
        arch = dataclasses.replace(base, W_um=W, Ndop_cm3=Ndop,
                                   tau_srh_ms=tau, J0s_fA=J0s, Rs_ohm_cm2=Rs)
        r = simulate_silicon(arch, auger_model=auger)
        m1, m2 = st.columns(2)
        m1.metric("PCE", f"{r['PCE']:.2f}%")
        m2.metric("Voc", f"{r['Voc']*1000:.1f} mV")
        m3, m4 = st.columns(2)
        m3.metric("Jsc", f"{r['Jsc']:.2f} mA/cm²")
        m4.metric("FF", f"{r['FF']*100:.2f}%")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=r["voltages"], y=r["currents"], name="J-V",
                        line=dict(width=3))
        fig.add_scatter(x=[r["Vmpp"]], y=[r["Jmpp"]], mode="markers",
                        name="MPP", marker=dict(size=12, symbol="star"))
        fig.update_layout(title="J-V curve", xaxis_title="Voltage (V)",
                          yaxis_title="Current density (mA/cm²)", height=330)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_scatter(x=r["lams_qe"], y=r["qe"], name="EQE")
        fig2.update_layout(title="External quantum efficiency",
                           xaxis_title="Wavelength (nm)",
                           yaxis_title="EQE (%)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    from utils.interop import export_jv_csv, export_device_json
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ J-V CSV", export_jv_csv(r), "silicon_jv.csv")
    d2.download_button("⬇️ Device JSON (interop)",
                       export_device_json("silicon",
                                          params=dataclasses.asdict(arch),
                                          result=r),
                       "silicon_device.json")

with tab_opt:
    st.subheader("Differential-evolution optimization")
    st.caption("Global search over wafer/passivation parameters with a "
               "random-forest sensitivity ranking of what matters most.")
    preset_name2 = st.selectbox("Baseline preset", list(SILICON_PRESETS),
                                key="opt_preset")
    iters = st.slider("DE iterations", 5, 40, 12)
    if st.button("🚀 Optimize", type="primary"):
        from ai.multi_tech_optimizer import optimize
        with st.spinner("Optimizing..."):
            res = optimize("silicon",
                           {"W_um": (60, 250), "Ndop_cm3": (5e14, 5e16),
                            "tau_srh_ms": (0.5, 50), "J0s_fA": (1, 50)},
                           SILICON_PRESETS[preset_name2], maxiter=iters)
        st.success(f"Best PCE: {res['best_value']:.2f}% "
                   f"({res['n_evaluations']} device evaluations)")
        st.json(res["best_params"])
        if res.get("sensitivity"):
            sens = res["sensitivity"]
            fig = go.Figure(go.Bar(x=list(sens.values()), y=list(sens.keys()),
                                   orientation="h"))
            fig.update_layout(title="Parameter sensitivity (RF importance)",
                              height=280)
            st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(go.Scatter(y=res["history_best"], mode="lines"))
        fig.update_layout(title="Best-so-far PCE vs evaluation",
                          xaxis_title="Evaluation", yaxis_title="PCE (%)",
                          height=260)
        st.plotly_chart(fig, use_container_width=True)


# ── 🧱 Layer builder: material selection per side, cited ───────────────────
with tab_layers:
    from physics.layer_library import (
        SILICON_FRONT_STACKS, SILICON_REAR_STACKS, SILICON_TEXTURES,
        build_silicon_from_layers, silicon_combination_status,
        silicon_budget_rows)
    st.subheader("Build a silicon cell layer by layer")
    st.caption("Pick the front and rear passivation/contact stacks — every "
               "entry carries per-layer J₀/Rs contributions from the cited "
               "literature, and certified pairs reconstruct the record "
               "devices exactly (enforced by tests/test_layer_library.py).")
    c1, c2, c3 = st.columns(3)
    with c1:
        fkey = st.selectbox("Front stack", list(SILICON_FRONT_STACKS))
    with c2:
        rkey = st.selectbox("Rear stack", list(SILICON_REAR_STACKS))
    with c3:
        tkey = st.selectbox("Texture / light trapping",
                            ["(inherit from certified device)"] +
                            list(SILICON_TEXTURES))
    c4, c5, c6 = st.columns(3)
    with c4:
        W = st.slider("Wafer thickness (µm)", 60, 300, 165)
    with c5:
        tau = st.slider("Bulk SRH lifetime τ (ms)", 0.2, 50.0, 15.0)
    with c6:
        Nd = 10 ** st.slider("Base doping log₁₀(N, cm⁻³)", 14.5, 16.7, 15.5)

    status, detail = silicon_combination_status(fkey, rkey)
    if status == "CERTIFIED":
        st.success(f"✅ CERTIFIED pair — reconstructs **{detail}** "
                   "(benchmarked in the 14-device suite).")
    else:
        st.warning(f"⚠️ EXTRAPOLATED pair — {detail}")

    arch = build_silicon_from_layers(
        fkey, rkey,
        texture_key=None if tkey.startswith("(inherit") else tkey,
        W_um=float(W), tau_srh_ms=float(tau), Ndop_cm3=float(Nd))
    res = simulate_silicon(arch)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PCE", f"{res['PCE']:.2f} %")
    m2.metric("Voc", f"{res['Voc']*1000:.1f} mV")
    m3.metric("Jsc", f"{res['Jsc']:.2f} mA/cm²")
    m4.metric("FF", f"{res['FF']*100:.2f} %")
    st.caption(f"Stack totals: J₀ₛ = {arch.J0s_fA:.1f} fA/cm² · "
               f"Rs = {arch.Rs_ohm_cm2:.2f} Ω·cm² · Z = {arch.Z_path} "
               f"(29.4% thermodynamic guard enforced in the engine).")
    import pandas as _pd
    st.markdown("**Per-layer budget (with references)**")
    st.dataframe(_pd.DataFrame(silicon_budget_rows(fkey, rkey)),
                 use_container_width=True, hide_index=True)
    import plotly.graph_objects as _go
    figL = _go.Figure()
    figL.add_trace(_go.Scatter(x=res["voltages"], y=res["currents"],
                               mode="lines", name="layer-built device"))
    figL.update_layout(height=330, xaxis_title="Voltage (V)",
                       yaxis_title="J (mA/cm²)",
                       title="J-V of the layer-built device")
    st.plotly_chart(figL, use_container_width=True)
