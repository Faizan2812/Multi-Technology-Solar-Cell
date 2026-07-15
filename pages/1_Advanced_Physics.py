"""Advanced physics : production TMM-in-DD optics, temperature, defects, COUPLED ion hysteresis, Rs/Rsh, impedance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics.advanced import optics_tmm as tmm, temperature as tmod, defects as dfx, \
    ion_migration as ion, impedance as imp

st.set_page_config(page_title="Advanced Physics", page_icon="🔬", layout="wide")
st.title("🔬 Advanced physics ")
st.caption("Capabilities added to close the gaps identified in the integrity assessment. "
           "Each panel is backed by a tested module under physics/advanced.")

tab = st.tabs(["Optics (TMM → DD)", "Temperature", "Defects", "Ion hysteresis (coupled DD)", "Rs / Rsh circuit", "Impedance / C–V"])

with tab[0]:
    st.subheader("Transfer-matrix generation — wired into the drift-diffusion solver ")
    st.caption("This page computes the coherent glass/ITO/ETL/absorber/HTL/Au field profile (Pettersson 1999) and feeds "
               "G(x) directly into the production DD solver: `simulate_iv_curve(..., optics='tmm')`.")
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    c1, c2, c3, c4 = st.columns(4)
    htl_n = c1.selectbox("HTL", list(HTL_DB), index=list(HTL_DB).index("Spiro-OMeTAD") if "Spiro-OMeTAD" in HTL_DB else 0)
    abs_n = c2.selectbox("Absorber", list(PEROVSKITE_DB), index=list(PEROVSKITE_DB).index("MAPbI3") if "MAPbI3" in PEROVSKITE_DB else 0)
    etl_n = c3.selectbox("ETL", list(ETL_DB), index=list(ETL_DB).index("SnO2") if "SnO2" in ETL_DB else 0)
    d_abs = c4.slider("Absorber thickness (nm)", 200, 900, 500, 50)
    if st.button("Run DD with TMM vs Beer-Lambert optics (~3 s)", key="tmm_dd"):
        from physics.device import simulate_iv_curve
        h, a, e = HTL_DB[htl_n], PEROVSKITE_DB[abs_n], ETL_DB[etl_n]
        with st.spinner("Running two full drift-diffusion J-V sweeps..."):
            r_bl = simulate_iv_curve(h, a, e, 150, d_abs, 50, None, 300, mode="dd")
            r_tm = simulate_iv_curve(h, a, e, 150, d_abs, 50, None, 300, mode="dd", optics="tmm")
        cA, cB = st.columns(2)
        cA.metric("Beer-Lambert PCE", f"{r_bl['PCE']:.2f} %", help=f"Jsc {r_bl['Jsc']:.2f}, Voc {r_bl['Voc']:.3f}, FF {r_bl['FF']:.3f}")
        cB.metric("TMM PCE", f"{r_tm['PCE']:.2f} %",
                  delta=f"{r_tm['PCE']-r_bl['PCE']:+.2f} vs BL",
                  help=f"Jsc {r_tm['Jsc']:.2f}, Voc {r_tm['Voc']:.3f}, FF {r_tm['FF']:.3f}")
        od = r_tm.get("optics_diagnostics") or {}
        st.caption(f"TMM optical Jsc bound: {od.get('Jsc_optical_mA_cm2', float('nan')):.2f} mA/cm2 | "
                   f"energy-conservation residual: {od.get('energy_conservation_max_residual', float('nan')):.1e} "
                   f"(machine precision) | model: {od.get('model','')}")
        fig = go.Figure()
        xg = r_tm["profiles"]["x"] * 1e4 if "profiles" in r_tm else None
        fig.add_trace(go.Scatter(x=r_bl["profiles"]["x"]*1e4, y=r_bl["profiles"]["G"], name="Beer-Lambert G(x)", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=r_tm["profiles"]["x"]*1e4, y=r_tm["profiles"]["G"], name="TMM G(x)", line=dict(width=3, dash="dash")))
        fig.update_layout(xaxis_title="x (um)", yaxis_title="G (cm-3 s-1)", height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width='stretch')
    st.info("TMM captures interference, glass/ITO front reflection and parasitic absorption "
            "that Beer-Lambert misses; energy is conserved to machine precision (validated in tests/test_v3_upgrades.py).")

with tab[1]:
    st.subheader("Temperature-dependent material physics")
    mat = st.selectbox("Material (Varshni)", list(tmod.VARSHNI))
    Ts = np.linspace(250, 400, 60); p = tmod.VARSHNI[mat]
    Eg = tmod.eg_varshni(Ts, **p)
    nis = np.array([tmod.ni(T, 2.2e18, 1.8e19, **p) for T in Ts])
    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Scatter(x=Ts, y=Eg, line=dict(color="#C00000", width=3)))
    f1.update_layout(title="Band gap Eg(T)", xaxis_title="T (K)", yaxis_title="Eg (eV)", height=320)
    c1.plotly_chart(f1, width='stretch')
    f2 = go.Figure(go.Scatter(x=Ts, y=nis, line=dict(color="#2E7D32", width=3)))
    f2.update_layout(title="Intrinsic density ni(T)", xaxis_title="T (K)", yaxis_title="ni (cm⁻³)",
                     yaxis_type="log", height=320)
    c2.plotly_chart(f2, width='stretch')

with tab[2]:
    st.subheader("Defect distributions (multi-level / Gaussian)")
    c1, c2 = st.columns(2)
    Nt = c1.number_input("Trap density Nt (cm⁻³)", 1e13, 1e18, 1e15, format="%.1e")
    sigmaE = c2.slider("Gaussian width σ_E (eV)", 0.001, 0.2, 0.05, 0.005)
    n, p_, ni_ = 1e16, 1e14, 1e8
    single = dfx.srh_rate(n, p_, ni_, 0.0, Nt)
    gauss = dfx.gaussian_dos(n, p_, ni_, 0.0, sigmaE, Nt)
    st.metric("Single-level SRH rate", f"{single:.3e} cm⁻³s⁻¹")
    st.metric("Gaussian-band SRH rate", f"{gauss:.3e} cm⁻³s⁻¹")
    st.caption("A narrow Gaussian reduces to the discrete single level (validated).")

with tab[3]:
    st.subheader("J-V hysteresis from ions COUPLED into the Poisson equation ")
    st.caption("The mobile ionic charge is added to the Poisson RHS of the production Scharfetter-Gummel solver (quasi-static "
               "asymptotics of IonMonger/Driftfusion; Eames 2015 D_ion). Hysteresis is now a solver "
               "OUTPUT with exact limits: HI=0 for zero ions and for slow scans.")
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    c1, c2, c3 = st.columns(3)
    N_ion = c1.select_slider("Mobile ion density (cm⁻³)", options=[0.0, 1e16, 1e17, 5e17, 1e18], value=1e17, format_func=lambda v: f"{v:.0e}")
    rate = c2.select_slider("Scan rate (V/s)", options=[1e-4, 1e-2, 0.1, 1.0, 10.0], value=1.0)
    d_abs_i = c3.slider("Absorber (nm)", 300, 700, 500, 50, key="ion_dabs")
    st.warning("Coupled ion solve: expect 1-4 minutes for a full forward+reverse scan.")
    if st.button("Run coupled forward + reverse scans", key="ion_dd"):
        from physics.dd_ion import hysteresis_jv
        h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
        with st.spinner("Self-consistent electronic + ionic solve at every bias..."):
            r = hysteresis_jv(h, a, e, 150, d_abs_i, 50, N_ion=N_ion, scan_rate=rate, N_V=11)
        cA, cB, cC = st.columns(3)
        cA.metric("Hysteresis index", f"{r['hysteresis_index']:+.3f}")
        cB.metric("PCE forward", f"{r['metrics_forward']['PCE']:.2f} %")
        cC.metric("PCE reverse", f"{r['metrics_reverse']['PCE']:.2f} %")
        st.caption(f"ion relaxation fraction f = {r['f_ion_relaxation']:.3f} | tau_ion = {r['tau_ion_s']:.1f} s | "
                   f"t_scan = {r['t_scan_s']:.1f} s | {r['model']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["voltages"], y=-r["J_forward_mA"], name="forward scan", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=r["voltages"], y=-r["J_reverse_mA"], name="reverse scan", line=dict(width=3, dash="dash")))
        fig.update_layout(xaxis_title="V (V)", yaxis_title="J (mA/cm2)", height=380, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width='stretch')
    with st.expander("Quick single-time-constant estimate (simplified, deprecated)"):
        st.caption("Kept for comparison only; superseded by the coupled solver above.")

with tab[4]:
    st.subheader("External series / shunt resistance on the DD J-V ")
    st.caption("SCAPS-style post-processing: V_ext = V_int + J*Rs, J += V/Rsh. Ideal limits are exact identities (unit-tested).")
    c1, c2 = st.columns(2)
    Rs_v = c1.slider("Rs (Ohm*cm2)", 0.0, 15.0, 3.0, 0.5)
    Rsh_v = c2.select_slider("Rsh (Ohm*cm2)", options=[100.0, 300.0, 1000.0, 1e4, 1e12], value=1000.0, format_func=lambda v: "ideal" if v >= 1e10 else f"{v:.0f}")
    if st.button("Run DD with parasitic resistances (~3 s)", key="rs_dd"):
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from physics.device import simulate_iv_curve
        h, a, e = HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"], ETL_DB["SnO2"]
        with st.spinner("Running..."):
            r0 = simulate_iv_curve(h, a, e, 150, 500, 50, None, 300, mode="dd")
            r1 = simulate_iv_curve(h, a, e, 150, 500, 50, None, 300, mode="dd", Rs=Rs_v, Rsh=Rsh_v)
        cA, cB = st.columns(2)
        cA.metric("Ideal PCE / FF", f"{r0['PCE']:.2f} % / {r0['FF']:.3f}")
        cB.metric("With Rs/Rsh", f"{r1['PCE']:.2f} % / {r1['FF']:.3f}", delta=f"{r1['PCE']-r0['PCE']:+.2f} %")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r0["voltages"], y=-np.array(r0["currents"]), name="ideal", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=r1["voltages"], y=-np.array(r1["currents"]), name=f"Rs={Rs_v}, Rsh={Rsh_v:.0f}", line=dict(width=3, dash="dash")))
        fig.update_layout(xaxis_title="V (V)", yaxis_title="J (mA/cm2)", height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width='stretch')

with tab[5]:
    st.subheader("Mott–Schottky and impedance")
    Na = st.select_slider("Doping N_A (cm⁻³)", [1e15,1e16,1e17,1e18], 1e16)
    ms = imp.mott_schottky(Na, eps_r=6.5, Vbi=0.9); z = imp.impedance()
    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Scatter(x=ms["V"], y=1/ms["C"]**2, line=dict(color="#1F3864", width=3)))
    f1.update_layout(title=f"Mott–Schottky (N_A fit = {ms['Na_fit_cm3']:.1e})",
                     xaxis_title="V (V)", yaxis_title="1/C²", height=320)
    c1.plotly_chart(f1, width='stretch')
    f2 = go.Figure(go.Scatter(x=z["nyq_re"], y=z["nyq_im"], line=dict(color="#C00000", width=3)))
    f2.update_layout(title="Impedance (Nyquist)", xaxis_title="Re(Z) Ω", yaxis_title="−Im(Z) Ω", height=320)
    c2.plotly_chart(f2, width='stretch')
