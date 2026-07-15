"""Tandem designer: perovskite/Si, perovskite/perovskite, perovskite/organic."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Tandem Designer", page_icon="🔺", layout="wide")
st.title("🔺 Tandem Solar Cell Designer")
st.caption(
    "Monolithic (2T) tandems use proper series J-V voltage addition on the common "
    "current axis (not a min-Jsc heuristic), Beer-Lambert spectral filtering of the "
    "bottom cell, textured-optics path enhancement and an interconnect series "
    "resistance. Validated against the certified record lineage: 29.15% "
    "(Al-Ashouri 2020) → 31.25–32.5% (Chin/Mariotti 2023) → 34.58% (Jia 2025), "
    "reproduced to ≤1.3% PCE error at the corresponding interface-quality grades. "
    "DOIs on the 📚 References page."
)

from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
from physics.silicon import SILICON_PRESETS
from physics.tandem import (simulate_perovskite_silicon_tandem,
                            simulate_perovskite_organic_tandem,
                            current_matching_scan)
from physics.organic import ORGANIC_PRESETS

tab_rec, tab_design = st.tabs(["📊 Record-lineage validation", "🔧 Designer"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))

_htl = HTL_DB["2PACz"]
_etl = ETL_DB.get("C60", ETL_DB.get("PCBM"))
_wg = PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"]
_si = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]

with tab_rec:
    st.subheader("Perovskite/Si record lineage (2020 → 2025)")
    if st.button("Run record-lineage validation (~5 s)", type="primary"):
        rows = []
        with st.spinner("Simulating three record generations..."):
            for b in DB["benchmarks"]["tandem"]:
                p = b["params"]
                if "d_top_nm" in p:
                    r = simulate_perovskite_silicon_tandem(
                        _htl, _wg, _etl, p["d_top_nm"], _si,
                        Nt_top=p["Nt_top"], R_int=p["R_int"],
                        parasitic=p["parasitic"],
                        Rs_int_ohm_cm2=p["Rs_int_ohm_cm2"])
                else:
                    r = max((simulate_perovskite_silicon_tandem(
                        _htl, _wg, _etl, float(d), _si,
                        Nt_top=p["Nt_top"], R_int=p["R_int"],
                        parasitic=p["parasitic"],
                        Rs_int_ohm_cm2=p["Rs_int_ohm_cm2"])
                        for d in np.linspace(400, 1100, 8)),
                        key=lambda x: x["PCE"])
                tgt = b["target"].get("PCE", b["target"].get("PCE_range"))
                rows.append({
                    "Grade": b["grade"], "Published PCE": str(tgt),
                    "Model PCE (%)": round(r["PCE"], 2),
                    "Voc (V)": round(r["Voc"], 3),
                    "Jsc (mA/cm²)": round(r["Jsc"], 2),
                    "FF (%)": round(r["FF"] * 100, 1),
                    "DOI": DB["references"][b["reference"]]["doi"],
                })
        st.dataframe(rows, use_container_width=True)
        st.info("Disclosed model trade-off for the 2020 device: PCE and Voc match "
                "closely while the analytical engine runs ~7% low on Jsc and ~6% "
                "high on FF (compensating). Details in VALIDATION_MULTI_TECH.md.")

with tab_design:
    kind = st.radio("Tandem type",
                    ["Perovskite / Silicon", "Perovskite / Organic",
                     "Perovskite / Perovskite "], horizontal=True)
    terminal = st.radio("Terminals", ["2T (monolithic)", "4T (stacked)"],
                        horizontal=True)
    term = "2T" if terminal.startswith("2T") else "4T"

    if kind == "Perovskite / Silicon":
        c0, c1 = st.columns([1, 2])
        with c0:
            top_abs_name = st.selectbox(
                "Top absorber", [k for k in PEROVSKITE_DB
                                 if PEROVSKITE_DB[k].Eg >= 1.5])
            top_abs = PEROVSKITE_DB[top_abs_name]
            htl_name = st.selectbox("Top HTL", list(HTL_DB), index=list(HTL_DB).index("2PACz"))
            etl_name = st.selectbox("Top ETL", list(ETL_DB),
                                    index=list(ETL_DB).index("C60") if "C60" in ETL_DB else 0)
            si_name = st.selectbox("Bottom silicon", list(SILICON_PRESETS))
            d_top = st.slider("Top absorber thickness (nm)", 150, 1200, 600)
            Nt = 10 ** st.slider("Top absorber log₁₀(N_t) [cm⁻³]", 13.0, 16.0, 14.0, 0.1)
            from physics.layer_library import (TANDEM_INTERCONNECTS,
                                               interconnect_params)
            ic_key = st.selectbox("Interconnect (recombination junction)",
                                  list(TANDEM_INTERCONNECTS))
            _ic = TANDEM_INTERCONNECTS[ic_key]
            st.caption(f"{_ic.note}  [refs: {', '.join(_ic.refs)}]")
            Rs_int = st.slider("Interconnect R_s (Ω·cm²)", 0.5, 6.0,
                               float(_ic.Rs_int_ohm_cm2))
            lc_eta = st.slider(
                "Luminescent coupling η_LC (top→bottom photon recycling)",
                0.0, 0.9, 0.0, 0.05,
                help="Fraction of the top cell's excess recombination "
                     "(Jsc,top − Jmpp) returned as bottom-cell photocurrent. "
                     "Reduced one-pass model per Steiner & Geisz 2012; "
                     ">50% of excess pairs shown usable in perovskite/Si "
                     "tandems (Jäger 2020). Not a self-consistent emission "
                     "calculation.")
            r = simulate_perovskite_silicon_tandem(
                HTL_DB[htl_name], top_abs, ETL_DB[etl_name], d_top,
                SILICON_PRESETS[si_name], Nt_top=Nt, terminal=term,
                Rs_int_ohm_cm2=Rs_int, R_int=_ic.R_int,
                parasitic=_ic.parasitic, lc_eta=lc_eta)
            if r.get("lc_dJ_bot_mA_cm2"):
                st.caption(f"Luminescent coupling adds "
                           f"{r['lc_dJ_bot_mA_cm2']:.2f} mA/cm² to the "
                           f"bottom cell (refs: steiner_2012_apl, "
                           f"jager_2020_solrrl).")
            m1, m2 = st.columns(2)
            m1.metric("Tandem PCE", f"{r['PCE']:.2f}%")
            m2.metric("Voc", f"{r['Voc']:.3f} V" if term == "2T" else "—")
            m3, m4 = st.columns(2)
            m3.metric("Top Jsc", f"{r['top']['Jsc']:.2f}")
            m4.metric("Bottom Jsc (filtered)", f"{r['bottom']['Jsc']:.2f}")
        with c1:
            if term == "2T":
                fig = go.Figure()
                fig.add_scatter(x=r["V_axis"], y=r["J_axis"], name="tandem J-V")
                fig.add_scatter(x=r["top"]["voltages"], y=r["top"]["currents"],
                                name="top subcell", line=dict(dash="dot"))
                fig.add_scatter(x=r["bottom"]["voltages"], y=r["bottom"]["currents"],
                                name="bottom subcell", line=dict(dash="dot"))
                fig.update_layout(title=f"2T series J-V — {r['stack']}",
                                  xaxis_title="V (V)", yaxis_title="J (mA/cm²)",
                                  height=340)
                st.plotly_chart(fig, use_container_width=True)
            lams, T_top = r["T_top_spectrum"]
            fig2 = go.Figure(go.Scatter(x=lams, y=T_top * 100))
            fig2.update_layout(title="Spectral transmission to bottom cell",
                               xaxis_title="Wavelength (nm)",
                               yaxis_title="T (%)", height=280)
            st.plotly_chart(fig2, use_container_width=True)

        if term == "2T" and st.button("🔎 Current-matching scan (top thickness)"):
            with st.spinner("Scanning..."):
                rows, best = current_matching_scan(
                    HTL_DB[htl_name], top_abs, ETL_DB[etl_name],
                    SILICON_PRESETS[si_name], Nt_top=Nt,
                    Rs_int_ohm_cm2=Rs_int)
            fig = go.Figure()
            ds = [x["d_top_nm"] for x in rows]
            fig.add_scatter(x=ds, y=[x["Jsc_top"] for x in rows], name="J_top")
            fig.add_scatter(x=ds, y=[x["Jsc_bot"] for x in rows], name="J_bot")
            fig.add_scatter(x=ds, y=[x["PCE"] for x in rows], name="PCE (%)",
                            yaxis="y2")
            fig.update_layout(
                title=f"Current matching — best {best['PCE']:.2f}% at "
                      f"{best['d_top_nm']:.0f} nm",
                xaxis_title="Top thickness (nm)",
                yaxis_title="Jsc (mA/cm²)",
                yaxis2=dict(title="PCE (%)", overlaying="y", side="right"),
                height=340)
            st.plotly_chart(fig, use_container_width=True)

    elif kind == "Perovskite / Organic":
        c0, c1 = st.columns([1, 2])
        with c0:
            top_abs_name = st.selectbox(
                "Top perovskite", [k for k in PEROVSKITE_DB
                                   if PEROVSKITE_DB[k].Eg >= 1.6])
            blend_name = st.selectbox("Bottom organic blend",
                                      list(ORGANIC_PRESETS))
            d_top = st.slider("Top thickness (nm)", 100, 800, 300)
            r = simulate_perovskite_organic_tandem(
                HTL_DB["2PACz"], PEROVSKITE_DB[top_abs_name],
                ETL_DB.get("C60", ETL_DB.get("PCBM")), d_top,
                ORGANIC_PRESETS[blend_name], terminal=term)
            st.metric("Tandem PCE", f"{r['PCE']:.2f}%")
            if term == "2T":
                st.metric("Voc", f"{r['Voc']:.3f} V")
            st.caption("Perovskite/organic tandems are an emerging class "
                       "(no certified >26% device yet); this designer is for "
                       "exploration, not validated record replication.")
        with c1:
            if term == "2T":
                fig = go.Figure(go.Scatter(x=r["V_axis"], y=r["J_axis"]))
                fig.update_layout(title=f"2T J-V — {r['stack']}",
                                  xaxis_title="V (V)",
                                  yaxis_title="J (mA/cm²)", height=340)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Perovskite/perovskite tandems run on the perovskite tandem engine: "
                "`physics.device.simulate_tandem`. Use the main app's Tandem "
                "tab; it shares the same honest Beer-Lambert spectral filter.")
