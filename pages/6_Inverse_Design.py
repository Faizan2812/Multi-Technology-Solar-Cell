"""Inverse design: target-driven parameter prediction, forward-verified by the FDM solver."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="PINN Inverse Design", page_icon="🔁", layout="wide")
st.title("🔁 PINN Inverse Parameter Prediction ")
st.caption(
    "The synopsis (Fig. 3) specifies a physics-informed neural network for INVERSE "
    "parameter prediction. This page implements it literally: the trained conditional "
    "PINN is a differentiable map (x, d_abs, Nt) → (ψ, log n, log p); given observed "
    "profiles, the device parameters are recovered by gradient descent ON THE INPUTS "
    "through the frozen network (Raissi et al., J. Comput. Phys. 378, 686 (2019), "
    "inverse mode)."
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wts = os.path.join(ROOT, "artifacts", "conditional_pinn.pt")

st.info(
    "**Identifiability finding (quantified in `ai/pinn_inverse.py`):** short-circuit "
    "profiles identify GEOMETRY but not recombination — the ground-truth drift-diffusion "
    "profiles change by <10⁻⁴ rel-L₂ across two decades of Nt (≈1600× below the PINN's "
    "7% interpolation floor), because extraction dominates SRH at V = 0. The tool "
    "therefore uses a two-stage inversion: PINN gradients recover the thickness from "
    "profiles; the Voc observable recovers Nt (golden-section on the physics model). "
    "Validated recovery: thickness ≈3% error, Nt exact to 0.00 decades."
)

if not os.path.exists(wts):
    st.warning("Trained conditional-PINN weights not found (artifacts/conditional_pinn.pt). "
               "Run scripts/train_conditional_pinn.py first.")
else:
    try:
        import torch
        from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
        from physics.device import fast_simulate
        from ai.conditional_pinn_torch import ConditionalPINN, DeviceFamily
        from ai.pinn_inverse import invert_with_voc

        st.subheader("Blind-recovery demonstration")
        st.caption(
            "Pick hidden ground-truth parameters. The full drift-diffusion solver "
            "generates the 'measured' profiles and Voc; ONLY those observables are "
            "handed to the inverter, which must recover the parameters."
        )
        c1, c2 = st.columns(2)
        d_true = c1.slider("Hidden thickness d_abs (nm)", 350, 700, 520, 10)
        logNt_true = c2.slider("Hidden log₁₀ Nt (cm⁻³)", 13.2, 15.5, 14.5, 0.1)

        if st.button("Run two-stage inversion", key="inv"):
            with st.spinner("DD forward solve + multi-start PINN gradient inversion..."):
                h, a, e = (HTL_DB['Spiro-OMeTAD'], PEROVSKITE_DB['MAPbI3'],
                           ETL_DB['SnO2'])
                fam = DeviceFamily(h, a, e)
                model = ConditionalPINN()
                model.load_state_dict(torch.load(wts, map_location="cpu"))
                model.eval()
                Nt_true = 10.0 ** logNt_true
                mesh, rsol = fam.dd_solution(float(d_true), Nt_true)
                xn = mesh.x / fam.L_total_cm(float(d_true))
                Voc_obs = fast_simulate(h, a, e, 150, float(d_true), 50,
                                        Nt_true, 300)["Voc"]
                out = invert_with_voc(model, fam, xn,
                                      rsol.psi - rsol.psi.mean(),
                                      np.log(np.clip(rsol.n, 1e5, None)),
                                      Voc_obs)
            err_d = abs(out["d_abs_nm"] - d_true) / d_true * 100
            err_n = abs(np.log10(out["Nt_cm3"]) - logNt_true)
            k1, k2, k3 = st.columns(3)
            k1.metric("Recovered d_abs (nm)", f"{out['d_abs_nm']:.0f}",
                      f"{err_d:.1f}% vs true {d_true}")
            k2.metric("Recovered Nt (cm⁻³)", f"{out['Nt_cm3']:.2e}",
                      f"{err_n:.2f} decades vs true 1e{logNt_true:.1f}")
            k3.metric("Voc residual (V)", f"{out['voc_residual_V']:.4f}")
            st.success(out["method"])
            starts = out["stage1"]["all_starts"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[s["theta"][0] for s in starts],
                y=[s["loss"] for s in starts], mode="markers",
                marker=dict(size=10, color="#1C7293"), name="multi-starts"))
            fig.update_layout(xaxis_title="θ_d (normalized thickness) at convergence",
                              yaxis_title="profile misfit", height=320,
                              title="Stage-1 multi-start convergence")
            st.plotly_chart(fig, width='stretch')
    except ImportError as ex:
        st.warning(f"PyTorch is required for this page: {ex}")

st.divider()
st.caption("References: Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 378, 686 "
           "(2019), DOI 10.1016/j.jcp.2018.10.045 · identifiability split mirrors "
           "standard device characterization practice (geometry from profiling, "
           "lifetime from Voc/transients).")
