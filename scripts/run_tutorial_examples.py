"""
scripts/run_tutorial_examples.py — executable tutorial verification
===================================================================
Re-derives every mathematically solved example in docs/TUTORIAL.md by hand
(closed-form physics) and asserts agreement with the shipped engines.
If this script exits 0, every number printed in the tutorial is live.

Run:  python scripts/run_tutorial_examples.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

Q = 1.602176634e-19
K_B = 1.380649e-23
H = 6.62607015e-34
C0 = 2.99792458e8
T = 300.0
VT = K_B * T / Q
trapz = np.trapezoid

PASS = []


def check(name, ok, detail):
    PASS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")


print(f"Thermal voltage V_T = {VT:.6f} V (tutorial: 0.025852)")

# ===========================================================================
print("\n=== Example 1: Silicon SHJ-IBC by hand ===")
from physics.silicon import (SILICON_PRESETS, simulate_silicon,
                             intrinsic_recombination, NI_EFF_300K)
from physics.spectrum import photon_flux, AM15G_WAVELENGTHS

arch = SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"]
r = simulate_silicon(arch)

lam = AM15G_WAVELENGTHS.astype(float)
phi = photon_flux(lam)
m = lam <= 1200
Jmax = Q * trapz(phi[m], lam[m]) * 1e3
check("1.1 photocurrent ceiling", abs(Jmax - 46.41) < 0.2,
      f"q*Phi(<=1200nm) = {Jmax:.2f} mA/cm² (tutorial 46.41); "
      f"engine Jsc {r['Jsc']:.2f} = {r['Jsc']/Jmax*100:.1f}% utilization")

# hand recombination balance at the engine's dn_oc
ni = NI_EFF_300K
dn = r["dn_oc"]
n0, p0 = arch.Ndop_cm3, ni**2 / arch.Ndop_cm3
W = arch.W_um * 1e-4
R_int = intrinsic_recombination(dn, arch.Ndop_cm3, "n")
R_srh = dn / (arch.tau_srh_ms * 1e-3)
R_srf = (arch.J0s_fA * 1e-15 / Q) * (((n0 + dn) * (p0 + dn)) / ni**2 - 1) / W
bal = abs((R_int + R_srh + R_srf) - r["G_avg"]) / r["G_avg"]
check("1.2 open-circuit balance closes", bal < 1e-3,
      f"(R_intr+R_SRH+R_surf)/G - 1 = {bal:.1e} "
      f"({R_int:.3e}+{R_srh:.3e}+{R_srf:.3e} vs G={r['G_avg']:.3e})")

Voc_hand = VT * np.log((n0 + dn) * (p0 + dn) / ni**2)
check("1.3 hand Voc = engine Voc", abs(Voc_hand - r["implied_Voc"]) < 1e-4,
      f"hand {Voc_hand:.4f} V vs engine {r['implied_Voc']:.4f} V")

voc = r["Voc"] / (r["n"] * VT)
FF0 = (voc - np.log(voc + 0.72)) / (voc + 1)
rs = arch.Rs_ohm_cm2 * r["Jsc"] / (r["Voc"] * 1000)
FF_hand = FF0 * (1 - 1.1 * rs)
check("1.4 Green-1981 FF", abs(FF_hand - r["FF"]) < 0.003,
      f"FF0={FF0:.4f}, rs={rs:.4f} -> hand {FF_hand:.4f} vs engine {r['FF']:.4f}")

pce_hand = r["Voc"] * r["Jsc"] * r["FF"]
check("1.5 PCE assembles to certified device",
      abs(pce_hand - 26.7) / 26.7 < 0.03,
      f"{r['Voc']:.4f} x {r['Jsc']:.2f} x {r['FF']:.4f} = {pce_hand:.2f}% "
      "(certified 26.7%, Yoshikawa 2017)")

# ===========================================================================
print("\n=== Example 2: Organic PM6:Y6 by hand ===")
from physics.organic import ORGANIC_PRESETS, simulate_organic

b = ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"]
ro = simulate_organic(b)
check("2.1 Voc = (Eg - E_loss)/q",
      abs((b.Eg_opt_eV - b.E_loss_eV) - ro["Voc"]) < 0.005,
      f"1.33 - 0.50 = 0.830 V vs engine {ro['Voc']:.3f} V")

edge = 1239.84 / b.Eg_opt_eV
mw = (lam >= 350) & (lam <= edge)
J_window = Q * trapz(phi[mw], lam[mw]) * 1e3
check("2.2 spectral window integral",
      abs(J_window - 34.90) < 0.2,
      f"q∫Phi(350-{edge:.0f}nm) = {J_window:.2f}; xEQE(0.769) = "
      f"{J_window*0.769:.2f}; roll-off -> engine {ro['Jsc']:.2f} "
      "(certified 25.3)")

J0_hand = ro["Jsc"] / (np.exp(ro["Voc"] / (b.n_id * VT)) - 1)
check("2.3 diode J0 consistency",
      abs(J0_hand - ro["J0"]) / ro["J0"] < 0.05,
      f"hand {J0_hand:.3e} vs engine {ro['J0']:.3e} mA/cm²")

check("2.4 PCE vs certified device", abs(ro["PCE"] - 15.7) / 15.7 < 0.04,
      f"{ro['PCE']:.2f}% vs 15.7% (Yuan 2019)")

p_tmm = simulate_organic(b, optics="tmm")["PCE"]
check("2.5 independent TMM optics path",
      abs(p_tmm - ro["PCE"]) / ro["PCE"] < 0.08,
      f"TMM {p_tmm:.2f}% vs calibrated {ro['PCE']:.2f}% "
      f"({abs(p_tmm-ro['PCE'])/ro['PCE']*100:.1f}% dev; Rosa-2021 band ~16%)")

# ===========================================================================
print("\n=== Example 3: Perovskite MAPbI3 — bounds and absorption ===")
from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
from physics.device import fast_simulate

ab = PEROVSKITE_DB["MAPbI3"]
rp = fast_simulate(HTL_DB["Spiro-OMeTAD"], ab, ETL_DB["SnO2"],
                   150, 500, 50, 1e14)

E600 = 1239.84 / 600.0
alpha600 = ab.alpha_coeff * np.sqrt(E600 - ab.Eg) / E600
A600 = 1 - np.exp(-alpha600 * 500e-7)
check("3.1 Tauc absorption at 600 nm",
      abs(alpha600 - 5.22e4) / 5.22e4 < 0.02 and abs(A600 - 0.926) < 0.01,
      f"alpha = {alpha600:.3e} cm⁻¹, A(500 nm film) = {A600:.3f}")

mpk = lam <= 1239.84 / ab.Eg
J_pk_max = Q * trapz(phi[mpk], lam[mpk]) * 1e3
check("3.2 Jsc below above-gap ceiling", rp["Jsc"] < J_pk_max,
      f"engine {rp['Jsc']:.2f} < ceiling {J_pk_max:.2f} mA/cm² "
      f"({rp['Jsc']/J_pk_max*100:.1f}% utilization at Nt=1e14)")

Egj = ab.Eg * Q
J0_rad = Q * (2 * np.pi * K_B * T * Egj**2 / (H**3 * C0**2)) \
    * np.exp(-Egj / (K_B * T)) * 1e-1     # -> mA/cm²
Voc_rad = VT * np.log(J_pk_max / J0_rad + 1)
check("3.3 Voc below radiative (SQ) limit",
      rp["Voc"] < Voc_rad and abs(Voc_rad - 1.279) < 0.01,
      f"engine {rp['Voc']:.3f} V < V_oc,rad = {Voc_rad:.4f} V "
      f"(J0_rad = {J0_rad:.3e} mA/cm²)")

# ===========================================================================
print("\n=== Example 4: Tandem — additivity and current limiting ===")
from physics.tandem import simulate_perovskite_silicon_tandem

rt = simulate_perovskite_silicon_tandem(
    HTL_DB["2PACz"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"],
    ETL_DB.get("C60", ETL_DB.get("PCBM")), 650,
    SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"], Nt_top=2e14,
    R_int=0.07, parasitic=0.05, Rs_int_ohm_cm2=4.5)

v_sum = rt["top"]["Voc"] + rt["bottom"]["Voc"]
check("4.1 Voc additivity at J=0",
      abs(v_sum - rt["Voc"]) / rt["Voc"] < 0.005,
      f"{rt['top']['Voc']:.3f} + {rt['bottom']['Voc']:.3f} = {v_sum:.3f} V "
      f"vs tandem {rt['Voc']:.3f} V")

dv = VT * np.log(rt["bottom"]["Jsc"] / r["Jsc"])
check("4.2 bottom-cell intensity penalty ~ VT ln(J/J0)",
      abs((rt["bottom"]["Voc"] - r["Voc"]) - dv) < 0.012,
      f"observed {1000*(rt['bottom']['Voc']-r['Voc']):.1f} mV vs "
      f"VT·ln({rt['bottom']['Jsc']:.2f}/{r['Jsc']:.2f}) = {1000*dv:.1f} mV")

check("4.3 series stack limited by smaller subcell current",
      abs(rt["Jsc"] - min(rt["top"]["Jsc"], rt["bottom"]["Jsc"])) < 0.05,
      f"tandem {rt['Jsc']:.2f} = min({rt['top']['Jsc']:.2f}, "
      f"{rt['bottom']['Jsc']:.2f})")

check("4.4 PCE vs certified 29.15%", abs(rt["PCE"] - 29.15) / 29.15 < 0.05,
      f"{rt['PCE']:.2f}% (Al-Ashouri 2020)")

# ===========================================================================
print("\n=== Example 5: TMM — analytic optics ===")
from physics.tmm import tmm_single, nk, jsc_vs_thickness, \
    energy_conservation_error
from scipy.signal import argrelextrema

out = tmm_single(550.0, [np.complex128(1.52), np.complex128(1.0)], [])
fres = ((1.52 - 1) / (1.52 + 1)) ** 2
check("5.1 Fresnel limit", abs(out["R"] - fres) < 1e-12,
      f"R = {out['R']:.8f} vs ((0.52/2.52))² = {fres:.8f}; R+T-1 = "
      f"{out['R']+out['T']-1:.1e}")

n550 = nk("P3HT:PCBM", 550).real
d_qw = 550 / (4 * n550)
stk = [("ITO", 100), ("PEDOT:PSS", 40), ("P3HT:PCBM", 90), ("Al", 100)]
ds, js = jsc_vs_thickness(stk, "P3HT:PCBM", np.arange(30, 305, 5), IQE=0.8)
mx = argrelextrema(js, np.greater, order=3)[0]
check("5.2 quarter-wave rule anticipates first maximum",
      abs(d_qw - 66.1) < 1 and 60 <= ds[mx[0]] <= 95,
      f"lam/4n = {d_qw:.1f} nm; broadband TMM max at {ds[mx[0]]:.0f} nm "
      "(published window 70-90 nm)")

err = energy_conservation_error(stk)
check("5.3 energy conservation", err < 5e-4, f"max|1-(R+T+SumA)| = {err:.1e}")

# ===========================================================================
print("\n=== Example 6: PINN verification protocol ===")
import json
mpath = os.path.join(os.path.dirname(__file__), "..",
                     "artifacts", "conditional_pinn_metrics.json")
if os.path.exists(mpath):
    met = json.load(open(mpath))
    rel = [v["relL2_psi"] for v in met["interpolation_validation"].values()]
    check("6.1 PINN vs FDM held-out devices", max(rel) < 0.10,
          f"rel-L2(psi) on {len(rel)} held-out devices: "
          f"{', '.join(f'{v*100:.1f}%' for v in rel)} (<10%)")
else:
    check("6.1 PINN vs FDM held-out devices", False, "artifact missing")

eps0 = 8.8541878128e-12
W_D = np.sqrt(2 * 25 * eps0 * 1.0 / (Q * 1e21))   # N = 1e15 cm^-3 in m^-3
check("6.2 analytic depletion anchor",
      abs(W_D * 1e9 - 1662) < 20,
      f"W_D = {W_D*1e9:.0f} nm >> 500 nm -> fully depleted absorber, "
      "uniform field ~ V_bi/d = 20 kV/cm (PINN field check)")

# ===========================================================================
n_ok = sum(PASS)
print(f"\n=== Tutorial verification: {n_ok}/{len(PASS)} solved examples "
      f"confirmed against the live engines ===")
sys.exit(0 if n_ok == len(PASS) else 1)
