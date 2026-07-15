"""Tests for the v2.1 advanced-physics modules (physics.advanced)."""
import numpy as np
from physics.advanced import optics_tmm as o, temperature as t, defects as d, \
    ion_migration as im, impedance as z

def test_tmm_energy_conservation():
    wl, _ = o.am15g()
    for lam in wl[::20]:
        R, T, A = o.solve_stack([2.0+0j, 2.4+0.3j, 1.8+0j], [50,400,50], lam, 1.0, 3.5)
        assert abs(R+T+A-1.0) < 1e-9

def test_temperature_trends():
    p = t.VARSHNI["MAPbI3"]
    assert t.eg_varshni(250,**p) > t.eg_varshni(350,**p)
    assert t.ni(250,2.2e18,1.8e19,**p) < t.ni(350,2.2e18,1.8e19,**p)

def test_defects_reduce_to_single_level():
    s = d.srh_rate(1e16,1e14,1e8,0.0,1e15)
    m = d.multilevel_srh(1e16,1e14,1e8,[dict(Et_rel_mid=0.0,Nt=1e15)])
    assert abs(m-s)/s < 1e-12

def test_ion_hysteresis_limits():
    fast = im.simulate_scan(scan_rate_V_s=1.0, ion_density=1e17)["hysteresis_index"]
    none = im.simulate_scan(scan_rate_V_s=1.0, ion_density=0.0)["hysteresis_index"]
    assert fast > 1e-3 and none < 1e-6

def test_mott_schottky_recovers_doping():
    ms = z.mott_schottky(1e16, eps_r=6.5, Vbi=0.9)
    assert abs(ms["Na_fit_cm3"]-1e16)/1e16 < 0.05
