"""Measured-n,k ingestion: parsing, physicality gates, and the guarantee
that uploaded materials inherit the solver's energy-conservation property."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.tmm import _NK, solve_stack
from physics.tmm_custom import (parse_nk_csv, register_custom_nk,
                                load_nk_csv, validate_nk)


def _fake_blend_csv(k_scale=1.0):
    lam = np.linspace(350, 950, 40)
    n = 1.9 + 0.2 * np.exp(-((lam - 620) / 120.0) ** 2)
    k = k_scale * 0.45 * np.exp(-((lam - 600) / 110.0) ** 2)
    body = "wavelength_nm,n,k\n" + "\n".join(
        f"{l:.1f},{a:.4f},{b:.5f}" for l, a, b in zip(lam, n, k))
    return body


def test_parse_and_register_roundtrip():
    key = load_nk_csv("MyBlend-batch7", _fake_blend_csv().encode())
    assert key == "custom:MyBlend-batch7" and key in _NK
    lam, n, k = _NK[key]
    assert lam[0] <= 400 and lam[-1] >= 800 and np.all(n > 0)


def test_validation_gates():
    lam = np.linspace(500, 700, 20)              # insufficient coverage
    with pytest.raises(ValueError, match="cover at least"):
        validate_nk(lam, np.full(20, 2.0), np.zeros(20))
    lam = np.linspace(350, 950, 20)
    with pytest.raises(ValueError, match="n must be"):
        validate_nk(lam, np.zeros(20), np.zeros(20))
    with pytest.raises(ValueError, match="k must be"):
        validate_nk(lam, np.full(20, 2.0), np.full(20, -0.1))


def test_uploaded_material_conserves_energy_and_responds_physically():
    kA = load_nk_csv("blendA", _fake_blend_csv(1.0))
    kB = load_nk_csv("blendB", _fake_blend_csv(1.6))   # more absorbing
    for key in (kA, kB):
        sol = solve_stack([("ITO", 100), ("PEDOT:PSS", 40),
                           (key, 110), ("Ag", 100)])
        budget = sol["R"] + sol["T"] + np.sum(sol["A"], axis=0)
        assert float(np.max(np.abs(budget - 1))) < 1e-3
    from physics.tmm import jsc_vs_thickness
    _, jA = jsc_vs_thickness([("ITO", 100), ("PEDOT:PSS", 40),
                              (kA, 110), ("Ag", 100)], kA, [110], IQE=0.9)
    _, jB = jsc_vs_thickness([("ITO", 100), ("PEDOT:PSS", 40),
                              (kB, 110), ("Ag", 100)], kB, [110], IQE=0.9)
    assert jB[0] > jA[0]                     # higher k -> more Jsc


def test_builtin_citation_backed_entries_not_overwritable():
    lam = np.linspace(350, 950, 20)
    key = register_custom_nk("PM6:Y6", lam, np.full(20, 2.0), np.zeros(20))
    assert key == "custom:PM6:Y6"            # prefixed, original untouched
    assert not np.allclose(_NK["PM6:Y6"][1], 2.0)
