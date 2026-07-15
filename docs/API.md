# Scripting API — five lines per technology

All engines are importable pure functions. Install with `pip install .`
(or `pip install -e .` for development) from the repository root, then:

## Perovskite
```python
from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
from physics.device import fast_simulate
r = fast_simulate(HTL_DB["Spiro-OMeTAD"], PEROVSKITE_DB["MAPbI3"],
                  ETL_DB["SnO2"], 150, 500, 50, 1e14)
print(r["PCE"], r["Voc"], r["Jsc"], r["FF"])
```

## Silicon (layer-built)
```python
from physics.layer_library import build_silicon_from_layers
from physics.silicon import simulate_silicon
arch = build_silicon_from_layers("a-Si:H(i/n) heterocontact + ARC (SHJ front)",
                                 "IBC a-Si:H(i/n+p) rear contacts (SHJ-IBC)")
print(simulate_silicon(arch)["PCE"])            # ~26.77 (certified 26.7)
```

## Organic (rigorous TMM optics)
```python
from physics.organic import ORGANIC_PRESETS, simulate_organic
blend = ORGANIC_PRESETS["PM6:Y6 (Joule 2019, 15.7%)"]
print(simulate_organic(blend)["PCE"],            # calibrated path
      simulate_organic(blend, optics="tmm")["PCE"])  # Maxwell path
```

## Tandem (with luminescent coupling)
```python
from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
from physics.silicon import SILICON_PRESETS
from physics.tandem import simulate_perovskite_silicon_tandem
r = simulate_perovskite_silicon_tandem(
    HTL_DB["2PACz"], PEROVSKITE_DB["FA0.83Cs0.17Pb_I0.6Br0.4_3"],
    ETL_DB["C60"], 650, SILICON_PRESETS["SHJ-IBC (Kaneka 26.7%)"],
    Nt_top=2e14, R_int=0.07, parasitic=0.05, Rs_int_ohm_cm2=4.5, lc_eta=0.3)
print(r["PCE"], r.get("lc_dJ_bot_mA_cm2", 0.0))
```

## Fit a measured J-V (with uncertainty)
```python
from utils.measurement_fit import demo_measurement, fit_measured_jv
V, J, truth = demo_measurement()
fr = fit_measured_jv(V, J, n_bootstrap=100)
print(fr.params["n"], fr.ci_low["n"], fr.ci_high["n"], fr.warnings)
```

## Energy yield & stability
```python
from utils.energy_yield import annual_yield
from utils.stability import hysteresis_index, t80_report
print(annual_yield("Silicon (SHJ-IBC record class)",
                   "Hot desert (Riyadh-class)")["kWh_per_kWp_year"])
print(hysteresis_index(0.12), t80_report([85, 65, 45], [500, 2200, 12000]))
```

Every code block above is executed by `tests/test_api_examples.py` on each
CI run — the documentation cannot silently rot.
