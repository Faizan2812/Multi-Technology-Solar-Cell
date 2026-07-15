# Multi-Technology Validation Report

**Suite**: `scripts/run_multi_tech_validation.py` → `artifacts/validation_multi_tech.json`
**Result**: **14/14 benchmarks pass** · mean |PCE error| **0.73%** · median **0.73%** · max **1.91%**

Every benchmark target below is a **certified, fabricated record device** from the
peer-reviewed literature (not another group's simulation). Full citations with DOIs
live in `data/multi_technology_database.json` and on the 📚 References page.

## Silicon (physics/silicon.py)

| Benchmark | Published | Model | Error | Reference (DOI) |
|---|---|---|---|---|
| SHJ-IBC (Kaneka) | 26.7% | 26.77% | 0.26% | 10.1016/j.solmat.2017.06.024 |
| SHJ both-side (LONGi) | 26.81% | 26.75% | 0.21% | 10.1038/s41560-023-01255-2 |
| TOPCon back junction | 26.0% | 25.98% | 0.07% | 10.1038/s41560-021-00805-w |
| PERC industrial | 24.06% | 23.60% | 1.91% | 10.1002/pip.3371 |
| Al-BSF legacy | ~20.1% | 20.16% | 0.31% | 10.1002/pip.3371 |

Physics: Green (2008) α(λ) table, Tiedje-Yablonovitch Lambertian light trapping,
Richter (2012) intrinsic recombination (Niewelt 2022 selectable), implied-Voc
balance, Green (1981) FF, full single-diode J-V. A hard sanity test enforces that
no configuration exceeds the 29.4% single-junction limit (Niewelt 2022).

## Organic (physics/organic.py)

| Benchmark | Published | Model | Error | Reference (DOI) |
|---|---|---|---|---|
| PM6:Y6 | 15.7% | 15.57% | 0.84% | 10.1016/j.joule.2019.01.004 |
| PM6:Y6 @ 300 nm | 13.6% | 13.42% | 1.30% | 10.1016/j.joule.2019.01.004 |
| PM6:L8-BO | 18.32% | 18.20% | 0.63% | 10.1038/s41560-021-00820-x |
| PBQx-TF:eC9-2Cl | 19.0% | 19.19% | 1.02% | 10.1002/adma.202102420 |
| PM6:L8-BO-C4 | 20.42% | 20.67% | 1.23% | 10.1038/s41563-024-02087-5 |
| P3HT:PC61BM | ~4% | 4.05% | 1.24% | 10.1103/PhysRevB.72.085205 |

The 100 → 300 nm thickness roll-off is a genuine out-of-calibration check: only
the 100 nm device parameters were fit; the 300 nm prediction follows from the
drift-collection model.

## Tandem (physics/tandem.py)

| Record generation | Published | Model | Error | Reference (DOI) |
|---|---|---|---|---|
| 2020 (HZB) | 29.15% | 29.16% | 0.02% | 10.1126/science.abd4016 |
| 2023 (EPFL/CSEM, HZB) | 31.25–32.5% | 31.75% | in band | 10.1126/science.adg0091, 10.1126/science.adf5872 |
| 2025 (LONGi) | 34.58% | 34.16% | 1.21% | 10.1038/s41586-025-09333-z |

2T coupling is a proper series J-V voltage addition on the common current axis
(mismatched subcells operate off their own MPP), with Beer-Lambert spectral
filtering, textured-optics path enhancement (factor 1.6) and an interconnect
series resistance.

## Honest disclosure of model trade-offs

1. **Tandem 2020-grade internal split**: PCE (0.02% error) and Voc (1.922 vs
   1.92 V) match closely, but the engine runs ~7% low on Jsc (17.9 vs 19.26
   mA/cm²) and ~6% high on FF (84.7 vs 79.5%) — compensating errors of the
   analytical top-cell optics vs the real textured multilayer stack. The
   2023/2025 grades match all four parameters within ~5%.
2. **Grade presets are calibrated**, not ab-initio: N_t, R_int, parasitic and
   Rs_int per record generation are chosen inside physically documented ranges
   to represent 2020/2023/2025 interface quality. This mirrors how SCAPS
   baselines are constructed in the literature.
3. **Organic engine class**: semi-empirical spectral + equivalent-circuit +
   drift-collection. No transfer-matrix interference, no exciton/CT
   drift-diffusion; for those, export the device (Datasets & Interop) and
   cross-check in OghmaNano.
4. **Silicon engine class**: analytical (implied-Voc + diode), not a
   finite-element device solver like Quokka/Sentaurus; it captures wafer,
   passivation, optics and resistive physics but not 2/3-D geometry effects
   (busbar layout, local contacts).

## Reproduce

```bash
python scripts/run_multi_tech_validation.py   # writes artifacts/validation_multi_tech.json
pytest tests/test_multi_technology.py -q      # 10 tests, includes all benchmarks
```
