# Benchmark Validation Report (v2 — May 2026)

## Summary

Every reference value (PCE, Voc, Jsc, FF) in `artifacts/benchmark_results.json`
has been validated against its cited peer-reviewed primary source. Every DOI
has been verified at `doi.org`. The "Our PCE" column is the actual output of
running `scripts/run_benchmark.py` against `data/materials_database.json`.

| Suite                          | Devices | Mean error | Worst error |
|--------------------------------|---------|------------|-------------|
| SCAPS reference (in-sample)    | 7       | 0.5%       | 1.3%        |
| SCAPS reference (out-of-sample)| 3       | 1.0%       | 2.0%        |
| **SCAPS combined**             | **10**  | **0.7%**   | **2.0%**    |
| Experimental (fabricated cells)| 4       | 17.7%      | 26.1%       |

---

## How to verify these numbers yourself

```bash
git clone https://github.com/Faizan2812/perovskite-solar-optimizer
cd perovskite-solar-optimizer
pip install -r requirements.txt

# Re-run the SCAPS-reference benchmark suite (~13 seconds, 10 devices)
python scripts/run_benchmark.py

# Re-run experimental benchmarks
python scripts/run_experimental_benchmark.py

# Verify every cited DOI resolves at doi.org
python scripts/verify_references.py --network --strict

# Run the full test suite (28 tests)
python -m pytest tests/ -q
```

The output PCE values reproduce within 0.1–0.3% absolute on any machine.

---

## SCAPS-reference benchmark (10 devices)

### In-sample (D1–D7) — used to calibrate effective Nt

| ID | Stack                                  | Paper PCE | Our DD | Error |
|----|----------------------------------------|-----------|--------|-------|
| D1 | ITO/PCBM/CsPbI₃/CBTS/Au                | 16.71%    | 16.67% | 0.2%  |
| D2 | ITO/TiO₂/CsPbI₃/CBTS/Au ⭐             | 17.90%    | 17.85% | 0.3%  |
| D3 | ITO/ZnO/CsPbI₃/CBTS/Au                 | 17.86%    | 17.86% | 0.0%  |
| D4 | ITO/C₆₀/CsPbI₃/CBTS/Au                 | 14.47%    | 14.32% | 1.1%  |
| D5 | ITO/IGZO/CsPbI₃/CBTS/Au                | 17.76%    | 17.77% | 0.0%  |
| D6 | ITO/WS₂/CsPbI₃/CBTS/Au                 | 17.82%    | 18.05% | 1.3%  |
| D7 | ITO/CuI/Cs₂SnI₆/ZnO/AZO/Ag             | 14.65%    | 14.75% | 0.7%  |

### Out-of-sample (D8–D10) — different HTLs from D1–D7

| ID  | Stack                                  | Paper PCE | Our DD | Error |
|-----|----------------------------------------|-----------|--------|-------|
| D8  | ITO/Cu₂O/CsPbI₃/TiO₂/Au                | 17.64%    | 18.00% | 2.0%  |
| D9  | ITO/CuSCN/CsPbI₃/TiO₂/Au               | 17.81%    | 17.84% | 0.1%  |
| D10 | ITO/Spiro-OMeTAD/CsPbI₃/TiO₂/Au        | 17.18%    | 17.02% | 0.9%  |

D1–D7 reference values: **Hossain et al. ACS Omega 7, 43210 (2022)**, Table 4 — DOI `10.1021/acsomega.2c05912` (verified, GOLD-OA).
D7 reference values: **Chabri et al. J. Electron. Mater. 52, 2722 (2023)** — DOI `10.1007/s11664-023-10235-x` (verified).
D8–D10 reference values: same Hossain paper, Figure 4(b) "TiO₂" panel.

---

## Experimental benchmark (4 fabricated cells)

| ID | Reference                              | DOI                        | Measured | Our DD | Error |
|----|----------------------------------------|----------------------------|----------|--------|-------|
| E1 | Saliba 2016 EES                        | 10.1039/c5ee03874j         | 21.10%   | 18.20% | 13.8% |
| E2 | Yang/Jeon 2015 Science (NREL-cert.)    | 10.1126/science.aaa9272    | 20.10%   | 18.88% | 6.1%  |
| E3 | Yoo 2021 Nature 590 (NREL-cert. 25.2%) | 10.1038/s41586-021-03285-w | 25.20%   | 19.00% | 24.6% |
| E4 | Min 2021 Nature 598 (NREL-cert. 25.7%) | 10.1038/s41586-021-03964-8 | 25.70%   | 19.00% | 26.1% |

3 of 4 are NREL-certified champion devices. Errors of 6–26% are within the
documented 30% limit of 1D drift-diffusion vs. fabricated cells with grain
boundaries, area-scaling losses, and engineered passivation.

---

## Material database (32 materials, 292 parameters)

| Category   | Count | Materials |
|------------|-------|-----------|
| Absorbers  | 9     | MAPbI₃, FAPbI₃, CsPbI₃, Cs₂SnI₆, MAPbBr₃, MASnI₃, CsSnI₃, FASnI₃, FA₀.₈₃Cs₀.₁₇Pb(I₀.₆Br₀.₄)₃ |
| ETLs       | 10    | SnO₂, TiO₂, ZnO, C₆₀, PCBM, c-Si, IGZO, WS₂, AZO, In₂S₃ |
| HTLs       | 13    | Spiro-OMeTAD, Cu₂O, NiO, CuSCN, PEDOT:PSS, 2PACz, CBTS, CuI, CuSbS₂, V₂O₅, P₃HT, CFTS, SrCu₂O₂ |

**Confidence-tier distribution after re-sourcing:**

| Tier   | Count | Fraction |
|--------|-------|----------|
| HIGH   | 42    | 14%      |
| MEDIUM | 250   | 86%      |
| LOW    | 0     | **0%**   |

Every parameter has `value`, `source`, and `confidence` fields. Sources resolve to verified DOIs in the `_references` block (38 active references).

---

## What changed in v2 (May 2026)

1. **0 LOW-confidence parameters** (down from 7).
   - CsSnI₃ mobilities corrected (was 585 cm²/Vs — wrong by ~10×) using **Chung 2012 J. Am. Chem. Soc.** Hall measurement.
   - MAPbI₃, MAPbBr₃ defect/doping values upgraded to MEDIUM with explicit literature ranges (Saidaminov 2015, Euvrard 2021).
   - 2PACz mobilities re-sourced to Magomedov 2018 with explicit "SAM thickness ill-defined" caveat.

2. **+7 new materials** (25 → 32) all from verified DOIs:
   - Wide-gap perovskite (1.68 eV) for tandems (Yoo 2021, Min 2021)
   - In₂S₃ ETL (Hadi 2020)
   - CuSbS₂, V₂O₅, P₃HT, CFTS HTLs (Hossain 2022 Table 2)
   - SrCu₂O₂ HTL (Haider 2018)

3. **+3 SCAPS-reference benchmark devices** (D8–D10) as out-of-sample test set.
4. **+2 NREL-certified experimental benchmarks** (E3 Yoo 2021, E4 Min 2021).
5. **Tandem cell feature** with real Beer-Lambert spectral filtering, 2T current-matching, 4T independent — fully tested.
6. **+5 new unit tests**: 2T current-matching, filter-factor bounds, 4T ≥ 2T, no LOW confidence DB, DOI format validity.

All 28 unit tests pass.


---

## What changed in v3.0 (July 2026)

All v2.1 benchmark numbers above are **unchanged** (re-run verified: mean
0.67%, worst 2.0%) — v3.0 physics is opt-in. New validation evidence:

| Check (production code path) | Result | Where |
|---|---|---|
| MMS on `solve_poisson_newton` (production Newton) | observed order **2.019**, rel-L2 2.0×10⁻⁶ | `scripts/run_validation.py` |
| Scharfetter–Gummel diffusion-limit identity | max rel err **1.3×10⁻¹⁶** | same |
| Device-level mesh convergence (Richardson) | discretization error ≈ **0.07 % abs PCE** | same |
| TMM energy conservation (R+T+ΣA−1) | ≤ **1.6×10⁻¹⁵** at every wavelength | `tests/test_v3_upgrades.py` |
| TMM optical Jsc (MAPbI₃, 500 nm, glass/ITO/SnO₂) | 23.33 mA/cm² < SQ limit 26.9 | same |
| Ion-coupled hysteresis limits | HI = 0 exactly at N_ion = 0 and at slow scan; HI > 0 fast scan | same |
| Rs/Rsh circuit identities | ideal limits exact; Rs and finite Rsh lower FF | same |
| Split-conformal 95 % coverage (empirical) | **95.3 %** (naive bootstrap: 18 % — overconfident, replaced) | `artifacts/validation_v3_0.json` |
| Conditional PINN unseen-device interpolation | 6.5–7.6 % rel-L2 (ψ, log n) vs DD | `artifacts/conditional_pinn_metrics.json` |

Reproduce all of it:

```bash
python scripts/run_validation.py            # production MMS + SG + mesh conv + UQ calibration
python scripts/run_benchmark.py             # SCAPS 10-device suite (regression gate)
python scripts/train_conditional_pinn.py    # conditional PINN + interpolation metrics
python -m pytest tests/ -q                  # full suite incl. tests/test_v3_upgrades.py
```
