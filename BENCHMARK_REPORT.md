# SCAPS-Reference Benchmark Report

*Generated: 2026-07-05T05:05:54Z*

Validation of the drift-diffusion solver in this tool against 10 published SCAPS-1D results from 4 peer-reviewed papers.

## Summary

- **Mean PCE error**: 0.67%
- **Median PCE error**: 0.47%
- **Worst-case PCE error**: 2.02%
- **Convergence rate**: 100.0%

## Per-device results

| Device | Stack | SCAPS PCE | Our PCE | ΔPCE | Converged |
|---|---|---|---|---|---|
| D1 | CBTS/CsPbI3/PCBM | 16.71 | 16.67 | 0.2% | yes |
| D2 | CBTS/CsPbI3/TiO2 | 17.90 | 17.85 | 0.3% | yes |
| D3 | CBTS/CsPbI3/ZnO | 17.86 | 17.86 | 0.0% | yes |
| D4 | CBTS/CsPbI3/C60 | 14.47 | 14.32 | 1.1% | yes |
| D5 | CBTS/CsPbI3/IGZO | 17.76 | 17.77 | 0.0% | yes |
| D6 | CBTS/CsPbI3/WS2 | 17.82 | 18.05 | 1.3% | yes |
| D7 | CuI/Cs2SnI6/ZnO | 14.65 | 14.75 | 0.7% | yes |
| D8 | Cu2O/CsPbI3/TiO2 | 17.64 | 18.00 | 2.0% | yes |
| D9 | CuSCN/CsPbI3/TiO2 | 17.81 | 17.84 | 0.1% | yes |
| D10 | Spiro-OMeTAD/CsPbI3/TiO2 | 17.18 | 17.02 | 0.9% | yes |

## References

- Hossain et al., ACS Omega 7, 43210 (2022). DOI: 10.1021/acsomega.2c05912
- Chabri et al., J. Electron. Mater. 52, 2722 (2023). DOI: 10.1007/s11664-023-10235-x

## Notes

The PCE/Voc/Jsc/FF values in the 'SCAPS PCE' column are taken directly from the cited papers, not fits produced by this tool. Agreement with other groups' SCAPS output does not validate against experimental measurement — see `EXPERIMENTAL_BENCHMARK_REPORT.md` for that.

## Regenerating this report

```bash
# Full SCAPS-reference suite (7 devices)
python scripts/run_benchmark.py

# Single device (e.g. just D2 = ITO/TiO2/CsPbI3/CBTS/Au)
python scripts/run_benchmark.py --device D2

# Quick subset for CI
python scripts/run_benchmark.py --quick
```