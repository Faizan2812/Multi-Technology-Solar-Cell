# Cross-Tool Validation

**Suite**: `scripts/run_cross_tool_validation.py` → `artifacts/validation_cross_tool.json`
**Result**: **9/9 checks pass**

## Position statement

This tool's primary validation is against **certified experimental record
devices** (14/14 benchmarks, mean PCE error 0.73% — `VALIDATION_MULTI_TECH.md`),
because simulators can share systematic errors while certified devices cannot.
Cross-tool validation is the second pillar: it demonstrates that where this
tool and the incumbent simulators solve the same physics, they agree.

## The Lumerical question, answered precisely

Lumerical STACK (and FDTD in the planar limit) solves the coherent Maxwell
problem for multilayer thin films. For planar stacks that problem has an
exact solution: the transfer-matrix method (TMM), which is also what the
published organic-solar-cell optical literature uses (Pettersson 1999;
Sievers 2006; Kotlarski 2008; Monestier 2007; Im 2023; Rosa 2021 — DOIs in
the registry). `physics/tmm.py` implements it with complex refractive
indices, internal field profiles and per-layer absorption. **Agreement with
these published TMM/Lumerical-class results is therefore validation against
that tool class** — and the checks below prove it quantitatively. The one
regime that genuinely requires FDTD — textured/nanostructured 3-D light
trapping — is outside any 1-D tool (including SCAPS-1D) and is stated as
such, never imitated.

## Check table

| ID | Check | Result |
|---|---|---|
| C1a | Energy conservation R + T + ΣA = 1 across the spectrum | max error 9.7×10⁻⁵ |
| C1b | Analytic Fresnel limit, bare glass/air interface | machine precision |
| C1c | P3HT:PCBM Jsc(L) interference vs published TMM (Sievers 2006, Kotlarski 2008, Monestier 2007) | maxima at 80 & 215 nm (8.4, 10.1 mA/cm²), minimum 130 nm — inside all published windows |
| C1d | NFA-blend first optical optimum ≈100 nm (Im 2023) | 90 nm |
| C1e | PM6:Y6 absolute TMM Jsc vs certified device (Yuan 2019) | 24.0 vs 25.3 mA/cm² (5.1%; published TMM-vs-experiment band ~16%, Rosa 2021) |
| C1f | Independent-path PCE: TMM optics vs calibrated optics (PM6:Y6) | 14.88% vs 15.57% (4.4%) |
| C2a | SCAPS-1D `.def` export → re-import round trip | parameter-exact |
| C2b | Published-SCAPS numerical cross-check (v3, 10 devices) | mean PCE error 0.7%, worst 2.0% |
| C3a | Silicon: Richter-2012 vs Niewelt-2022 intrinsic recombination | 0.56% PCE deviation |

## How a reviewer reproduces any single device in SCAPS-1D or Lumerical

1. Design the device on any technology page → download **Device JSON** and
   (for layered stacks) the **SCAPS .def** export.
2. SCAPS-1D: open the .def, run — the shared 1-D parameter set transfers
   losslessly; anything SCAPS-specific is listed in the file header.
3. Lumerical STACK / any TMM code: the layer list + thicknesses map 1:1;
   n,k tables are in `physics/tmm.py` (`_NK`) with source DOIs, or use the
   reviewer's own ellipsometry data.
4. Compare against the J-V/EQE CSV exports.

```bash
python scripts/run_cross_tool_validation.py   # regenerates all 9 checks
pytest tests/test_multi_technology.py -q
```
