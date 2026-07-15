# AI-Driven Open-Source Tool for Design & Optimization of Solar Cells
### Final Researcher Edition

![CI](https://img.shields.io/badge/CI-pytest%20%2B%203%20validation%20suites-2E7D32) ![Benchmarks](https://img.shields.io/badge/certified%20devices-14%2F14-2E7D32) ![CrossTool](https://img.shields.io/badge/SCAPS%20%7C%20TMM%20checks-9%2F9-2E7D32) ![Math](https://img.shields.io/badge/hand--solved%20proofs-22%2F22-2E7D32) ![Refs](https://img.shields.io/badge/references-DOI--cited%20%2B%20Consensus--verified-1F4E79) ![Install](https://img.shields.io/badge/pip%20install-.%20%7C%20Docker-1F4E79)

**Install:** `pip install .` (repo root) or `docker build -t solar-tool . && docker run -p 8501:8501 solar-tool` — then `streamlit run app.py`. Scripting API: `docs/API.md` (examples are executed by CI).

An open-source, physics-based design and optimization environment for solar
cells, delivered as a Streamlit application backed by this repository. Four
technology families run on shared, provenance-tracked numerical foundations:
**Perovskite** (full 1-D drift-diffusion plus a calibrated fast simulator),
**Tandem** (perovskite/Si, perovskite/organic and perovskite/perovskite, 2T
and 4T), **Organic BHJ** (calibrated spectral model plus rigorous
transfer-matrix wave optics), and **Crystalline Silicon** (Al-BSF, PERC,
TOPCon, SHJ, SHJ-IBC). CdTe and related thin-film materials remain available
as absorbers in the main workbench.

Every engine is validated against **certified, fabricated record devices**
with DOI-cited references — never merely against other simulators — and every
headline number regenerates from a script into a versioned artifact.

## Validation status (regenerate any of these yourself)

| Suite | Command | Result |
|---|---|---|
| Certified experimental devices (14) | `python scripts/run_multi_tech_validation.py` | **14/14 pass** · mean \|PCE error\| 0.73% · max 1.91% |
| Cross-tool: SCAPS-1D + Lumerical-class TMM + redundancy (9 checks) | `python scripts/run_cross_tool_validation.py` | **9/9 pass** |
| Hand-solved mathematical examples (22) | `python scripts/run_tutorial_examples.py` | **22/22 asserted** against the live engines |
| Published-SCAPS replication (10 devices) | pre-computed artifact, re-asserted by the cross-tool suite | mean PCE error 0.7%, worst 2.0% |
| Regression tests | `pytest tests/ -q` | full suite green |

Reports: `VALIDATION_MULTI_TECH.md`, `docs/CROSS_TOOL_VALIDATION.md`,
`docs/TUTORIAL.md` (worked mathematics), `MAJOR_CONCERNS.md` (disclosed
limitations with workarounds), `docs/Solar_Cell_Data_Book.xlsx`
(complete parameter/result data book).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## The application

**Main workbench** — perovskite/thin-film device design: stack builder over
the provenance-tracked material database, fast simulator and full
finite-difference drift-diffusion solver, J-V / EQE / band diagram / loss
analysis, temperature and defect studies, perovskite/perovskite tandem tab.

| Page | Purpose |
|---|---|
| **1 · Advanced Physics** | The numerical core exposed: Scharfetter-Gummel drift-diffusion, mesh/convergence controls, recombination-channel breakdown, coherent optics. |
| **2 · Validation & Benchmarks** | The tool grading itself: database provenance audit, solver validation (MMS on the production solver), calibrated uncertainty, published-SCAPS replication. |
| **3 · Silicon Technology** | Five certified-calibrated architectures (Al-BSF → SHJ-IBC), design explorer (wafer, doping, lifetime, J₀ₛ, Rs; Richter/Niewelt recombination switch), **🧱 layer builder** (front/rear passivation-contact stacks with cited per-layer J₀/Rs budgets; certified pairs reconstruct the record devices exactly, hybrids flagged EXTRAPOLATED), DE optimization with sensitivity ranking. |
| **4 · Organic Technology** | Six certified blend benchmarks (PM6:Y6 → PM6:L8-BO-C4), **🧱 donor:acceptor blend builder** (certified pairs reproduce benchmarks exactly; novel pairs via Scharber 2006 design rules with sourced energetics, flagged EXTRAPOLATED, unsourced components refused) with one-click DE optimization, thickness roll-off, transfer-matrix wave-optics tab with **layer-material selection** (interlayer, top electrode, ITO thickness — all with cited n,k; certified stacks labeled) and **📤 measured-n,k upload** (ellipsometry CSV, validated on ingest, session-scoped), DE optimization. |
| **5 · Tandem Designer** | Perovskite/Si, perovskite/organic, perovskite/perovskite; true 2T series J-V coupling and 4T; **interconnect material selection** (ITO RJ / nc-Si:H RJ / 2025-class, each cited); **luminescent coupling** (Steiner-Geisz formulation, Jäger-2020 magnitude anchor); spectral splitting plots; one-click current-matching scans; certified record-lineage validation (29.15% → 34.58%). |
| **6 · Inverse Design** | Target-driven design (e.g., Voc ≥ 1.15 V at PCE ≥ 20%): differentiable-surrogate search with every candidate re-verified through the finite-difference solver before display. |
| **7 · Datasets & Interop** | Perovskite Database Project ingestion; SCAPS-1D `.def` import **and** export (round-trip tested); device-spec JSON; J-V/EQE CSV; **☀️ all-technology energy yield** (engine-derived temperature coefficients, NOCT model per Skoplaki 2009); **⏳ stability tools** (reduced-order scan-rate hysteresis per Richardson 2016; Arrhenius T80 projection per the ISOS consensus, Khenkin 2020); **📈 fit measured J-V** (exact Lambert-W diode model, bootstrap confidence intervals, degeneracy/non-uniqueness report, optional joint light+dark fitting; blind-recovery verified). |
| **8 · References Library** | Every citation in the tool — searchable, with DOI, confidence tier and verification status; CSV export. |

## Physics engines

`physics/device.py` + `physics/dd_solver.py` (perovskite drift-diffusion,
Scharfetter-Gummel), `physics/silicon.py` (Green-2008 optics, Lambertian
trapping, Richter-2012 / Niewelt-2022 intrinsic recombination, implied-Voc
balance, hard 29.4% thermodynamic guard), `physics/organic.py` (spectral EQE +
energy-loss Voc + drift-collection; `optics="tmm"` mode), `physics/tmm.py`
(complex-index transfer-matrix solver, Pettersson-1999/Burkhard-2010
formalism — the coherent Maxwell solution Lumerical STACK computes for planar
stacks), `physics/tandem.py` (series V(J) addition with spectral filtering and
interconnect resistance), `physics/cdte.py` (CdTe on the same drift-diffusion
core).

## AI, used with guardrails

Differential-evolution optimization with random-forest sensitivity ranking on
every technology page (`ai/multi_tech_optimizer.py`); conditional
physics-informed surrogate powering Inverse Design, verified three ways
(held-out FDM comparison, closed-form anchors, forward re-simulation of every
candidate); uncertainty quantification (`ai/uncertainty.py`). Hard
thermodynamic bounds are enforced in code; optimizer boundary solutions are
reported as constraints, not recipes.

## Data, references, reproducibility

All parameters and benchmark targets live in machine-readable registries
(`data/materials_database.json`, `data/multi_technology_database.json`) with
per-entry citation, DOI, confidence tier and verification flag (audited
builds cross-check entries against the Consensus academic index; a
registry-integrity test rejects any benchmark citing an unregistered source
or malformed DOI). Artifacts in `artifacts/` are the single source of truth
the UI renders and the tests assert against. To add a certified benchmark:
one JSON entry (targets + tolerance + DOI) — the validation script, the
technology page and the tests pick it up automatically.

## Interoperability

SCAPS-1D `.def` import/export covering the shared 1-D electro-optical
parameter set (skipped tool-specific extras are listed, never silently
dropped; the round trip is a unit test); OghmaNano-friendly device-spec JSON;
J-V/EQE CSV; TMM layer lists map 1:1 to Lumerical STACK with source-cited n,k
tables. Details: `docs/INTEROPERABILITY.md`.

## Known limitations (disclosed by design)

1-D physics (no busbar geometry or 3-D textures — export to Quokka/FDTD);
organic transport is semi-empirical (exciton/CT drift-diffusion →
OghmaNano via JSON); TMM assumes planar coherent stacks; steady-state only
(no hysteresis/degradation kinetics — benchmark targets are stabilized
certified values). Full list with workarounds: `MAJOR_CONCERNS.md`.

## Citing

See `CITATION.cff` for the tool, and the References Library (page 8) for the
original physics whose DOIs travel with every number.
