# Interoperability

Goal: a device designed here should be re-checkable in the incumbent tools, and
devices defined elsewhere should be importable here. "Compatibility" below means
**parameter-set compatibility**: the fraction of the other tool's 1-D
electro-optical device description this tool can consume or emit without loss.

## Compatibility matrix

| Tool | Direction | Mechanism | Coverage of shared 1-D parameter set |
|---|---|---|---|
| **SCAPS-1D** (Burgelman 2000, DOI 10.1016/S0040-6090(99)00825-1) | import | `utils/scaps_import.parse_def` | thickness, Eg, χ, ε, Nc, Nv, μn, μp, Na, Nd, dominant bulk defect, absorption constant — **>90%** of a typical published .def; skipped items (grading, multi-defect spectra, interface-defect blocks, metastability) are returned as explicit warnings |
| **SCAPS-1D** | export | `utils/interop.export_scaps_def` | same key set, layer-block text; round-trip verified in `tests/test_multi_technology.py::test_interop_roundtrip` |
| **OghmaNano / gpvdm** | export | `utils/interop.export_device_json` (self-describing JSON) | full stack + parameters + result; import into OghmaNano via its JSON material editor |
| **Any J-V / EQE analysis tool** (e.g. NREL/Fraunhofer cert. workflows, Origin, Igor) | export | `export_jv_csv`, `export_eqe_csv` | universal CSV |
| **Perovskite Database Project** (Jacobsson 2022) | import | `utils/dataset_io.load_perovskite_database` | device-level records for ML training (existing feature) |
| **PC1D-class silicon tools** | conceptual | `physics/silicon.py` parameter names follow the community convention (W, N_dop, τ_SRH, J0s, Rs) so a PC1D/Quokka deck maps 1:1 by hand or script | wafer, doping, lifetime, surface recombination, optics |

The one-line honest statement: for the **shared 1-D drift-diffusion parameter
vocabulary** (which is what SCAPS-1D, OghmaNano, PC1D and this tool all speak),
the import/export layer covers essentially the entire set; the things that do
not transfer are each tool's proprietary extras (SCAPS metastability, OghmaNano
exciton modules, Quokka 3-D geometry), and every skipped item is reported, never
silently dropped.

## Round-trip guarantee

`export_scaps_def → parse_def` is tested on every commit: layer count, Eg and
the full key set must survive the round trip bit-cleanly (float-formatted).

## File formats

* `.def` — SCAPS-1D layer-block text (import + export)
* `solarcell-device-spec` JSON — this tool's exchange format (schema in
  `utils/interop.py`), covers all four technologies
* `.csv` — J-V and EQE curves
* `data/*.json` — machine-readable material + reference databases with
  per-parameter provenance (source, DOI, method, confidence tier)
