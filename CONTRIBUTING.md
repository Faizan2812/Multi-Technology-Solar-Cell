# Contributing

## Add a certified benchmark (the most valuable contribution)
One JSON entry in `data/multi_technology_database.json`:
1. Add the reference to `references` (citation, DOI, `used_for`). Only
   peer-reviewed or certification-lab sources; the registry-integrity test
   rejects malformed DOIs, and audited builds cross-check entries against
   the Consensus academic index.
2. Add the device under `benchmarks.<technology>` with certified targets
   (PCE, Voc, Jsc, FF) and a pre-registered tolerance justified by the
   published measurement uncertainty.
3. Run `python scripts/run_multi_tech_validation.py` — the suite, the
   technology page, and the tests pick the entry up automatically.
Open a PR using the benchmark template. Red suites are discussed, not
hidden: if the model misses your device, that is a finding.

## Code contributions
- Every physics change must keep `pytest tests/` and all three validation
  scripts green (CI enforces this).
- New parameters require a registry citation. No DOI, no merge.
- New models must state their model class and limits in the docstring and
  the UI (see `utils/stability.py` for the expected disclosure style).
