"""
physics/tmm_custom.py — measured n,k (ellipsometry) ingestion for the TMM.
===========================================================================
Lets a researcher replace any built-in optical dataset with their OWN
measured complex refractive index — the correct answer to batch-to-batch
n,k spread in organic blends (±5-10 %, Kerremans 2020, registry
kerremans_2020_aom): don't trust a library value, measure and load.

CSV format: three columns  wavelength_nm, n, k  (header optional; comma /
semicolon / tab / whitespace delimited). Validation before registration:

  * ≥ 8 rows, strictly increasing wavelength after sorting/dedup
  * coverage: data must span at least [400, 800] nm (the AM1.5G core);
    the TMM never extrapolates silently — the required range is enforced
  * physicality: n > 0 everywhere; k ≥ 0 everywhere (passive material)

Registered materials live in the same _NK table the solver reads, so every
guarantee that holds for built-ins (energy conservation R+T+ΣA=1 to <1e-3,
log-linear k interpolation) holds identically for uploads. Registration is
per-process (session-scoped in the UI) — the shipped, cited database is
never mutated on disk.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np

from physics.tmm import _NK

REQUIRED_RANGE_NM = (400.0, 800.0)


def parse_nk_csv(text_or_bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse (wavelength_nm, n, k) rows; tolerant of headers/delimiters."""
    if isinstance(text_or_bytes, bytes):
        text_or_bytes = text_or_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in io.StringIO(text_or_bytes):
        line = line.strip().replace(";", ",").replace("\t", ",")
        if not line:
            continue
        parts = [p for p in (line.split(",") if "," in line else line.split())
                 if p != ""]
        try:
            lam, n_, k_ = float(parts[0]), float(parts[1]), float(parts[2])
            rows.append((lam, n_, k_))
        except (ValueError, IndexError):
            continue                                  # header / junk line
    if len(rows) < 8:
        raise ValueError("could not parse ≥8 numeric (wavelength, n, k) rows")
    arr = np.array(sorted(rows), float)
    lam, keep = np.unique(arr[:, 0], return_index=True)
    return lam, arr[keep, 1], arr[keep, 2]


def validate_nk(lam: np.ndarray, n: np.ndarray, k: np.ndarray) -> None:
    lo, hi = REQUIRED_RANGE_NM
    if lam.min() > lo or lam.max() < hi:
        raise ValueError(
            f"n,k data must cover at least {lo:.0f}-{hi:.0f} nm "
            f"(got {lam.min():.0f}-{lam.max():.0f}); the TMM does not "
            "extrapolate optical constants silently")
    if np.any(n <= 0):
        raise ValueError("refractive index n must be > 0 everywhere")
    if np.any(k < 0):
        raise ValueError("extinction k must be ≥ 0 (passive material)")
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(k)):
        raise ValueError("non-finite values in n,k data")


def register_custom_nk(name: str, lam, n, k,
                       source_note: str = "user-measured (session)") -> str:
    """Validate and register a measured dataset under `name`.

    Returns the registered key. Built-in, citation-backed entries cannot be
    overwritten — uploads get a 'custom:' prefix so provenance classes never
    mix.
    """
    lam = np.asarray(lam, float)
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    validate_nk(lam, n, k)
    key = name if name.startswith("custom:") else f"custom:{name}"
    _NK[key] = (lam, n, k)
    return key


def load_nk_csv(name: str, text_or_bytes) -> str:
    """One-call convenience: parse + validate + register."""
    lam, n, k = parse_nk_csv(text_or_bytes)
    return register_custom_nk(name, lam, n, k)
