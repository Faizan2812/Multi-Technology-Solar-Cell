"""
utils/dataset_io.py — external / experimental dataset integration
(v3.1, Phase 3)
==================================================================

The synopsis (Objective 2) calls for the AI framework to operate on "large
datasets covering various material characteristics ... and performances of
available solar cells" — i.e. REAL published devices, not only the tool's
own synthetic physics samples. This module provides that pathway:

1. `load_perovskite_database(path)` — loader for the Perovskite Database
   Project export CSV (Jacobsson et al., "An open-access database and
   analysis tool for perovskite solar cells based on the FAIR data
   principles", Nature Energy 7, 107-115 (2022),
   DOI 10.1038/s41560-021-00941-3; data: perovskitedatabase.com, ~43,000
   devices). The full CSV is distributed via NOMAD and must be downloaded
   by the user (it is not redistributed here); this loader maps its column
   schema onto the tool's feature space and cleans/filters records.

2. `load_seed_dataset()` — a small, fully-cited seed dataset bundled with
   the repository (data/experimental_devices.csv): the 10 SCAPS reference
   devices and 4 certified experimental cells already used by the
   validation suite, each row carrying its DOI. Guarantees the dataset
   pathway works out-of-the-box with traceable data only.

3. `train_surrogate_on_dataset(df)` — trains the tool's from-scratch
   surrogates on any loaded dataset (features -> PCE) and reports held-out
   metrics, so users can compare physics-trained vs literature-trained
   models.

No fabricated device records: every bundled row cites its source.
"""
from __future__ import annotations
import io
import os
import csv
import numpy as np

# ── Perovskite Database Project schema (Jacobsson 2022, Nature Energy) ──────
# Column names as exported by perovskitedatabase.com ("all data" CSV).
PDP_COLUMNS = {
    "bandgap":   "Perovskite_band_gap",
    "abs_thick": "Perovskite_thickness",
    "etl":       "ETL_stack_sequence",
    "htl":       "HTL_stack_sequence",
    "Voc":       "JV_default_Voc",
    "Jsc":       "JV_default_Jsc",
    "FF":        "JV_default_FF",
    "PCE":       "JV_default_PCE",
    "doi":       "Ref_DOI_number",
}

_SEED_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "experimental_devices.csv")


def load_perovskite_database(path, min_pce=1.0, max_pce=30.0,
                             require_bandgap=True, max_rows=None):
    """Load + clean a Perovskite Database Project CSV export.

    Filters: PCE in (min_pce, max_pce); finite Voc/Jsc/FF; optional finite
    bandgap; drops rows with FF quoted in percent inconsistently (>1.05
    treated as % and divided by 100).

    Returns list of dicts with keys: bandgap, abs_thick_nm, etl, htl,
    Voc, Jsc, FF, PCE, doi.
    """
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    cols = {k: v for k, v in PDP_COLUMNS.items() if v in df.columns}
    missing = set(PDP_COLUMNS) - set(cols)
    if {"PCE", "Voc", "Jsc", "FF"} & missing:
        raise ValueError(f"Not a Perovskite-Database export (missing {missing}); "
                         "download the 'all data' CSV from perovskitedatabase.com")
    out = []
    for _, r in df.iterrows():
        try:
            pce = float(r[cols["PCE"]])
            voc = float(r[cols["Voc"]]); jsc = float(r[cols["Jsc"]])
            ff = float(r[cols["FF"]])
        except (ValueError, TypeError, KeyError):
            continue
        if not (min_pce < pce < max_pce) or not all(np.isfinite([voc, jsc, ff])):
            continue
        if ff > 1.05:
            ff /= 100.0
        eg = r.get(cols.get("bandgap", ""), np.nan)
        try:
            eg = float(eg)
        except (ValueError, TypeError):
            eg = np.nan
        if require_bandgap and not np.isfinite(eg):
            continue
        th = r.get(cols.get("abs_thick", ""), np.nan)
        try:
            th = float(th)
        except (ValueError, TypeError):
            th = np.nan
        out.append({
            "bandgap": eg, "abs_thick_nm": th,
            "etl": str(r.get(cols.get("etl", ""), "")),
            "htl": str(r.get(cols.get("htl", ""), "")),
            "Voc": voc, "Jsc": jsc, "FF": ff, "PCE": pce,
            "doi": str(r.get(cols.get("doi", ""), "")),
        })
        if max_rows and len(out) >= max_rows:
            break
    return out


def load_seed_dataset():
    """Bundled, fully-cited seed dataset (each row has a DOI/source)."""
    rows = []
    with open(_SEED_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k in ("bandgap", "abs_thick_nm",
                                               "Voc", "Jsc", "FF", "PCE")
                             and v not in ("", "NA") else v)
                         for k, v in r.items()})
    return rows


def dataset_to_features(rows):
    """Map dataset records to a numeric feature matrix for the surrogates.

    Features: bandgap [eV], absorber thickness [nm], Voc [V], Jsc [mA/cm2],
    FF [-].  Target: PCE [%]. Rows with missing values are dropped.
    """
    X, y, kept = [], [], []
    for r in rows:
        f = [r.get("bandgap"), r.get("abs_thick_nm"),
             r.get("Voc"), r.get("Jsc"), r.get("FF")]
        if any(v is None or (isinstance(v, float) and not np.isfinite(v))
               or isinstance(v, str) for v in f):
            continue
        X.append(f); y.append(r["PCE"]); kept.append(r)
    return (np.array(X, float), np.array(y, float),
            ["bandgap_eV", "abs_thickness_nm", "Voc_V", "Jsc_mA_cm2", "FF"],
            kept)


def train_surrogate_on_dataset(rows, test_fraction=0.25, seed=0):
    """Train the tool's from-scratch Gradient Boosting surrogate on a
    loaded dataset; report held-out R2 / RMSE. Works with the seed dataset
    (small-N sanity) or a full Perovskite-Database export (large-N)."""
    from ai.ml_models import GradientBoostingRegressor, compute_metrics
    X, y, feat_names, kept = dataset_to_features(rows)
    n = len(y)
    if n < 8:
        raise ValueError(f"dataset too small to train on ({n} usable rows)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_te = max(2, int(n * test_fraction))
    te, tr = idx[:n_te], idx[n_te:]
    model = GradientBoostingRegressor(
        n_estimators=min(200, 20 * max(1, n // 10)),
        learning_rate=0.05, max_depth=3)
    model.fit(X[tr], y[tr])
    yp = model.predict(X[te])
    m = compute_metrics(y[te], yp)
    return {"model": model, "n_train": int(len(tr)), "n_test": int(len(te)),
            "feature_names": feat_names, "test_metrics": m,
            "y_test": y[te], "pred_test": yp}
