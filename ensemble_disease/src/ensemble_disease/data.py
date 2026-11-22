"""Dataset registry and loader (cross-platform, relative paths, local CSV cache).

* UCI datasets are fetched once with `ucimlrepo` and cached in data/raw/<name>.csv
  (features + the raw target in a column called `__target__`).
* Kaggle datasets cannot be downloaded automatically: put the CSV in
  data/raw/<name>.csv (see README).

The positive class (y = 1) is ALWAYS the *disease* class and is defined explicitly
per dataset below, so that precision / recall / F1 are meaningful later.
Items marked VERIFY must be confirmed in the EDA notebook.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .config import ROOT

RAW = ROOT / "data" / "raw"


def _s(y: pd.Series) -> pd.Series:
    """Normalise a raw target to lower-case stripped strings (handles 'notckd\\t')."""
    return y.astype(str).str.strip().str.lower()


REGISTRY = {
    # name: source, uci_id / csv name, target column (case-insensitive), positive rule, note
    "D2_heart_cleveland": dict(
        source="uci", uci_id=45, target="num",
        positive=lambda y: pd.to_numeric(y, errors="coerce") > 0,
        note="num 0 = healthy, 1..4 = disease  ->  disease = num > 0"),
    "D5_liver_disorders": dict(
        source="uci", uci_id=60, target="selector", extra_features=["drinks"],
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="VERIFY: label = 'selector' (1: 145, 2: 200), 'drinks' kept as a feature. Selector is a train/test split flag in UCI, not a clean diagnosis"),
    "D6_ilpd": dict(
        source="uci", uci_id=225, target="selector",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Selector 1 = liver patient (416), 2 = non-patient (167)"),
    "D8_early_diabetes": dict(
        source="uci", uci_id=529, target="class",
        positive=lambda y: _s(y) == "positive",
        note="class Positive / Negative"),
    "D11_debrecen": dict(
        source="uci", uci_id=329, target="class",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Class 1 = signs of diabetic retinopathy"),
    "D12_ckd": dict(
        source="uci", uci_id=336, target="class",
        positive=lambda y: _s(y) == "ckd",
        note="class ckd / notckd (raw file has stray tabs/spaces)"),
    "D16_dermatology": dict(
        source="uci", uci_id=33, target="class",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="VERIFY: 6 classes. Binarised as psoriasis (class 1) vs rest; decide multiclass vs binary"),
    "D4_heart_failure": dict(
        source="csv", target="DEATH_EVENT",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Kaggle: heart_failure_clinical_records_dataset.csv ; DEATH_EVENT 1 = died"),
    "D10_diabetes_2019": dict(
        source="csv", target="Diabetic",
        positive=lambda y: _s(y) == "yes",
        note="VERIFY: Kaggle diabetes 2019 file; column name and yes/no spelling"),
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names / text cells and turn '?' or '' into NaN."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            col = df[c].astype(object).where(df[c].notna(), None)
            col = col.map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = col.map(lambda v: np.nan if v in (None, "", "?") else v)
    return _tidy_types(df)


def _tidy_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that are actually numeric back to numbers."""
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum():  # nothing lost
                df[c] = conv
    return df


def _find_col(columns, wanted: str):
    for c in columns:
        if str(c).strip().lower() == wanted.lower():
            return c
    return None


def _fetch_uci(spec, cache):
    """Download from UCI. Uses `ds.data.original` because some columns (e.g. BUPA 'selector')
    have role 'Other' and are in neither `features` nor `targets`."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=spec["uci_id"])
    feats, targs = ds.data.features.copy(), ds.data.targets.copy()
    orig = getattr(ds.data, "original", None)
    if orig is None:
        orig = pd.concat([feats, targs], axis=1)
    tcol = _find_col(orig.columns, spec["target"])
    if tcol is None:
        raise KeyError(f"Target '{spec['target']}' not found. Columns: {list(orig.columns)}")
    y = orig[tcol].reset_index(drop=True)
    X = feats.reset_index(drop=True).copy()
    for c in spec.get("extra_features", []):          # columns we keep although not role 'Feature'
        X[c] = orig[c].reset_index(drop=True)
    # never leave a target column among the features (leakage)
    X = X.drop(columns=[c for c in X.columns
                        if c == tcol or (c in targs.columns and c not in spec.get("extra_features", []))],
               errors="ignore")
    X.assign(__target__=y.values).to_csv(cache, index=False)


def load_dataset(name: str):
    """Return (X, y): X = cleaned feature DataFrame, y = 0/1 int Series (1 = disease)."""
    spec = REGISTRY[name]
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"{name}.csv"
    if not cache.exists():
        if spec["source"] == "uci":
            _fetch_uci(spec, cache)
        else:
            raise FileNotFoundError(f"Place the Kaggle CSV at {cache} (see README).")
    df = pd.read_csv(cache, skipinitialspace=True)
    tname = "__target__" if spec["source"] == "uci" else _find_col(df.columns, spec["target"])
    if tname is None or tname not in df.columns:
        raise KeyError(f"Target column '{spec['target']}' not in {cache.name}: {list(df.columns)}")
    y_raw = df.pop(tname)
    mask = y_raw.notna()                       # rows with no label are unusable
    df, y_raw = df[mask], y_raw[mask]
    y = spec["positive"](y_raw).astype(int)
    X = clean_frame(df).reset_index(drop=True)
    return X, y.reset_index(drop=True)


def basic_report(X: pd.DataFrame, y: pd.Series) -> dict:
    """Quick EDA numbers incl. duplicate rows (possible leakage source)."""
    return dict(
        n=len(X), n_features=X.shape[1],
        pos=int(y.sum()), neg=int((1 - y).sum()),
        missing=int(X.isna().sum().sum()),
        duplicate_rows=int(X.duplicated().sum()),
    )