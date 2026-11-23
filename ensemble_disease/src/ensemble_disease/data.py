"""Dataset registry and loader (cross-platform, relative paths, local CSV cache).

* UCI datasets are fetched once with `ucimlrepo` and cached in data/raw/<name>.csv
  (features + the raw target in a column called `__target__`).
* Kaggle datasets cannot be downloaded automatically: put the CSV in
  data/raw/<name>.csv (see README).

The positive class (y = 1) is ALWAYS the *disease* class and is defined explicitly
per dataset below, so that precision / recall / F1 are meaningful later.
Items marked VERIFY must be confirmed in the EDA notebook.

The registry covers the 16 datasets (D1..D16) of Table 2 of the reference article
(Mahajan et al., 2024). D2 and D12 are loaded from their UCI versions (same patients as the
Kaggle files of the article). D15 (HAM10000) is multiclass and D16 is multiclass in the article
(binarised here for the binary protocol): `task="multiclass"` datasets are skipped by the binary
pipeline until multiclass metrics are implemented.
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
        source="uci", uci_id=45, target="num", article=dict(n=303, n_features=13, pos=165, neg=138),
        positive=lambda y: pd.to_numeric(y, errors="coerce") > 0,
        note="num 0 = healthy, 1..4 = disease  ->  disease = num > 0"),
    "D5_liver_disorders": dict(
        source="uci", uci_id=60, target="selector", extra_features=["drinks"],
        article=dict(n=345, n_features=7, pos=200, neg=145),
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="VERIFY: label = 'selector' (1: 145, 2: 200), 'drinks' kept as a feature. Selector is a train/test split flag in UCI, not a clean diagnosis"),
    "D6_ilpd": dict(
        source="uci", uci_id=225, target="selector", article=dict(n=583, n_features=10, pos=167, neg=416),
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Selector 1 = liver patient (416), 2 = non-patient (167)"),
    "D8_early_diabetes": dict(
        source="uci", uci_id=529, target="class", article=dict(n=520, n_features=16, pos=320, neg=200),
        positive=lambda y: _s(y) == "positive",
        note="class Positive / Negative"),
    "D11_debrecen": dict(
        source="uci", uci_id=329, target="class", article=dict(n=1151, n_features=18, pos=611, neg=540),
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Class 1 = signs of diabetic retinopathy"),
    "D12_ckd": dict(
        source="uci", uci_id=336, target="class", article=dict(n=400, n_features=25, pos=150, neg=250),
        positive=lambda y: _s(y) == "ckd",
        note="class ckd / notckd (raw file has stray tabs/spaces)"),
    "D16_dermatology": dict(
        source="uci", uci_id=33, target="class", article=dict(n=366, n_features=35),
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="VERIFY: 6 classes. Binarised as psoriasis (class 1) vs rest; decide multiclass vs binary"),
    "D4_heart_failure": dict(
        source="csv", target="DEATH_EVENT", article=dict(n=299, n_features=12, pos=96, neg=203),
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        note="Kaggle: heart_failure_clinical_records_dataset.csv ; DEATH_EVENT 1 = died"),
    "D10_diabetes_2019": dict(
        source="csv", target="Diabetic", article=dict(n=952, n_features=17, pos=267, neg=685),
        positive=lambda y: _s(y) == "yes",
        note="VERIFY: Kaggle diabetes 2019 file; column name and yes/no spelling"),
    # ---- Kaggle datasets of Table 2 (CSV to be placed by hand in data/raw/<name>.csv) ----
    "D1_heart_comprehensive": dict(
        source="csv", target="target",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=1190, n_features=11, pos=629, neg=561),
        note="VERIFY: Kaggle 'Heart Disease Dataset (Comprehensive)', 1190 rows; target 1 = heart disease"),
    "D3_heart_disease": dict(
        source="csv", target="target",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=1025, n_features=13, pos=526, neg=499),
        note="VERIFY: Kaggle johnsmith88/heart-disease-dataset (heart.csv), 1025 rows; many duplicate rows expected"),
    "D7_covid_liver_cancer": dict(
        source="csv", target="Result",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=450, n_features=25, pos=310, neg=140),
        note="VERIFY: Kaggle fedesoriano 'COVID-19 effect on liver cancer prediction'; target column 'Result'"),
    "D9_diabetes_pima": dict(
        source="csv", target="Outcome",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=768, n_features=7, pos=268, neg=500),
        note="VERIFY: Kaggle abdallamahgoub/diabetes (Pima, 768 rows); Outcome 1 = diabetes"),
    "D13_ckd_kaggle": dict(
        source="csv", target="Class",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=400, n_features=13, pos=250, neg=150),
        note="VERIFY: Kaggle abhia1999/chronic-kidney-disease; Class 1 expected = ckd (250 rows)"),
    "D14_kidney_stone": dict(
        source="csv", target="target",
        positive=lambda y: pd.to_numeric(y, errors="coerce") == 1,
        article=dict(n=90, n_features=7, pos=45, neg=45),
        note="VERIFY: Kaggle harshghadiya/kidneystone (urine analysis); very small"),
    "D15_ham10000": dict(
        source="csv", task="multiclass", target="dx", drop_cols=["lesion_id", "image_id"],
        article=dict(n=10015, n_features=6),
        note="Kaggle kmader/skin-cancer-mnist-ham10000, HAM10000_metadata.csv; 7 classes (dx); identifier columns dropped"),
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
    for c in spec.get("drop_cols", []):          # identifiers that must not be used as features
        k = _find_col(df.columns, c)
        if k is not None:
            df = df.drop(columns=[k])
    mask = y_raw.notna()                       # rows with no label are unusable
    df, y_raw = df[mask], y_raw[mask]
    if spec.get("task") == "multiclass":       # class codes 0..K-1 in sorted label order
        y = pd.Series(pd.factorize(y_raw.astype(str).str.strip(), sort=True)[0], index=y_raw.index)
    else:
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