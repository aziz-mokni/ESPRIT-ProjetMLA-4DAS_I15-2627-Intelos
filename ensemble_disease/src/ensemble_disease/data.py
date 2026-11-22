"""Dataset registry and loader. Cross-platform: relative paths, local CSV cache.

UCI datasets are fetched once with `ucimlrepo` and cached in data/raw/.
Kaggle datasets cannot be auto-downloaded without credentials: put the CSV in
data/raw/<name>.csv (see README). Target is always returned as a 0/1 int series.
NOTE: target/column conventions must be verified in the EDA notebook.
"""
from __future__ import annotations
import pandas as pd
from .config import ROOT

RAW = ROOT / "data" / "raw"

REGISTRY = {
    "D2_heart_cleveland": dict(source="uci", uci_id=45),
    "D5_liver_disorders": dict(source="uci", uci_id=60),
    "D6_ilpd": dict(source="uci", uci_id=225),
    "D8_early_diabetes": dict(source="uci", uci_id=529),
    "D11_debrecen": dict(source="uci", uci_id=329),
    "D12_ckd": dict(source="uci", uci_id=336),
    "D16_dermatology": dict(source="uci", uci_id=33),
    "D4_heart_failure": dict(source="csv", target="DEATH_EVENT"),
    "D10_diabetes_2019": dict(source="csv", target="Diabetic"),
}


def _binarize(y: pd.Series) -> pd.Series:
    """Map any target to 0/1 (numeric: > min; text: first-seen class = 0)."""
    if pd.api.types.is_numeric_dtype(y):
        return (y > y.min()).astype(int)
    codes = pd.factorize(y.astype(str).str.strip().str.lower())[0]
    return pd.Series((codes > 0).astype(int), index=y.index)


def load_dataset(name: str):
    spec = REGISTRY[name]
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"{name}.csv"
    if spec["source"] == "uci":
        if not cache.exists():
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=spec["uci_id"])
            X, y = ds.data.features, ds.data.targets.iloc[:, 0]
            X.assign(__target__=y.values).to_csv(cache, index=False)
        df = pd.read_csv(cache)
        y = _binarize(df.pop("__target__"))
        return df, y
    if not cache.exists():
        raise FileNotFoundError(f"Place the Kaggle CSV at {cache} (see README).")
    df = pd.read_csv(cache)
    y = _binarize(df.pop(spec["target"]))
    return df, y


def basic_report(X: pd.DataFrame, y: pd.Series) -> dict:
    """Quick EDA numbers incl. duplicate rows (possible leakage source)."""
    return dict(
        n=len(X), n_features=X.shape[1],
        pos=int(y.sum()), neg=int((1 - y).sum()),
        missing=int(X.isna().sum().sum()),
        duplicate_rows=int(X.duplicated().sum()),
    )
