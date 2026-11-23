"""EDA helpers used by notebooks/02_eda.ipynb (kept in a module so they can be tested)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import ROOT
from .data import REGISTRY, load_dataset

RESULTS = ROOT / "results"


# ----------------------------------------------------------------------------- loading
def load_all(names=None, binary_only=True):
    """Load every dataset that is available. Returns (data, errors).
    Multiclass datasets are skipped when binary_only=True (the binary pipeline cannot use them yet)."""
    data, errors = {}, {}
    for name in (names or list(REGISTRY)):
        if binary_only and REGISTRY[name].get("task") == "multiclass":
            errors[name] = "skipped: multiclass dataset (binary pipeline only)"
            continue
        try:
            data[name] = load_dataset(name)
        except Exception as e:  # keep going: one missing file must not stop the EDA
            errors[name] = f"{type(e).__name__}: {e}"
    return data, errors


def split_columns(X: pd.DataFrame):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


# ----------------------------------------------------------------------------- tables
def summary_table(data: dict) -> pd.DataFrame:
    rows = []
    for name, (X, y) in data.items():
        num, cat = split_columns(X)
        n_pos, n = int(y.sum()), len(y)
        n_neg = n - n_pos
        cells = X.shape[0] * X.shape[1]
        dup = int(X.duplicated().sum())
        dup_xy = int(pd.concat([X, y.rename("__y__")], axis=1).duplicated().sum())
        rows.append(dict(
            dataset=name, n=n, n_features=X.shape[1], n_numeric=len(num), n_categorical=len(cat),
            pos=n_pos, neg=n_neg, pos_pct=round(100 * n_pos / n, 1),
            imbalance_ratio=round(max(n_pos, n_neg) / max(1, min(n_pos, n_neg)), 2),
            missing_cells=int(X.isna().sum().sum()), missing_pct=round(100 * X.isna().sum().sum() / cells, 2),
            rows_with_missing=int(X.isna().any(axis=1).sum()),
            duplicate_rows=dup, duplicate_pct=round(100 * dup / n, 2),
            constant_cols=int((X.nunique(dropna=True) <= 1).sum()),
        ))
    return pd.DataFrame(rows).set_index("dataset")


def compare_with_article(summary: pd.DataFrame, article: dict) -> pd.DataFrame:
    """article = {dataset: dict(n=..., n_features=...)}; leave empty entries out."""
    rows = []
    for name, ref in article.items():
        if name in summary.index:
            for k, v in ref.items():
                mine = summary.loc[name, k]
                rows.append(dict(dataset=name, quantity=k, ours=mine, article=v,
                                 diff=mine - v if isinstance(v, (int, float)) else None))
    return pd.DataFrame(rows)


def categorical_report(X: pd.DataFrame, max_levels: int = 12) -> pd.DataFrame:
    """Distinct values of text columns -> reveals typos, case/whitespace variants, odd codes."""
    _, cat = split_columns(X)
    rows = [dict(column=c, n_unique=X[c].nunique(), values=dict(X[c].value_counts().head(max_levels)))
            for c in cat]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- leakage checks
def duplicate_leakage(X: pd.DataFrame, y: pd.Series, n_repeats: int = 50,
                      test_size: float = 0.2, seed: int = 42) -> dict:
    """How often does a TEST row have an exact copy (same features) in TRAIN?

    Such rows are 'free' for any memorising model and inflate the test score.
    Also counts duplicate groups with CONFLICTING labels (irreducible noise)."""
    key = pd.util.hash_pandas_object(X, index=False).reset_index(drop=True)  # NaN-safe row hash
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
    rates = []
    for tr, te in sss.split(X, y):
        train_keys = set(key.iloc[tr])
        rates.append(np.mean([k in train_keys for k in key.iloc[te]]))
    g = pd.DataFrame({"k": key, "y": y.values}).groupby("k")["y"].agg(["size", "nunique"])
    return dict(test_rows_with_train_copy_pct=round(100 * float(np.mean(rates)), 2),
                dup_groups=int((g["size"] > 1).sum()),
                dup_groups_conflicting_labels=int(((g["size"] > 1) & (g["nunique"] > 1)).sum()))


def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """AUC of each feature alone (as a score). Values near 1.0 (or 0.0) = suspiciously
    strong feature -> possible target leakage or an almost-trivial dataset."""
    out = []
    for c in X.columns:
        s = X[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.Series(pd.factorize(s)[0], index=s.index).astype(float)
            s[X[c].isna()] = np.nan
        s = s.astype(float)
        s = s.fillna(s.median())
        if s.nunique() < 2:
            continue
        auc = roc_auc_score(y, s)
        out.append(dict(feature=c, auc=round(auc, 3), strength=round(max(auc, 1 - auc), 3)))
    return pd.DataFrame(out).sort_values("strength", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- baselines
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num, cat = split_columns(X)
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def quick_baselines(X, y, seed: int = 42, n_splits: int = 5) -> pd.Series:
    """Sanity check only (NOT the final protocol): accuracy / F1 of 3 default models.
    Imputation + encoding are inside the pipeline, so they are fit on train folds only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = {"majority": DummyClassifier(strategy="most_frequent"),
              "tree": DecisionTreeClassifier(random_state=seed),
              "forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)}
    res = {}
    for k, m in models.items():
        r = cross_validate(Pipeline([("prep", make_preprocessor(X)), ("m", m)]), X, y, cv=cv,
                           scoring=["accuracy", "f1"])
        res[f"{k}_acc"] = round(r["test_accuracy"].mean(), 3)
        res[f"{k}_f1"] = round(r["test_f1"].mean(), 3)
        res[f"{k}_acc_std"] = round(r["test_accuracy"].std(), 3)
    return pd.Series(res)


# ----------------------------------------------------------------------------- plots
def _save(fig, name):
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / name, dpi=150, bbox_inches="tight")


def plot_class_balance(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(summary) + 1.5))
    s = summary.sort_values("pos_pct")
    ax.barh(s.index, 100 - s["pos_pct"], label="negative (0)", color="#9ecae1")
    ax.barh(s.index, s["pos_pct"], left=100 - s["pos_pct"], label="positive = disease (1)", color="#de2d26")
    ax.axvline(50, color="k", lw=0.8, ls=":")
    for i, (n, p) in enumerate(zip(s["n"], s["pos_pct"])):
        ax.text(101, i, f"n={n}  pos={p}%", va="center", fontsize=8)
    ax.set_xlim(0, 125); ax.set_xlabel("% of samples"); ax.set_title("Class balance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=8, frameon=False)
    _save(fig, "eda_class_balance.png")
    return fig


def plot_missing(data: dict, top: int = 12):
    rates = {}
    for name, (X, _) in data.items():
        r = X.isna().mean().mul(100)
        for c, v in r[r > 0].items():
            rates[f"{name.split('_')[0]}:{c}"] = v
    if not rates:
        print("No missing values in any dataset.")
        return None
    s = pd.Series(rates).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, color="#fd8d3c"); ax.set_xlabel("% missing")
    ax.set_title("Columns with most missing values")
    _save(fig, "eda_missing.png")
    return fig


def plot_corr(X: pd.DataFrame, y: pd.Series, name: str, max_features: int = 25):
    num, _ = split_columns(X)
    d = X[num].copy(); d["TARGET"] = y.values
    corr = d.corr()
    if corr.shape[0] > max_features:  # keep the features most correlated with the target
        keep = corr["TARGET"].abs().sort_values(ascending=False).head(max_features).index
        corr = corr.loc[keep, keep]
    fig, ax = plt.subplots(figsize=(0.45 * len(corr) + 3, 0.45 * len(corr) + 2.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7); ax.set_title(f"{name}: correlation (numeric features)")
    _save(fig, f"eda_corr_{name}.png")
    return fig