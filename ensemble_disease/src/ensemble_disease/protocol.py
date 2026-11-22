"""Evaluation protocols and cross-validation runner.

Two protocols are compared (same models, same folds logic):
  * "naive": repeated stratified K-fold on the data as downloaded (duplicate rows kept).
             A copy of a test row can sit in the training set -> optimistic scores.
  * "clean": exact duplicate rows (features + label) are removed first, then the same CV.
             No test row has an identical copy in the training set.
Preprocessing (median imputation, one-hot encoding) is inside the pipeline, so it is
fitted on the training part of each fold only (no leakage).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .config import ROOT
from .eda import make_preprocessor

RESULTS = ROOT / "results"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "auprc"]


def prepare(X: pd.DataFrame, y: pd.Series, protocol: str):
    """Apply the protocol's data preparation. Returns (X, y) with a clean 0..n-1 index."""
    if protocol == "naive":
        return X.reset_index(drop=True), y.reset_index(drop=True)
    if protocol == "clean":
        dup = pd.concat([X, y.rename("__y__")], axis=1).duplicated(keep="first")
        return X[~dup.values].reset_index(drop=True), y[~dup.values].reset_index(drop=True)
    raise ValueError(f"unknown protocol {protocol!r}")


def _fold(estimator, Xp, yp, tr, te, fold_id, n_splits):
    pipe = Pipeline([("prep", make_preprocessor(Xp)), ("m", clone(estimator))])
    t0 = time.time()
    pipe.fit(Xp.iloc[tr], yp.iloc[tr])
    fit_s = time.time() - t0
    y_te = yp.iloc[te]
    pred = pipe.predict(Xp.iloc[te])
    proba = pipe.predict_proba(Xp.iloc[te])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return dict(
        repeat=fold_id // n_splits, fold=fold_id % n_splits, n_train=len(tr), n_test=len(te),
        accuracy=accuracy_score(y_te, pred),
        precision=precision_score(y_te, pred, zero_division=0),
        recall=recall_score(y_te, pred, zero_division=0),
        f1=f1_score(y_te, pred, zero_division=0),
        roc_auc=roc_auc_score(y_te, proba) if y_te.nunique() == 2 else np.nan,
        auprc=average_precision_score(y_te, proba) if y_te.nunique() == 2 else np.nan,
        tn=tn, fp=fp, fn=fn, tp=tp, fit_seconds=fit_s)


def evaluate(estimator, X, y, protocol="clean", n_splits=5, n_repeats=3, seed=42, n_jobs=-1):
    """Per-fold metrics of one model on one dataset under one protocol (DataFrame, 1 row / fold)."""
    Xp, yp = prepare(X, y, protocol)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(estimator, Xp, yp, tr, te, i, n_splits)
        for i, (tr, te) in enumerate(cv.split(Xp, yp)))
    out = pd.DataFrame(rows)
    out["n_rows_used"] = len(Xp)
    return out


def run_all(data: dict, models: dict, protocols=("naive", "clean"), n_splits=5, n_repeats=3,
            seed=42, n_jobs=-1, verbose=True) -> pd.DataFrame:
    """Every dataset x protocol x model. A failing model is reported and skipped."""
    parts = []
    for ds, (X, y) in data.items():
        for proto in protocols:
            for name, est in models.items():
                t0 = time.time()
                try:
                    r = evaluate(est, X, y, proto, n_splits, n_repeats, seed, n_jobs)
                except Exception as e:
                    print(f"[FAILED] {ds} | {proto} | {name}: {type(e).__name__}: {e}")
                    continue
                r.insert(0, "model", name); r.insert(0, "protocol", proto); r.insert(0, "dataset", ds)
                parts.append(r)
                if verbose:
                    print(f"{ds:20s} {proto:5s} {name:16s} F1={r.f1.mean():.3f} "
                          f"acc={r.accuracy.mean():.3f}  ({time.time()-t0:.1f}s)")
    return pd.concat(parts, ignore_index=True)


def summarize(results: pd.DataFrame, metric="f1", protocol="clean", std=False) -> pd.DataFrame:
    """Mean (over folds and repeats) of one metric: rows = datasets, columns = models."""
    d = results[results.protocol == protocol]
    g = d.groupby(["dataset", "model"])[metric]
    t = (g.std() if std else g.mean()).unstack("model")
    return t[[m for m in d.model.drop_duplicates() if m in t.columns]].round(3)


def inflation(results: pd.DataFrame, metric="f1") -> pd.DataFrame:
    """naive - clean: how much the duplicate-keeping protocol overestimates the score."""
    return (summarize(results, metric, "naive") - summarize(results, metric, "clean")).round(3)


def mean_ranks(table: pd.DataFrame) -> pd.Series:
    """Average rank of each model over datasets (1 = best). Input for Friedman/Nemenyi (step 5)."""
    return table.rank(axis=1, ascending=False).mean().sort_values().round(2)


def confusion_totals(results: pd.DataFrame, protocol="clean") -> pd.DataFrame:
    d = results[results.protocol == protocol]
    return d.groupby(["dataset", "model"])[["tn", "fp", "fn", "tp"]].sum()


def plot_heatmap(table: pd.DataFrame, title: str, fname: str, center=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 3, 0.5 * table.shape[0] + 2))
    kw = dict(vmin=-abs(table.values).max(), vmax=abs(table.values).max()) if center == 0 else {}
    im = ax.imshow(table.values, cmap=cmap, aspect="auto", **kw)
    ax.set_xticks(range(table.shape[1])); ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0])); ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(title)
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / fname, dpi=150, bbox_inches="tight")
    return fig
