"""Model registry for the ensemble comparison.

All hyper-parameters below are FIXED, reasonable defaults (step 3 = first comparison).
Systematic tuning with an equal budget for every model comes in step 4.
Every model exposes predict_proba (needed for ROC-AUC / AUPRC).
"""
from __future__ import annotations
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # xgboost optional: the comparison still runs without it
    XGBClassifier = None


def get_models(seed: int = 42) -> dict:
    """Return {name: unfitted estimator}. n_jobs=1 inside models: parallelism is done over CV folds."""
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                             n_estimators=100, random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50,
                                     random_state=seed, n_jobs=1),
        "RandomForest": rf,
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "AdaBoost": ada,
        "GradientBoosting": gb,
    }
    base = [("rf", rf), ("ada", ada), ("gb", gb)]
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                          eval_metric="logloss", random_state=seed, n_jobs=1,
                                          verbosity=0)
        base.append(("xgb", models["XGBoost"]))
    models["Stacking"] = StackingClassifier(
        estimators=[(n, clone(m)) for n, m in base],
        final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1)
    return models
