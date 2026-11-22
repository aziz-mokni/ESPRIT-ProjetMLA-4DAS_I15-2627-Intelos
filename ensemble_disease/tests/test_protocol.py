import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ensemble_disease import protocol
from ensemble_disease.models import get_models


def _toy(n=120, dup=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    y = pd.Series((X.a + 0.5 * rng.normal(size=n) > 0).astype(int))
    X = pd.concat([X, X.iloc[:dup]], ignore_index=True)
    y = pd.concat([y, y.iloc[:dup]], ignore_index=True)
    return X, y


def test_clean_removes_duplicates_naive_keeps_them():
    X, y = _toy()
    Xn, _ = protocol.prepare(X, y, "naive")
    Xc, yc = protocol.prepare(X, y, "clean")
    assert len(Xn) == 160 and len(Xc) == 120 and len(Xc) == len(yc)
    assert not pd.concat([Xc, yc.rename("y")], axis=1).duplicated().any()


def test_evaluate_shape_and_columns():
    X, y = _toy()
    r = protocol.evaluate(DecisionTreeClassifier(random_state=0), X, y, "clean",
                          n_splits=3, n_repeats=2, n_jobs=1)
    assert len(r) == 6
    for m in protocol.METRICS + ["tn", "fp", "fn", "tp"]:
        assert m in r.columns
    assert (r[["tn", "fp", "fn", "tp"]].sum(axis=1) == r.n_test).all()


def test_naive_is_optimistic_on_duplicated_data():
    X, y = _toy(n=80, dup=70, seed=1)
    kw = dict(estimator=DecisionTreeClassifier(random_state=0), X=X, y=y, n_splits=5, n_repeats=3, n_jobs=1)
    naive = protocol.evaluate(protocol="naive", **kw).accuracy.mean()
    clean = protocol.evaluate(protocol="clean", **kw).accuracy.mean()
    assert naive > clean


def test_models_have_predict_proba():
    for name, m in get_models().items():
        assert hasattr(m, "predict_proba"), name
