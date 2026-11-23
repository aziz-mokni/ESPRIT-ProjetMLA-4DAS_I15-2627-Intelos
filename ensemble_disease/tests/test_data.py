import pandas as pd
from ensemble_disease.data import REGISTRY, clean_frame
from ensemble_disease import eda


def test_positive_rules_on_raw_values():
    assert REGISTRY["D2_heart_cleveland"]["positive"](pd.Series([0, 1, 4])).tolist() == [False, True, True]
    assert REGISTRY["D12_ckd"]["positive"](pd.Series(["ckd", "notckd\t", "ckd "])).tolist() == [True, False, True]
    assert REGISTRY["D8_early_diabetes"]["positive"](pd.Series(["Positive", "Negative"])).tolist() == [True, False]
    assert REGISTRY["D6_ilpd"]["positive"](pd.Series([1, 2])).tolist() == [True, False]


def test_clean_frame_strips_and_converts():
    df = pd.DataFrame({" pcv ": ["44", "\t?", " 52"], "al": ["yes", " no", "?"]})
    out = clean_frame(df)
    assert list(out.columns) == ["pcv", "al"]
    assert pd.api.types.is_numeric_dtype(out["pcv"]) and out["pcv"].isna().sum() == 1
    assert out["al"].isna().sum() == 1 and out["al"].iloc[1] == "no"


def test_duplicate_leakage_detects_copies():
    X = pd.DataFrame({"a": [1, 2, 3, 4] * 25, "b": list(range(4)) * 25})
    y = pd.Series([0, 1] * 50)
    r = eda.duplicate_leakage(X, y, n_repeats=5)
    assert r["test_rows_with_train_copy_pct"] > 90


def test_fetch_uci_uses_original_for_bupa(tmp_path, monkeypatch):
    """BUPA: label 'selector' has role Other -> must come from ds.data.original; 'drinks' stays a feature."""
    import types, sys
    from ensemble_disease import data as D
    orig = pd.DataFrame({"mcv": [85, 90, 88], "drinks": [0.5, 4.0, 2.0], "selector": [1, 2, 1]})
    ds = types.SimpleNamespace(data=types.SimpleNamespace(
        features=orig[["mcv"]], targets=orig[["drinks"]], original=orig))
    monkeypatch.setitem(sys.modules, "ucimlrepo", types.SimpleNamespace(fetch_ucirepo=lambda id: ds))
    cache = tmp_path / "D5.csv"
    D._fetch_uci(D.REGISTRY["D5_liver_disorders"], cache)
    out = pd.read_csv(cache)
    assert list(out.columns) == ["mcv", "drinks", "__target__"]
    assert out["__target__"].tolist() == [1, 2, 1]


def test_registry_covers_the_16_datasets_of_the_article():
    ids = sorted(int(k.split("_")[0][1:]) for k in REGISTRY)
    assert ids == list(range(1, 17))
    for k, v in REGISTRY.items():
        assert "note" in v and "article" in v, k
        if v.get("task") != "multiclass":
            assert callable(v["positive"]), k


def test_multiclass_loader_and_drop_cols(tmp_path, monkeypatch):
    from ensemble_disease import data as D
    monkeypatch.setattr(D, "RAW", tmp_path)
    pd.DataFrame({"lesion_id": ["a", "b", "c"], "image_id": ["x", "y", "z"], "dx": ["nv", "mel", "bkl"],
                  "age": [30, 40, 50]}).to_csv(tmp_path / "D15_ham10000.csv", index=False)
    X, y = D.load_dataset("D15_ham10000")
    assert list(X.columns) == ["age"] and sorted(y.unique()) == [0, 1, 2]