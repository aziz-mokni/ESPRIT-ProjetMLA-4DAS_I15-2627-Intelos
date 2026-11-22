# Ensemble Learning for Disease Prediction (AML – Project 2)

Reference article: Mahajan et al., *A comparative evaluation of machine learning
ensemble approaches for disease prediction using multiple datasets*,
Health Technol. 14, 597–613 (2024). https://doi.org/10.1007/s12553-024-00835-w

## Install (Windows / macOS / Linux, Python >= 3.9)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q            # 14 tests must pass
jupyter notebook notebooks/
```

## Data
- **UCI** (D2, D5, D6, D8, D11, D12, D16): downloaded automatically on first use (`ucimlrepo`, internet needed) and cached in `data/raw/`.
- **Kaggle** (manual, free account needed). Put the CSVs in `data/raw/` with EXACTLY these names:

| Dataset | Kaggle page (search this name) | Save as |
|---|---|---|
| D4 Heart failure clinical records | "Heart Failure Prediction" by *andrewmvd* | `data/raw/D4_heart_failure.csv` |
| D10 Diabetes dataset 2019 | "Diabetes Dataset 2019" by *tigganeha4* | `data/raw/D10_diabetes_2019.csv` |

  Download the zip from the dataset page, extract the CSV, rename it. If D10 fails to load, the error message lists
  the real column names (the target is expected to be `Diabetic`): send them and we adjust `REGISTRY` in `data.py`.
- The positive class (y = 1) is always the disease class; the rule for each dataset is in `REGISTRY` (`data.py`).

## Structure
```
configs/config.yaml     seed, CV settings
src/ensemble_disease/   cart.py (CART from scratch), data.py, eda.py, models.py, protocol.py, config.py, toy_data.py
tests/                  unit tests (CART vs course annexes and scikit-learn)
notebooks/              02_eda.ipynb, 03_models_protocol.ipynb ; annex/A1_cart_from_scratch.ipynb (theory annex)
results/                outputs (figures, tables)
```

## Status
- [x] Step 1: skeleton, config, data loader, CART from scratch validated on the course examples
- [x] Step 2 (code): data loader with explicit positive class, EDA notebook + leakage checks
- [ ] Step 2 (run): run `02_eda.ipynb` on the real data and review the outputs
- [x] Step 3 (code): evaluation protocols (naive vs clean) + 8 models + metrics, notebook 03
- [ ] Step 3 (run): run `03_models_protocol.ipynb` (10-20 min; set QUICK=True for a fast test)
- [ ] Step 4+: hyper-parameter tuning with equal budget, Friedman/Nemenyi tests, SHAP, comparison with the article, MLOps