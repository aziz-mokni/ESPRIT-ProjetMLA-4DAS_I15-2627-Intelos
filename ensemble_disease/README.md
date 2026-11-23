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
pytest -q            # 16 tests must pass
jupyter notebook notebooks/
```

## Data
The project covers the **16 datasets of Table 2** of the reference article (Mahajan et al., 2024).
UCI datasets are downloaded automatically on first use (`ucimlrepo`, internet needed) and cached in `data/raw/`.
Kaggle datasets must be downloaded by hand (free account) and saved in `data/raw/` **with exactly these names**.
The positive class (y = 1) is always the disease class; the rule of each dataset is in `REGISTRY` (`src/ensemble_disease/data.py`).

| ID | Dataset | Source | Rows | File to create in `data/raw/` |
|---|---|---|---|---|
| D1 | Heart Disease Dataset (Comprehensive) | Kaggle | 1190 | `D1_heart_comprehensive.csv` |
| D2 | Heart disease (Cleveland) | UCI 45 (auto) | 303 | - |
| D3 | Heart Disease Dataset (johnsmith88) | Kaggle | 1025 | `D3_heart_disease.csv` |
| D4 | Heart Failure Prediction | Kaggle | 299 | `D4_heart_failure.csv` |
| D5 | Liver Disorders (BUPA) | UCI 60 (auto) | 345 | - |
| D6 | Indian Liver Patient (ILPD) | UCI 225 (auto) | 583 | - |
| D7 | COVID-19 effect on liver cancer prediction (fedesoriano) | Kaggle | 450 | `D7_covid_liver_cancer.csv` |
| D8 | Early-stage diabetes risk | UCI 529 (auto) | 520 | - |
| D9 | Diabetes prediction, KNN (abdallamahgoub, Pima) | Kaggle | 768 | `D9_diabetes_pima.csv` |
| D10 | Diabetes Dataset 2019 (tigganeha4) | Kaggle | 952 | `D10_diabetes_2019.csv` |
| D11 | Diabetic Retinopathy Debrecen | UCI 329 (auto) | 1151 | - |
| D12 | Chronic Kidney Disease | UCI 336 (auto; Kaggle `mansoordaku/ckdisease` has the same patients) | 400 | - |
| D13 | Chronic Kidney Disease (abhia1999) | Kaggle | 400 | `D13_ckd_kaggle.csv` |
| D14 | Kidney Stone (harshghadiya) | Kaggle | 90 | `D14_kidney_stone.csv` |
| D15 | Skin Cancer MNIST HAM10000 (metadata CSV, multiclass) | Kaggle `kmader/skin-cancer-mnist-ham10000` | 10015 | `D15_ham10000.csv` (= `HAM10000_metadata.csv`) |
| D16 | Dermatology (6 classes; binarised for the binary protocol) | UCI 33 (auto) | 366 | - |

If a Kaggle file has a different target column than expected, the loader error lists the real column names; adjust `REGISTRY`.
D15 and the original six-class D16 need multiclass metrics (planned); the binary pipeline skips D15.

## Structure
```
configs/config.yaml     seed, CV settings
src/ensemble_disease/   cart.py (CART from scratch), data.py, eda.py, models.py, protocol.py, config.py, toy_data.py
tests/                  unit tests (CART vs course annexes and scikit-learn)
notebooks/              02_eda.ipynb, 03_models_protocol.ipynb ; annex/A1_cart_from_scratch.ipynb (theory annex)
results/                outputs (figures, tables)
```
