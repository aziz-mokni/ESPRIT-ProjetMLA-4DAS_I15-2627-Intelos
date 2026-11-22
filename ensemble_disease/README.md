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
pytest -q            # 6 tests must pass
jupyter notebook notebooks/
```

## Data
- UCI datasets download automatically on first use (`ucimlrepo`) and are cached in `data/raw/`.
- Kaggle datasets (D4 `D4_heart_failure.csv`, D10 `D10_diabetes_2019.csv`) must be placed manually in `data/raw/`.

## Structure
```
configs/config.yaml     seed, CV settings
src/ensemble_disease/   cart.py (CART from scratch), data.py, config.py, toy_data.py
tests/                  unit tests (CART vs course annexes and scikit-learn)
notebooks/              01_cart_from_scratch.ipynb
results/                outputs (figures, tables)
```

## Status
- [x] Step 1: skeleton, config, data loader, CART from scratch validated on the course examples
- [ ] Step 2: EDA + leakage checks (duplicates) on the datasets
- [ ] Step 3+: models, tuning, evaluation, statistical tests, comparison with the article, MLOps
