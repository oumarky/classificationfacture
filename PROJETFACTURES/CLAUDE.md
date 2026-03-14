# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Invoice classification system for medical billing. Uses ML to automatically classify invoices as `validée` (validated) or `à_corriger` (needs correction).

status_verification=>observations_verification
a_corriger=>Quantité anormale d’acte Réduire la quantité à 1
validee=>RAS
a_corriger=>Bien et service non éligible pour la catégorie de patient
a_corriger=>L’acte ne correspond pas avec le sexe du patient
a_corriger=>Absence de: registre_number
a_corriger=>Montant d'évacuation anormal
a_corriger=>Absence de: nom_patient
a_corriger=>Chevauchement de date d'hospitalisation détecté
a_corriger=>Facture saisie avant date d'entrée; Date de sortie est avant la date d’entrée
a_corriger=>Prestation non éligible pour la Planification familiale avec type de prestation Sayana Press

All code is in Jupyter notebooks — no Python scripts.

## Running Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Or with JupyterLab
jupyter lab
```

Run notebooks in order: **ML_FEATURE.ipynb first**, then **ML_MODEL.ipynb**.

## Dependencies

No requirements.txt exists. Inferred from notebook imports:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost torch imbalanced-learn matplotlib seaborn joblib
```

PyTorch is configured to use Apple MPS (M2 Pro GPU). No CUDA needed.

## Architecture & Data Pipeline

```
data/fisprod_acorig.csv   (raw invoices ~10k rows)
data/fisprod_valide.csv   (validated invoices ~10k rows)
         ↓
ML_FEATURE.ipynb          (feature engineering → 40 boolean flags)
         ↓
data/dfforml2.csv         (1,434 rows, balanced dataset input)
         ↓
ML_MODEL.ipynb            (LightGBM / XGBoost / PyTorch MLP)
         ↓
models/                   (trained model artifacts)
```

### ML_FEATURE.ipynb — Feature Engineering

Applies three categories of validation rules to raw invoice data:

- **Formal rules (A)**: Missing required fields (nom_patient, registre_number, age_patient, etc.)
- **Temporal rules (B)**: Date parsing and coherence (entry/exit/visit ordering, hospitalization overlap)
- **Categorical rules (C)**: Domain-specific checks (sex-act compatibility, child eligibility, evacuation costs, act/cost existence)

Output: `dfforml2.csv` with 40 boolean feature columns + `status_verification` target.

### ML_MODEL.ipynb — Model Training

Loads `dfforml2.csv`, balances to 5,000 per class, trains three models:

- **LightGBM**: Best benchmark — ~73% F1, ~73% AUC (minimal overfitting)
- **XGBoost**: Same metrics tracked
- **PyTorch MLP**: 40 → 64 → 32 → 1, dropout 0.3/0.2, BCELoss, Adam, early stopping, MPS backend

Key utility functions in the notebook:

- `train_pipeline()` — full preprocessing + model training
- `predict_one()` — classify a single invoice dict

## Key Data Files

| File                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| `data/fisprod_acorig.csv` | Raw input (à corriger side)                   |
| `data/fisprod_valide.csv` | Raw input (validée side)                      |
| `data/dfforml2.csv`       | Feature-engineered ML-ready dataset           |
| `data/dfforml.csv`        | Intermediate format (pre-feature engineering) |

## Git Branches

- `main` — production
- `dev_claude` — current development branch
- `dev_ml_facturegartuite` — ML for free invoices feature
- `cursor_ml` — parallel ML development
