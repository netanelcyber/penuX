# System Architecture

## Overview

```
User / Researcher
    │
    ├── scripts/sanitize_datasets.py   ← sanitize local/public data
    ├── scripts/summarize_datasets.py  ← inspect dataset
    ├── scripts/run_baseline.py        ← train & evaluate baseline models
    └── scripts/evaluate_model.py      ← evaluate saved model on new data
         │
         ▼
    src/penux_ap/
    ├── datasets.py       load CSV/XLSX, detect target, sanitize identifiers
    ├── preprocessing.py  impute, encode, scale (fit on train only)
    ├── features.py       AP feature definitions, column aliases
    ├── labels.py         target detection, binarization
    ├── models.py         LR, RF, HGBT, XGB, LGB, MLP
    ├── calibration.py    sigmoid/isotonic calibration, reliability curves
    ├── evaluation.py     AUROC, AUPRC, confusion matrices, bootstrap CI
    ├── explainability.py permutation importance, optional SHAP
    ├── clinical_scores.py SIRS, BISAP, Ranson, APACHE II, Modified CTSI
    ├── leadtime.py       horizon detection, operating-point frontier
    └── utils.py          logging, I/O
         │
         ▼
    outputs/
    ├── models/best_model.joblib
    └── reports/metrics.json, threshold_table.csv, confusion_matrices.json
         │
         ▼
    api/main.py           FastAPI research-only endpoint (PENUX_AP_MODEL_PATH)
```

## Data Flow

1. Raw local data → `sanitize_datasets.py` → `data/public_sanitized/`
2. Sanitized data → `run_baseline.py` → train/test split → model pipeline
3. Fitted model → evaluation metrics + confusion matrices → `outputs/reports/`
4. Best model → `outputs/models/best_model.joblib`
5. Saved model → `api/main.py` → `/predict` endpoint (research only)

## Anti-Leakage Design

- `build_preprocessor` returns an unfitted ColumnTransformer
- `fit_transform` is called only on training data
- `transform` is called on validation/test data separately
- No information from the test set influences preprocessing parameters
