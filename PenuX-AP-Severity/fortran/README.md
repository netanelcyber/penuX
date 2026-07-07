# Logistic regression from scratch (Fortran)

A from-scratch, dependency-free implementation of L2-regularized logistic
regression for SAP prediction, written to answer "how would this actually
learn something, implemented directly, no libraries" -- as a companion to
the scikit-learn/XGBoost/LightGBM/CatBoost pipeline in `src/penux_ap` and
the 784-model benchmark in `scripts/benchmark_model_zoo.py`.

Everything is implemented by hand in `logreg_from_scratch.f90`: CSV
parsing, stratified train/test split, feature standardization, batch
gradient descent, and evaluation (confusion matrix, sensitivity,
specificity, PPV, NPV, F1, AUROC via the exact Mann-Whitney U statistic).
No BLAS/LAPACK, no ML library, no external dependency beyond a Fortran
2008 compiler.

This is a teaching/reference implementation, not a substitute for the
project's actual research pipeline -- see
`docs/sap_severity_gbdt_analysis_he.md` for the validated GBDT results and
`docs/dataset_sources.md` for the reversed-label caveat this program
already accounts for (raw value 0 = SAP in both registered datasets).

## Build

```bash
gfortran -O2 -o logreg_from_scratch logreg_from_scratch.f90
```

## Run

```bash
./logreg_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv
./logreg_from_scratch ../data/public_sanitized/ap_lnn_sanitized.csv
```

Optional positional arguments: `<csv> [n_iter] [learning_rate] [l2_lambda]`
(defaults: `n_iter=3000`, `learning_rate=0.3`, `l2_lambda=1e-3`).

## What it assumes about the input

- Last CSV column is the raw binary target; all other columns are numeric
  features (true for both sanitized datasets in `data/public_sanitized/`).
- Raw target value `0` denotes SAP (the flip is applied internally,
  matching `scripts/benchmark_model_zoo.py --positive-value 0`).
- No missing values (true for both sanitized datasets as committed).

## Sample output (ap_multiml_sanitized.csv, 80/20 stratified split, seed 42)

```
Target distribution: 204 SAP / 1085 non-SAP
Train size: 1031   Test size: 258
AUROC       =  0.8489
Accuracy    =  0.8837
Sensitivity =  0.4390
Specificity =  0.9677
PPV         =  0.7200
NPV         =  0.9013
F1          =  0.5455
```

Consistent with the scikit-learn logistic regression configurations in
`scripts/model_zoo.py` (AUROC in the 0.80-0.84 range on this dataset) --
this from-scratch Fortran version isn't more accurate than sklearn's, it's
meant to demonstrate the algorithm's mechanics transparently.
