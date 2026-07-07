# From-scratch ML implementations (Fortran)

Dependency-free Fortran implementations written to answer "how would this
actually learn something, implemented directly, no libraries" -- as a
companion to the scikit-learn/XGBoost/LightGBM/CatBoost pipeline in
`src/penux_ap` and the 784-model benchmark in
`scripts/benchmark_model_zoo.py`. Both are teaching/reference
implementations, not a substitute for the project's actual research
pipeline -- see `docs/sap_severity_gbdt_analysis_he.md` for the validated
results and `docs/dataset_sources.md` for the reversed-label caveat both
programs already account for (raw value `0` = SAP in both registered
datasets).

No BLAS/LAPACK, no ML library, no external dependency beyond a Fortran
2008 compiler (`gfortran`).

## 1. Logistic regression (`logreg_from_scratch.f90`)

CSV parsing, stratified train/test split, feature standardization, batch
gradient descent (L2-regularized), and evaluation (confusion matrix,
sensitivity, specificity, PPV, NPV, F1, AUROC via the exact Mann-Whitney U
statistic) are all implemented by hand.

### Build & run

```bash
gfortran -O2 -o logreg_from_scratch logreg_from_scratch.f90
./logreg_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv
./logreg_from_scratch ../data/public_sanitized/ap_lnn_sanitized.csv
```

Optional positional arguments: `<csv> [n_iter] [learning_rate] [l2_lambda]`
(defaults: `n_iter=3000`, `learning_rate=0.3`, `l2_lambda=1e-3`).

### Sample output (ap_multiml_sanitized.csv, 80/20 stratified split, seed 42)

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
`scripts/model_zoo.py` (AUROC in the 0.80-0.84 range on this dataset).

## 2. Gradient-boosted trees, XGBoost-style (`xgboost_from_scratch.f90`)

Implements the core of XGBoost's *Exact Greedy Algorithm for Split
Finding* (Chen & Guestrin, 2016, Algorithm 1): each of `n_estimators` trees
is fit to the first- and second-order gradient statistics (`g_i`, `h_i`)
of the logistic loss against the ensemble's current log-odds prediction,
splitting nodes on the regularized gain

```
Gain = 0.5*[GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda)] - gamma
```

with leaf weights `w* = -G/(H+lambda)` and additive shrinkage
(`learning_rate`) across boosting rounds. Also hand-written: CSV parsing,
stratified split, a recursive quicksort used for exact greedy split
search, tree building/prediction, and the same evaluation suite as the
logistic regression program above.

This is a from-scratch reference implementation of XGBoost's core
algorithm, not a performance-competitive reimplementation: no histogram
binning, no column subsampling, no sparsity-aware split finding -- just
the exact greedy algorithm over every feature at every node. That's
tractable at this project's dataset sizes (~700-1300 rows, depth <=4) but
would not scale to XGBoost's actual target regime (millions of rows).

### Build & run

```bash
gfortran -O2 -o xgboost_from_scratch xgboost_from_scratch.f90
./xgboost_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv
./xgboost_from_scratch ../data/public_sanitized/ap_lnn_sanitized.csv
```

Optional positional arguments: `<csv> [n_estimators] [max_depth] [learning_rate]`
(defaults: `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`;
`lambda=1.0`, `gamma=0.0`, and `min_child_weight=1.0` are fixed in
`main` -- edit the source to change them).

### Sample output (ap_multiml_sanitized.csv, 80/20 stratified split, seed 42)

```
Target distribution: 204 SAP / 1085 non-SAP
Train size: 1031   Test size: 258
Training time:     0.94 s (CPU)
AUROC       =  0.8650
Accuracy    =  0.8798
Sensitivity =  0.4146
Specificity =  0.9677
PPV         =  0.7083
NPV         =  0.8974
F1          =  0.5231
```

And on `ap_lnn_sanitized.csv`: AUROC 0.8720, F1 0.6087. Both are consistent
with the Python XGBoost configurations benchmarked in
`scripts/model_zoo.py` (AUROC 0.84 on multiml, 0.88 on lnn) --
training/inference logic is independently implemented, not linked against
the `xgboost` package.

## What both programs assume about the input

- Last CSV column is the raw binary target; all other columns are numeric
  features (true for both sanitized datasets in `data/public_sanitized/`).
- Raw target value `0` denotes SAP (the flip is applied internally,
  matching `scripts/benchmark_model_zoo.py --positive-value 0`).
- No missing values (true for both sanitized datasets as committed).
