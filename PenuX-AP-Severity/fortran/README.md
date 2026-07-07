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

## Pseudocode

### Logistic regression (`logreg_from_scratch.f90`)

```text
load CSV -> X (n x p), y_raw (n)              # last column = target
y <- 1 - y_raw                                 # raw 0 = SAP, flip so 1 = SAP

(train_idx, test_idx) <- stratified_split(y, train_frac=0.8)
Xtr, ytr <- X[train_idx], y[train_idx]
Xte, yte <- X[test_idx],  y[test_idx]

mu, sigma <- column_mean(Xtr), column_std(Xtr)        # fit on train only
Xtr <- (Xtr - mu) / sigma
Xte <- (Xte - mu) / sigma                              # apply train stats to test

w <- zeros(p); b <- 0
for it in 1..n_iter:
    z    <- Xtr @ w + b
    p̂    <- sigmoid(z)
    grad_w <- Xtr^T @ (p̂ - ytr) / n_train + lambda * w   # L2 penalty
    grad_b <- mean(p̂ - ytr)
    w <- w - lr * grad_w
    b <- b - lr * grad_b

proba_test <- sigmoid(Xte @ w + b)
report AUROC(proba_test, yte), confusion_matrix(proba_test, yte, threshold=0.5)
```

### Gradient-boosted trees, XGBoost-style (`xgboost_from_scratch.f90`)

```text
load CSV -> X (n x p), y_raw (n)
y <- 1 - y_raw
(train_idx, test_idx) <- stratified_split(y, train_frac=0.8)

rate  <- mean(ytr) clipped to [0.01, 0.99]
F     <- log(rate / (1 - rate))  for every training row     # log-odds init

for m in 1..n_estimators:
    p̂    <- sigmoid(F)
    g    <- p̂ - ytr                        # first-order gradient  (per row)
    h    <- p̂ * (1 - p̂)                    # second-order gradient (per row)
    tree <- BUILD_TREE(Xtr, g, h, depth=0)
    F    <- F + learning_rate * tree.predict(Xtr)

function BUILD_TREE(X, g, h, idx, depth):
    G <- sum(g[idx]); H <- sum(h[idx])
    if depth >= max_depth or |idx| < 2 or H < 2*min_child_weight:
        return LEAF(value = -G / (H + lambda))

    best_gain <- 0; best_split <- none
    for each feature f in 1..p:
        sort idx by X[idx, f]
        GL <- 0; HL <- 0
        for each candidate split point k (between distinct sorted values):
            GL += g[k]; HL += h[k]
            GR <- G - GL; HR <- H - HL
            if HL < min_child_weight or HR < min_child_weight: continue
            gain <- 0.5 * ( GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda) ) - gamma
            if gain > best_gain: best_gain, best_split <- gain, (f, threshold_k)

    if best_split is none:
        return LEAF(value = -G / (H + lambda))

    left_idx, right_idx <- partition idx by X[idx, best_split.f] <= best_split.threshold
    return NODE(best_split.f, best_split.threshold,
                left  = BUILD_TREE(X, g, h, left_idx,  depth+1),
                right = BUILD_TREE(X, g, h, right_idx, depth+1))

proba_test <- sigmoid( F_init + learning_rate * sum(tree.predict(Xte) for tree in trees) )
report AUROC(proba_test, yte), confusion_matrix(proba_test, yte, threshold=0.5)
```

## What both programs assume about the input

- Last CSV column is the raw binary target; all other columns are numeric
  features (true for both sanitized datasets in `data/public_sanitized/`).
- Raw target value `0` denotes SAP (the flip is applied internally,
  matching `scripts/benchmark_model_zoo.py --positive-value 0`).
- No missing values (true for both sanitized datasets as committed).
