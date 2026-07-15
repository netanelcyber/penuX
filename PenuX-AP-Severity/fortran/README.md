# From-scratch ML implementations (Fortran)

Dependency-free Fortran implementations written to answer "how would this
actually learn something, implemented directly, no libraries" -- as a
companion to the scikit-learn/XGBoost/LightGBM/CatBoost pipeline in
`src/penux_ap` and the 784-model benchmark in
`scripts/benchmark_model_zoo.py`. All three are teaching/reference
implementations, not a substitute for the project's actual research
pipeline -- see `docs/sap_severity_gbdt_analysis_he.md` for the validated
results and `docs/dataset_sources.md` for the reversed-label caveat every
program already accounts for (raw value `0` = SAP in both registered
datasets).

No BLAS/LAPACK, no ML library, no external dependency beyond a Fortran
2008 compiler (`gfortran`). Both programs' `sigmoid()` even avoids the
compiler's intrinsic `exp()`. Two from-scratch implementations of e^x are
included, both using the same range-reduction scheme (repeatedly halve x
until the remainder r has |r|<0.5, approximate e^r there, then undo the
reduction by squaring the result back up: e^x = (e^(x/2^k))^(2^k)):

- `taylor_exp(x)`: a 20-term Taylor series for e^r. Accurate to ~1e-15
  relative error against the intrinsic `exp()` across x in [-20, 20].
- `pade_exp(x)`: the **[3/3] Pade approximant** for e^r -- a *ratio of two
  cubic polynomials*, P(r)/Q(r) = (1 + r/2 + r^2/10 + r^3/120) /
  (1 - r/2 + r^2/10 - r^3/120), rather than a single polynomial. A
  rational function of this degree captures e^x with far fewer terms than
  a Taylor polynomial would need for the same accuracy. Accurate to
  ~1e-7 relative error (>=7 correct significant digits, matching to all
  4 decimal places requested) across x in [-20, 20].

`sigmoid()` currently calls `pade_exp`. Both were verified end-to-end: the
AUROC/F1 results reported below are identical regardless of which of the
two (or the intrinsic `exp()`) computes the exponential.

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

## 3. Feedforward network as a chain of polynomial/rational functions (`dnn_from_scratch.f90`)

The model is literally the composition `F(x) = f_L(f_{L-1}(...f_1(x)...))`:
`f_1` takes the full parameter vector as input, and every single `f_l` --
hidden layers *and* the output layer alike -- has the same form
`f_l(a) = sigmoid(W_l a + b_l)`, ending with a single number in (0,1) (the
predicted SAP probability). This is a deliberate constraint: every `f_l`
must be a polynomial or a rational function, so there's no ReLU (piecewise,
not a single polynomial/rational expression) anywhere. `sigmoid` is built
entirely from `pade_exp` (see above) -- a ratio of two polynomials -- so
every stage of the chain is provably polynomial/rational, all the way
through. Training is textbook backpropagation: the chain rule applied
layer by layer, backwards through the same composition, with class-balanced
gradient weighting (like `class_weight='balanced'`) since without it the
network settles into predicting everyone negative on these imbalanced
datasets (~16-19% SAP prevalence).

### Build & run

```bash
gfortran -O2 -o dnn_from_scratch dnn_from_scratch.f90
./dnn_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv 64,32
```

Optional positional arguments: `<csv> [hidden_sizes, comma-separated] [n_epochs] [lr]`
(defaults: `hidden_sizes=64,32`, `n_epochs=300`, `lr=0.05`).

### Sample output (ap_multiml_sanitized.csv, 80/20 stratified split, seed 42)

```
Network: composition of 3 functions f_1..f_L (all rational/sigmoid):
  f_1: R^59 -> R^64   f_i(a) = sigmoid(W_i a + b_i)
  f_2: R^64 -> R^32   f_i(a) = sigmoid(W_i a + b_i)
  f_3: R^32 -> R^1   f_i(a) = sigmoid(W_i a + b_i)
AUROC       =  0.8321
Accuracy    =  0.7597
Sensitivity =  0.7317
Specificity =  0.7650
PPV         =  0.3704
NPV         =  0.9379
F1          =  0.4918
```

On `ap_lnn_sanitized.csv`: AUROC 0.7575, F1 0.4225. An all-sigmoid network
trains slower than the ReLU-based `TorchDNNClassifier` in
`src/penux_ap/torch_models.py` (a known historical reason ReLU displaced
sigmoid in deep nets -- vanishing gradients compound across many sigmoid
layers), but the polynomial/rational constraint on every `f_l` was the
explicit point of this program, not raw accuracy.

## Pseudocode

### exp(x) via Taylor series with range reduction (used by both programs' `sigmoid`)

```text
function taylor_exp(x):
    r <- x; k <- 0
    while |r| > 0.5:            # range reduction: shrink to where the series converges fast
        r <- r / 2; k <- k + 1
    sum <- 1; term <- 1
    for i in 1..20:              # Taylor series for e^r: sum_{i=0..20} r^i / i!
        term <- term * r / i
        sum <- sum + term
    result <- sum
    for _ in 1..k:               # undo the reduction: e^x = (e^r)^(2^k)
        result <- result * result
    return result
```

### exp(x) via the [3/3] Pade approximant -- ratio of two polynomials (used by `sigmoid`)

```text
function pade_exp(x):
    r <- x; k <- 0
    while |r| > 0.5:             # same range reduction as taylor_exp
        r <- r / 2; k <- k + 1
    num <- 1 + r/2 + r^2/10 + r^3/120     # numerator: cubic polynomial P(r)
    den <- 1 - r/2 + r^2/10 - r^3/120     # denominator: cubic polynomial Q(r)
    result <- num / den                   # e^r  ~=  P(r) / Q(r)
    for _ in 1..k:
        result <- result * result
    return result
```

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

### Feedforward network as a chain of rational functions (`dnn_from_scratch.f90`)

```text
load CSV -> X (n x p), y_raw (n)
y <- 1 - y_raw
(train_idx, test_idx) <- stratified_split(y, train_frac=0.8)
standardize Xtr/Xte using Xtr's mean/std

# layer_sizes = [p, h_1, h_2, ..., h_{L-1}, 1]; every layer uses sigmoid.
initialize W_l, b_l for l in 1..L  (Xavier-scaled random weights, b=0)

w_pos <- n / (2 * n_positive);  w_neg <- n / (2 * n_negative)   # class balance

for epoch in 1..n_epochs:
    # forward pass: the composition F(x) = f_L(...f_1(x)...)
    A_0 <- Xtr
    for l in 1..L:
        Z_l <- A_{l-1} @ W_l + b_l
        A_l <- sigmoid(Z_l)                     # every f_l, hidden AND output
    proba <- A_L

    # backward pass: the chain rule, applied layer by layer, backwards
    weight_i <- w_pos if y_i == 1 else w_neg
    dZ_L <- weight * (A_L - y) / n               # combined sigmoid+BCE gradient, class-weighted
    for l in L..1:
        dW_l <- A_{l-1}^T @ dZ_l;  db_l <- column_sums(dZ_l)
        W_l <- W_l - lr * dW_l;   b_l <- b_l - lr * db_l
        if l > 1:
            dA_{l-1} <- dZ_l @ W_l^T
            dZ_{l-1} <- dA_{l-1} * A_{l-1} * (1 - A_{l-1})   # sigmoid'(z) = a*(1-a)

proba_test <- forward(Xte)
report AUROC(proba_test, yte), confusion_matrix(proba_test, yte, threshold=0.5)
```

## What all three programs assume about the input

- Last CSV column is the raw binary target; all other columns are numeric
  features (true for both sanitized datasets in `data/public_sanitized/`).
- Raw target value `0` denotes SAP (the flip is applied internally,
  matching `scripts/benchmark_model_zoo.py --positive-value 0`).
- No missing values (true for both sanitized datasets as committed).
