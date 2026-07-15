"""Gradient Boosted Decision Trees implemented from scratch.

No reliance on any existing GBDT library or tree implementation --
not scikit-learn's GradientBoostingClassifier/HistGradientBoostingClassifier,
not sklearn.tree, not XGBoost/LightGBM/CatBoost. Only numpy.

Follows the classical Friedman (2001) gradient boosting algorithm, as
summarized in Google's "Intro to Gradient Boosted Decision Trees"
(https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt):

  1. Start with a constant initial prediction (log-odds of the base rate).
  2. For each boosting round:
     a. Compute the negative gradient ("pseudo-residual") of the binomial
        log-loss w.r.t. the current prediction, for every training example:
        residual_i = y_i - sigmoid(F_i).
     b. Fit a regression tree to predict these pseudo-residuals from the
        input features -- an ordinary CART regression tree, splitting to
        minimize the sum of squared residuals within each child, built
        from scratch in this module.
     c. At each leaf, replace the raw mean-residual value with the
        loss-optimal value via a single Newton step (Friedman's TreeBoost
        refinement): leaf_value = sum(residual) / sum(p*(1-p)) over the
        leaf's training examples.
     d. Add learning_rate * tree_output to the running prediction F.
  3. Final prediction is sigmoid(F).

This is what distinguishes it algorithmically from XGBoost (see
fortran/xgboost_from_scratch.f90 for that variant): XGBoost's Exact Greedy
Algorithm evaluates every candidate split directly against the regularized
gain formula built from gradient/Hessian sums, baking the Newton step into
split *selection* itself. Here, split selection is ordinary variance
reduction on the residual target (as in classic Friedman TreeBoost /
scikit-learn's GradientBoostingClassifier design), and the Newton
correction is applied only afterwards, to fix up each already-built leaf's
output value.

A teaching/reference implementation -- see scripts/model_zoo.py for the
production-grade GBDT libraries (XGBoost, LightGBM, CatBoost, sklearn's own
GradientBoostingClassifier/HistGradientBoostingClassifier) actually used
for this project's headline results in docs/sap_severity_gbdt_analysis_he.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self):
        self.is_leaf = True
        self.value = 0.0
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None


def _best_split(X_col_sorted_idx, X_col, residual, min_samples_leaf):
    """Find the split point minimizing total SSE for a single feature.

    X_col_sorted_idx: indices (into the node's sample set) sorted by X_col value.
    Returns (best_sse_reduction, threshold) or (0.0, None) if no valid split.
    """
    n = len(X_col_sorted_idx)
    if n < 2 * min_samples_leaf:
        return 0.0, None

    vals = X_col[X_col_sorted_idx]
    res = residual[X_col_sorted_idx]

    cum_sum = np.cumsum(res)
    cum_sq = np.cumsum(res * res)
    total_sum = cum_sum[-1]
    total_sq = cum_sq[-1]

    # SSE of a set = sum(r^2) - (sum(r))^2 / n  (variance * n)
    best_gain = 0.0
    best_threshold = None
    parent_sse = total_sq - total_sum * total_sum / n

    for k in range(min_samples_leaf - 1, n - min_samples_leaf):
        if vals[k] == vals[k + 1]:
            continue
        n_left = k + 1
        n_right = n - n_left
        left_sum = cum_sum[k]
        left_sq = cum_sq[k]
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        left_sse = left_sq - left_sum * left_sum / n_left
        right_sse = right_sq - right_sum * right_sum / n_right
        gain = parent_sse - (left_sse + right_sse)
        if gain > best_gain:
            best_gain = gain
            best_threshold = 0.5 * (vals[k] + vals[k + 1])

    return best_gain, best_threshold


def _build_tree(X, residual, sample_idx, depth, max_depth, min_samples_leaf):
    node = _Node()
    n = len(sample_idx)
    node.value = float(np.mean(residual[sample_idx])) if n > 0 else 0.0

    if depth >= max_depth or n < 2 * min_samples_leaf:
        return node

    best_feature, best_threshold, best_gain = -1, None, 0.0
    for f in range(X.shape[1]):
        col = X[sample_idx, f]
        order = np.argsort(col, kind="mergesort")
        gain, threshold = _best_split(order, col, residual[sample_idx], min_samples_leaf)
        if gain > best_gain:
            best_gain, best_feature, best_threshold = gain, f, threshold

    if best_feature == -1:
        return node

    mask = X[sample_idx, best_feature] <= best_threshold
    left_idx = sample_idx[mask]
    right_idx = sample_idx[~mask]
    if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
        return node

    node.is_leaf = False
    node.feature = best_feature
    node.threshold = best_threshold
    node.left = _build_tree(X, residual, left_idx, depth + 1, max_depth, min_samples_leaf)
    node.right = _build_tree(X, residual, right_idx, depth + 1, max_depth, min_samples_leaf)
    return node


def _assign_leaves(node, X, sample_idx, leaf_members):
    """Walk each sample to its leaf node, recording membership for the Newton step."""
    if node.is_leaf:
        leaf_members.append((node, sample_idx))
        return
    mask = X[sample_idx, node.feature] <= node.threshold
    _assign_leaves(node.left, X, sample_idx[mask], leaf_members)
    _assign_leaves(node.right, X, sample_idx[~mask], leaf_members)


def _predict_tree(node, x_row):
    while not node.is_leaf:
        node = node.left if x_row[node.feature] <= node.threshold else node.right
    return node.value


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ScratchGBDTClassifier(BaseEstimator, ClassifierMixin):
    """Binary GBDT classifier, implemented from scratch (see module docstring).

    Parameters mirror the common subset of XGBoost/LightGBM/sklearn's GBM
    for ease of comparison in scripts/model_zoo.py.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.classes_ = np.unique(y)
        rate = np.clip(y.mean(), 0.01, 0.99)
        self.F0_ = float(np.log(rate / (1.0 - rate)))

        n = len(y)
        F = np.full(n, self.F0_)
        self.trees_ = []
        sample_idx_all = np.arange(n)

        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            residual = y - p
            hessian = p * (1.0 - p)

            root = _build_tree(X, residual, sample_idx_all, 0, self.max_depth, self.min_samples_leaf)

            # Newton step: overwrite each leaf's raw mean-residual value with
            # sum(residual) / sum(hessian) over that leaf's members.
            leaf_members = []
            _assign_leaves(root, X, sample_idx_all, leaf_members)
            for leaf_node, idx in leaf_members:
                h_sum = hessian[idx].sum()
                leaf_node.value = float(residual[idx].sum() / h_sum) if h_sum > 1e-12 else 0.0

            tree_pred = np.array([_predict_tree(root, X[i]) for i in range(n)])
            F += self.learning_rate * tree_pred
            self.trees_.append(root)

        return self

    def decision_function(self, X):
        check_is_fitted(self, "trees_")
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * np.array([_predict_tree(tree, X[i]) for i in range(X.shape[0])])
        return F

    def predict_proba(self, X):
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
