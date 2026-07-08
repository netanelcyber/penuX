"""Tests for the from-scratch GBDT classifier, in-memory fixtures only."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from penux_ap.scratch_gbdt import ScratchGBDTClassifier


@pytest.fixture
def separable_binary_dataset():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    # y depends mostly on X[:, 0] and X[:, 1], plus noise -- learnable but not trivial.
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    return X, y


def test_fit_predict_shapes(separable_binary_dataset):
    X, y = separable_binary_dataset
    clf = ScratchGBDTClassifier(n_estimators=10, max_depth=2, learning_rate=0.3)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    pred = clf.predict(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}


def test_learns_separable_signal(separable_binary_dataset):
    X, y = separable_binary_dataset
    clf = ScratchGBDTClassifier(n_estimators=30, max_depth=3, learning_rate=0.2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    # Should fit the training data well above chance -- this is a from-scratch
    # implementation check, not a generalization benchmark.
    assert roc_auc_score(y, proba) > 0.85


def test_constant_target_does_not_crash():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = np.zeros(30, dtype=int)
    clf = ScratchGBDTClassifier(n_estimators=5, max_depth=2, learning_rate=0.1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (30, 2)
    assert np.all(np.isfinite(proba))
