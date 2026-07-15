"""Tests for the hybrid DNN+ConvNet+GBDT classifier, in-memory fixtures only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from penux_ap.hybrid_models import HybridDNNConvGBDTClassifier


@pytest.fixture
def separable_binary_dataset():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 10)).astype(np.float32)
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    return X, y


@pytest.mark.parametrize("combo_method", ["average", "gbdt_heavy", "nn_heavy"])
def test_fit_predict_shapes(separable_binary_dataset, combo_method):
    X, y = separable_binary_dataset
    clf = HybridDNNConvGBDTClassifier(
        dnn_hidden=(16,), conv_channels=(8,), gbdt_n_estimators=20, combo_method=combo_method,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= {0, 1}


def test_learns_separable_signal(separable_binary_dataset):
    from sklearn.metrics import roc_auc_score
    X, y = separable_binary_dataset
    clf = HybridDNNConvGBDTClassifier(
        dnn_hidden=(32, 16), conv_channels=(8, 16), gbdt_n_estimators=50, combo_method="average",
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    assert roc_auc_score(y, proba) > 0.8


def test_invalid_combo_method_raises():
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = np.array([0, 1] * 15)
    clf = HybridDNNConvGBDTClassifier(combo_method="bogus")
    with pytest.raises(ValueError):
        clf.fit(X, y)
