"""Tests for the PyTorch DNN/ConvNet classifiers, in-memory fixtures only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from penux_ap.torch_models import TorchConvNetClassifier, TorchDNNClassifier


@pytest.fixture
def separable_binary_dataset():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 10)).astype(np.float32)
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    return X, y


@pytest.mark.parametrize("cls,kwargs", [
    (TorchDNNClassifier, {"hidden_sizes": (16,), "max_epochs": 20}),
    (TorchConvNetClassifier, {"channels": (4, 8), "max_epochs": 20}),
])
def test_fit_predict_shapes(separable_binary_dataset, cls, kwargs):
    X, y = separable_binary_dataset
    clf = cls(**kwargs)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= {0, 1}


def test_dnn_learns_separable_signal(separable_binary_dataset):
    from sklearn.metrics import roc_auc_score
    X, y = separable_binary_dataset
    clf = TorchDNNClassifier(hidden_sizes=(32, 16), max_epochs=100)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    assert roc_auc_score(y, proba) > 0.8
