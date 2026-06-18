"""Tests for model training and evaluation using in-memory DataFrames only."""
import numpy as np
import pandas as pd
import pytest

from penux_ap.models import get_model_registry, predict_proba_safe, train_model
from penux_ap.preprocessing import build_preprocessor, infer_feature_types
from penux_ap.evaluation import evaluate_binary_classifier


@pytest.fixture
def tiny_dataset():
    np.random.seed(42)
    n = 40
    df = pd.DataFrame({
        "age": np.random.uniform(30, 80, n),
        "wbc": np.random.uniform(5, 20, n),
        "crp": np.random.uniform(10, 200, n),
        "sex": np.random.choice(["M", "F"], n),
        "severe": np.random.randint(0, 2, n),
    })
    return df


def test_get_model_registry():
    registry = get_model_registry()
    assert "logistic_regression" in registry
    assert "random_forest" in registry


def test_train_logistic_regression(tiny_dataset):
    X = tiny_dataset.drop(columns=["severe"])
    y = tiny_dataset["severe"]
    types = infer_feature_types(tiny_dataset, "severe")
    preprocessor = build_preprocessor(types["numeric"], types["categorical"])
    model = train_model("logistic_regression", X, y, preprocessor)
    assert hasattr(model, "predict_proba")


def test_predict_proba_safe(tiny_dataset):
    X = tiny_dataset.drop(columns=["severe"])
    y = tiny_dataset["severe"]
    types = infer_feature_types(tiny_dataset, "severe")
    preprocessor = build_preprocessor(types["numeric"], types["categorical"])
    model = train_model("logistic_regression", X, y, preprocessor)
    proba = predict_proba_safe(model, X)
    assert proba.shape == (len(X),)
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_evaluate_binary_classifier():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = evaluate_binary_classifier(y_true, y_proba)
    for key in ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier_score"]:
        assert key in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
