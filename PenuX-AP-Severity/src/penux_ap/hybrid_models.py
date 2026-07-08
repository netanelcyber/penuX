"""Hybrid classifier combining a DNN, a 1D ConvNet, and a GBDT (LightGBM).

Trains all three sub-models on the same data and combines their predicted
probabilities via one of three fixed-weight schemes:
    - "average":    equal 1/3 weight to each sub-model
    - "gbdt_heavy": 0.6 GBDT + 0.2 DNN + 0.2 ConvNet
    - "nn_heavy":   0.2 GBDT + 0.4 DNN + 0.4 ConvNet

This is an exploratory architecture -- a simple fixed-weight ensemble
rather than a learned stacking meta-model (which would need an internal
cross-fitting loop to avoid leakage; out of scope for this benchmarking
utility). See scripts/model_zoo.py for how this is parameterized into 300
configurations (DNN architecture x ConvNet architecture x GBDT
hyperparameters x combination method).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from penux_ap.torch_models import TorchConvNetClassifier, TorchDNNClassifier

_COMBO_WEIGHTS = {
    "average": (1 / 3, 1 / 3, 1 / 3),
    "gbdt_heavy": (0.2, 0.2, 0.6),
    "nn_heavy": (0.4, 0.4, 0.2),
}


class HybridDNNConvGBDTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        dnn_hidden=(64, 32), dnn_dropout=0.2, dnn_lr=1e-3,
        conv_channels=(16, 32), conv_kernel=3, conv_dropout=0.2, conv_lr=1e-3,
        gbdt_n_estimators=100, gbdt_max_depth=3, gbdt_learning_rate=0.1,
        combo_method="average",
    ):
        self.dnn_hidden = dnn_hidden
        self.dnn_dropout = dnn_dropout
        self.dnn_lr = dnn_lr
        self.conv_channels = conv_channels
        self.conv_kernel = conv_kernel
        self.conv_dropout = conv_dropout
        self.conv_lr = conv_lr
        self.gbdt_n_estimators = gbdt_n_estimators
        self.gbdt_max_depth = gbdt_max_depth
        self.gbdt_learning_rate = gbdt_learning_rate
        self.combo_method = combo_method

    def fit(self, X, y):
        if self.combo_method not in _COMBO_WEIGHTS:
            raise ValueError(f"Unknown combo_method: {self.combo_method}")
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        self.dnn_ = TorchDNNClassifier(
            hidden_sizes=self.dnn_hidden, dropout=self.dnn_dropout, lr=self.dnn_lr,
            max_epochs=30, patience=5,
        )
        self.dnn_.fit(X, y)

        self.conv_ = TorchConvNetClassifier(
            channels=self.conv_channels, kernel_size=self.conv_kernel, dropout=self.conv_dropout,
            lr=self.conv_lr, max_epochs=30, patience=5,
        )
        self.conv_.fit(X, y)

        self.gbdt_ = LGBMClassifier(
            n_estimators=self.gbdt_n_estimators, max_depth=self.gbdt_max_depth,
            learning_rate=self.gbdt_learning_rate, random_state=42, verbose=-1,
        )
        self.gbdt_.fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "gbdt_")
        X = check_array(X)
        w_dnn, w_conv, w_gbdt = _COMBO_WEIGHTS[self.combo_method]
        p_dnn = self.dnn_.predict_proba(X)[:, 1]
        p_conv = self.conv_.predict_proba(X)[:, 1]
        p_gbdt = self.gbdt_.predict_proba(X)[:, 1]
        p1 = w_dnn * p_dnn + w_conv * p_conv + w_gbdt * p_gbdt
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
