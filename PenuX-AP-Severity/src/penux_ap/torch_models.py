"""PyTorch-based deep learning classifiers, wrapped as scikit-learn estimators.

Two architectures for tabular SAP prediction:
- TorchDNNClassifier: a configurable feedforward network (depth, width,
  dropout, batch norm).
- TorchConvNetClassifier: a 1D convolutional network that treats the
  standardized feature vector as a length-p sequence (1 channel), with a
  stack of Conv1d + BatchNorm1d + ReLU + pooling blocks followed by a
  linear head. Convolution over a tabular feature vector has no natural
  spatial/temporal structure (unlike an image or time series), but this
  is included as a from-first-principles architectural exploration.

CPU-only, no GPU dependency. Trained with early stopping on a held-out
validation split carved out of the training data to avoid overfitting on
these small datasets (~700-1300 rows).

Used only in scripts/model_zoo.py for benchmarking against the GBDT
models and everything else in the zoo -- not part of the production
model registry in penux_ap.models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _DNN(nn.Module):
    def __init__(self, n_features, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ConvNet(nn.Module):
    def __init__(self, n_features, channels, kernel_size, dropout):
        super().__init__()
        conv_layers = []
        in_ch = 1
        length = n_features
        for out_ch in channels:
            pad = kernel_size // 2
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
            length = length // 2
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * max(length, 1), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x).squeeze(-1)


class _TorchClassifierBase(BaseEstimator, ClassifierMixin):
    """Shared fit/predict machinery for the DNN and ConvNet wrappers."""

    def _build_model(self, n_features) -> nn.Module:
        raise NotImplementedError

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)
        torch.manual_seed(42)

        n = len(y)
        if n >= 40 and len(np.unique(y)) == 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y

        self.model_ = self._build_model(X.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        best_val_loss = float("inf")
        best_state = None
        patience_left = self.patience
        batch_size = min(self.batch_size, len(X_tr))

        for _ in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(X_tr_t))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                logits = self.model_(X_tr_t[idx])
                loss = criterion(logits, y_tr_t[idx])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X)
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            p1 = torch.sigmoid(logits).numpy()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchDNNClassifier(_TorchClassifierBase):
    """Feedforward deep neural network classifier (PyTorch, CPU)."""

    def __init__(
        self, hidden_sizes=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
        max_epochs=200, batch_size=32, patience=15,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def _build_model(self, n_features):
        return _DNN(n_features, self.hidden_sizes, self.dropout)


class TorchConvNetClassifier(_TorchClassifierBase):
    """1D convolutional network classifier over the tabular feature vector (PyTorch, CPU)."""

    def __init__(
        self, channels=(16, 32), kernel_size=3, dropout=0.2, lr=1e-3, weight_decay=1e-4,
        max_epochs=200, batch_size=32, patience=15,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def _build_model(self, n_features):
        return _ConvNet(n_features, self.channels, self.kernel_size, self.dropout)
