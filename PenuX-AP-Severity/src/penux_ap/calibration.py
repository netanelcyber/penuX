"""Probability calibration: Platt/sigmoid, isotonic, reliability curves."""
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def calibrate_model(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Wrap a fitted model with CalibratedClassifierCV using validation data.

    Args:
        model: A fitted sklearn Pipeline.
        X_val: Validation features (not used in training).
        y_val: Validation labels.
        method: 'sigmoid' (Platt) or 'isotonic'.
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"method must be 'sigmoid' or 'isotonic', got '{method}'")
    cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    log.info("Calibrated model with method='%s'.", method)
    return cal


def calibration_curve_data(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Return reliability curve data (fraction of positives vs mean predicted probability)."""
    y_true = np.asarray(y_true, dtype=int)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "brier_score": brier_score(y_true, y_proba),
    }


def brier_score(y_true: np.ndarray | pd.Series, y_proba: np.ndarray) -> float:
    """Compute Brier score (lower is better, 0.0 is perfect)."""
    return float(brier_score_loss(np.asarray(y_true, dtype=int), y_proba))
