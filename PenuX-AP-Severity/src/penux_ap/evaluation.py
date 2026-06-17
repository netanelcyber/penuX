"""Binary classification evaluation: metrics, thresholds, bootstrapping.

Confusion matrices are computed at multiple thresholds to support
time-based or threshold-sweep analysis (e.g. every 4-hour horizon).
"""
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from penux_ap.config import DEFAULT_THRESHOLD, RANDOM_SEED

log = logging.getLogger(__name__)


def _safe_metric(func, *args, default=float("nan"), **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log.warning("Metric computation failed: %s", e)
        return default


def evaluate_binary_classifier(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute a full suite of binary classification metrics.

    Returns AUROC, AUPRC, accuracy, sensitivity, specificity,
    PPV, NPV, F1, Brier score, and confusion matrix at the given threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
        if len(np.unique(y_true)) == 2 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")

    return {
        "auroc":       _safe_metric(roc_auc_score, y_true, y_proba),
        "auprc":       _safe_metric(average_precision_score, y_true, y_proba),
        "brier_score": _safe_metric(brier_score_loss, y_true, y_proba),
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv":         ppv,
        "npv":         npv,
        "f1":          _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "threshold":   threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    }


def threshold_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute metrics at multiple decision thresholds.

    Useful for plotting operating-point curves and for comparing
    confusion matrices across decision boundaries (e.g. every 4-hour horizon).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()
    rows = []
    for t in thresholds:
        m = evaluate_binary_classifier(y_true, y_proba, threshold=t)
        rows.append({
            "threshold":   t,
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "ppv":         m["ppv"],
            "npv":         m["npv"],
            "f1":          m["f1"],
            "accuracy":    m["accuracy"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        })
    return pd.DataFrame(rows)


def confusion_matrix_at_thresholds(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Return a list of confusion matrices, one per threshold.

    Each entry has: threshold, TP, TN, FP, FN, sensitivity, specificity, PPV, NPV.
    Designed to support reporting confusion matrices at multiple operating points
    (e.g. every 4-hour time horizon or every 0.1 threshold step).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1).tolist()
    results = []
    for t in thresholds:
        m = evaluate_binary_classifier(y_true, y_proba, threshold=t)
        results.append({
            "threshold":   t,
            "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
            "sensitivity": round(m["sensitivity"], 4) if not np.isnan(m["sensitivity"]) else None,
            "specificity": round(m["specificity"], 4) if not np.isnan(m["specificity"]) else None,
            "ppv":         round(m["ppv"], 4) if not np.isnan(m["ppv"]) else None,
            "npv":         round(m["npv"], 4) if not np.isnan(m["npv"]) else None,
        })
    return results


def bootstrap_auc_ci(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=int)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return {
        "mean": float(np.mean(aucs)),
        "ci_lower": float(np.percentile(aucs, 2.5)),
        "ci_upper": float(np.percentile(aucs, 97.5)),
        "n_bootstraps": len(aucs),
    }
