"""Leakage-resistant validation utilities for SAP prediction research.

This module focuses on model-development methodology rather than bedside use:
- choose operating thresholds on development predictions only;
- support a high-sensitivity operating point;
- report uncertainty with stratified bootstrap intervals;
- quantify net benefit with decision-curve analysis.

RESEARCH USE ONLY. These utilities do not establish clinical validity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from penux_ap.config import RANDOM_SEED
from penux_ap.evaluation import evaluate_binary_classifier


def _as_binary_inputs(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize binary labels and probabilities."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_proba, dtype=float).reshape(-1)

    if y.size == 0:
        raise ValueError("y_true must not be empty")
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and y_proba must have the same length")
    if not np.isfinite(p).all():
        raise ValueError("y_proba contains NaN or infinite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("y_proba must contain probabilities in [0, 1]")
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"y_true must be binary 0/1; observed labels: {sorted(labels)}")
    return y, p


def fbeta_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float,
    beta: float = 2.5,
) -> float:
    """Compute F-beta at a fixed probability threshold."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y, p = _as_binary_inputs(y_true, y_proba)
    pred = (p >= threshold).astype(int)
    return float(fbeta_score(y, pred, beta=beta, zero_division=0))


def select_threshold_for_sensitivity(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    target_sensitivity: float = 0.98,
    beta: float = 2.5,
) -> dict:
    """Choose the most specific threshold that reaches target sensitivity.

    Call this on development/validation predictions, never on the final held-out
    test set. Among thresholds meeting the requested sensitivity, the selection
    rule maximizes specificity, then PPV, then the threshold itself.
    """
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    if beta <= 0:
        raise ValueError("beta must be > 0")

    y, p = _as_binary_inputs(y_true, y_proba)
    if int(np.sum(y == 1)) == 0:
        raise ValueError("At least one positive case is required to target sensitivity")

    # Sensitivity changes only when the threshold crosses an observed score.
    # Include zero so a feasible 100%-sensitivity operating point always exists.
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    rows: list[dict] = []
    for threshold in candidates:
        metrics = evaluate_binary_classifier(y, p, threshold=float(threshold))
        sensitivity = metrics["sensitivity"]
        if np.isnan(sensitivity) or sensitivity < target_sensitivity:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(metrics["specificity"])
                if not np.isnan(metrics["specificity"])
                else float("nan"),
                "ppv": float(metrics["ppv"]) if not np.isnan(metrics["ppv"]) else float("nan"),
                "npv": float(metrics["npv"]) if not np.isnan(metrics["npv"]) else float("nan"),
                "f_beta": fbeta_at_threshold(y, p, float(threshold), beta=beta),
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
            }
        )

    if not rows:
        raise RuntimeError("No threshold reached the requested sensitivity")

    table = pd.DataFrame(rows)
    table["_specificity_sort"] = table["specificity"].fillna(-1.0)
    table["_ppv_sort"] = table["ppv"].fillna(-1.0)
    best = table.sort_values(
        ["_specificity_sort", "_ppv_sort", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    return {
        "target_sensitivity": float(target_sensitivity),
        "threshold": float(best["threshold"]),
        "achieved_sensitivity": float(best["sensitivity"]),
        "specificity": float(best["specificity"]),
        "ppv": float(best["ppv"]),
        "npv": float(best["npv"]),
        "f_beta": float(best["f_beta"]),
        "beta": float(beta),
        "tp": int(best["tp"]),
        "tn": int(best["tn"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "selection_rule": "max_specificity_then_ppv_then_threshold_subject_to_sensitivity",
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float,
    beta: float = 2.5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Stratified bootstrap CIs for discrimination and locked-threshold metrics.

    Positive and negative cases are resampled separately so each bootstrap
    replicate retains both classes and the observed class prevalence.
    """
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y, p = _as_binary_inputs(y_true, y_proba)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap intervals require both positive and negative cases")

    metric_names = [
        "auroc",
        "auprc",
        "brier_score",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "f_beta",
    ]
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(random_state)

    point = evaluate_binary_classifier(y, p, threshold=threshold)
    point["f_beta"] = fbeta_at_threshold(y, p, threshold=threshold, beta=beta)

    for _ in range(n_bootstraps):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate((sampled_pos, sampled_neg))
        rng.shuffle(idx)

        metrics = evaluate_binary_classifier(y[idx], p[idx], threshold=threshold)
        metrics["f_beta"] = fbeta_at_threshold(
            y[idx], p[idx], threshold=threshold, beta=beta
        )
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                samples[name].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, dict] = {}
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        if values.size == 0:
            intervals[name] = {
                "point": float(point[name]),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_valid": 0,
            }
            continue
        intervals[name] = {
            "point": float(point[name]),
            "ci_lower": float(np.quantile(values, alpha)),
            "ci_upper": float(np.quantile(values, 1.0 - alpha)),
            "n_valid": int(values.size),
        }

    return {
        "threshold": float(threshold),
        "beta": float(beta),
        "confidence_level": float(confidence_level),
        "n_bootstraps": int(n_bootstraps),
        "stratified": True,
        "metrics": intervals,
    }


def decision_curve_analysis(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold_probabilities: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate decision-curve net benefit for research evaluation.

    Net benefit is reported for the model, a treat-all strategy, and a
    treat-none strategy. This is an analytic research output, not a treatment
    recommendation.
    """
    y, p = _as_binary_inputs(y_true, y_proba)
    if threshold_probabilities is None:
        threshold_probabilities = np.round(np.arange(0.01, 0.51, 0.01), 2)

    thresholds = np.asarray(threshold_probabilities, dtype=float).reshape(-1)
    if thresholds.size == 0:
        raise ValueError("threshold_probabilities must not be empty")
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("Decision-curve thresholds must be strictly between 0 and 1")

    n = float(len(y))
    prevalence = float(np.mean(y))
    rows: list[dict] = []

    for pt in thresholds:
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        odds = float(pt / (1.0 - pt))

        model_nb = (tp / n) - (fp / n) * odds
        treat_all_nb = prevalence - (1.0 - prevalence) * odds

        rows.append(
            {
                "threshold_probability": float(pt),
                "model_net_benefit": float(model_nb),
                "treat_all_net_benefit": float(treat_all_nb),
                "treat_none_net_benefit": 0.0,
                "prevalence": prevalence,
                "tp": tp,
                "fp": fp,
            }
        )

    return pd.DataFrame(rows)
