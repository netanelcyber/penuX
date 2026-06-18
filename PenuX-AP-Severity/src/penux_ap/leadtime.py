"""Lead-time vs confidence analysis across temporal horizons.

Supports confusion-matrix reporting at multiple time points
(e.g. admission, 6h, 12h, 24h) and threshold operating-point frontiers.
"""
import logging

import numpy as np
import pandas as pd

from penux_ap.evaluation import confusion_matrix_at_thresholds, evaluate_binary_classifier

log = logging.getLogger(__name__)

SUPPORTED_HORIZONS = ["admission", "6h", "12h", "24h"]


def detect_horizons(df: pd.DataFrame) -> list[str]:
    """Detect time horizons based on column suffixes."""
    found = []
    for h in SUPPORTED_HORIZONS:
        suffix = f"_{h}"
        if any(col.endswith(suffix) for col in df.columns):
            found.append(h)
    if not found:
        log.info("No horizon-specific columns detected. Defaulting to 24h (all features).")
        found = ["24h"]
    return found


def select_horizon_features(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Select columns for a given time horizon.

    If horizon-specific columns exist (e.g. wbc_6h), select those.
    Otherwise return the full DataFrame.
    """
    suffix = f"_{horizon}"
    horizon_cols = [c for c in df.columns if c.endswith(suffix)]
    if horizon_cols:
        log.info("Horizon '%s': using %d columns", horizon, len(horizon_cols))
        return df[horizon_cols]
    log.info("Horizon '%s': no suffix columns found, using all features.", horizon)
    return df


def operating_point_frontier(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute sensitivity/specificity/PPV/NPV at multiple thresholds.

    Returns a DataFrame suitable for plotting an operating-point curve.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()

    rows = []
    for t in thresholds:
        m = evaluate_binary_classifier(y_true, y_proba, threshold=t)
        rows.append({
            "threshold":              t,
            "sensitivity":            m["sensitivity"],
            "specificity":            m["specificity"],
            "ppv":                    m["ppv"],
            "npv":                    m["npv"],
            "f1":                     m["f1"],
            "predicted_positive_rate": (m["tp"] + m["fp"]) / max(m["tp"] + m["tn"] + m["fp"] + m["fn"], 1),
            "auroc":                  m["auroc"],
            "auprc":                  m["auprc"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        })
    return pd.DataFrame(rows)


def confusion_matrices_by_horizon(
    horizon_results: dict[str, tuple],
    thresholds: list[float] | None = None,
) -> dict[str, list[dict]]:
    """Return confusion matrices at each threshold for each time horizon.

    Args:
        horizon_results: dict mapping horizon name -> (y_true, y_proba)
        thresholds: list of decision thresholds (default 0.1 steps)

    Returns:
        dict mapping horizon -> list of confusion matrix dicts
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1).tolist()

    output = {}
    for horizon, (y_true, y_proba) in horizon_results.items():
        output[horizon] = confusion_matrix_at_thresholds(y_true, y_proba, thresholds)
        log.info("Confusion matrices computed for horizon '%s' at %d thresholds.", horizon, len(thresholds))
    return output
