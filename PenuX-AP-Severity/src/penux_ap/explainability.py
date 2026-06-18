"""Feature importance: permutation importance (default) and optional SHAP."""
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)


def permutation_importance_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance and return as a sorted DataFrame."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc"
    )
    names = feature_names if feature_names is not None else (list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])])
    df = pd.DataFrame({
        "feature": names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def shap_report_if_available(model, X: pd.DataFrame) -> dict | None:
    """Compute SHAP values if the shap package is installed; otherwise return None."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed. Install with: pip install shap")
        return None

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        return {
            "feature_names": names,
            "mean_abs_shap": mean_abs.tolist(),
        }
    except Exception as e:
        log.warning("SHAP computation failed: %s", e)
        return None
