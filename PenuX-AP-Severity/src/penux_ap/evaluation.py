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

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

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


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    title: str | None = None,
    ax=None,
    cmap: str = "Blues",
) -> "plt.Axes":
    """Plot a single annotated confusion matrix for SAP prediction.

    Each cell shows raw count, row-percentage, and clinical label
    (TP=Correctly flagged severe, TN=Correctly cleared, etc.).
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    mat = np.array([[tn, fp], [fn, tp]])
    row_totals = mat.sum(axis=1, keepdims=True)
    mat_pct = np.where(row_totals > 0, mat / row_totals * 100, 0)

    im = ax.imshow(mat, cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    clinical_labels = [
        ["TN\nCleared correctly", "FP\nFalse alarm"],
        ["FN\nMissed severe",     "TP\nCaught severe"],
    ]
    thresh_color = mat.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = mat[i, j]
            pct = mat_pct[i, j]
            color = "white" if count > thresh_color else "black"
            ax.text(j, i,
                    f"{count}\n({pct:.1f}%)\n{clinical_labels[i][j]}",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-severe", "Predicted\nSevere"], fontsize=9)
    ax.set_yticklabels(["Actual\nNon-severe", "Actual\nSevere"], fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("Actual label", fontsize=10)
    ax.set_title(title or f"Confusion Matrix  (threshold = {threshold:.2f})", fontsize=11)
    return ax


def plot_confusion_matrix_sweep(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> "plt.Figure":
    """Plot a grid of confusion matrices across multiple decision thresholds.

    Shows how TP/FP/FN/TN shift as the operating threshold changes —
    useful for clinical decision-making (high sensitivity vs high specificity).
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting")

    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ncols = 4
    nrows = (len(thresholds) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (ncols * 4.5, nrows * 4),
    )
    axes_flat = np.array(axes).flatten()

    for i, t in enumerate(thresholds):
        plot_confusion_matrix(y_true, y_proba, threshold=t, ax=axes_flat[i])

    for j in range(len(thresholds), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "SAP Prediction — Confusion Matrices at Multiple Thresholds",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved confusion matrix figure → %s", save_path)

    return fig


def plot_metrics_vs_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
    highlight_threshold: float = DEFAULT_THRESHOLD,
    save_path: str | None = None,
) -> "plt.Figure":
    """Line plot of Sensitivity, Specificity, PPV, NPV, F1 across thresholds.

    Helps clinicians visually select an operating point that balances
    sensitivity (catching all severe cases) and specificity (avoiding ICU overload).
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting")

    df = threshold_table(y_true, y_proba, thresholds)

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["sensitivity", "specificity", "ppv", "npv", "f1"]
    colors  = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    labels  = ["Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV", "F1 Score"]

    for metric, color, label in zip(metrics, colors, labels):
        ax.plot(df["threshold"], df[metric], color=color, linewidth=2, label=label)

    ax.axvline(highlight_threshold, color="gray", linestyle="--", linewidth=1.2,
               label=f"Default threshold ({highlight_threshold})")
    ax.fill_betweenx([0, 1], highlight_threshold - 0.005, highlight_threshold + 0.005,
                     color="gray", alpha=0.15)

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("SAP Prediction — Metrics vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_xlim(df["threshold"].min(), df["threshold"].max())
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved metrics figure → %s", save_path)

    return fig


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
