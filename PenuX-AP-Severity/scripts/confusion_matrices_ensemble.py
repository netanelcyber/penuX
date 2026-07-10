"""Generates confusion matrices (at threshold=0.5, the project default) for
the best single model and the best ensemble combination on each dataset,
using real out-of-fold predictions (same 5-fold CV/seed as the rest of the
project). Renders PNG plots via matplotlib.

Usage:
    python scripts/confusion_matrices_ensemble.py \\
        --data data/public_sanitized/ap_multiml_sanitized.csv \\
        --target-column "Diagnostic Result" \\
        --checkpoint outputs/multiml/model_zoo_checkpoint.csv \\
        --outdir outputs/multiml \\
        --label multiml
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import apply_positive_value, binarize_target, describe_target, infer_positive_value
from penux_ap.preprocessing import build_preprocessor, infer_feature_types
from penux_ap.models import predict_proba_safe
from penux_ap.utils import setup_logging, ensure_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import build_model_zoo
from ensemble_model_zoo import family, get_oof

log = setup_logging()

N_SPLITS = 5
RANDOM_SEED = 42
THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", required=True, help="Dataset label for filenames/titles")
    parser.add_argument("--n-select", type=int, default=15)
    parser.add_argument("--top-k-combo", type=int, default=None, help="K for the simple-average ensemble (defaults to n-select)")
    parser.add_argument("--positive-value", default="auto", choices=["auto", "0", "1"])
    args = parser.parse_args()
    top_k_combo = args.top_k_combo or args.n_select

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.data)
    target_col = args.target_column or detect_target_column(df)
    y = binarize_target(df[target_col])
    y = y.dropna().astype(int)
    positive_value = infer_positive_value(target_col, args.data, args.positive_value)
    y = apply_positive_value(y, positive_value)
    df = df.loc[y.index]
    log.info("Target distribution (1=SAP): %s", describe_target(y))
    feature_types = infer_feature_types(df, target_col)
    X = df.drop(columns=[target_col])
    y_arr = y.values

    ckpt = pd.read_csv(args.checkpoint)
    ckpt = ckpt[ckpt["status"] == "ok"].copy()
    ckpt["family"] = ckpt["model"].apply(family)
    diverse = ckpt.sort_values("auroc", ascending=False).groupby("family").head(2)
    diverse = diverse.sort_values("auroc", ascending=False).head(args.n_select)
    selected_names = diverse["model"].tolist()

    zoo = dict(build_model_zoo())
    base_oof = {}
    for name in selected_names:
        base_oof[name] = get_oof(zoo[name], X, y, feature_types, RANDOM_SEED)
        log.info("  %s -> AUROC=%.4f", name, roc_auc_score(y_arr, base_oof[name]))

    P = np.column_stack([base_oof[n] for n in selected_names])
    aurocs = np.array([roc_auc_score(y_arr, P[:, i]) for i in range(P.shape[1])])
    order = np.argsort(-aurocs)
    best_single_idx = order[0]
    best_single_name = selected_names[best_single_idx]
    best_single_oof = P[:, best_single_idx]

    combo_idx = order[:top_k_combo]
    combo_oof = P[:, combo_idx].mean(axis=1)
    combo_auroc = roc_auc_score(y_arr, combo_oof)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, proba) in zip(
        axes,
        [
            (f"Best single model\n{best_single_name}\nAUROC={roc_auc_score(y_arr, best_single_oof):.4f}", best_single_oof),
            (f"Ensemble: simple average of top {top_k_combo}\nAUROC={combo_auroc:.4f}", combo_oof),
        ],
    ):
        y_pred = (proba >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-SAP", "SAP"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=9)
    fig.suptitle(f"{args.label}: confusion matrices at threshold={THRESHOLD} (out-of-fold, 5-fold CV)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(outdir) / f"confusion_matrices_{args.label}.png"
    fig.savefig(out_path, dpi=150)
    log.info("Saved %s", out_path)

    for name, proba in [("best_single:" + best_single_name, best_single_oof), (f"ensemble_top{top_k_combo}", combo_oof)]:
        y_pred = (proba >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        log.info(
            "%-45s TP=%d TN=%d FP=%d FN=%d  sens=%.3f spec=%.3f ppv=%.3f npv=%.3f",
            name, tp, tn, fp, fn, sens, spec, ppv, npv,
        )


if __name__ == "__main__":
    main()
