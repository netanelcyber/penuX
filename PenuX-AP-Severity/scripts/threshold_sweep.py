"""Sweeps the decision threshold to find the value that MINIMIZES the overall
misclassification rate (1 - accuracy), for the best single model and the best
simple-average ensemble on a dataset, using real out-of-fold predictions
(same 5-fold CV/seed as the rest of the project).

Requested follow-up to the confusion-matrix analysis: minimizing
misclassification rate is NOT the same target as minimizing false negatives
(missed severe cases). With an imbalanced dataset (~16-19% positive), the
misclassification-minimizing threshold typically converges toward predicting
almost everyone negative -- this script reports that tradeoff explicitly
rather than silently optimizing a metric that would be clinically harmful.

Usage:
    python scripts/threshold_sweep.py \\
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
from sklearn.metrics import confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import apply_positive_value, binarize_target, describe_target, infer_positive_value
from penux_ap.preprocessing import infer_feature_types
from penux_ap.utils import setup_logging, ensure_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import build_model_zoo
from ensemble_model_zoo import family, get_oof

log = setup_logging()

RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.5


def metrics_at_threshold(y, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    n = len(y)
    misclass_rate = (fp + fn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return dict(threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
                misclass_rate=misclass_rate, accuracy=1 - misclass_rate,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv)


def sweep(y, proba):
    candidates = np.unique(np.concatenate([[0.0], proba, [1.0]]))
    rows = [metrics_at_threshold(y, proba, t) for t in candidates]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-select", type=int, default=15)
    parser.add_argument("--top-k-combo", type=int, default=None)
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
    best_single_name = selected_names[order[0]]
    best_single_oof = P[:, order[0]]
    combo_oof = P[:, order[:top_k_combo]].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    summary_rows = []
    for ax, (name, proba) in zip(axes, [
        (f"single best ({best_single_name})", best_single_oof),
        (f"ensemble (top {top_k_combo})", combo_oof),
    ]):
        table = sweep(y_arr, proba)
        best_row = table.loc[table["misclass_rate"].idxmin()]
        default_row = table.iloc[(table["threshold"] - DEFAULT_THRESHOLD).abs().argmin()]

        log.info("--- %s ---", name)
        log.info(
            "  default threshold=0.5:        misclass=%.4f (acc=%.4f) sens=%.4f spec=%.4f",
            default_row["misclass_rate"], default_row["accuracy"], default_row["sensitivity"], default_row["specificity"],
        )
        log.info(
            "  misclass-minimizing threshold=%.4f: misclass=%.4f (acc=%.4f) sens=%.4f spec=%.4f  [TP=%d TN=%d FP=%d FN=%d]",
            best_row["threshold"], best_row["misclass_rate"], best_row["accuracy"],
            best_row["sensitivity"], best_row["specificity"],
            best_row["tp"], best_row["tn"], best_row["fp"], best_row["fn"],
        )
        always_negative_rate = y_arr.mean()
        log.info(
            "  (reference: always predicting 'not severe' gives misclass=%.4f, sensitivity=0.0)",
            always_negative_rate,
        )

        ax.plot(table["threshold"], table["misclass_rate"], label="misclassification rate", color="crimson")
        ax.plot(table["threshold"], 1 - table["sensitivity"], label="false-negative rate (1-sensitivity)", color="navy", linestyle="--")
        ax.axvline(best_row["threshold"], color="crimson", alpha=0.4, linestyle=":")
        ax.axvline(0.5, color="gray", alpha=0.6, linestyle=":")
        ax.set_xlabel("decision threshold")
        ax.set_ylabel("rate")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)

        summary_rows.append({"strategy": name, "criterion": "default_0.5", **{k: default_row[k] for k in ["threshold", "misclass_rate", "accuracy", "sensitivity", "specificity", "ppv", "npv"]}})
        summary_rows.append({"strategy": name, "criterion": "misclass_minimizing", **{k: best_row[k] for k in ["threshold", "misclass_rate", "accuracy", "sensitivity", "specificity", "ppv", "npv"]}})

    fig.suptitle(f"{args.label}: misclassification rate vs. false-negative rate across thresholds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = Path(outdir) / f"threshold_sweep_{args.label}.png"
    fig.savefig(fig_path, dpi=150)
    log.info("Saved %s", fig_path)

    out_path = Path(outdir) / f"threshold_sweep_{args.label}.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
