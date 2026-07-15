"""Sweeps the decision threshold for the best single model and the best
simple-average ensemble on a dataset, using real out-of-fold predictions
(same 5-fold CV/seed as the rest of the project). Reports three operating
points:
  1. The default threshold (0.5)
  2. The threshold that MINIMIZES overall misclassification rate
  3. The least-restrictive threshold that achieves a target negative
     likelihood ratio (LR-), default target 0.35

Minimizing misclassification rate is NOT the same target as minimizing false
negatives (missed severe cases): with an imbalanced dataset (~16-19%
positive), the misclassification-minimizing threshold typically drifts
toward predicting almost everyone negative. This script reports that
tradeoff explicitly, and separately reports what threshold is needed to hit
a target LR-, since that was the actual clinical question asked.

Base model out-of-fold predictions are cached to
outputs/<label>/oof_cache_<label>.npz after the first run, since refitting
15 models x 5 folds is expensive (minutes) and every threshold-only question
after the first one needs none of that recomputation.

Usage:
    python scripts/threshold_sweep.py \\
        --data data/public_sanitized/ap_multiml_sanitized.csv \\
        --target-column "Diagnostic Result" \\
        --checkpoint outputs/multiml/model_zoo_checkpoint.csv \\
        --outdir outputs/multiml \\
        --label multiml \\
        --target-lr-minus 0.35
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
DEFAULT_TARGET_LR_MINUS = 0.35


def metrics_at_threshold(y, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    n = len(y)
    misclass_rate = (fp + fn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    lr_plus = sens / (1 - spec) if spec is not None and spec < 1 else float("inf")
    lr_minus = (1 - sens) / spec if spec else float("inf")
    return dict(threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
                misclass_rate=misclass_rate, accuracy=1 - misclass_rate,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                lr_plus=lr_plus, lr_minus=lr_minus)


def sweep(y, proba):
    candidates = np.unique(np.concatenate([[0.0], proba, [1.0]]))
    rows = [metrics_at_threshold(y, proba, t) for t in candidates]
    return pd.DataFrame(rows)


def load_or_compute_oof(cache_path, selected_names, zoo, X, y, feature_types):
    if cache_path.exists():
        log.info("Loading cached OOF predictions from %s (delete this file to force recompute)", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        cached_names = list(cached["names"])
        if set(selected_names).issubset(set(cached_names)):
            return {name: cached[name] for name in selected_names}
        log.info("Cache does not cover all requested models -- recomputing")

    base_oof = {}
    y_arr = y.values
    for name in selected_names:
        base_oof[name] = get_oof(zoo[name], X, y, feature_types, RANDOM_SEED)
        log.info("  %s -> AUROC=%.4f", name, roc_auc_score(y_arr, base_oof[name]))

    np.savez(cache_path, names=np.array(selected_names, dtype=object), **base_oof)
    log.info("Cached OOF predictions to %s", cache_path)
    return base_oof


def best_row_for_lr_target(table, target_lr_minus):
    """Least-restrictive (highest) threshold achieving LR- <= target."""
    satisfying = table[table["lr_minus"] <= target_lr_minus]
    if satisfying.empty:
        return None
    return satisfying.loc[satisfying["threshold"].idxmax()]


def best_row_for_dor(table):
    """Threshold maximizing the diagnostic odds ratio LR+/LR-, restricted to
    'interior' thresholds where all four confusion-matrix cells are nonzero.

    Unconstrained, DOR is gameable at the extremes: near threshold=0 (flag
    almost everyone) sensitivity->1 drives LR- near 0, and near threshold=1
    (flag almost no one) specificity->1 drives LR+ very high -- both can
    produce a huge DOR while being clinically useless (see the "interior"
    vs raw comparison logged by the caller). Restricting to interior rows
    still allows extreme-but-populated operating points, so the raw result
    is reported AND flagged, not silently accepted.
    """
    interior = table[(table["tp"] > 0) & (table["tn"] > 0) & (table["fp"] > 0) & (table["fn"] > 0)].copy()
    if interior.empty:
        return None
    interior["dor"] = interior["lr_plus"] / interior["lr_minus"]
    return interior.loc[interior["dor"].idxmax()]


def best_row_for_custom_metric(table, auc):
    """Threshold maximizing LR+ / sqrt(LR- * (1 - AUC)), interior thresholds only.

    IMPORTANT: AUC is NOT a function of threshold -- it is computed by
    ranking predictions across every possible threshold at once, so (1-AUC)
    is a single fixed number for a given model, not something that varies
    across this sweep. Dividing by sqrt(constant) rescales every row by the
    same factor, so the ARGMAX threshold is mathematically identical to
    maximizing LR+/sqrt(LR-) alone -- the AUC term only rescales the metric
    value for comparison *across* models/datasets, it cannot change which
    threshold wins *within* one model. It also does not fix the DOR-style
    gaming problem: LR- -> 0 (near threshold=0) or LR+ -> large (near
    threshold=1) still dominate, just with a sqrt instead of a linear rate.
    """
    interior = table[(table["tp"] > 0) & (table["tn"] > 0) & (table["fp"] > 0) & (table["fn"] > 0)].copy()
    if interior.empty:
        return None, None
    interior["custom_metric"] = interior["lr_plus"] / np.sqrt(interior["lr_minus"] * (1 - auc))
    best = interior.loc[interior["custom_metric"].idxmax()]
    return best, best["custom_metric"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-select", type=int, default=15)
    parser.add_argument("--top-k-combo", type=int, default=None)
    parser.add_argument("--target-lr-minus", type=float, default=DEFAULT_TARGET_LR_MINUS)
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
    cache_path = Path(outdir) / f"oof_cache_{args.label}.npz"
    base_oof = load_or_compute_oof(cache_path, selected_names, zoo, X, y, feature_types)

    P = np.column_stack([base_oof[n] for n in selected_names])
    aurocs = np.array([roc_auc_score(y_arr, P[:, i]) for i in range(P.shape[1])])
    order = np.argsort(-aurocs)
    best_single_name = selected_names[order[0]]
    best_single_oof = P[:, order[0]]
    combo_oof = P[:, order[:top_k_combo]].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    summary_rows = []
    full_tables = []
    for ax, (name, proba, model_auc) in zip(axes, [
        (f"single best ({best_single_name})", best_single_oof, roc_auc_score(y_arr, best_single_oof)),
        (f"ensemble (top {top_k_combo})", combo_oof, roc_auc_score(y_arr, combo_oof)),
    ]):
        table = sweep(y_arr, proba)
        table.insert(0, "strategy", name)
        full_tables.append(table)

        best_misclass_row = table.loc[table["misclass_rate"].idxmin()]
        default_row = table.iloc[(table["threshold"] - DEFAULT_THRESHOLD).abs().argmin()]
        lr_row = best_row_for_lr_target(table, args.target_lr_minus)

        log.info("--- %s ---", name)
        log.info(
            "  default threshold=0.5:        misclass=%.4f (acc=%.4f) sens=%.4f spec=%.4f LR-=%.3f LR+=%.3f",
            default_row["misclass_rate"], default_row["accuracy"], default_row["sensitivity"],
            default_row["specificity"], default_row["lr_minus"], default_row["lr_plus"],
        )
        log.info(
            "  misclass-minimizing threshold=%.4f: misclass=%.4f sens=%.4f spec=%.4f LR-=%.3f  [TP=%d TN=%d FP=%d FN=%d]",
            best_misclass_row["threshold"], best_misclass_row["misclass_rate"],
            best_misclass_row["sensitivity"], best_misclass_row["specificity"], best_misclass_row["lr_minus"],
            best_misclass_row["tp"], best_misclass_row["tn"], best_misclass_row["fp"], best_misclass_row["fn"],
        )
        if lr_row is None:
            log.info(
                "  NO threshold achieves LR- <= %.2f for this model -- even threshold=0 (flag everyone) "
                "does not reach it. This model cannot hit the target on its own.",
                args.target_lr_minus,
            )
        else:
            log.info(
                "  LR- <= %.2f at threshold=%.4f: misclass=%.4f sens=%.4f spec=%.4f ppv=%.4f LR-=%.3f LR+=%.3f  [TP=%d TN=%d FP=%d FN=%d]",
                args.target_lr_minus, lr_row["threshold"], lr_row["misclass_rate"], lr_row["sensitivity"],
                lr_row["specificity"], lr_row["ppv"], lr_row["lr_minus"], lr_row["lr_plus"],
                lr_row["tp"], lr_row["tn"], lr_row["fp"], lr_row["fn"],
            )

        dor_row = best_row_for_dor(table)
        if dor_row is not None:
            dor = dor_row["lr_plus"] / dor_row["lr_minus"]
            log.info(
                "  MAX DOR (LR+/LR-)=%.2f at threshold=%.4f: misclass=%.4f sens=%.4f spec=%.4f ppv=%.4f LR-=%.3f LR+=%.3f  [TP=%d TN=%d FP=%d FN=%d]",
                dor, dor_row["threshold"], dor_row["misclass_rate"], dor_row["sensitivity"],
                dor_row["specificity"], dor_row["ppv"], dor_row["lr_minus"], dor_row["lr_plus"],
                dor_row["tp"], dor_row["tn"], dor_row["fp"], dor_row["fn"],
            )
            if dor_row["sensitivity"] < 0.5 or dor_row["specificity"] < 0.5:
                log.warning(
                    "  ^ FLAGGED: this DOR-maximizing threshold has sensitivity=%.3f, specificity=%.3f -- "
                    "an extreme, likely clinically unusable operating point despite the high ratio.",
                    dor_row["sensitivity"], dor_row["specificity"],
                )

        custom_row, custom_value = best_row_for_custom_metric(table, model_auc)
        if custom_row is not None:
            log.info(
                "  MAX LR+/sqrt(LR-*(1-AUC)) [AUC=%.4f]=%.2f at threshold=%.4f: misclass=%.4f sens=%.4f spec=%.4f ppv=%.4f LR-=%.3f LR+=%.3f  [TP=%d TN=%d FP=%d FN=%d]",
                model_auc, custom_value, custom_row["threshold"], custom_row["misclass_rate"], custom_row["sensitivity"],
                custom_row["specificity"], custom_row["ppv"], custom_row["lr_minus"], custom_row["lr_plus"],
                custom_row["tp"], custom_row["tn"], custom_row["fp"], custom_row["fn"],
            )
            sanity_row_at_auc_half, _ = best_row_for_custom_metric(table, 0.5)
            if sanity_row_at_auc_half is not None and sanity_row_at_auc_half["threshold"] != custom_row["threshold"]:
                log.warning("  ^ unexpected: argmax threshold changed when AUC was swapped for 0.5 -- investigate.")
            else:
                log.info(
                    "  ^ NOTE: re-ran with AUC swapped to 0.5 (vs actual %.4f) and got the SAME argmax threshold -- "
                    "confirms (1-AUC) is a constant multiplier here, not a function of threshold. It rescales the "
                    "metric's *value* (useful only for comparing across models with different AUCs), but cannot "
                    "change which threshold wins for a single model. This is a different formula from DOR "
                    "(LR+/LR-, no sqrt), so its argmax threshold is not expected to match DOR's.",
                    model_auc,
                )
            if custom_row["sensitivity"] < 0.5 or custom_row["specificity"] < 0.5:
                log.warning(
                    "  ^ FLAGGED: this threshold has sensitivity=%.3f, specificity=%.3f -- "
                    "same extreme-operating-point problem as MAX DOR, not fixed by including AUC.",
                    custom_row["sensitivity"], custom_row["specificity"],
                )

        ax.plot(table["threshold"], table["misclass_rate"], label="misclassification rate", color="crimson")
        ax.plot(table["threshold"], 1 - table["sensitivity"], label="false-negative rate (1-sensitivity)", color="navy", linestyle="--")
        ax.plot(table["threshold"], table["lr_minus"].clip(upper=2), label="LR- (clipped at 2)", color="darkorange", linestyle="-.")
        ax.axhline(args.target_lr_minus, color="darkorange", alpha=0.4, linestyle=":")
        ax.axvline(best_misclass_row["threshold"], color="crimson", alpha=0.4, linestyle=":")
        ax.axvline(0.5, color="gray", alpha=0.6, linestyle=":")
        if lr_row is not None:
            ax.axvline(lr_row["threshold"], color="darkorange", alpha=0.6, linestyle=":")
        ax.set_xlabel("decision threshold")
        ax.set_ylabel("rate")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=6)

        cols = ["threshold", "misclass_rate", "accuracy", "sensitivity", "specificity", "ppv", "npv", "lr_plus", "lr_minus"]
        summary_rows.append({"strategy": name, "criterion": "default_0.5", **{k: default_row[k] for k in cols}})
        summary_rows.append({"strategy": name, "criterion": "misclass_minimizing", **{k: best_misclass_row[k] for k in cols}})
        if lr_row is not None:
            summary_rows.append({"strategy": name, "criterion": f"lr_minus_leq_{args.target_lr_minus}", **{k: lr_row[k] for k in cols}})
        if dor_row is not None:
            summary_rows.append({"strategy": name, "criterion": "max_dor", **{k: dor_row[k] for k in cols}})
        if custom_row is not None:
            summary_rows.append({"strategy": name, "criterion": "max_lrplus_over_sqrt_lrminus_times_1minusauc",
                                  "auc": model_auc, "metric_value": custom_value,
                                  **{k: custom_row[k] for k in cols}})

    fig.suptitle(f"{args.label}: misclassification rate / false-negative rate / LR- across thresholds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = Path(outdir) / f"threshold_sweep_{args.label}.png"
    fig.savefig(fig_path, dpi=150)
    log.info("Saved %s", fig_path)

    out_path = Path(outdir) / f"threshold_sweep_{args.label}.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    log.info("Saved %s", out_path)

    full_path = Path(outdir) / f"threshold_sweep_full_{args.label}.csv"
    pd.concat(full_tables, ignore_index=True).to_csv(full_path, index=False)
    log.info("Saved full per-threshold table to %s", full_path)


if __name__ == "__main__":
    main()
