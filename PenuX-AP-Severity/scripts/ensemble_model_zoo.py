"""Builds new combination/ensemble models out of the top-ranked diverse
performers from the model zoo, on the REAL sanitized public dataset.

Directly answers: "can these models be combined for more accurate SAP
detection?" Strategies tested (matching the psychosis project's sibling
script for methodological consistency):
  1. Simple averaging of the top-K models' out-of-fold probabilities
  2. AUROC-weighted averaging
  3. Stacking: logistic-regression meta-learner on the base out-of-fold
     probabilities, evaluated via an outer 5-fold CV

Unlike the earlier significance_vs_baseline.py screen (which used an
UNPAIRED Hanley-McNeil approximation because raw OOF vectors were not
persisted), this script keeps the actual OOF probability arrays in memory,
so the final ensemble-vs-best-single-model and ensemble-vs-baseline
comparisons use a proper PAIRED bootstrap test -- more statistical power,
same real data, same 5-fold split (seed=42) as every other benchmark in
this project.

Usage:
    python scripts/ensemble_model_zoo.py \\
        --data data/public_sanitized/ap_multiml_sanitized.csv \\
        --checkpoint outputs/multiml/model_zoo_checkpoint.csv \\
        --outdir outputs/multiml
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

log = setup_logging()

N_SPLITS = 5
RANDOM_SEED = 42
BASELINE_NAME = "logreg_C1_None_lbfgs"
N_BOOTSTRAP = 2000


def family(name):
    if name.startswith("hybrid"):
        return "hybrid"
    if name.startswith("dnn_v2") or name.startswith("dnn_") or name.startswith("torch_dnn"):
        return "dnn"
    if name.startswith("convnet"):
        return "convnet"
    if name.startswith("xgboost_dart"):
        return "xgboost_dart"
    if name.startswith("xgboost"):
        return "xgboost"
    if name.startswith("lightgbm_dart"):
        return "lightgbm_dart"
    if name.startswith("lightgbm_goss"):
        return "lightgbm_goss"
    if name.startswith("lightgbm"):
        return "lightgbm"
    if name.startswith("catboost_plain"):
        return "catboost_plain"
    if name.startswith("catboost"):
        return "catboost"
    if name.startswith("scratch_gbdt"):
        return "scratch_gbdt"
    if name.startswith("gbdt"):
        return "gbdt_sklearn"
    if name.startswith("histgbdt"):
        return "histgbdt"
    if "huge" in name:
        return "huge_trees"
    if name.startswith("rf"):
        return "random_forest"
    if name.startswith("extra_tree"):
        return "extra_trees"
    return "other"


def get_oof(estimator, X, y, feature_types, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        preprocessor = build_preprocessor(feature_types["numeric"], feature_types["categorical"])
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clone(estimator))])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = predict_proba_safe(pipe, X.iloc[test_idx])
    return oof


def paired_bootstrap_p(y, p_a, p_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """One-sided paired bootstrap test: P(AUROC(p_b) <= AUROC(p_a)) resampling
    rows jointly (preserves the pairing between the two prediction vectors)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            auc_a = roc_auc_score(y[idx], p_a[idx])
            auc_b = roc_auc_score(y[idx], p_b[idx])
        except ValueError:
            diffs[i] = 0.0
            continue
        diffs[i] = auc_b - auc_a
    p_one_sided = float((diffs <= 0).mean())
    return p_one_sided, diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--checkpoint", required=True, help="Path to model_zoo_checkpoint.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-select", type=int, default=15, help="Number of diverse top models to combine")
    parser.add_argument("--positive-value", default="auto", choices=["auto", "0", "1"])
    args = parser.parse_args()

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
    log.info("Selected %d diverse base models: %s", len(selected_names), selected_names)

    zoo = dict(build_model_zoo())
    missing = [n for n in selected_names if n not in zoo]
    if missing:
        raise SystemExit(f"Models not found in current zoo (renamed/removed?): {missing}")

    base_oof = {}
    for name in selected_names:
        oof = get_oof(zoo[name], X, y, feature_types, RANDOM_SEED)
        base_oof[name] = oof
        log.info("  %s -> AUROC=%.4f", name, roc_auc_score(y_arr, oof))

    baseline_oof = get_oof(zoo[BASELINE_NAME], X, y, feature_types, RANDOM_SEED)
    baseline_auroc = roc_auc_score(y_arr, baseline_oof)
    log.info("Baseline %s -> AUROC=%.4f", BASELINE_NAME, baseline_auroc)

    P = np.column_stack([base_oof[n] for n in selected_names])
    aurocs = np.array([roc_auc_score(y_arr, P[:, i]) for i in range(P.shape[1])])
    order = np.argsort(-aurocs)
    best_single_idx = order[0]
    best_single_name = selected_names[best_single_idx]
    best_single_auroc = aurocs[best_single_idx]
    best_single_oof = P[:, best_single_idx]

    results = {}
    combo_oofs = {}
    for k in [3, 5, 10, args.n_select]:
        idx = order[:k]
        avg = P[:, idx].mean(axis=1)
        results[f"simple_average_top{k}"] = roc_auc_score(y_arr, avg)
        combo_oofs[f"simple_average_top{k}"] = avg
    for k in [5, 10, args.n_select]:
        idx = order[:k]
        w = aurocs[idx]
        w = (w - w.min() + 1e-6)
        w = w / w.sum()
        weighted = (P[:, idx] * w).sum(axis=1)
        results[f"auroc_weighted_top{k}"] = roc_auc_score(y_arr, weighted)
        combo_oofs[f"auroc_weighted_top{k}"] = weighted

    outer_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    stack_oof = np.zeros(len(y_arr), dtype=float)
    for train_idx, test_idx in outer_skf.split(P, y_arr):
        meta = LogisticRegression(max_iter=2000)
        meta.fit(P[train_idx], y_arr[train_idx])
        stack_oof[test_idx] = meta.predict_proba(P[test_idx])[:, 1]
    results[f"stacking_logreg_all{args.n_select}"] = roc_auc_score(y_arr, stack_oof)
    combo_oofs[f"stacking_logreg_all{args.n_select}"] = stack_oof

    best_combo_name = max(results, key=results.get)
    best_combo_auroc = results[best_combo_name]
    best_combo_oof = combo_oofs[best_combo_name]

    log.info("Best single base model: %s AUROC=%.4f", best_single_name, best_single_auroc)
    for name, auroc in sorted(results.items(), key=lambda kv: -kv[1]):
        log.info("  %-30s AUROC=%.4f", name, auroc)
    log.info("Best combination: %s AUROC=%.4f (%+.4f vs best single)",
              best_combo_name, best_combo_auroc, best_combo_auroc - best_single_auroc)

    p_vs_single, _ = paired_bootstrap_p(y_arr, best_single_oof, best_combo_oof)
    p_vs_baseline, _ = paired_bootstrap_p(y_arr, baseline_oof, best_combo_oof)
    log.info("Paired bootstrap (n=%d) one-sided p, combo > best single model: p=%.4f", N_BOOTSTRAP, p_vs_single)
    log.info("Paired bootstrap (n=%d) one-sided p, combo > logreg baseline:   p=%.4f", N_BOOTSTRAP, p_vs_baseline)

    out_rows = [{"strategy": f"single:{n}", "auroc": roc_auc_score(y_arr, base_oof[n])} for n in selected_names]
    out_rows.append({"strategy": f"baseline:{BASELINE_NAME}", "auroc": baseline_auroc})
    for name, auroc in results.items():
        out_rows.append({"strategy": name, "auroc": auroc})
    out_rows.append({"strategy": "paired_bootstrap_p_combo_vs_best_single", "auroc": p_vs_single})
    out_rows.append({"strategy": "paired_bootstrap_p_combo_vs_baseline", "auroc": p_vs_baseline})
    pd.DataFrame(out_rows).to_csv(outdir / "ensemble_results.csv", index=False)
    log.info("Saved %s", outdir / "ensemble_results.csv")


if __name__ == "__main__":
    main()
