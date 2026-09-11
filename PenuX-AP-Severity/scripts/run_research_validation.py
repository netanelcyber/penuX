"""Leakage-resistant research validation for PenuX-AP-Severity.

The script keeps the final test split untouched during model selection:
1. Create a stratified development/test split.
2. Generate out-of-fold (OOF) probabilities inside the development split.
3. Select the model by development OOF AUPRC (default).
4. Lock a high-sensitivity operating threshold using development OOF predictions.
5. Fit the selected pipeline on all development data.
6. Evaluate exactly once on the held-out test split.

RESEARCH USE ONLY. This workflow does not establish clinical validity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penux_ap.datasets import detect_target_column, load_dataset
from penux_ap.evaluation import (
    confusion_matrix_at_thresholds,
    evaluate_binary_classifier,
    threshold_table,
)
from penux_ap.explainability import permutation_importance_report
from penux_ap.labels import binarize_target, describe_target
from penux_ap.models import build_pipeline, get_model_registry, predict_proba_safe
from penux_ap.preprocessing import (
    build_preprocessor,
    infer_feature_types,
    make_train_test_split,
    summarize_missingness,
)
from penux_ap.research_validation import (
    bootstrap_metric_intervals,
    decision_curve_analysis,
    fbeta_at_threshold,
    select_threshold_for_sensitivity,
)
from penux_ap.utils import ensure_dir, save_json, setup_logging

log = setup_logging()


def _add_optional_filter(pipe, alpha: float | None):
    """Insert a fold-local univariate filter without leaking across CV folds."""
    if alpha is None:
        return pipe
    if not 0.0 < alpha <= 1.0:
        raise ValueError("--univariate-alpha must be in (0, 1]")
    steps = list(pipe.steps)
    steps.insert(-1, ("univariate_filter", SelectFpr(score_func=f_classif, alpha=alpha)))
    pipe.steps = steps
    return pipe


def _make_pipeline(model_name: str, preprocessor, univariate_alpha: float | None):
    pipe = build_pipeline(model_name, clone(preprocessor))
    return _add_optional_filter(pipe, univariate_alpha)


def _valid_cv_folds(y: pd.Series, requested: int) -> int:
    if requested < 2:
        raise ValueError("--cv-folds must be >= 2")
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("Development split must contain both outcome classes")
    folds = min(requested, int(counts.min()))
    if folds < 2:
        raise ValueError("Not enough minority-class observations for cross-validation")
    if folds != requested:
        log.warning(
            "Reducing CV folds from %d to %d because the minority class has only %d cases.",
            requested,
            folds,
            int(counts.min()),
        )
    return folds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant research validation for SAP severity prediction."
    )
    parser.add_argument("--data", required=True, help="Path to a sanitized CSV/XLSX dataset.")
    parser.add_argument("--target-column", default=None, help="Binary SAP target column.")
    parser.add_argument("--outdir", default="outputs/research_validation")
    parser.add_argument(
        "--selection-metric",
        choices=["auprc", "auroc"],
        default="auprc",
        help="OOF development metric used to select the model.",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.98)
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
        help="F-beta weighting used at the locked operating point.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Parallel CV jobs; keep at 1 when estimators already use internal parallelism.",
    )
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument(
        "--univariate-alpha",
        type=float,
        default=None,
        help=(
            "Optional fold-local SelectFpr alpha. Example: 0.30. "
            "Filtering is fit inside each CV fold to prevent leakage."
        ),
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Feature column to exclude explicitly; repeat for multiple columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.target_sensitivity <= 1.0:
        raise ValueError("--target-sensitivity must be in (0, 1]")
    if args.beta <= 0:
        raise ValueError("--beta must be > 0")
    if args.bootstraps < 1:
        raise ValueError("--bootstraps must be >= 1")

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.data)

    target_col = args.target_column or detect_target_column(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = binarize_target(df[target_col]).dropna().astype(int)
    df = df.loc[y.index].copy()

    drop_columns = [c for c in args.drop_column if c in df.columns and c != target_col]
    missing_drop_columns = [c for c in args.drop_column if c not in df.columns]
    if missing_drop_columns:
        log.warning("Requested drop columns not found: %s", missing_drop_columns)

    X = df.drop(columns=[target_col] + drop_columns)
    analysis_df = X.copy()
    analysis_df[target_col] = y

    feature_types = infer_feature_types(analysis_df, target_col)
    preprocessor = build_preprocessor(
        feature_types["numeric"],
        feature_types["categorical"],
    )

    X_dev, X_test, y_dev, y_test = make_train_test_split(X, y)
    cv_folds = _valid_cv_folds(y_dev, args.cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    log.info("Target distribution: %s", describe_target(y))
    log.info(
        "Development=%d, test=%d, features=%d, target=%s",
        len(X_dev),
        len(X_test),
        X.shape[1],
        target_col,
    )

    missingness = summarize_missingness(X_dev)
    missingness.to_csv(outdir / "development_missingness.csv")

    registry = get_model_registry()
    development_results: dict[str, dict] = {}
    best_model_name: str | None = None
    best_score = -np.inf
    best_oof: np.ndarray | None = None
    best_operating_point: dict | None = None

    for model_name in registry:
        try:
            pipeline = _make_pipeline(model_name, preprocessor, args.univariate_alpha)
            oof_proba = cross_val_predict(
                pipeline,
                X_dev,
                y_dev,
                cv=cv,
                method="predict_proba",
                n_jobs=args.cv_jobs,
            )[:, 1]

            metrics = evaluate_binary_classifier(y_dev.values, oof_proba, threshold=0.5)
            operating_point = select_threshold_for_sensitivity(
                y_dev.values,
                oof_proba,
                target_sensitivity=args.target_sensitivity,
                beta=args.beta,
            )
            score = float(metrics[args.selection_metric])

            development_results[model_name] = {
                "oof_metrics_at_0_5": metrics,
                "locked_operating_point_candidate": operating_point,
                "selection_metric": args.selection_metric,
                "selection_score": score,
            }
            log.info(
                "%s OOF: AUPRC=%.4f AUROC=%.4f; %.1f%%-sensitivity threshold=%.6f",
                model_name,
                metrics["auprc"],
                metrics["auroc"],
                100.0 * args.target_sensitivity,
                operating_point["threshold"],
            )

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_model_name = model_name
                best_oof = oof_proba
                best_operating_point = operating_point
        except Exception as exc:
            log.exception("Development validation failed for %s: %s", model_name, exc)
            development_results[model_name] = {"error": str(exc)}

    save_json(development_results, outdir / "development_model_selection.json")

    if best_model_name is None or best_oof is None or best_operating_point is None:
        raise RuntimeError("No candidate model completed development validation")

    # Lock model identity and operating threshold before inspecting final test metrics.
    locked_threshold = float(best_operating_point["threshold"])
    final_model = _make_pipeline(best_model_name, preprocessor, args.univariate_alpha)
    final_model.fit(X_dev, y_dev)
    test_proba = predict_proba_safe(final_model, X_test)

    test_metrics = evaluate_binary_classifier(
        y_test.values,
        test_proba,
        threshold=locked_threshold,
    )
    test_metrics["f_beta"] = fbeta_at_threshold(
        y_test.values,
        test_proba,
        threshold=locked_threshold,
        beta=args.beta,
    )
    test_metrics["beta"] = float(args.beta)
    test_metrics["locked_from_development"] = True

    metric_intervals = bootstrap_metric_intervals(
        y_test.values,
        test_proba,
        threshold=locked_threshold,
        beta=args.beta,
        n_bootstraps=args.bootstraps,
    )

    joblib.dump(final_model, outdir / "best_model.joblib")
    save_json(test_metrics, outdir / "test_metrics.json")
    save_json(metric_intervals, outdir / "test_metric_intervals.json")
    save_json(best_operating_point, outdir / "locked_operating_point.json")

    threshold_table(y_test.values, test_proba).to_csv(
        outdir / "test_threshold_table.csv",
        index=False,
    )
    save_json(
        confusion_matrix_at_thresholds(y_test.values, test_proba),
        outdir / "test_confusion_matrices.json",
    )
    decision_curve_analysis(y_test.values, test_proba).to_csv(
        outdir / "decision_curve.csv",
        index=False,
    )

    try:
        importance = permutation_importance_report(final_model, X_test, y_test)
        importance.to_csv(outdir / "feature_importance_test.csv", index=False)
    except Exception as exc:
        log.warning("Permutation importance unavailable: %s", exc)

    manifest = {
        "workflow": "development_oof_model_selection_then_locked_holdout_test",
        "research_use_only": True,
        "data_file": str(Path(args.data).name),
        "target_column": target_col,
        "dropped_columns": drop_columns,
        "n_total": int(len(X)),
        "n_development": int(len(X_dev)),
        "n_test": int(len(X_test)),
        "n_features_input": int(X.shape[1]),
        "overall_prevalence": float(y.mean()),
        "development_prevalence": float(y_dev.mean()),
        "test_prevalence": float(y_test.mean()),
        "cv_folds": int(cv_folds),
        "selection_metric": args.selection_metric,
        "selected_model": best_model_name,
        "development_selection_score": float(best_score),
        "target_sensitivity": float(args.target_sensitivity),
        "locked_threshold": locked_threshold,
        "beta": float(args.beta),
        "univariate_alpha": args.univariate_alpha,
        "test_used_for_model_or_threshold_selection": False,
        "bootstrap_iterations": int(args.bootstraps),
        "decision_curve_note": (
            "Exploratory net-benefit analysis; interpret only after checking probability calibration "
            "and before any prospective clinical evaluation."
        ),
    }
    save_json(manifest, outdir / "validation_manifest.json")

    log.info(
        "Selected %s by development OOF %s=%.4f. Locked threshold=%.6f. "
        "Held-out test: AUPRC=%.4f AUROC=%.4f sensitivity=%.4f specificity=%.4f.",
        best_model_name,
        args.selection_metric,
        best_score,
        locked_threshold,
        test_metrics["auprc"],
        test_metrics["auroc"],
        test_metrics["sensitivity"],
        test_metrics["specificity"],
    )
    log.info("Research validation outputs written to %s", outdir)


if __name__ == "__main__":
    main()
