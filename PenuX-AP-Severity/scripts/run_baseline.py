"""Train baseline models on a sanitized dataset and save results.

Usage:
    python scripts/run_baseline.py --data data/public_sanitized/file.csv \
        --target-column severe --outdir outputs/demo

No dataset is bundled. Add a legally usable, de-identified dataset to
data/public_sanitized/ before running this script.
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import (
    apply_positive_value,
    binarize_target,
    describe_target,
    infer_positive_value,
)
from penux_ap.preprocessing import (
    build_preprocessor, infer_feature_types, make_train_test_split, summarize_missingness
)
from penux_ap.models import get_model_registry, train_model, predict_proba_safe
from penux_ap.evaluation import evaluate_binary_classifier, threshold_table, confusion_matrix_at_thresholds
from penux_ap.explainability import permutation_importance_report
from penux_ap.utils import setup_logging, ensure_dir, save_json

log = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Train baseline models for SAP severity prediction.")
    parser.add_argument("--data", required=True, help="Path to sanitized dataset.")
    parser.add_argument("--target-column", default=None, help="Target column name.")
    parser.add_argument("--outdir", default="outputs/demo", help="Output directory.")
    parser.add_argument(
        "--positive-value",
        default="auto",
        choices=["auto", "0", "1"],
        help=(
            "Which raw binarized value denotes the positive SAP class. "
            "Use 'auto' to flip the registered public AP datasets where raw 0=SAP; "
            "unknown datasets default to raw 1=SAP."
        ),
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.data)

    target_col = args.target_column or detect_target_column(df)
    log.info("Target column: '%s'", target_col)

    y = binarize_target(df[target_col])
    y = y.dropna().astype(int)
    positive_value = infer_positive_value(target_col, args.data, args.positive_value)
    y = apply_positive_value(y, positive_value)
    df = df.loc[y.index]
    log.info("Positive raw value for SAP: %s", positive_value)
    log.info("Target distribution (1=SAP): %s", describe_target(y))

    feature_types = infer_feature_types(df, target_col)
    X = df.drop(columns=[target_col])
    preprocessor = build_preprocessor(feature_types["numeric"], feature_types["categorical"])

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    registry = get_model_registry()
    all_metrics = {}
    best_auroc = -1
    best_model = None
    best_model_name = None

    for name in registry:
        try:
            model = train_model(name, X_train, y_train, preprocessor)
            y_proba = predict_proba_safe(model, X_test)
            metrics = evaluate_binary_classifier(y_test.values, y_proba)
            metrics["positive_value"] = positive_value
            all_metrics[name] = metrics
            log.info("%s -> AUROC=%.3f AUPRC=%.3f", name, metrics["auroc"], metrics["auprc"])
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_model = model
                best_model_name = name
        except Exception as e:
            log.error("Failed to train %s: %s", name, e)

    save_json(all_metrics, outdir / "metrics.json")
    log.info("Metrics saved to %s/metrics.json", outdir)

    if best_model is not None:
        joblib.dump(best_model, outdir / "best_model.joblib")
        log.info("Best model ('%s', AUROC=%.3f) saved.", best_model_name, best_auroc)

        y_proba_best = predict_proba_safe(best_model, X_test)

        tt = threshold_table(y_test.values, y_proba_best)
        tt.insert(0, "positive_value", positive_value)
        tt.to_csv(outdir / "threshold_table.csv", index=False)

        # Confusion matrices at standard thresholds
        cm_data = confusion_matrix_at_thresholds(y_test.values, y_proba_best)
        save_json(cm_data, outdir / "confusion_matrices.json")
        log.info("Confusion matrices saved to %s/confusion_matrices.json", outdir)

        try:
            fi = permutation_importance_report(best_model, X_test, y_test)
            fi.to_csv(outdir / "feature_importance.csv", index=False)
            log.info("Feature importance saved.")
        except Exception as e:
            log.warning("Could not compute feature importance: %s", e)

    log.info("Done. Outputs in %s/", outdir)


if __name__ == "__main__":
    main()
