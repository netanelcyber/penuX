"""Evaluate a saved model on a new dataset.

Usage:
    python scripts/evaluate_model.py --model outputs/demo/best_model.joblib \\
        --data data/public_sanitized/file.csv --target-column severe --outdir outputs/eval
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import binarize_target
from penux_ap.models import predict_proba_safe
from penux_ap.evaluation import evaluate_binary_classifier, threshold_table, confusion_matrix_at_thresholds
from penux_ap.utils import setup_logging, ensure_dir, save_json

log = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SAP prediction model.")
    parser.add_argument("--model", required=True, help="Path to saved .joblib model.")
    parser.add_argument("--data", required=True, help="Path to evaluation dataset.")
    parser.add_argument("--target-column", default=None, help="Target column name.")
    parser.add_argument("--outdir", default="outputs/eval", help="Output directory.")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    model = joblib.load(args.model)
    df = load_dataset(args.data)
    target_col = args.target_column or detect_target_column(df)

    y = binarize_target(df[target_col]).dropna().astype(int)
    X = df.drop(columns=[target_col]).loc[y.index]

    y_proba = predict_proba_safe(model, X)
    metrics = evaluate_binary_classifier(y.values, y_proba)
    save_json(metrics, outdir / "eval_metrics.json")
    log.info("AUROC=%.3f AUPRC=%.3f F1=%.3f", metrics["auroc"], metrics["auprc"], metrics["f1"])

    threshold_table(y.values, y_proba).to_csv(outdir / "threshold_table.csv", index=False)

    cm = confusion_matrix_at_thresholds(y.values, y_proba)
    save_json(cm, outdir / "confusion_matrices.json")
    log.info("Evaluation complete. Results in %s/", outdir)


if __name__ == "__main__":
    main()
