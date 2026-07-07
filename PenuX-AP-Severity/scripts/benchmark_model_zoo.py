"""Benchmark the 3 headline GBDT models against a large zoo of ~171 other
classifier configurations, using out-of-fold predictions, matching the
methodology described in docs/sap_severity_gbdt_analysis_he.md.

Usage:
    python scripts/benchmark_model_zoo.py \\
        --data data/public_sanitized/ap_multiml_sanitized.csv \\
        --target-column "Diagnostic Result" \\
        --outdir outputs/multiml

No dataset is bundled. Add a legally usable, de-identified dataset to
data/public_sanitized/ before running this script.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import binarize_target, describe_target
from penux_ap.preprocessing import build_preprocessor, infer_feature_types
from penux_ap.evaluation import evaluate_binary_classifier
from penux_ap.models import predict_proba_safe
from penux_ap.utils import setup_logging, ensure_dir
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import build_model_zoo

log = setup_logging()

GBDT_NAMES = {
    "xgboost_n200_d5_lr0.1": "XGBoost (headline config)",
    "lightgbm_n200_leaves31_lr0.1": "LightGBM (headline config)",
    "catboost_n200_d6_lr0.1": "CatBoost (headline config)",
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark GBDT models against a large classifier zoo.")
    parser.add_argument("--data", required=True, help="Path to sanitized dataset.")
    parser.add_argument("--target-column", default=None, help="Target column name.")
    parser.add_argument("--outdir", default="outputs/demo", help="Output directory.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds for out-of-fold predictions.")
    parser.add_argument(
        "--positive-value", type=int, default=1, choices=[0, 1],
        help="Which raw binarized value denotes the positive (SAP) class. "
             "Both registered public datasets store SAP as raw value 0 "
             "(see docs/dataset_sources.md caveat on reversed raw label direction) "
             "-- pass --positive-value 0 for those.",
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.data)

    target_col = args.target_column or detect_target_column(df)
    log.info("Target column: '%s'", target_col)

    y = binarize_target(df[target_col])
    y = y.dropna().astype(int)
    if args.positive_value == 0:
        y = 1 - y
    df = df.loc[y.index]
    log.info("Target distribution (1=SAP): %s", describe_target(y))

    feature_types = infer_feature_types(df, target_col)
    X = df.drop(columns=[target_col])

    zoo = build_model_zoo()
    log.info("Model zoo size: %d configurations", len(zoo))

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    rows = []

    for i, (name, estimator) in enumerate(zoo, start=1):
        t0 = time.time()
        try:
            y_proba = np.full(len(y), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_idx, test_idx in cv.split(X, y):
                    preprocessor = build_preprocessor(feature_types["numeric"], feature_types["categorical"])
                    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clone(estimator))])
                    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                    y_proba[test_idx] = predict_proba_safe(pipe, X.iloc[test_idx])
            metrics = evaluate_binary_classifier(y.values, y_proba)
            metrics["model"] = name
            metrics["status"] = "ok"
            metrics["seconds"] = round(time.time() - t0, 2)
            rows.append(metrics)
            log.info("[%d/%d] %s -> AUC=%.4f (%.1fs)", i, len(zoo), name, metrics["auroc"], metrics["seconds"])
        except Exception as e:
            log.warning("[%d/%d] %s failed: %s", i, len(zoo), name, e)
            rows.append({"model": name, "status": f"failed: {e}", "auroc": float("nan")})

    results = pd.DataFrame(rows)
    results = results.sort_values("auroc", ascending=False, na_position="last").reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    results.to_csv(outdir / "model_zoo_benchmark.csv", index=False)
    log.info("Full benchmark saved to %s/model_zoo_benchmark.csv", outdir)

    ok = results[results["status"] == "ok"]
    log.info("Evaluated %d/%d model configurations successfully.", len(ok), len(zoo))

    highlight = ok[ok["model"].isin(GBDT_NAMES)].copy()
    highlight["label"] = highlight["model"].map(GBDT_NAMES)
    print("\n=== Headline GBDT models vs. full zoo (n=%d evaluated) ===" % len(ok))
    for _, r in highlight.iterrows():
        print(f"  {r['label']:30s} rank {int(r['rank']):3d}/{len(ok)}  AUC={r['auroc']:.4f}")
    print(f"\nTop 5 overall:")
    print(ok[["rank", "model", "auroc", "auprc", "f1"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
