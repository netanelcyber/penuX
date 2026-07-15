"""Capstone model: best hybrid DNN+ConvNet+GBDT architecture, augmented with
the quasi-SOFA engineered feature (Sepsis-3 methodology adapted to
lab-only data, src/penux_ap/clinical_scores.py), evaluated out-of-fold
with the same 5-fold Stratified CV used throughout scripts/model_zoo.py.

Combines every finding from this project's model search:
  - The winning architecture (HybridDNNConvGBDTClassifier, #1/1981 on multiml)
  - The quasi-SOFA organ-dysfunction feature (evaluate_quasi_sofa.py)
  - The gbdt_heavy combination weighting (best-performing combo method)

Usage:
    python scripts/build_capstone_model.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.clinical_scores import compute_quasi_sofa_labs
from penux_ap.datasets import load_dataset
from penux_ap.evaluation import evaluate_binary_classifier
from penux_ap.hybrid_models import HybridDNNConvGBDTClassifier
from penux_ap.labels import binarize_target
from penux_ap.models import predict_proba_safe
from penux_ap.preprocessing import build_preprocessor, infer_feature_types
from penux_ap.utils import ensure_dir, setup_logging

log = setup_logging()

DATASETS = {
    "multiml": dict(
        path="data/public_sanitized/ap_multiml_sanitized.csv",
        target_col="Diagnostic Result",
        colmap={"creatinine_umol_l": "Cr", "bilirubin_umol_l": "TBIL", "platelets_e9_l": "PLT", "wbc_e9_l": "WBC"},
        best_hybrid=dict(dnn_hidden=(64,), conv_channels=(8, 16), gbdt_n_estimators=100,
                          gbdt_max_depth=5, gbdt_learning_rate=0.05, combo_method="gbdt_heavy"),
    ),
    "lnn": dict(
        path="data/public_sanitized/ap_lnn_sanitized.csv",
        target_col="严重程度",
        colmap={"creatinine_umol_l": "肌酐", "bilirubin_umol_l": "总胆红素", "platelets_e9_l": "血小板",
                "wbc_e9_l": "白细胞", "lactate_mmol_l": "乳酸"},
        best_hybrid=dict(dnn_hidden=(32,), conv_channels=(32, 64), gbdt_n_estimators=100,
                          gbdt_max_depth=3, gbdt_learning_rate=0.1, combo_method="gbdt_heavy"),
    ),
}


def run_cv(X, y, estimator_factory, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_proba = np.full(len(y), np.nan)
    feature_types = infer_feature_types(pd.concat([X, y.rename("__target__")], axis=1), "__target__")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(X, y):
            pre = build_preprocessor(feature_types["numeric"], feature_types["categorical"])
            pipe = Pipeline([("preprocessor", pre), ("classifier", clone(estimator_factory()))])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_proba[test_idx] = predict_proba_safe(pipe, X.iloc[test_idx])
    return y_proba


def main():
    root = Path(__file__).resolve().parents[1]
    outdir = ensure_dir(root / "outputs" / "capstone")
    rows = []

    for name, cfg in DATASETS.items():
        log.info("=== %s ===", name)
        df = load_dataset(root / cfg["path"])
        y = binarize_target(df[cfg["target_col"]]).dropna().astype(int)
        y = 1 - y  # raw 0 = SAP
        df = df.loc[y.index]

        X_base = df.drop(columns=[cfg["target_col"]])

        mapped = pd.DataFrame(index=df.index)
        for std_name, raw_col in cfg["colmap"].items():
            mapped[std_name] = df[raw_col]
        quasi_sofa = mapped.apply(compute_quasi_sofa_labs, axis=1)

        X_aug = X_base.copy()
        X_aug["quasi_sofa_labs"] = quasi_sofa

        def make_estimator(params=cfg["best_hybrid"]):
            return HybridDNNConvGBDTClassifier(**params)

        for variant_name, X in [("baseline_no_quasi_sofa", X_base), ("capstone_with_quasi_sofa", X_aug)]:
            log.info("Running %s / %s ...", name, variant_name)
            y_proba = run_cv(X, y, make_estimator)
            metrics = evaluate_binary_classifier(y.values, y_proba)
            metrics["dataset"] = name
            metrics["variant"] = variant_name
            rows.append(metrics)
            log.info(
                "%s / %s -> AUC=%.4f F1=%.4f sens=%.3f spec=%.3f FN=%d",
                name, variant_name, metrics["auroc"], metrics["f1"],
                metrics["sensitivity"], metrics["specificity"], metrics["fn"],
            )

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "capstone_results.csv", index=False)
    print("\n=== Capstone model: quasi-SOFA feature ablation ===")
    print(results[["dataset", "variant", "auroc", "f1", "sensitivity", "specificity", "tp", "fp", "tn", "fn"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
