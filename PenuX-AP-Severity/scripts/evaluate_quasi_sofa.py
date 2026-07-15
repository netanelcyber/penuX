"""Evaluate the lab-only quasi-SOFA score (penux_ap.clinical_scores.compute_quasi_sofa_labs)
as a standalone SAP predictor on both registered public datasets.

Methodology note: this does NOT use the PhysioNet sepsis dataset's actual
patient data -- only the Sepsis-3/SOFA scoring *methodology* (organ
dysfunction via renal/hepatic/coagulation labs, plus optional SIRS-style
WBC and lactate terms), applied to the existing sanitized AP datasets'
own lab columns. See docs/dataset_sources.md and
src/penux_ap/clinical_scores.py's compute_quasi_sofa_labs docstring.

Usage:
    python scripts/evaluate_quasi_sofa.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.clinical_scores import compute_quasi_sofa_labs
from penux_ap.datasets import load_dataset
from penux_ap.evaluation import evaluate_binary_classifier, threshold_table
from penux_ap.labels import binarize_target
from penux_ap.utils import setup_logging

log = setup_logging()


def evaluate(name, df, target_col, colmap, positive_value):
    y = binarize_target(df[target_col]).dropna().astype(int)
    if positive_value == 0:
        y = 1 - y
    df = df.loc[y.index]

    mapped = pd.DataFrame(index=df.index)
    for std_name, raw_col in colmap.items():
        if raw_col in df.columns:
            mapped[std_name] = df[raw_col]
    scores = mapped.apply(compute_quasi_sofa_labs, axis=1)

    valid = scores.notna()
    log.info("%s: quasi-SOFA computed for %d/%d rows", name, valid.sum(), len(df))

    y_valid = y[valid]
    scores_valid = scores[valid]
    auc = roc_auc_score(y_valid, scores_valid)
    metrics = evaluate_binary_classifier(y_valid.values, scores_valid.values / scores_valid.max())
    tt = threshold_table(y_valid.values, (scores_valid / scores_valid.max()).values,
                          thresholds=sorted(set(scores_valid / scores_valid.max())))
    best_f1_row = tt.loc[tt["f1"].idxmax()]

    print(f"\n=== {name}: quasi-SOFA (lab-only) as standalone SAP score ===")
    print(f"AUC = {auc:.4f}  (n={valid.sum()})")
    print(f"Best-F1 threshold row: {best_f1_row.to_dict()}")
    return auc


def main():
    root = Path(__file__).resolve().parents[1]

    df_m = load_dataset(root / "data/public_sanitized/ap_multiml_sanitized.csv")
    colmap_m = {
        "creatinine_umol_l": "Cr", "bilirubin_umol_l": "TBIL", "platelets_e9_l": "PLT",
        "wbc_e9_l": "WBC",
    }
    evaluate("multiml", df_m, "Diagnostic Result", colmap_m, positive_value=0)

    df_l = load_dataset(root / "data/public_sanitized/ap_lnn_sanitized.csv")
    colmap_l = {
        "creatinine_umol_l": "肌酐", "bilirubin_umol_l": "总胆红素", "platelets_e9_l": "血小板",
        "wbc_e9_l": "白细胞", "lactate_mmol_l": "乳酸",
    }
    evaluate("lnn", df_l, "严重程度", colmap_l, positive_value=0)


if __name__ == "__main__":
    main()
