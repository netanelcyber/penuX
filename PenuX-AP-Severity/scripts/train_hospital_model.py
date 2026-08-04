#!/usr/bin/env python3
"""Train the hospital-oriented PenuX-AP model bundle.

Example:
    python scripts/train_hospital_model.py \
        --input hospital_ap.csv \
        --target persistent_organ_failure_gt_48h \
        --output models/hospital_ap_model.joblib

The input must already be pseudonymised and must contain the fields needed to
define the operational AP cohort. No imaging column is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from api.hospital_model import train_hospital_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Pseudonymised CSV or XLSX file")
    parser.add_argument(
        "--target",
        default="persistent_organ_failure_gt_48h",
        help="Binary 0/1 outcome column",
    )
    parser.add_argument("--output", required=True, help="Output .joblib model bundle")
    parser.add_argument(
        "--minimum-availability",
        type=float,
        default=0.80,
        help="Minimum variable availability for the primary model",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.98,
        help="OOF sensitivity target used to choose the research threshold",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--audit-output",
        help="Optional CSV path for the case-level eligibility audit",
    )
    parser.add_argument(
        "--metrics-output",
        help="Optional JSON path; defaults beside the model bundle",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be CSV, XLSX or XLS")


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)

    frame = load_table(source)
    bundle, audit = train_hospital_model(
        frame,
        target_column=args.target,
        minimum_availability=args.minimum_availability,
        target_sensitivity=args.target_sensitivity,
        cv_folds=args.cv_folds,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)

    audit_path = Path(args.audit_output) if args.audit_output else output.with_suffix(".eligibility.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    metrics_path = Path(args.metrics_output) if args.metrics_output else output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": bundle.model_version,
        "feature_names": bundle.feature_names,
        "feature_availability": bundle.feature_availability,
        "metrics": bundle.metrics,
        "warning": bundle.warning,
        "data_policy": {
            "imaging_required": False,
            "minimum_core_groups_per_case": 8,
            "primary_variable_availability": args.minimum_availability,
            "target_sensitivity": args.target_sensitivity,
            "imputation_fitted_inside_cv": True,
            "complete_case_only": False,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved model: {output}")
    print(f"Saved eligibility audit: {audit_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(bundle.metrics, indent=2))


if __name__ == "__main__":
    main()
