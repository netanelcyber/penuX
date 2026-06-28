"""Extract acute pancreatitis cohort from MIMIC-III/IV and evaluate PenuX-AP-Severity.

Produces a CSV compatible with the rest of the PenuX-AP pipeline, then runs
evaluation (confusion matrices, threshold sweep, metrics curve) using the
existing penux_ap.evaluation module.

Usage:
    python scripts/extract_mimic_ap.py --mimic-dir ../mimic-iv-clinical-database-demo-2.2
    python scripts/extract_mimic_ap.py --mimic-dir /data/mimic-iv --out outputs/mimic_eval
    python scripts/extract_mimic_ap.py --mimic-dir /data/mimic-iv --mimic-version 3

IMPORTANT: MIMIC data requires credentialed PhysioNet access.
           Research / educational use only.
"""
import argparse
import csv
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from penux_ap.evaluation import (
    evaluate_binary_classifier,
    confusion_matrix_at_thresholds,
    plot_confusion_matrix_sweep,
    plot_metrics_vs_threshold,
    threshold_table,
)
from penux_ap.utils import ensure_dir, save_json, setup_logging

log = setup_logging()

# ── MIMIC lab itemid → PenuX feature name ─────────────────────────────────
# MIMIC-IV itemids (hosp/labevents)
MIMIC4_ITEM_MAP: dict[str, str] = {
    "51300": "wbc",          "51301": "wbc",
    "50889": "crp",
    "50912": "creatinine",
    "51006": "bun",
    "50931": "glucose",      "50809": "glucose",
    "50954": "ldh",
    "50878": "ast",
    "50861": "alt",
    "51221": "hematocrit",   "51222": "hematocrit",  "50810": "hematocrit",
    "50893": "calcium",
    "50862": "albumin",
    "50885": "bilirubin_total",
    "50956": "lipase",
    "50867": "amylase",
    "51000": "triglycerides",
}

# MIMIC-III uses different itemids (labevents table, same schema)
MIMIC3_ITEM_MAP: dict[str, str] = {
    "51300": "wbc",          "51301": "wbc",
    "50149": "crp",
    "50090": "creatinine",
    "50177": "bun",
    "50112": "glucose",
    "50300": "ldh",
    "50073": "ast",
    "50062": "alt",
    "50030": "hematocrit",
    "50080": "calcium",
    "50060": "albumin",
    "50170": "bilirubin_total",
    "50203": "lipase",
    "50037": "amylase",
}

# ICD codes for acute pancreatitis
AP_ICD9  = {"5770"}
AP_ICD10_PREFIX = "K85"

# Severity proxy: organ failure / sepsis comorbidities in same admission
SEVERE_ICD9_PREFIXES  = {"584", "518", "785", "99591", "99592"}
SEVERE_ICD10_PREFIXES = {"N17", "J96", "A41", "R65"}

# PenuX scoring heuristic (logistic, BISAP/Ranson-weighted)
WEIGHTS = {
    "wbc":             0.08,
    "crp":             0.004,
    "creatinine":      0.30,
    "bun":             0.015,
    "glucose":         0.002,
    "ldh":             0.001,
    "hematocrit":      0.03,
    "ast":             0.002,
    "albumin":        -0.30,
    "calcium":        -0.40,
    "bilirubin_total": 0.05,
}
THRESHOLDS = {
    "wbc": 12.0, "crp": 150.0, "creatinine": 1.5, "bun": 25.0,
    "glucose": 200.0, "ldh": 250.0, "hematocrit": 44.0, "ast": 250.0,
    "albumin": 3.5, "calcium": 8.0, "bilirubin_total": 3.0,
}
BASE_LOGIT = -1.8
AGE_WEIGHT = 0.015


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def _find(mimic_dir: Path, *parts) -> Path:
    """Find a file, trying .gz and plain variants."""
    base = mimic_dir.joinpath(*parts)
    for candidate in [base, Path(str(base) + ".gz"),
                      Path(str(base).replace(".csv.gz", ".csv"))]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {base}[.gz] in {mimic_dir}")


def _score(labs: dict, age: float, sex: str) -> float:
    logit = BASE_LOGIT + AGE_WEIGHT * max(0, age - 55)
    if sex.upper() in ("M", "MALE"):
        logit += 0.15
    for feat, w in WEIGHTS.items():
        v = labs.get(feat)
        if v is None:
            continue
        t = THRESHOLDS[feat]
        logit += w * (max(0, t - v) if feat in ("albumin", "calcium") else max(0, v - t))
    return round(1.0 / (1.0 + math.exp(-logit)), 4)


def _is_severe(icd_codes: list[str]) -> bool:
    for code in icd_codes:
        for pfx in SEVERE_ICD9_PREFIXES:
            if code.startswith(pfx):
                return True
        for pfx in SEVERE_ICD10_PREFIXES:
            if code.startswith(pfx):
                return True
    return False


def extract_cohort(mimic_dir: Path, item_map: dict[str, str]) -> pd.DataFrame:
    """Return a DataFrame with one row per AP admission."""
    hosp = mimic_dir / "hosp"

    # AP admissions
    ap_hadm: dict[str, str] = {}  # hadm_id → subject_id
    with _open(_find(hosp, "diagnoses_icd.csv")) as f:
        for r in csv.DictReader(f):
            code = r["icd_code"].strip()
            if code in AP_ICD9 or code.startswith(AP_ICD10_PREFIX):
                ap_hadm[r["hadm_id"]] = r["subject_id"]

    log.info("AP admissions found: %d", len(ap_hadm))
    if not ap_hadm:
        raise RuntimeError("No AP admissions found — check ICD code filtering.")

    # All diagnoses per AP admission (for severity label)
    all_dx: dict[str, list] = defaultdict(list)
    with _open(_find(hosp, "diagnoses_icd.csv")) as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] in ap_hadm:
                all_dx[r["hadm_id"]].append(r["icd_code"].strip())

    # Demographics
    patients: dict[str, dict] = {}
    with _open(_find(hosp, "patients.csv")) as f:
        for r in csv.DictReader(f):
            patients[r["subject_id"]] = r

    # Lab values
    labs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with _open(_find(hosp, "labevents.csv")) as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] not in ap_hadm:
                continue
            feat = item_map.get(r["itemid"])
            if feat is None:
                continue
            try:
                labs[r["hadm_id"]][feat].append(float(r["valuenum"]))
            except (ValueError, TypeError):
                pass

    # Build rows
    rows = []
    for hadm_id, subject_id in ap_hadm.items():
        pt   = patients.get(subject_id, {})
        age  = float(pt.get("anchor_age", pt.get("dob", 55)) or 55)
        sex  = pt.get("gender", "M")
        mean_labs = {feat: sum(vals) / len(vals)
                     for feat, vals in labs.get(hadm_id, {}).items()}
        severe  = int(_is_severe(all_dx.get(hadm_id, [])))
        prob    = _score(mean_labs, age, sex)
        row = {
            "hadm_id": hadm_id, "subject_id": subject_id,
            "age": age, "sex": sex, "severe": severe,
            "sap_probability": prob,
            "n_features": len(mean_labs),
        }
        row.update(mean_labs)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MIMIC AP cohort and evaluate PenuX-AP-Severity"
    )
    parser.add_argument("--mimic-dir",     required=True, help="Path to MIMIC-IV (or III) root")
    parser.add_argument("--mimic-version", default="4", choices=["3", "4"],
                        help="MIMIC version (default: 4)")
    parser.add_argument("--out",           default="outputs/mimic_eval",
                        help="Output directory")
    parser.add_argument("--threshold",     type=float, default=0.40,
                        help="Primary decision threshold (default 0.40)")
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    out_dir   = ensure_dir(args.out)
    item_map  = MIMIC4_ITEM_MAP if args.mimic_version == "4" else MIMIC3_ITEM_MAP

    log.info("MIMIC-%s dir: %s", args.mimic_version, mimic_dir)

    # ── Extract ──────────────────────────────────────────────────────────
    df = extract_cohort(mimic_dir, item_map)
    log.info("Cohort: %d admissions | %d severe | %d non-severe",
             len(df), df["severe"].sum(), (df["severe"] == 0).sum())

    csv_path = out_dir / "mimic_ap_cohort.csv"
    df.to_csv(csv_path, index=False)
    log.info("Cohort CSV → %s", csv_path)

    y_true  = df["severe"].values
    y_proba = df["sap_probability"].values

    # ── Evaluate ─────────────────────────────────────────────────────────
    metrics = evaluate_binary_classifier(y_true, y_proba, threshold=args.threshold)
    save_json(metrics, out_dir / "eval_metrics.json")
    log.info("AUROC=%.3f  Sens=%.3f  Spec=%.3f  F1=%.3f",
             metrics["auroc"], metrics["sensitivity"],
             metrics["specificity"], metrics["f1"])
    log.info("TP=%d  FP=%d  TN=%d  FN=%d",
             metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])

    tt = threshold_table(y_true, y_proba)
    tt.to_csv(out_dir / "threshold_table.csv", index=False)

    cms = confusion_matrix_at_thresholds(y_true, y_proba)
    save_json(cms, out_dir / "confusion_matrices.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        plot_confusion_matrix_sweep(
            y_true, y_proba,
            thresholds=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            save_path=str(out_dir / "confusion_matrix_sweep.png"),
        )
        plot_metrics_vs_threshold(
            y_true, y_proba,
            highlight_threshold=args.threshold,
            save_path=str(out_dir / "metrics_vs_threshold.png"),
        )
    except ImportError:
        log.warning("matplotlib not available — skipping plots")

    log.info("Done. Results in %s/", out_dir)


if __name__ == "__main__":
    main()
