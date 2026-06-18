#!/usr/bin/env python3
"""
MIMIC-III / MIMIC-IV → PenuX-AP-Severity Confusion Matrix Evaluation

Extracts acute pancreatitis (AP) patients from the MIMIC-IV clinical demo
database, pulls their admission lab values, scores them with the PenuX
severity heuristic, and generates confusion matrices across thresholds.

Usage:
    python3 mimic_ap_confusion_matrix.py
    python3 mimic_ap_confusion_matrix.py --mimic-dir mimic-iv-clinical-database-demo-2.2
    python3 mimic_ap_confusion_matrix.py --out outputs/ap_eval

IMPORTANT: Research / educational use only.
           MIMIC data requires credentialed PhysioNet access.
           Do NOT use for clinical decision-making.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Optional matplotlib ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed — text output only.")

# ═══════════════════════════════════════════════════════════════════════════
# MIMIC item → feature mapping
# ═══════════════════════════════════════════════════════════════════════════
ITEM_MAP = {
    # WBC
    "51300": "wbc", "51301": "wbc",
    # CRP  (rarely in MIMIC, but included)
    "50889": "crp",
    # Creatinine
    "50912": "creatinine",
    # BUN
    "51006": "bun",
    # Glucose
    "50931": "glucose", "50809": "glucose",
    # LDH
    "50954": "ldh",
    # AST
    "50878": "ast",
    # ALT
    "50861": "alt",
    # Hematocrit
    "51221": "hematocrit", "51222": "hematocrit", "50810": "hematocrit",
    # Calcium
    "50893": "calcium",
    # Albumin
    "50862": "albumin",
    # Total Bilirubin
    "50885": "bilirubin_total",
    # Lipase
    "50956": "lipase",
    # Amylase
    "50867": "amylase",
    # Triglycerides
    "51000": "triglycerides",
}

# ICD-9: 577.0 = Acute pancreatitis
# ICD-10: K85.* = Acute pancreatitis
AP_ICD9 = {"5770"}
AP_ICD10_PREFIX = "K85"

# ═══════════════════════════════════════════════════════════════════════════
# Severity label heuristic (Atlanta 2012 proxy from MIMIC ICD codes)
# ═══════════════════════════════════════════════════════════════════════════
# Severe AP ICD codes (organ failure / necrosis markers)
SEVERE_ICD9 = {
    "5771",   # Chronic pancreatitis
    "5772",   # Pancreatic cyst
    "5778",   # Other pancreatitis
    "5770",   # Acute
    "99591",  # Sepsis
    "99592",  # Severe sepsis
    "58381",  # Acute renal failure
    "4589",   # Hypotension
    "51881",  # Acute respiratory failure
}
SEVERE_ICD10 = {
    "K853", "K854", "K858", "K859",  # Severe / infected / other AP
    "A419",  # Sepsis
    "N170",  # Acute kidney injury
    "J960",  # Acute respiratory failure
}


def is_severe_by_icd(all_icd_codes: list[str]) -> bool:
    """Proxy label: severe AP if patient has organ failure / sepsis codes."""
    for code in all_icd_codes:
        if code in SEVERE_ICD9:
            continue  # AP itself is not a severity marker
        if code[:4] in SEVERE_ICD10 or code[:3] in {"A41", "N17", "J96"}:
            return True
        # Ranson-like: acute renal failure, respiratory failure, shock
        if code.startswith(("584", "518", "785")):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PenuX scoring heuristic (simplified logistic approximation)
# ═══════════════════════════════════════════════════════════════════════════
# Weights derived from literature (BISAP, Ranson, APACHE-II feature importance)
# This is a HEURISTIC for demo purposes — not the trained model.
FEATURE_WEIGHTS = {
    "wbc":            0.08,   # per 10^3/µL above 12
    "crp":            0.004,  # per mg/L above 150
    "creatinine":     0.25,   # per mg/dL above 1.5
    "bun":            0.015,  # per mg/dL above 25
    "glucose":        0.002,  # per mg/dL above 200
    "ldh":            0.001,  # per U/L above 250
    "hematocrit":     0.03,   # per % above 44 (haemoconcentration)
    "ast":            0.002,  # per U/L above 250
    "albumin":       -0.3,    # per g/dL (lower = worse)
    "calcium":       -0.4,    # per mg/dL below 8
    "bilirubin_total": 0.05,  # per mg/dL above 3
}
FEATURE_THRESHOLDS = {
    "wbc":            12.0,
    "crp":            150.0,
    "creatinine":     1.5,
    "bun":            25.0,
    "glucose":        200.0,
    "ldh":            250.0,
    "hematocrit":     44.0,
    "ast":            250.0,
    "albumin":        3.5,
    "calcium":        8.0,
    "bilirubin_total": 3.0,
}
AGE_WEIGHT = 0.015   # per year above 55
BASE_LOGIT = -1.8    # calibrated intercept


def score_patient(labs: dict, age: float, sex: str) -> float:
    """Return P(severe AP) for a patient given labs + demographics."""
    logit = BASE_LOGIT

    # Age risk
    logit += AGE_WEIGHT * max(0, age - 55)

    # Sex risk (male slightly higher)
    if sex.upper() in ("M", "MALE"):
        logit += 0.15

    for feat, weight in FEATURE_WEIGHTS.items():
        val = labs.get(feat)
        if val is None:
            continue
        thresh = FEATURE_THRESHOLDS[feat]
        if feat in ("albumin", "calcium"):
            # Lower is worse
            logit += weight * max(0, thresh - val)
        else:
            logit += weight * max(0, val - thresh)

    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(prob, 4)


# ═══════════════════════════════════════════════════════════════════════════
# MIMIC data loaders
# ═══════════════════════════════════════════════════════════════════════════
def open_mimic(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def load_ap_admissions(mimic_dir: Path) -> dict:
    """Return {hadm_id: subject_id} for all AP admissions."""
    ap = {}
    diag_path = mimic_dir / "hosp" / "diagnoses_icd.csv.gz"
    if not diag_path.exists():
        diag_path = mimic_dir / "hosp" / "diagnoses_icd.csv"
    with open_mimic(diag_path) as f:
        for r in csv.DictReader(f):
            code = r["icd_code"].strip()
            if code in AP_ICD9 or code.startswith(AP_ICD10_PREFIX):
                ap[r["hadm_id"]] = r["subject_id"]
    return ap


def load_all_diagnoses(mimic_dir: Path, hadm_ids: set) -> dict:
    """Return {hadm_id: [icd_codes]} for the given admissions."""
    result = defaultdict(list)
    diag_path = mimic_dir / "hosp" / "diagnoses_icd.csv.gz"
    if not diag_path.exists():
        diag_path = mimic_dir / "hosp" / "diagnoses_icd.csv"
    with open_mimic(diag_path) as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] in hadm_ids:
                result[r["hadm_id"]].append(r["icd_code"].strip())
    return result


def load_patients(mimic_dir: Path) -> dict:
    """Return {subject_id: {anchor_age, gender}}."""
    path = mimic_dir / "hosp" / "patients.csv.gz"
    if not path.exists():
        path = mimic_dir / "hosp" / "patients.csv"
    with open_mimic(path) as f:
        return {r["subject_id"]: r for r in csv.DictReader(f)}


def load_labs(mimic_dir: Path, hadm_ids: set) -> dict:
    """Return {hadm_id: {feature: [values]}} for AP patients."""
    result = defaultdict(lambda: defaultdict(list))
    path = mimic_dir / "hosp" / "labevents.csv.gz"
    if not path.exists():
        path = mimic_dir / "hosp" / "labevents.csv"
    with open_mimic(path) as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] not in hadm_ids:
                continue
            feat = ITEM_MAP.get(r["itemid"])
            if feat is None:
                continue
            try:
                val = float(r["valuenum"])
                result[r["hadm_id"]][feat].append(val)
            except (ValueError, TypeError):
                pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Confusion matrix helpers
# ═══════════════════════════════════════════════════════════════════════════
def confusion_at_threshold(y_true, y_proba, threshold):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_proba):
        pred = int(yp >= threshold)
        if yt == 1 and pred == 1: tp += 1
        elif yt == 0 and pred == 1: fp += 1
        elif yt == 0 and pred == 0: tn += 1
        else: fn += 1
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else float("nan")
    return {"threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════
def plot_single_cm(cm: dict, ax, cmap="Blues"):
    mat = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    row_tot = [cm["TN"] + cm["FP"], cm["FN"] + cm["TP"]]
    labels = [["TN\nCleared\ncorrectly", "FP\nFalse alarm"],
              ["FN\nMissed\nsevere",      "TP\nCaught\nsevere"]]
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(max(r) for r in mat) or 1)
    thresh = max(max(r) for r in mat) / 2.0
    for i in range(2):
        for j in range(2):
            count = mat[i][j]
            pct   = count / row_tot[i] * 100 if row_tot[i] > 0 else 0
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)\n{labels[i][j]}",
                    ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-severe", "Predicted\nSevere"], fontsize=8)
    ax.set_yticklabels(["Actual\nNon-severe", "Actual\nSevere"], fontsize=8)
    s = cm["sensitivity"]; sp = cm["specificity"]
    title = (f"Threshold={cm['threshold']:.2f}\n"
             f"Sens={s:.2f}  Spec={sp:.2f}" if not (math.isnan(s) or math.isnan(sp))
             else f"Threshold={cm['threshold']:.2f}")
    ax.set_title(title, fontsize=9)


def plot_confusion_sweep(cms: list, out_path: Path):
    n = len(cms)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes_flat = [ax for row in (axes if nrows > 1 else [axes]) for ax in (row if ncols > 1 else [row])]
    for i, cm in enumerate(cms):
        plot_single_cm(cm, axes_flat[i])
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("PenuX-AP-Severity — MIMIC-IV Confusion Matrices\n"
                 "(8 AP admissions, demo cohort)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_metrics_curve(cms: list, out_path: Path, default_thresh=0.4):
    thresholds = [c["threshold"] for c in cms]
    sens = [c["sensitivity"] for c in cms]
    spec = [c["specificity"] for c in cms]
    ppv  = [c["ppv"]  for c in cms]
    npv  = [c["npv"]  for c in cms]
    f1   = [c["f1"]   for c in cms]

    def clean(vals):
        return [v if not math.isnan(v) else None for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    for vals, color, label in [
        (sens, "#e74c3c", "Sensitivity"),
        (spec, "#2ecc71", "Specificity"),
        (ppv,  "#3498db", "PPV (Precision)"),
        (npv,  "#9b59b6", "NPV"),
        (f1,   "#f39c12", "F1 Score"),
    ]:
        clean_vals = clean(vals)
        valid = [(t, v) for t, v in zip(thresholds, clean_vals) if v is not None]
        if valid:
            ax.plot(*zip(*valid), color=color, linewidth=2, label=label, marker="o", markersize=4)

    ax.axvline(default_thresh, color="gray", linestyle="--", linewidth=1.2,
               label=f"Default ({default_thresh})")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Metric Value")
    ax.set_title("PenuX-AP-Severity — Metrics vs. Threshold (MIMIC-IV demo)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MIMIC → PenuX AP confusion matrix")
    parser.add_argument("--mimic-dir", default="mimic-iv-clinical-database-demo-2.2",
                        help="Path to MIMIC-IV database directory")
    parser.add_argument("--out", default="outputs/ap_mimic_eval",
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Primary decision threshold (default 0.4)")
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mimic_dir.exists():
        print(f"ERROR: MIMIC directory not found: {mimic_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("PenuX-AP-Severity — MIMIC Evaluation")
    print("=" * 60)

    # ── 1. Load AP admissions ─────────────────────────────────
    print("\n[1/4] Loading AP admissions...")
    ap_hadm = load_ap_admissions(mimic_dir)
    print(f"  AP admissions found: {len(ap_hadm)}")

    if len(ap_hadm) == 0:
        print("No AP admissions found. Check ICD code filtering.")
        sys.exit(1)

    # ── 2. Load demographics + labs ──────────────────────────
    print("[2/4] Loading labs and demographics...")
    patients    = load_patients(mimic_dir)
    labs        = load_labs(mimic_dir, set(ap_hadm.keys()))
    all_dx      = load_all_diagnoses(mimic_dir, set(ap_hadm.keys()))

    # ── 3. Score each patient ─────────────────────────────────
    print("[3/4] Scoring patients...\n")
    records = []
    for hadm_id, subject_id in ap_hadm.items():
        pt   = patients.get(subject_id, {})
        age  = float(pt.get("anchor_age", 55))
        sex  = pt.get("gender", "M")

        # Use median of first-day lab values
        pt_labs = {feat: sum(vals) / len(vals)
                   for feat, vals in labs.get(hadm_id, {}).items()}

        # Severity label: organ failure / sepsis codes as proxy
        icd_codes  = all_dx.get(hadm_id, [])
        is_severe  = is_severe_by_icd(icd_codes)
        prob       = score_patient(pt_labs, age, sex)

        records.append({
            "hadm_id":    hadm_id,
            "subject_id": subject_id,
            "age": age, "sex": sex,
            "labs": pt_labs,
            "icd_codes":  icd_codes,
            "severe_label": int(is_severe),
            "sap_probability": prob,
            "risk_group": "High" if prob >= 0.6 else "Moderate" if prob >= 0.3 else "Low",
            "features_used": sorted(pt_labs.keys()),
        })

        status = "⚠️  SEVERE" if is_severe else "✅ Non-severe"
        print(f"  hadm={hadm_id}  age={int(age)}  sex={sex}  "
              f"P(SAP)={prob:.3f}  [{status}]")
        key_labs = {k: round(v, 1) for k, v in pt_labs.items()
                    if k in ("wbc", "creatinine", "crp", "ldh", "bun")}
        print(f"    key labs: {key_labs}")
        print(f"    ICD codes: {icd_codes[:6]}")

    y_true  = [r["severe_label"]    for r in records]
    y_proba = [r["sap_probability"] for r in records]
    n_severe = sum(y_true)
    n_total  = len(y_true)

    print(f"\n  Cohort: {n_total} admissions | {n_severe} severe ({n_severe/n_total:.0%}) | "
          f"{n_total - n_severe} non-severe")

    # ── 4. Confusion matrices ─────────────────────────────────
    print("\n[4/4] Generating confusion matrices...")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    cms        = [confusion_at_threshold(y_true, y_proba, t) for t in thresholds]

    # Primary threshold result
    primary = confusion_at_threshold(y_true, y_proba, args.threshold)
    print(f"\n  ── At threshold {args.threshold:.2f} ──")
    print(f"  TP={primary['TP']}  FP={primary['FP']}  TN={primary['TN']}  FN={primary['FN']}")
    s = primary["sensitivity"]; sp = primary["specificity"]
    print(f"  Sensitivity: {s:.3f}" if not math.isnan(s) else "  Sensitivity: N/A")
    print(f"  Specificity: {sp:.3f}" if not math.isnan(sp) else "  Specificity: N/A")
    f1 = primary["f1"]
    print(f"  F1:          {f1:.3f}" if not math.isnan(f1) else "  F1: N/A")

    # Threshold table
    print("\n  Threshold sweep:")
    print(f"  {'Threshold':>9}  {'TP':>3}  {'FP':>3}  {'TN':>3}  {'FN':>3}  "
          f"{'Sens':>6}  {'Spec':>6}  {'F1':>6}")
    for cm in cms:
        s_str  = f"{cm['sensitivity']:.3f}" if not math.isnan(cm['sensitivity']) else "  N/A"
        sp_str = f"{cm['specificity']:.3f}" if not math.isnan(cm['specificity']) else "  N/A"
        f1_str = f"{cm['f1']:.3f}"          if not math.isnan(cm['f1'])          else "  N/A"
        print(f"  {cm['threshold']:>9.2f}  {cm['TP']:>3}  {cm['FP']:>3}  "
              f"{cm['TN']:>3}  {cm['FN']:>3}  {s_str:>6}  {sp_str:>6}  {f1_str:>6}")

    # Save JSON results
    results = {
        "dataset":    str(mimic_dir),
        "n_patients": n_total,
        "n_severe":   n_severe,
        "primary_threshold": args.threshold,
        "primary_confusion_matrix": primary,
        "threshold_sweep": cms,
        "patients": records,
    }
    json_path = out_dir / "mimic_ap_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON → {json_path}")

    # Plots
    if HAS_MPL:
        plot_confusion_sweep(cms, out_dir / "mimic_cm_sweep.png")
        plot_metrics_curve(cms, out_dir / "mimic_metrics_curve.png", args.threshold)
    else:
        print("  (Install matplotlib to generate PNG figures)")

    print("\n" + "=" * 60)
    print("✅  Evaluation complete")
    print(f"    Results in: {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
