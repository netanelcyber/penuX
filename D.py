#!/usr/bin/env python3
"""
PenuX - Rapid-Lab Separate Binary AMR/Pathogen Targets
=========================================

Uses only fast prediction-time features:
  * rapid labs: WBC, lactate, CRP, procalcitonin, CBC/basic chemistry markers
  * vitals: temperature, SpO2
  * age

Excludes microbiology interpretation, organism names as input, antibiotic-panel
fields, ab_name, dilution, susceptibility, and any value available after T0 +
prediction window.

Usage:
  python penux_three_binary_targets.py --dir /path/to/mimic-demo --window_hours 6

Main goal:
  Evaluate three separate clinically actionable binary targets:
    1) MRSA vs non-MRSA
    2) FUNGAL/YEAST vs non-FUNGAL
    3) HIGH_RISK_GRAM_NEGATIVE vs other

  This is preferred over one broad HIGH_RISK_DANGEROUS label when N is small.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CANDIDATES = [
    SCRIPT_DIR / "dataset" / "mimic" / "mimic-iii-clinical-database-demo-1.4",
    SCRIPT_DIR / "dataset" / "mimic" / "mimic-iv-clinical-database-demo-2.2",
    SCRIPT_DIR.parent / "dataset" / "mimic" / "mimic-iii-clinical-database-demo-1.4",
    SCRIPT_DIR.parent / "dataset" / "mimic" / "mimic-iv-clinical-database-demo-2.2",
    SCRIPT_DIR,
]

# Seed itemid map. The script also expands this using D_LABITEMS labels.
SEED_FAST_LAB_ITEMIDS: Dict[int, str] = {
    # CBC / infection markers
    51300: "wbc",
    51301: "wbc",
    51256: "neutrophils_pct",
    51244: "bands_pct",
    51265: "platelets",
    51222: "hemoglobin",
    51221: "hematocrit",

    # Rapid inflammatory / sepsis markers
    50813: "lactate",
    52442: "lactate",
    50889: "crp",
    227444: "crp",
    227464: "procalcitonin",

    # Basic chemistry often available quickly
    50912: "creatinine",
    51006: "bun",
    50882: "bicarbonate",
    50868: "anion_gap",
    50931: "glucose",
    50983: "sodium",
    50971: "potassium",
    50902: "chloride",
}

CHART_ITEMIDS_III = {
    676: "temperature_c",
    678: "temperature_f",
    646: "spo2",
}
CHART_ITEMIDS_IV = {
    223761: "temperature_f",
    223762: "temperature_c",
    220277: "spo2",
}


def pick_mimic_dir(user_dir: Optional[str]) -> Path:
    if user_dir:
        return Path(user_dir).expanduser().resolve()
    for p in DEFAULT_CANDIDATES:
        if p.exists() and list(p.rglob("*.csv*")):
            return p.resolve()
    return SCRIPT_DIR.resolve()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def find_table_path(mimic_dir: Path, name: str) -> Path:
    target = name.lower()
    candidates: List[Path] = []
    for ext in ("*.csv", "*.csv.gz", "*.parquet"):
        candidates.extend(mimic_dir.rglob(ext))
    for p in candidates:
        stem = p.name.lower()
        if stem in {f"{target}.csv", f"{target}.csv.gz", f"{target}.parquet"}:
            return p
    raise FileNotFoundError(f"Cannot find table {name!r} under {mimic_dir}")


def load_table(name: str, mimic_dir: Path, required: bool = True) -> Optional[pd.DataFrame]:
    try:
        p = find_table_path(mimic_dir, name)
    except FileNotFoundError:
        if required:
            raise
        print(f"  [warn] {name} not found")
        return None
    try:
        shown = p.relative_to(mimic_dir)
    except ValueError:
        shown = p.name
    print(f"  Loading {shown}...")
    if p.suffix == ".parquet":
        return normalize_columns(pd.read_parquet(p))
    return normalize_columns(pd.read_csv(p, low_memory=False))


def load_all(mimic_dir: Path):
    print(f"[1/7] Loading tables from {mimic_dir}...")
    patients = load_table("patients", mimic_dir)
    admissions = load_table("admissions", mimic_dir)
    icustays = load_table("icustays", mimic_dir)
    labevents = load_table("labevents", mimic_dir)
    micro = load_table("microbiologyevents", mimic_dir)
    chartevents = load_table("chartevents", mimic_dir, required=False)
    d_labitems = load_table("d_labitems", mimic_dir, required=False)
    return patients, admissions, icustays, labevents, chartevents, micro, d_labitems


def clean_label(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def infer_fast_lab_itemids(d_labitems: Optional[pd.DataFrame]) -> Dict[int, str]:
    mapping = dict(SEED_FAST_LAB_ITEMIDS)
    if d_labitems is None or "itemid" not in d_labitems.columns or "label" not in d_labitems.columns:
        return mapping

    rules: List[Tuple[str, str]] = [
        (r"\bwhite blood cells?\b|\bwbc\b", "wbc"),
        (r"\blactate\b", "lactate"),
        (r"\bprocalcitonin\b", "procalcitonin"),
        (r"\bc[\-\s]?reactive protein\b|\bcrp\b", "crp"),
        (r"\bneutrophils?\b", "neutrophils_pct"),
        (r"\bbands?\b", "bands_pct"),
        (r"\bplatelet count\b|\bplatelets?\b", "platelets"),
        (r"\bhemoglobin\b", "hemoglobin"),
        (r"\bhematocrit\b", "hematocrit"),
        (r"\bcreatinine\b", "creatinine"),
        (r"\burea nitrogen\b|\bbun\b", "bun"),
        (r"\bbicarbonate\b|\btotal co2\b", "bicarbonate"),
        (r"\banion gap\b", "anion_gap"),
        (r"\bglucose\b", "glucose"),
        (r"\bsodium\b", "sodium"),
        (r"\bpotassium\b", "potassium"),
        (r"\bchloride\b", "chloride"),
    ]

    tmp = d_labitems[["itemid", "label"]].dropna().copy()
    for _, row in tmp.iterrows():
        try:
            itemid = int(row["itemid"])
        except Exception:
            continue
        label = clean_label(row["label"])
        for pattern, feature in rules:
            if re.search(pattern, label):
                mapping[itemid] = feature
                break
    return mapping


def coalesce_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for c in cols:
        if c in df.columns:
            v = pd.to_datetime(df[c], errors="coerce")
            out = out.fillna(v)
    return out

def safe_add_hours(dt: pd.Series, hours: int, label: str = "datetime") -> pd.Series:
    """Safely add hours to pandas datetimes without int64 overflow.

    Pandas datetime64[ns] overflows near Timestamp.max/min. Instead of doing
    vectorized dt + Timedelta on every row, this function masks unsafe rows and
    returns NaT for them. Those rows are later dropped from the cohort.
    """
    s = pd.to_datetime(dt, errors="coerce")
    delta = pd.to_timedelta(int(hours), unit="h")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    max_allowed = pd.Timestamp.max - delta - pd.Timedelta(seconds=1)
    min_allowed = pd.Timestamp.min + pd.Timedelta(seconds=1)
    good = s.notna() & (s >= min_allowed) & (s <= max_allowed)
    unsafe = int((s.notna() & ~good).sum())
    if unsafe:
        print(f"  [warn] {unsafe} {label} value(s) too close to pandas datetime bounds; marking cutoff as NaT")

    out.loc[good] = s.loc[good] + delta
    return out


def safe_age_from_dob(t0: pd.Series, dob: pd.Series) -> pd.Series:
    """Compute age without subtracting datetimes, avoiding int64 overflow."""
    t0s = pd.to_datetime(t0, errors="coerce")
    dobs = pd.to_datetime(dob, errors="coerce")
    age = pd.Series(np.nan, index=t0s.index, dtype="float64")

    good = t0s.notna() & dobs.notna()
    if good.any():
        t = t0s.loc[good]
        d = dobs.loc[good]
        years = t.dt.year - d.dt.year
        had_birthday = (t.dt.month > d.dt.month) | ((t.dt.month == d.dt.month) & (t.dt.day >= d.dt.day))
        vals = years - (~had_birthday).astype(int)
        vals = vals.where(vals.between(0, 110), np.nan)
        age.loc[good] = vals.astype(float)
    return age


def build_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    micro: pd.DataFrame,
    window_hours: int,
) -> pd.DataFrame:
    print("[2/7] Building ICU-positive-culture cohort...")

    icu = icustays.copy()
    icu["t0"] = coalesce_datetime(icu, ["intime"])
    icu["icu_outtime"] = pd.to_datetime(icu["outtime"], errors="coerce") if "outtime" in icu.columns else pd.NaT

    stay_col = "stay_id" if "stay_id" in icu.columns else ("icustay_id" if "icustay_id" in icu.columns else None)
    keep_cols = ["subject_id", "hadm_id", "t0", "icu_outtime"]
    if stay_col:
        keep_cols.append(stay_col)

    pos = micro[micro["org_name"].notna()].copy()
    pos["culture_time"] = coalesce_datetime(pos, ["charttime", "chartdate"])
    if "spec_type_desc" not in pos.columns:
        pos["spec_type_desc"] = ""

    merged = pos.merge(icu[keep_cols], on=["subject_id", "hadm_id"], how="inner")
    before = len(merged)
    merged = merged[merged["culture_time"].notna() & merged["t0"].notna()]
    merged = merged[merged["culture_time"] >= merged["t0"]]
    if merged["icu_outtime"].notna().any():
        merged = merged[(merged["icu_outtime"].isna()) | (merged["culture_time"] <= merged["icu_outtime"])]
    print(f"  Temporal target guard: {before} -> {len(merged)} positive cultures after ICU admission")

    group_cols = ["subject_id", "hadm_id"]
    if stay_col:
        group_cols.append(stay_col)

    cohort = (
        merged.sort_values("culture_time")
        .groupby(group_cols, as_index=False)
        .agg(
            org_name=("org_name", "first"),
            spec_type_desc=("spec_type_desc", "first"),
            culture_time=("culture_time", "first"),
            t0=("t0", "first"),
            icu_outtime=("icu_outtime", "first"),
        )
    )

    pts = patients.copy()
    if "anchor_age" in pts.columns:
        cohort = cohort.merge(pts[["subject_id", "anchor_age"]], on="subject_id", how="left")
        cohort.rename(columns={"anchor_age": "age"}, inplace=True)
    elif "dob" in pts.columns:
        pts["dob"] = pd.to_datetime(pts["dob"], errors="coerce")
        cohort = cohort.merge(pts[["subject_id", "dob"]], on="subject_id", how="left")
        cohort["age"] = safe_age_from_dob(cohort["t0"], cohort["dob"])
        cohort.drop(columns=["dob"], inplace=True)
    else:
        cohort["age"] = np.nan

    cohort["t_cutoff"] = safe_add_hours(cohort["t0"], window_hours, label="ICU intime/t0")
    before_cutoff = len(cohort)
    cohort = cohort[cohort["t_cutoff"].notna()].copy()
    if len(cohort) < before_cutoff:
        print(f"  Dropped {before_cutoff - len(cohort)} row(s) with unsafe/invalid T0 cutoff")

    # Unique row id prevents duplicate (subject_id, hadm_id) indices from expanding
    # during feature/label alignment. This fixes boolean-mask length mismatches
    # when one admission has multiple ICU stays or repeated culture rows.
    cohort = cohort.reset_index(drop=True)
    cohort["cohort_row_id"] = np.arange(len(cohort), dtype=int)

    print(f"  Cohort: {len(cohort)} ICU stays/admissions with positive cultures")
    return cohort


def build_labels(cohort: pd.DataFrame, min_class_n: int = 3, top_species_n: int = 5) -> pd.DataFrame:
    print("[3/7] Building labels...")
    cohort = cohort.copy()

    vc = cohort["org_name"].value_counts()
    keep = vc[vc >= min_class_n].index
    cohort["label_species"] = cohort["org_name"].where(cohort["org_name"].isin(keep), "OTHER")
    top_species = vc.nlargest(top_species_n).index
    cohort["label_species_top"] = cohort["org_name"].where(cohort["org_name"].isin(top_species), "OTHER")

    def upper(org: object) -> str:
        return str(org).upper()

    def is_fungal(org: object) -> bool:
        u = upper(org)
        return any(x in u for x in ["YEAST", "CANDIDA", "ASPERGILLUS", "FUNGUS", "CRYPTOCOCCUS", "CANDIDA AURIS"])

    def is_mrsa(org: object) -> bool:
        u = upper(org)
        return "METHICILLIN RESISTANT STAPH AUREUS" in u or "MRSA" in u

    def is_staph_aureus(org: object) -> bool:
        u = upper(org)
        return is_mrsa(org) or "STAPH AUREUS" in u or "STAPHYLOCOCCUS AUREUS" in u

    def is_coag_neg_staph(org: object) -> bool:
        u = upper(org)
        return "COAGULASE NEGATIVE" in u or "COAG -" in u

    def is_enterococcus(org: object) -> bool:
        return "ENTEROCOCCUS" in upper(org)

    def is_vre_like(org: object) -> bool:
        u = upper(org)
        return "VANCOMYCIN RESISTANT" in u or "VRE" in u

    def is_high_risk_gram_negative(org: object) -> bool:
        u = upper(org)
        # These are clinically actionable, often AMR-concern organisms in ICU settings.
        # Susceptibility is NOT used as an input feature; organism name is used only to build targets.
        return any(x in u for x in [
            "ACINETOBACTER",
            "PSEUDOMONAS",
            "KLEBSIELLA",
            "ESCHERICHIA",
            "E. COLI",
            "ENTEROBACTER",
            "GRAM NEGATIVE",
            "PROTEUS",
            "SERRATIA",
            "CITROBACTER",
            "MORGANELLA",
        ])

    def is_necrotizing_risk(org: object) -> bool:
        u = upper(org)
        # Approximate pathogen-based necrotizing soft-tissue infection risk.
        # This is NOT a diagnosis of necrotizing fasciitis; true detection needs site, exam, surgery, imaging, CK, etc.
        return any(x in u for x in [
            "STREPTOCOCCUS PYOGENES",
            "GROUP A STREP",
            "BETA HEMOLYTIC STREPTOCOCCUS GROUP A",
            "CLOSTRIDIUM",
            "VIBRIO",
            "AEROMONAS",
        ])

    def to_3class(org: object) -> str:
        if is_fungal(org):
            return "FUNGAL"
        u = upper(org)
        if any(x in u for x in ["VIRUS", "VIRAL", "INFLUENZA", "RSV", "COVID", "CMV", "HSV"]):
            return "VIRAL"
        return "BACTERIAL"

    def to_clinical_group(org: object) -> str:
        if is_fungal(org):
            return "FUNGAL"
        if is_mrsa(org):
            return "MRSA"
        if is_staph_aureus(org):
            return "STAPH_AUREUS"
        if is_coag_neg_staph(org):
            return "COAG_NEG_STAPH"
        if is_vre_like(org):
            return "VRE_LIKE"
        if is_enterococcus(org):
            return "ENTEROCOCCUS"
        if is_high_risk_gram_negative(org):
            return "HIGH_RISK_GRAM_NEGATIVE"
        if is_necrotizing_risk(org):
            return "NECROTIZING_RISK_PATHOGEN"
        return "OTHER"

    def to_danger_group(org: object) -> str:
        # Ordered by clinical actionability for early triage.
        if is_necrotizing_risk(org):
            return "NECROTIZING_RISK_PATHOGEN"
        if is_mrsa(org):
            return "MRSA"
        if is_vre_like(org):
            return "VRE_LIKE"
        if is_high_risk_gram_negative(org):
            return "HIGH_RISK_GRAM_NEGATIVE"
        if is_fungal(org):
            return "FUNGAL_OR_YEAST"
        if is_staph_aureus(org):
            return "STAPH_AUREUS_NON_MRSA"
        return "LOWER_RISK_OTHER"

    def to_high_risk_binary(org: object) -> str:
        # The FIRST target to train/evaluate with small N.
        # Positive class = organisms where early escalation / isolation / stewardship attention is most actionable.
        if (
            is_necrotizing_risk(org)
            or is_mrsa(org)
            or is_vre_like(org)
            or is_high_risk_gram_negative(org)
            or is_fungal(org)
        ):
            return "HIGH_RISK_DANGEROUS"
        return "LOWER_RISK_OTHER"

    def to_high_risk_pathogen_type(org: object) -> str:
        # Stage-2 target. Only meaningful after Stage 1 says HIGH_RISK_DANGEROUS.
        # LOWER_RISK_OTHER is kept so the cascade can be evaluated end-to-end.
        if is_necrotizing_risk(org):
            return "NECROTIZING_RISK_PATHOGEN"
        if is_mrsa(org):
            return "MRSA"
        if is_vre_like(org):
            return "VRE_LIKE"
        if is_high_risk_gram_negative(org):
            return "HIGH_RISK_GRAM_NEGATIVE"
        if is_fungal(org):
            return "FUNGAL_OR_YEAST"
        return "LOWER_RISK_OTHER"

    cohort["label_3class"] = cohort["org_name"].apply(to_3class)
    cohort["label_clinical_group"] = cohort["org_name"].apply(to_clinical_group)
    cohort["label_danger_group"] = cohort["org_name"].apply(to_danger_group)
    cohort["label_high_risk_binary"] = cohort["org_name"].apply(to_high_risk_binary)
    cohort["label_high_risk_pathogen_type"] = cohort["org_name"].apply(to_high_risk_pathogen_type)

    # Preferred small-N targets: separate binary questions instead of one broad high-risk bucket.
    cohort["label_mrsa_binary"] = cohort["org_name"].apply(lambda x: "MRSA" if is_mrsa(x) else "NON_MRSA")
    cohort["label_fungal_binary"] = cohort["org_name"].apply(lambda x: "FUNGAL_YEAST" if is_fungal(x) else "NON_FUNGAL")
    cohort["label_high_risk_gram_negative_binary"] = cohort["org_name"].apply(
        lambda x: "HIGH_RISK_GRAM_NEGATIVE" if is_high_risk_gram_negative(x) else "OTHER"
    )

    # Put the new most-clinically-actionable target first.
    for lbl in [
        "label_mrsa_binary",
        "label_fungal_binary",
        "label_high_risk_gram_negative_binary",
        "label_high_risk_binary",
        "label_high_risk_pathogen_type",
        "label_danger_group",
        "label_3class",
        "label_clinical_group",
        "label_species_top",
        "label_species",
    ]:
        print(f"\n  {lbl}:")
        for cls, n in cohort[lbl].value_counts().items():
            print(f"    {cls:45s} n={n}")
    return cohort

def summarize_timeseries(df: pd.DataFrame, id_cols: List[str], feature_col: str, value_col: str, time_col: str, prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=[feature_col, value_col]).copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(time_col)
    grouped = df.groupby(id_cols + [feature_col])[value_col]
    agg = grouped.agg(["first", "last", "min", "max", "mean", "count"]).reset_index()
    wide = agg.pivot_table(index=id_cols, columns=feature_col, values=["first", "last", "min", "max", "mean", "count"])
    wide.columns = [f"{prefix}{feature}_{stat}" for stat, feature in wide.columns]
    wide = wide.reset_index().set_index(id_cols)

    for c in [c for c in wide.columns if c.endswith("_count")]:
        wide[c.replace("_count", "_has")] = (wide[c].fillna(0) > 0).astype(float)
    return wide


def extract_rapid_labs(cohort: pd.DataFrame, labevents: pd.DataFrame, lab_itemids: Dict[int, str], window_hours: int) -> pd.DataFrame:
    print(f"[4/7] Extracting rapid labs available by T0+{window_hours}h...")
    lab = labevents.copy()
    if "itemid" not in lab.columns:
        print("  [warn] LABEVENTS has no itemid column")
        return pd.DataFrame()
    lab["itemid"] = pd.to_numeric(lab["itemid"], errors="coerce").astype("Int64")
    lab = lab[lab["itemid"].isin(list(lab_itemids.keys()))].copy()
    if lab.empty:
        print("  [warn] No matching rapid lab itemids found")
        return pd.DataFrame()

    lab["feature"] = lab["itemid"].astype(int).map(lab_itemids)
    lab["charttime"] = coalesce_datetime(lab, ["charttime", "chartdate"])
    if "storetime" in lab.columns:
        storetime = pd.to_datetime(lab["storetime"], errors="coerce")
        lab["available_time"] = storetime.fillna(lab["charttime"])
    else:
        lab["available_time"] = lab["charttime"]

    if "valuenum" not in lab.columns:
        print("  [warn] LABEVENTS has no valuenum column")
        return pd.DataFrame()

    id_cols = ["subject_id", "hadm_id"]
    lab = lab.merge(cohort[id_cols + ["t0", "t_cutoff"]], on=id_cols, how="inner")
    before = len(lab)
    lab = lab[(lab["available_time"].notna()) & (lab["available_time"] >= lab["t0"]) & (lab["available_time"] <= lab["t_cutoff"])].copy()
    print(f"  Availability-time guard: {before} -> {len(lab)} rows (dropped {before - len(lab)})")
    lab["valuenum"] = pd.to_numeric(lab["valuenum"], errors="coerce")

    print("  Coverage by rapid lab:")
    base_idx = cohort.set_index(id_cols).index
    for feature in sorted(set(lab_itemids.values())):
        sub = lab[lab["feature"] == feature]
        n = sub.groupby(id_cols).size().reindex(base_idx, fill_value=0).gt(0).sum()
        print(f"    {feature:20s}: {n}/{len(base_idx)} ({100*n/max(len(base_idx),1):.0f}%)")

    return summarize_timeseries(lab, id_cols=id_cols, feature_col="feature", value_col="valuenum", time_col="available_time", prefix="lab_")


def extract_vitals(cohort: pd.DataFrame, chartevents: Optional[pd.DataFrame], window_hours: int) -> pd.DataFrame:
    if chartevents is None:
        print("[5/7] Skipping vitals: CHARTEVENTS not available")
        return pd.DataFrame()
    print(f"[5/7] Extracting rapid vitals by T0+{window_hours}h...")
    ce = chartevents.copy()
    if "itemid" not in ce.columns:
        print("  [warn] CHARTEVENTS has no itemid column")
        return pd.DataFrame()
    chart_map = {}
    chart_map.update(CHART_ITEMIDS_III)
    chart_map.update(CHART_ITEMIDS_IV)
    ce["itemid"] = pd.to_numeric(ce["itemid"], errors="coerce").astype("Int64")
    ce = ce[ce["itemid"].isin(list(chart_map.keys()))].copy()
    if ce.empty:
        print("  [warn] No matching vital itemids found")
        return pd.DataFrame()

    ce["feature"] = ce["itemid"].astype(int).map(chart_map)
    ce["charttime"] = coalesce_datetime(ce, ["charttime", "chartdate"])
    if "valuenum" not in ce.columns:
        print("  [warn] CHARTEVENTS has no valuenum column")
        return pd.DataFrame()

    id_cols = ["subject_id", "hadm_id"]
    ce = ce.merge(cohort[id_cols + ["t0", "t_cutoff"]], on=id_cols, how="inner")
    before = len(ce)
    ce = ce[(ce["charttime"].notna()) & (ce["charttime"] >= ce["t0"]) & (ce["charttime"] <= ce["t_cutoff"])].copy()
    print(f"  Temporal guard: {before} -> {len(ce)} rows (dropped {before - len(ce)})")
    ce["valuenum"] = pd.to_numeric(ce["valuenum"], errors="coerce")

    f_mask = ce["feature"].eq("temperature_f")
    ce.loc[f_mask, "valuenum"] = (ce.loc[f_mask, "valuenum"] - 32.0) * 5.0 / 9.0
    ce.loc[f_mask, "feature"] = "temperature_c"
    ce.loc[ce["feature"].eq("temperature_c") & ~ce["valuenum"].between(25, 45), "valuenum"] = np.nan
    ce.loc[ce["feature"].eq("spo2") & ~ce["valuenum"].between(40, 100), "valuenum"] = np.nan

    print("  Coverage by vital:")
    base_idx = cohort.set_index(id_cols).index
    for feature in sorted(ce["feature"].dropna().unique()):
        sub = ce[ce["feature"] == feature]
        n = sub.groupby(id_cols).size().reindex(base_idx, fill_value=0).gt(0).sum()
        print(f"    {feature:20s}: {n}/{len(base_idx)} ({100*n/max(len(base_idx),1):.0f}%)")

    return summarize_timeseries(ce, id_cols=id_cols, feature_col="feature", value_col="valuenum", time_col="charttime", prefix="vital_")


def build_feature_matrix(cohort: pd.DataFrame, lab_feat: pd.DataFrame, vital_feat: pd.DataFrame) -> pd.DataFrame:
    print("[6/7] Building rapid-lab feature matrix...")
    id_cols = ["subject_id", "hadm_id"]

    cohort = cohort.copy().reset_index(drop=True)
    if "cohort_row_id" not in cohort.columns:
        cohort["cohort_row_id"] = np.arange(len(cohort), dtype=int)

    # IMPORTANT: use a unique row id as the feature index. Do not use only
    # (subject_id, hadm_id), because those keys can repeat in MIMIC and then
    # pandas .loc/reindex can silently expand rows, causing boolean mask errors.
    ci = cohort.set_index("cohort_row_id", drop=True)
    feat = ci[["age"]].copy()

    cohort_keys = pd.MultiIndex.from_frame(ci[id_cols])
    cohort_keys.names = id_cols

    def align_event_features(event_feat: pd.DataFrame, name: str) -> pd.DataFrame:
        if event_feat is None or event_feat.empty:
            return pd.DataFrame(index=ci.index)
        ef = event_feat.copy()
        if not isinstance(ef.index, pd.MultiIndex):
            if all(c in ef.columns for c in id_cols):
                ef = ef.set_index(id_cols)
            else:
                print(f"  [warn] {name} features have no subject_id/hadm_id index; skipping")
                return pd.DataFrame(index=ci.index)
        # One row per (subject_id, hadm_id). If duplicates exist, keep first;
        # lab/vital summaries are already aggregated by these keys.
        if ef.index.has_duplicates:
            before = len(ef)
            ef = ef.groupby(level=list(range(ef.index.nlevels))).first()
            print(f"  [warn] Collapsed duplicate {name} feature keys: {before} -> {len(ef)}")
        aligned = ef.reindex(cohort_keys)
        aligned.index = ci.index
        return aligned

    if lab_feat is not None and not lab_feat.empty:
        feat = feat.join(align_event_features(lab_feat, "lab"), how="left")
    if vital_feat is not None and not vital_feat.empty:
        feat = feat.join(align_event_features(vital_feat, "vital"), how="left")

    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    print(f"  Feature rows aligned to cohort rows: {len(feat)}")
    print(f"  Features: {len(feat.columns)}")
    print("  Missingness / zero diagnostics:")
    diag = pd.DataFrame({
        "non_missing": feat.notna().sum(),
        "missing_pct": feat.isna().mean() * 100,
        "zero_pct_among_non_missing": [((feat[c] == 0).sum() / max(feat[c].notna().sum(), 1)) * 100 for c in feat.columns],
    }).sort_values("missing_pct", ascending=False)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(diag.round(1).to_string())
    return feat

def make_cv(y: np.ndarray, groups: np.ndarray, seed: int = 42, max_splits: int = 5):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        return None, n_splits
    n_unique_groups = len(pd.unique(groups))
    n_splits = min(n_splits, n_unique_groups)
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits
    if n_unique_groups >= n_splits:
        return GroupKFold(n_splits=n_splits), n_splits
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits


def get_models(seed: int = 42) -> Dict[str, Pipeline]:
    return {
        "DummyMostFreq": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "LogRegBalanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.5, random_state=seed)),
        ]),
        "RandomForestBalanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        "ExtraTreesBalanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
    }


def manual_oof_predict(model: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv, n_classes: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, float]]]:
    y_pred = np.full_like(y, fill_value=-1)
    y_proba = np.zeros((len(y), n_classes), dtype=float)
    has_proba = hasattr(model, "predict_proba")
    fold_metrics = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        mdl = clone(model)
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        y_pred[te] = pred
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X[te])
            clf = mdl.named_steps.get("clf", mdl)
            classes_seen = getattr(clf, "classes_", np.arange(p.shape[1]))
            for j, cls in enumerate(classes_seen):
                if int(cls) < n_classes:
                    y_proba[te, int(cls)] = p[:, j]
        else:
            has_proba = False
        fold_metrics.append({
            "fold": fold,
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
        })
    if np.any(y_pred < 0):
        raise RuntimeError("Some samples did not receive out-of-fold predictions")
    return y_pred, (y_proba if has_proba else None), fold_metrics


def align_cohort_to_feat(feat: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Return cohort rows aligned 1:1 to feat rows using cohort_row_id."""
    cohort = cohort.copy().reset_index(drop=True)
    if "cohort_row_id" not in cohort.columns:
        cohort["cohort_row_id"] = np.arange(len(cohort), dtype=int)

    if "cohort_row_id" in cohort.columns and feat.index.is_unique:
        ci_all = cohort.set_index("cohort_row_id", drop=False)
        missing_ids = feat.index.difference(ci_all.index)
        if len(missing_ids) > 0:
            raise ValueError(f"Feature matrix contains cohort_row_id values not found in cohort: {list(missing_ids[:10])}")
        ci = ci_all.loc[feat.index]
    else:
        if len(feat) != len(cohort):
            raise ValueError(f"Cannot align feat and cohort: len(feat)={len(feat)}, len(cohort)={len(cohort)}")
        ci = cohort.copy()
        ci.index = feat.index

    if len(ci) != len(feat):
        raise RuntimeError(f"Feature/label alignment failed: len(feat)={len(feat)}, len(labels)={len(ci)}")
    return ci


def get_positive_label_for_binary_task(label_col: str, classes: np.ndarray) -> Optional[str]:
    """Return the clinically positive class for binary tasks where ranking/AP matters."""
    preferred = {
        "label_high_risk_binary": "HIGH_RISK_DANGEROUS",
        "label_mrsa_binary": "MRSA",
        "label_fungal_binary": "FUNGAL_YEAST",
        "label_high_risk_gram_negative_binary": "HIGH_RISK_GRAM_NEGATIVE",
    }
    pos = preferred.get(label_col)
    if pos in set(map(str, classes)):
        return pos
    return None


def evaluate_task(feat: pd.DataFrame, cohort: pd.DataFrame, label_col: str, out_dir: Path, seed: int = 42, min_class_n: int = 3) -> List[Dict[str, object]]:
    ci = align_cohort_to_feat(feat, cohort)

    y_raw = ci[label_col].astype(str).reset_index(drop=True)
    counts = y_raw.value_counts()
    valid_classes = counts[counts >= min_class_n].index
    mask = y_raw.isin(valid_classes).to_numpy()
    if mask.shape[0] != len(feat):
        raise RuntimeError(f"Internal mask length mismatch: mask={mask.shape[0]}, feat={len(feat)}")
    if mask.sum() < 10 or len(valid_classes) < 2:
        print(f"\n  [{label_col}] Skipping: too few valid samples/classes")
        return []

    X = feat.to_numpy()[mask]
    y_labels = y_raw.to_numpy()[mask]
    ci_e = ci.reset_index(drop=True).loc[mask].reset_index(drop=True)
    groups = ci_e["subject_id"].astype(str).to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_
    n_classes = len(classes)
    cv, n_splits = make_cv(y, groups, seed=seed)
    if cv is None:
        print(f"\n  [{label_col}] Skipping: smallest class has only {n_splits} sample")
        return []
    baseline = pd.Series(y).value_counts().max() / len(y)

    print("\n" + "=" * 78)
    print(f"Task: {label_col} | N={len(y)} | classes={n_classes} | folds={n_splits} | majority baseline={baseline:.3f}")
    print("=" * 78)
    print("Class counts:")
    for cls, n in pd.Series(y_labels).value_counts().items():
        print(f"  {cls:45s} n={n}")

    rows = []
    for name, model in get_models(seed=seed).items():
        y_pred, y_proba, fold_metrics = manual_oof_predict(model, X, y, groups, cv, n_classes)
        metrics = {
            "task": label_col,
            "model": name,
            "n": len(y),
            "n_classes": n_classes,
            "folds": n_splits,
            "majority_baseline": baseline,
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "macro_auc_ovr": np.nan,
            "average_precision": np.nan,
            "top3_accuracy": np.nan,
        }
        if y_proba is not None and n_classes > 2:
            try:
                metrics["top3_accuracy"] = top_k_accuracy_score(y, y_proba, k=min(3, n_classes), labels=np.arange(n_classes))
            except Exception:
                pass
            try:
                y_bin = label_binarize(y, classes=np.arange(n_classes))
                metrics["macro_auc_ovr"] = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            except Exception:
                pass
        elif y_proba is not None and n_classes == 2:
            try:
                metrics["macro_auc_ovr"] = roc_auc_score(y, y_proba[:, 1])
            except Exception:
                pass
            try:
                pos_label = get_positive_label_for_binary_task(label_col, classes)
                pos_idx = list(classes).index(pos_label) if pos_label is not None else 1
                metrics["average_precision"] = average_precision_score((y == pos_idx).astype(int), y_proba[:, pos_idx])
            except Exception:
                pass

        rows.append(metrics)
        print(
            f"\n{name}: acc={metrics['accuracy']:.3f} | macroF1={metrics['macro_f1']:.3f} | "
            f"balAcc={metrics['balanced_accuracy']:.3f} | top3={metrics['top3_accuracy'] if not np.isnan(metrics['top3_accuracy']) else 'NA'} | "
            f"AUC={metrics['macro_auc_ovr'] if not np.isnan(metrics['macro_auc_ovr']) else 'NA'} | "
            f"AP={metrics['average_precision'] if not np.isnan(metrics['average_precision']) else 'NA'}"
        )
        print(classification_report(y, y_pred, target_names=classes, zero_division=0))

        pred_df = pd.DataFrame({
            "cohort_row_id": ci_e["cohort_row_id"].to_numpy() if "cohort_row_id" in ci_e.columns else np.arange(len(ci_e)),
            "subject_id": ci_e["subject_id"].to_numpy(),
            "hadm_id": ci_e["hadm_id"].to_numpy(),
            "label": y_labels,
            "y_true_idx": y,
            "y_pred_idx": y_pred,
            "y_pred_label": le.inverse_transform(y_pred),
        })
        if y_proba is not None:
            for i, cls in enumerate(classes):
                safe_cls = re.sub(r"[^A-Za-z0-9_]+", "_", str(cls))[:40]
                pred_df[f"prob_{i}_{safe_cls}"] = y_proba[:, i]
        pred_path = out_dir / f"oof_predictions_{label_col}_{name}.csv"
        pred_df.to_csv(pred_path, index=False)

        pos_label = get_positive_label_for_binary_task(label_col, classes)
        if n_classes == 2 and y_proba is not None and pos_label is not None:
            pos_idx = list(classes).index(pos_label)
            safe_pos = re.sub(r"[^A-Za-z0-9_]+", "_", str(pos_label))[:50]
            rank_df = pred_df.copy()
            rank_col = f"risk_score_{safe_pos}"
            rank_df[rank_col] = y_proba[:, pos_idx]
            rank_df = rank_df.sort_values(rank_col, ascending=False)
            rank_path = out_dir / f"priority_rankings_{label_col}_{name}.csv"
            rank_df.to_csv(rank_path, index=False)
    return rows


def evaluate_stage2_true_high_risk(
    feat: pd.DataFrame,
    cohort: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    min_class_n: int = 3,
) -> List[Dict[str, object]]:
    """Evaluate pathogen-type classification only inside TRUE high-risk cases.

    This answers: if the first step already identified the patient as high risk,
    can rapid labs/vitals separate MRSA vs high-risk Gram-negative vs fungal, etc.?
    This is an oracle-stage-1 analysis, not a deployment cascade.
    """
    ci = align_cohort_to_feat(feat, cohort)
    high = ci["label_high_risk_binary"].astype(str).eq("HIGH_RISK_DANGEROUS").to_numpy()
    y_raw_all = ci["label_high_risk_pathogen_type"].astype(str).reset_index(drop=True)
    not_lower = ~y_raw_all.eq("LOWER_RISK_OTHER").to_numpy()
    stage2_mask = high & not_lower

    counts = y_raw_all[stage2_mask].value_counts()
    valid_classes = counts[counts >= min_class_n].index
    mask = stage2_mask & y_raw_all.isin(valid_classes).to_numpy()

    if mask.sum() < 10 or len(valid_classes) < 2:
        print("\n  [stage2_true_high_risk_pathogen_type] Skipping: too few high-risk samples/classes")
        print(f"  High-risk subtype counts before filtering: {counts.to_dict()}")
        return []

    X = feat.to_numpy()[mask]
    y_labels = y_raw_all.to_numpy()[mask]
    ci_e = ci.reset_index(drop=True).loc[mask].reset_index(drop=True)
    groups = ci_e["subject_id"].astype(str).to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_
    n_classes = len(classes)
    cv, n_splits = make_cv(y, groups, seed=seed)
    if cv is None:
        print(f"\n  [stage2_true_high_risk_pathogen_type] Skipping: smallest class has only {n_splits} sample")
        return []

    baseline = pd.Series(y).value_counts().max() / len(y)
    print("\n" + "=" * 78)
    print(f"Task: stage2_true_high_risk_pathogen_type | N={len(y)} | classes={n_classes} | folds={n_splits} | majority baseline={baseline:.3f}")
    print("=" * 78)
    print("Class counts within TRUE HIGH_RISK_DANGEROUS:")
    for cls, n in pd.Series(y_labels).value_counts().items():
        print(f"  {cls:45s} n={n}")

    rows: List[Dict[str, object]] = []
    for name, model in get_models(seed=seed).items():
        if name == "DummyMostFreq":
            # Keep dummy for baseline, but stage2 model ranking is mainly from real models.
            pass
        y_pred, y_proba, _ = manual_oof_predict(model, X, y, groups, cv, n_classes)
        metrics = {
            "task": "stage2_true_high_risk_pathogen_type",
            "model": name,
            "n": len(y),
            "n_classes": n_classes,
            "folds": n_splits,
            "majority_baseline": baseline,
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "macro_auc_ovr": np.nan,
            "average_precision": np.nan,
            "top3_accuracy": np.nan,
        }
        if y_proba is not None and n_classes > 2:
            try:
                metrics["top3_accuracy"] = top_k_accuracy_score(y, y_proba, k=min(3, n_classes), labels=np.arange(n_classes))
            except Exception:
                pass
            try:
                y_bin = label_binarize(y, classes=np.arange(n_classes))
                metrics["macro_auc_ovr"] = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            except Exception:
                pass
        rows.append(metrics)
        print(
            f"\n{name}: acc={metrics['accuracy']:.3f} | macroF1={metrics['macro_f1']:.3f} | "
            f"balAcc={metrics['balanced_accuracy']:.3f} | top3={metrics['top3_accuracy'] if not np.isnan(metrics['top3_accuracy']) else 'NA'} | "
            f"AUC={metrics['macro_auc_ovr'] if not np.isnan(metrics['macro_auc_ovr']) else 'NA'}"
        )
        print(classification_report(y, y_pred, target_names=classes, zero_division=0))

        pred_df = pd.DataFrame({
            "cohort_row_id": ci_e["cohort_row_id"].to_numpy() if "cohort_row_id" in ci_e.columns else np.arange(len(ci_e)),
            "subject_id": ci_e["subject_id"].to_numpy(),
            "hadm_id": ci_e["hadm_id"].to_numpy(),
            "label_high_risk_pathogen_type": y_labels,
            "y_true_idx": y,
            "y_pred_idx": y_pred,
            "y_pred_label": le.inverse_transform(y_pred),
        })
        if y_proba is not None:
            for i, cls in enumerate(classes):
                safe_cls = re.sub(r"[^A-Za-z0-9_]+", "_", str(cls))[:40]
                pred_df[f"prob_{i}_{safe_cls}"] = y_proba[:, i]
        pred_df.to_csv(out_dir / f"oof_predictions_stage2_true_high_risk_pathogen_type_{name}.csv", index=False)
    return rows


def evaluate_two_stage_cascade(
    feat: pd.DataFrame,
    cohort: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    min_class_n: int = 3,
    high_risk_threshold: float = 0.5,
    stage1_model_name: str = "LogRegBalanced",
) -> List[Dict[str, object]]:
    """End-to-end cascade: Stage 1 high-risk detection, then Stage 2 subtype.

    In each CV fold:
      1) Train binary high-risk model on train fold.
      2) Train subtype model only on TRUE high-risk train samples.
      3) On test fold, run Stage 2 only for cases predicted high-risk by Stage 1.

    Output is a deployment-like prediction: LOWER_RISK_OTHER, MRSA,
    HIGH_RISK_GRAM_NEGATIVE, FUNGAL_OR_YEAST, etc.
    """
    ci = align_cohort_to_feat(feat, cohort).reset_index(drop=True)
    X = feat.to_numpy()
    y1_raw = ci["label_high_risk_binary"].astype(str).to_numpy()
    y2_raw_full = ci["label_high_risk_pathogen_type"].astype(str).to_numpy()
    groups = ci["subject_id"].astype(str).to_numpy()

    # Remove rare subtype classes from the cascade by mapping them to HIGH_RISK_OTHER_TYPE.
    subtype_counts = pd.Series(y2_raw_full[y2_raw_full != "LOWER_RISK_OTHER"]).value_counts()
    rare_subtypes = set(subtype_counts[subtype_counts < min_class_n].index)
    y_cascade_raw = np.array([
        "HIGH_RISK_OTHER_TYPE" if y in rare_subtypes else y for y in y2_raw_full
    ], dtype=object)

    le1 = LabelEncoder()
    y1 = le1.fit_transform(y1_raw)
    if len(le1.classes_) < 2 or "HIGH_RISK_DANGEROUS" not in list(le1.classes_):
        print("\n  [two_stage_cascade] Skipping: binary high-risk target lacks both classes")
        return []
    pos_idx = list(le1.classes_).index("HIGH_RISK_DANGEROUS")

    cv, n_splits = make_cv(y1, groups, seed=seed)
    if cv is None:
        print("\n  [two_stage_cascade] Skipping: insufficient folds for Stage 1")
        return []

    models = get_models(seed=seed)
    if stage1_model_name not in models:
        raise ValueError(f"Unknown stage1_model_name={stage1_model_name}; choose from {list(models)}")
    stage1_template = models[stage1_model_name]
    stage2_model_names = ["LogRegBalanced", "RandomForestBalanced", "ExtraTreesBalanced"]

    rows: List[Dict[str, object]] = []
    print("\n" + "=" * 78)
    print(f"Task: two_stage_cascade_high_risk_then_type | Stage1={stage1_model_name} | threshold={high_risk_threshold:.2f} | folds={n_splits}")
    print("=" * 78)
    print("Cascade true-label counts:")
    for cls, n in pd.Series(y_cascade_raw).value_counts().items():
        print(f"  {cls:45s} n={n}")

    for stage2_name in stage2_model_names:
        stage2_template = models[stage2_name]
        y1_pred_all = np.empty(len(y1_raw), dtype=object)
        y1_score_all = np.full(len(y1_raw), np.nan, dtype=float)
        cascade_pred = np.array(["UNASSIGNED"] * len(y1_raw), dtype=object)
        stage2_pred = np.array([""] * len(y1_raw), dtype=object)
        stage2_conf = np.full(len(y1_raw), np.nan, dtype=float)

        for fold, (tr, te) in enumerate(cv.split(X, y1, groups), start=1):
            s1 = clone(stage1_template)
            s1.fit(X[tr], y1[tr])
            if hasattr(s1, "predict_proba"):
                p1 = s1.predict_proba(X[te])
                clf1 = s1.named_steps.get("clf", s1)
                classes_seen = list(getattr(clf1, "classes_", np.arange(p1.shape[1])))
                if pos_idx in classes_seen:
                    seen_pos_j = classes_seen.index(pos_idx)
                    score = p1[:, seen_pos_j]
                else:
                    score = np.zeros(len(te), dtype=float)
            else:
                pred1_num = s1.predict(X[te])
                score = (pred1_num == pos_idx).astype(float)
            pred_high = score >= high_risk_threshold
            y1_score_all[te] = score
            y1_pred_all[te] = np.where(pred_high, "HIGH_RISK_DANGEROUS", "LOWER_RISK_OTHER")

            # Stage 2 is trained only on TRUE high-risk cases from the train fold.
            train_high = y1_raw[tr] == "HIGH_RISK_DANGEROUS"
            tr2 = tr[train_high]
            y2_train_labels = y_cascade_raw[tr2]
            y2_train_labels = y2_train_labels[y2_train_labels != "LOWER_RISK_OTHER"]
            tr2 = tr2[y_cascade_raw[tr2] != "LOWER_RISK_OTHER"]

            if len(pd.unique(y2_train_labels)) < 2 or len(tr2) < 5:
                fallback = pd.Series(y2_train_labels).mode().iloc[0] if len(y2_train_labels) else "HIGH_RISK_OTHER_TYPE"
                for idx, is_high in zip(te, pred_high):
                    if is_high:
                        cascade_pred[idx] = fallback
                        stage2_pred[idx] = fallback
                        stage2_conf[idx] = np.nan
                    else:
                        cascade_pred[idx] = "LOWER_RISK_OTHER"
                continue

            le2 = LabelEncoder()
            y2_train = le2.fit_transform(y2_train_labels)
            s2 = clone(stage2_template)
            s2.fit(X[tr2], y2_train)

            high_te = te[pred_high]
            low_te = te[~pred_high]
            cascade_pred[low_te] = "LOWER_RISK_OTHER"
            if len(high_te):
                p2_pred = s2.predict(X[high_te])
                labels_pred = le2.inverse_transform(p2_pred)
                stage2_pred[high_te] = labels_pred
                cascade_pred[high_te] = labels_pred
                if hasattr(s2, "predict_proba"):
                    p2 = s2.predict_proba(X[high_te])
                    stage2_conf[high_te] = p2.max(axis=1)

        # Metrics
        y_true_binary_high = y1_raw == "HIGH_RISK_DANGEROUS"
        y_pred_binary_high = y1_pred_all == "HIGH_RISK_DANGEROUS"
        stage1_recall = recall_score(y_true_binary_high.astype(int), y_pred_binary_high.astype(int), zero_division=0)
        stage1_precision = precision_score(y_true_binary_high.astype(int), y_pred_binary_high.astype(int), zero_division=0)
        lower_true = ~y_true_binary_high
        stage1_specificity = ((~y_pred_binary_high) & lower_true).sum() / max(lower_true.sum(), 1)

        cascade_classes = sorted(pd.unique(np.concatenate([y_cascade_raw, cascade_pred])))
        cascade_acc = accuracy_score(y_cascade_raw, cascade_pred)
        cascade_macro_f1 = f1_score(y_cascade_raw, cascade_pred, labels=cascade_classes, average="macro", zero_division=0)
        cascade_bal_acc = balanced_accuracy_score(y_cascade_raw, cascade_pred)

        flagged_true_high = y_true_binary_high & y_pred_binary_high
        if flagged_true_high.sum() > 0:
            subtype_acc_flagged = accuracy_score(y_cascade_raw[flagged_true_high], cascade_pred[flagged_true_high])
            subtype_macro_f1_flagged = f1_score(y_cascade_raw[flagged_true_high], cascade_pred[flagged_true_high], average="macro", zero_division=0)
        else:
            subtype_acc_flagged = np.nan
            subtype_macro_f1_flagged = np.nan

        metrics = {
            "task": "two_stage_cascade_high_risk_then_type",
            "model": f"{stage1_model_name}->{stage2_name}",
            "n": len(y1_raw),
            "n_classes": len(cascade_classes),
            "folds": n_splits,
            "majority_baseline": pd.Series(y_cascade_raw).value_counts().max() / len(y_cascade_raw),
            "accuracy": cascade_acc,
            "balanced_accuracy": cascade_bal_acc,
            "macro_f1": cascade_macro_f1,
            "weighted_f1": f1_score(y_cascade_raw, cascade_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_cascade_raw, cascade_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_cascade_raw, cascade_pred, average="macro", zero_division=0),
            "macro_auc_ovr": np.nan,
            "average_precision": np.nan,
            "top3_accuracy": np.nan,
            "stage1_high_risk_recall": stage1_recall,
            "stage1_high_risk_precision": stage1_precision,
            "stage1_lower_risk_specificity": stage1_specificity,
            "stage2_subtype_accuracy_on_flagged_true_high_risk": subtype_acc_flagged,
            "stage2_subtype_macro_f1_on_flagged_true_high_risk": subtype_macro_f1_flagged,
            "true_high_risk_flagged_n": int(flagged_true_high.sum()),
            "true_high_risk_total_n": int(y_true_binary_high.sum()),
        }
        rows.append(metrics)

        print(
            f"\nCascade {stage1_model_name}->{stage2_name}: "
            f"stage1_recall={stage1_recall:.3f} | stage1_precision={stage1_precision:.3f} | "
            f"cascade_acc={cascade_acc:.3f} | cascade_macroF1={cascade_macro_f1:.3f} | "
            f"stage2_acc_on_flagged_true_high={subtype_acc_flagged if not np.isnan(subtype_acc_flagged) else 'NA'}"
        )
        print(classification_report(y_cascade_raw, cascade_pred, labels=cascade_classes, zero_division=0))

        pred_df = pd.DataFrame({
            "cohort_row_id": ci["cohort_row_id"].to_numpy() if "cohort_row_id" in ci.columns else np.arange(len(ci)),
            "subject_id": ci["subject_id"].to_numpy(),
            "hadm_id": ci["hadm_id"].to_numpy(),
            "org_name": ci["org_name"].astype(str).to_numpy(),
            "true_high_risk_binary": y1_raw,
            "stage1_high_risk_score": y1_score_all,
            "stage1_pred_binary": y1_pred_all,
            "true_high_risk_pathogen_type": y_cascade_raw,
            "stage2_pred_pathogen_type_if_flagged": stage2_pred,
            "stage2_confidence_if_flagged": stage2_conf,
            "cascade_pred": cascade_pred,
        }).sort_values("stage1_high_risk_score", ascending=False)
        pred_df.to_csv(out_dir / f"two_stage_cascade_predictions_{stage1_model_name}_to_{stage2_name}.csv", index=False)
    return rows

def evaluate_all(cohort: pd.DataFrame, feat: pd.DataFrame, out_dir: Path, seed: int = 42, min_class_n: int = 3, high_risk_threshold: float = 0.5) -> pd.DataFrame:
    print("[7/7] Evaluating rapid-lab-only models...")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    # Primary small-N questions: clinically distinct binary targets.
    for label_col in [
        "label_mrsa_binary",
        "label_fungal_binary",
        "label_high_risk_gram_negative_binary",
    ]:
        all_rows.extend(evaluate_task(feat, cohort, label_col, out_dir, seed=seed, min_class_n=min_class_n))

    # Optional broad target retained for comparison; prior runs suggested it is too heterogeneous.
    all_rows.extend(evaluate_task(feat, cohort, "label_high_risk_binary", out_dir, seed=seed, min_class_n=min_class_n))

    # Stage 2/cascade retained as exploratory secondary analysis.
    all_rows.extend(evaluate_stage2_true_high_risk(feat, cohort, out_dir, seed=seed, min_class_n=min_class_n))
    all_rows.extend(evaluate_two_stage_cascade(feat, cohort, out_dir, seed=seed, min_class_n=min_class_n, high_risk_threshold=high_risk_threshold))

    # Secondary/exploratory labels.
    for label_col in [
        "label_high_risk_pathogen_type",
        "label_danger_group",
        "label_3class",
        "label_clinical_group",
        "label_species_top",
        "label_species",
    ]:
        all_rows.extend(evaluate_task(feat, cohort, label_col, out_dir, seed=seed, min_class_n=min_class_n))
    summary = pd.DataFrame(all_rows)
    if not summary.empty:
        summary_path = out_dir / "rapid_lab_results_summary.csv"
        summary.to_csv(summary_path, index=False)
        print("\n" + "=" * 78)
        print("Summary")
        print("=" * 78)
        with pd.option_context("display.max_columns", 100, "display.width", 180):
            print(summary.sort_values(["task", "macro_f1"], ascending=[True, False]).round(3).to_string(index=False))
        print(f"\nSaved summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="PenuX rapid-lab leakage-free evaluation")
    parser.add_argument("--dir", default=os.environ.get("DIR"), help="Path to MIMIC dataset directory")
    parser.add_argument("--window_hours", type=int, default=6, help="Prediction-time window after ICU intime")
    parser.add_argument("--min_class_n", type=int, default=3, help="Minimum samples per class")
    parser.add_argument("--top_species_n", type=int, default=5, help="Top-N species to keep for label_species_top")
    parser.add_argument("--out_dir", default="penux_rapid_lab_outputs", help="Directory for CSV outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high_risk_threshold", type=float, default=0.5,
                        help="Stage-1 threshold for running Stage-2 subtype model; lower values improve recall")
    args = parser.parse_args()

    mimic_dir = pick_mimic_dir(args.dir)
    out_dir = Path(args.out_dir)

    print("=" * 78)
    print("PenuX - Rapid-Lab Separate Binary Targets")
    print("=" * 78)
    print(f"MIMIC dir      : {mimic_dir}")
    print(f"Prediction time: ICU intime + {args.window_hours}h")
    print("Inputs         : rapid labs + vitals + age only")
    print("Excluded       : microbiology interpretation, ab_name, dilution, susceptibility")
    print("CV guard       : grouped by subject_id")
    print("=" * 78)

    patients, admissions, icustays, labevents, chartevents, micro, d_labitems = load_all(mimic_dir)

    lab_itemids = infer_fast_lab_itemids(d_labitems)
    print(f"\nRapid lab itemids mapped: {len(lab_itemids)}")
    by_feature = pd.Series(lab_itemids).value_counts()
    for feature, n in by_feature.items():
        print(f"  {feature:20s}: {n} itemid(s)")

    cohort = build_cohort(patients, admissions, icustays, micro, args.window_hours)
    cohort = build_labels(cohort, min_class_n=args.min_class_n, top_species_n=args.top_species_n)
    lab_feat = extract_rapid_labs(cohort, labevents, lab_itemids, args.window_hours)
    vital_feat = extract_vitals(cohort, chartevents, args.window_hours)
    feat = build_feature_matrix(cohort, lab_feat, vital_feat)

    non_age_cols = [c for c in feat.columns if c != "age"]
    if non_age_cols:
        usable_rows = feat[non_age_cols].notna().any(axis=1).sum()
        print(f"\nRows with at least one non-age rapid feature: {usable_rows}/{len(feat)}")
        if usable_rows < max(10, 0.20 * len(feat)):
            print("[warn] Very low rapid-lab coverage. Results will be unstable.")

    evaluate_all(cohort, feat, out_dir, seed=args.seed, min_class_n=args.min_class_n, high_risk_threshold=args.high_risk_threshold)

    print("\n" + "=" * 78)
    print("DONE")
    print("Use rapid_lab_results_summary.csv and the OOF prediction CSVs for the manuscript.")
    print("For small N, prioritize: 1) label_high_risk_binary, 2) stage2_true_high_risk_pathogen_type, 3) two_stage_cascade. Treat species-level as exploratory.")
    print("=" * 78)


if __name__ == "__main__":
    main()
