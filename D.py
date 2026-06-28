#!/usr/bin/env python3
"""
penux_all_in_one_schema_eval.py

ONE FILE for PenuX demo workflows:

Modes
-----
1) mimic-eval
   Run leakage-aware MIMIC-III/MIMIC-IV evaluation directly on MIMIC CSV/CSV.GZ.

2) mimic-to-eicu
   Convert MIMIC-III Demo and/or MIMIC-IV Demo into an eICU-like schema:
     patient.csv
     lab.csv
     vitalPeriodic.csv
     medication.csv
     microLab.csv

3) eicu-eval
   Run the PenuX/eICU adapter on real eICU Demo OR MIMIC converted to eICU-like schema.

4) all
   Convert MIMIC Demo -> eICU-like schema, then run eICU-style evaluation.

No leakage rule
---------------
- microbiology organism/result text is used only for labels.
- microbiology interpretation/sensitivity fields are never used as model inputs.
- specimen/site fields are optional and should be treated cautiously.

Examples
--------
# Direct MIMIC evaluation
python3 penux_all_in_one_schema_eval.py mimic-eval \
  --dir dataset/mimic/mimic-iii-clinical-database-demo-1.4

# Convert both MIMIC demos to eICU-like schema
python3 penux_all_in_one_schema_eval.py mimic-to-eicu \
  --mimiciii-dir dataset/mimic/mimic-iii-clinical-database-demo-1.4 \
  --mimiciv-dir dataset/mimic/mimic-iv-demo-2.2 \
  --out dataset/eicu_like_from_mimic_demo

# Evaluate converted schema
python3 penux_all_in_one_schema_eval.py eicu-eval \
  --dir dataset/eicu_like_from_mimic_demo \
  --output-csv data/processed/eicu_like_from_mimic_features.csv.gz

# All in one: convert then evaluate
python3 penux_all_in_one_schema_eval.py all \
  --mimiciii-dir dataset/mimic/mimic-iii-clinical-database-demo-1.4 \
  --mimiciv-dir dataset/mimic/mimic-iv-demo-2.2 \
  --out dataset/eicu_like_from_mimic_demo \
  --output-csv data/processed/eicu_like_from_mimic_features.csv.gz

Research only. Not clinical decision support.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_STRATIFIED_GROUP_KFOLD = True
except Exception:
    StratifiedGroupKFold = None
    HAVE_STRATIFIED_GROUP_KFOLD = False

try:
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:
    raise RuntimeError("scikit-learn OneHotEncoder is required") from exc

warnings.filterwarnings("ignore")


# =============================================================================
# Global helpers
# =============================================================================

def banner(msg: str) -> None:
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def parse_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().upper())


def first_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False, table: str = "table") -> Optional[str]:
    cmap = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in cmap:
            return cmap[key]
    if required:
        raise KeyError(f"Missing one of {list(candidates)} in {table}. Columns: {list(df.columns)[:60]}")
    return None


def find_file(root: Path, names: Iterable[str]) -> Optional[Path]:
    suffixes = [".csv", ".csv.gz"]
    roots = [
        root,
        root / "extracted_csv",
        root / "hosp",
        root / "icu",
        root / "core",
        root / "csv",
        root / "data",
    ]
    for r in roots:
        if not r.exists():
            continue
        for name in names:
            variants = {name, name.lower(), name.upper(), name.capitalize()}
            for v in variants:
                for s in suffixes:
                    p = r / f"{v}{s}"
                    if p.exists():
                        return p

    target_lowers = set()
    for name in names:
        target_lowers.add(f"{name.lower()}.csv")
        target_lowers.add(f"{name.lower()}.csv.gz")
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in target_lowers:
            return p
    return None


def read_csv_any(path: Path, usecols=None) -> pd.DataFrame:
    print(f"  Loading {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def minute_offset(event_time, intime) -> pd.Series:
    return ((to_dt(event_time) - to_dt(intime)).dt.total_seconds() / 60.0).round().astype("Int64")


def clean_age_years(age) -> pd.Series:
    a = to_num(age)
    return a.clip(lower=0, upper=90)


def safe_num(s):
    if s is None:
        return np.nan
    return pd.to_numeric(s, errors="coerce")


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def normalize_join_id_value(x):
    """Normalize ID values for safe merges across CSVs.

    CSV readers often infer the same key differently across files:
      12345        -> int64
      12345.0      -> float64
      "12345"      -> object/string

    Pandas refuses merges on object vs float64, so all join IDs are normalized
    to a stable string representation with trailing .0 removed when appropriate.
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s:
        return pd.NA
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    if s.endswith(".0") and re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]
    return s


def normalize_join_id_series(s):
    return s.map(normalize_join_id_value).astype("string")


def safe_merge_patientunitstayid(left, right, how="left"):
    """Merge on patientunitstayid after normalizing both sides."""
    left = left.copy()
    right = right.copy()
    if "patientunitstayid" in left.columns:
        left["patientunitstayid"] = normalize_join_id_series(left["patientunitstayid"])
    if "patientunitstayid" in right.columns:
        right["patientunitstayid"] = normalize_join_id_series(right["patientunitstayid"])
    return left.merge(right, on="patientunitstayid", how=how)



# =============================================================================
# Shared microbiology/labels
# =============================================================================

NO_GROWTH_PATTERNS = [
    "NO GROWTH", "NO ORGANISM", "NEGATIVE", "NONE", "CANCELLED", "CANCELED",
    "CONTAMINANT", "NORMAL FLORA", "MIXED FLORA", "NOT DETECTED", "N/A", "NA",
]

FUNGAL_PATTERNS = [
    "YEAST", "CANDIDA", "FUNG", "ASPERGILL", "CRYPTOCOCC", "MOLD", "MOULD",
    "HISTOPLAS", "BLASTOMYC", "COCCIDIO", "PNEUMOCYST",
]

VIRAL_PATTERNS = [
    "VIRUS", "VIRAL", "INFLUENZA", "RSV", "ADENOVIR", "CMV", "HSV", "VZV",
    "ENTEROVIR", "PARAINFLUENZA", "CORONAVIR", "SARS", "COVID",
]


def is_positive_organism(org_name: object) -> bool:
    s = clean_text(org_name)
    if not s:
        return False
    for pat in NO_GROWTH_PATTERNS:
        if pat in s:
            return False
    return True


def pathogen_class(org_name: object) -> str:
    s = clean_text(org_name)
    if any(p in s for p in FUNGAL_PATTERNS):
        return "FUNGAL"
    if any(p in s for p in VIRAL_PATTERNS):
        return "VIRAL"
    return "BACTERIAL"


def normalize_species(org_name: object) -> str:
    s = clean_text(org_name)
    if not s:
        return "UNKNOWN"
    replacements = {
        "METHICILLIN RESISTANT STAPH AUREUS": "MRSA",
        "POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS": "MRSA",
        "STAPHYLOCOCCUS AUREUS": "STAPH_AUREUS",
        "STAPH AUREUS": "STAPH_AUREUS",
        "ESCHERICHIA COLI": "ESCHERICHIA_COLI",
        "E. COLI": "ESCHERICHIA_COLI",
        "KLEBSIELLA": "KLEBSIELLA",
        "PSEUDOMONAS": "PSEUDOMONAS",
        "PROTEUS MIRABILIS": "PROTEUS_MIRABILIS",
        "ENTEROCOCCUS": "ENTEROCOCCUS",
        "STREP": "STREPTOCOCCUS",
        "STAPHYLOCOCCUS, COAGULASE NEGATIVE": "COAG_NEG_STAPH",
        "STAPHYLOCOCCUS EPIDERMIDIS": "COAG_NEG_STAPH",
        "CORYNEBACTERIUM": "CORYNEBACTERIUM",
        "GRAM NEGATIVE ROD": "GRAM_NEGATIVE_ROD",
        "GRAM POSITIVE": "GRAM_POSITIVE",
        "YEAST": "YEAST",
        "CANDIDA": "YEAST",
    }
    for key, val in replacements.items():
        if key in s:
            return val
    return s[:80]


def normalize_specimen_family(spec_type_desc: object) -> str:
    s = clean_text(spec_type_desc)
    if not s:
        return "UNKNOWN"
    if "BLOOD" in s:
        return "BLOOD"
    if "URINE" in s:
        return "URINE"
    if any(k in s for k in ["SPUTUM", "BRONCH", "RESP", "TRACHEAL", "PLEURAL", "BAL"]):
        return "RESPIRATORY"
    if any(k in s for k in ["WOUND", "ABSCESS", "TISSUE", "SKIN", "SOFT"]):
        return "SKIN_SOFT_TISSUE"
    if any(k in s for k in ["CSF", "CEREBROSPINAL"]):
        return "CSF"
    if any(k in s for k in ["STOOL", "RECTAL", "FECES", "FAECES"]):
        return "GI"
    if any(k in s for k in ["CATHETER", "LINE", "TIP"]):
        return "CATHETER"
    if "FLUID" in s:
        return "BODY_FLUID"
    return "OTHER_SPECIMEN"


# =============================================================================
# MIMIC direct evaluation item maps
# =============================================================================

LAB_ITEM_MAP: Dict[int, str] = {
    50813: "lactate",
    51300: "wbc",
    51301: "wbc",
    50912: "creatinine",
    51006: "bun",
    50983: "sodium",
    50824: "sodium",
    50971: "potassium",
    50822: "potassium",
    50902: "chloride",
    50806: "chloride",
    50882: "bicarbonate",
    50803: "bicarbonate",
    50885: "bilirubin_total",
    51222: "hemoglobin",
    50811: "hemoglobin",
    51221: "hematocrit",
    50810: "hematocrit",
    51265: "platelets",
    51237: "inr",
    51274: "pt",
    51275: "ptt",
    50931: "glucose",
    50809: "glucose",
    50820: "ph",
    50821: "po2",
    50818: "pco2",
}


@dataclass(frozen=True)
class ChartSpec:
    name: str
    transform: str = "identity"


CHART_ITEM_MAP: Dict[int, ChartSpec] = {
    646: ChartSpec("spo2"),
    220277: ChartSpec("spo2"),
    676: ChartSpec("temperature_c"),
    223762: ChartSpec("temperature_c"),
    678: ChartSpec("temperature_c", "f_to_c"),
    223761: ChartSpec("temperature_c", "f_to_c"),
    211: ChartSpec("heart_rate"),
    220045: ChartSpec("heart_rate"),
    618: ChartSpec("resp_rate"),
    615: ChartSpec("resp_rate"),
    220210: ChartSpec("resp_rate"),
    224690: ChartSpec("resp_rate"),
    52: ChartSpec("mean_bp"),
    456: ChartSpec("mean_bp"),
    6702: ChartSpec("mean_bp"),
    443: ChartSpec("mean_bp"),
    220181: ChartSpec("mean_bp"),
    220052: ChartSpec("mean_bp"),
    51: ChartSpec("sys_bp"),
    442: ChartSpec("sys_bp"),
    455: ChartSpec("sys_bp"),
    6701: ChartSpec("sys_bp"),
    220179: ChartSpec("sys_bp"),
    220050: ChartSpec("sys_bp"),
    8368: ChartSpec("dias_bp"),
    8440: ChartSpec("dias_bp"),
    8441: ChartSpec("dias_bp"),
    8555: ChartSpec("dias_bp"),
    220180: ChartSpec("dias_bp"),
    220051: ChartSpec("dias_bp"),
    807: ChartSpec("glucose"),
    811: ChartSpec("glucose"),
    1529: ChartSpec("glucose"),
    3745: ChartSpec("glucose"),
    3744: ChartSpec("glucose"),
    225664: ChartSpec("glucose"),
    220621: ChartSpec("glucose"),
    226537: ChartSpec("glucose"),
}

MIMIC_CHARTITEMS_EICU = {
    211: "heartrate", 220045: "heartrate",
    618: "respiration", 615: "respiration", 220210: "respiration", 224690: "respiration",
    646: "sao2", 220277: "sao2",
    678: "temperature_f", 223761: "temperature_f",
    676: "temperature", 223762: "temperature",
    51: "systemicsystolic", 442: "systemicsystolic", 455: "systemicsystolic", 6701: "systemicsystolic",
    220179: "systemicsystolic", 220050: "systemicsystolic",
    8368: "systemicdiastolic", 8440: "systemicdiastolic", 8441: "systemicdiastolic", 8555: "systemicdiastolic",
    220180: "systemicdiastolic", 220051: "systemicdiastolic",
    456: "systemicmean", 52: "systemicmean", 6702: "systemicmean", 443: "systemicmean",
    220052: "systemicmean", 220181: "systemicmean",
}

LAB_NAME_NORMALIZE = {
    "wbc": "WBC",
    "white blood cells": "WBC",
    "lactate": "lactate",
    "creatinine": "creatinine",
    "urea nitrogen": "BUN",
    "bun": "BUN",
    "sodium": "sodium",
    "potassium": "potassium",
    "chloride": "chloride",
    "bicarbonate": "bicarbonate",
    "hemoglobin": "hemoglobin",
    "platelet count": "platelets",
    "platelets": "platelets",
    "bilirubin, total": "bilirubin_total",
    "bilirubin total": "bilirubin_total",
    "c-reactive protein": "CRP",
    "crp": "CRP",
}


def normalize_lab_label(x) -> str:
    if pd.isna(x):
        return "UNKNOWN_LAB"
    s = str(x).strip()
    sl = s.lower()
    for k, v in LAB_NAME_NORMALIZE.items():
        if k in sl:
            return v
    return re.sub(r"\s+", "_", s.lower())


# =============================================================================
# Direct MIMIC evaluation: table discovery/loading
# =============================================================================

def find_table_file(base_dir: str | Path, table_name: str) -> Path:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"MIMIC directory does not exist: {base}")

    target_names = {
        f"{table_name.lower()}.csv",
        f"{table_name.lower()}.csv.gz",
        f"{table_name.upper()}.csv",
        f"{table_name.upper()}.csv.gz",
    }

    matches: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.name.lower() in {x.lower() for x in target_names}:
            matches.append(p)

    if not matches:
        raise FileNotFoundError(f"Cannot find {table_name} in {base}")

    matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]


def read_csv_lower(path: Path, usecols_lower: Optional[Sequence[str]] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    actual_cols = list(header.columns)
    lower_to_actual = {str(c).strip().lower(): c for c in actual_cols}

    usecols_actual = None
    if usecols_lower is not None:
        usecols_actual = [lower_to_actual[c] for c in usecols_lower if c in lower_to_actual]
        if not usecols_actual:
            raise ValueError(f"None of requested columns {usecols_lower} found in {path}")

    if chunksize is None:
        df = pd.read_csv(path, usecols=usecols_actual, low_memory=False)
        return lower_columns(df)

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols_actual, chunksize=chunksize, low_memory=False):
        chunks.append(lower_columns(chunk))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_table(base_dir: str | Path, table_name: str, usecols_lower: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table_file(base_dir, table_name)
    print(f"  Loading {table_name} from {path}...")
    return read_csv_lower(path, usecols_lower=usecols_lower)


def load_event_table_filtered(
    base_dir: str | Path,
    table_name: str,
    wanted_itemids: Optional[Iterable[int]],
    usecols_lower: Sequence[str],
    chunksize: int,
) -> pd.DataFrame:
    path = find_table_file(base_dir, table_name)
    print(f"  Loading {table_name} from {path} with itemid filter...")

    header = pd.read_csv(path, nrows=0)
    actual_cols = list(header.columns)
    lower_to_actual = {str(c).strip().lower(): c for c in actual_cols}
    usecols_actual = [lower_to_actual[c] for c in usecols_lower if c in lower_to_actual]
    if "itemid" not in lower_to_actual:
        raise ValueError(f"{table_name} has no itemid column")

    wanted = set(int(x) for x in wanted_itemids) if wanted_itemids is not None else None
    chunks: List[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=usecols_actual, chunksize=chunksize, low_memory=False):
        chunk = lower_columns(chunk)
        if wanted is not None and "itemid" in chunk.columns:
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["itemid"].isin(wanted)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=[c for c in usecols_lower if c in lower_to_actual])
    return pd.concat(chunks, ignore_index=True)


# =============================================================================
# Direct MIMIC evaluation: normalization/features
# =============================================================================

def normalize_stay_id(icustays: pd.DataFrame) -> pd.DataFrame:
    icu = icustays.copy()
    if "icustay_id" not in icu.columns and "stay_id" in icu.columns:
        icu["icustay_id"] = icu["stay_id"]
    if "stay_id" not in icu.columns and "icustay_id" in icu.columns:
        icu["stay_id"] = icu["icustay_id"]
    return icu


def standardize_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "icustay_id" not in out.columns and "stay_id" in out.columns:
        out["icustay_id"] = out["stay_id"]
    if "stay_id" not in out.columns and "icustay_id" in out.columns:
        out["stay_id"] = out["icustay_id"]
    return out


def build_stays(patients: pd.DataFrame, admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    icu = normalize_stay_id(icustays)
    required = ["subject_id", "hadm_id", "icustay_id", "intime"]
    missing = [c for c in required if c not in icu.columns]
    if missing:
        raise ValueError(f"ICUSTAYS missing required columns: {missing}")

    keep = [c for c in ["subject_id", "hadm_id", "icustay_id", "stay_id", "intime", "outtime", "first_careunit", "last_careunit"] if c in icu.columns]
    stays = icu[keep].copy()
    stays["intime"] = parse_datetime_series(stays["intime"])
    stays["outtime"] = parse_datetime_series(stays["outtime"]) if "outtime" in stays.columns else pd.NaT

    adm = admissions.copy()
    if "admittime" in adm.columns:
        adm["admittime"] = parse_datetime_series(adm["admittime"])
    adm_keep = [c for c in ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "insurance", "ethnicity", "race", "diagnosis"] if c in adm.columns]
    stays = stays.merge(adm[adm_keep].drop_duplicates(["subject_id", "hadm_id"]), on=["subject_id", "hadm_id"], how="left")

    pat = patients.copy()
    pat_keep = [c for c in ["subject_id", "gender", "dob", "anchor_age", "anchor_year"] if c in pat.columns]
    stays = stays.merge(pat[pat_keep].drop_duplicates("subject_id"), on="subject_id", how="left")

    if "anchor_age" in stays.columns:
        stays["age"] = pd.to_numeric(stays["anchor_age"], errors="coerce")
        if "anchor_year" in stays.columns and "admittime" in stays.columns:
            admit_year = stays["admittime"].dt.year
            anchor_year = pd.to_numeric(stays["anchor_year"], errors="coerce")
            stays["age"] = stays["age"] + (admit_year - anchor_year)
    elif "dob" in stays.columns and "admittime" in stays.columns:
        stays["dob"] = parse_datetime_series(stays["dob"])
        stays["age"] = (stays["admittime"] - stays["dob"]).dt.days / 365.25
        stays.loc[stays["age"] > 89, "age"] = 90
    else:
        stays["age"] = np.nan

    stays["age"] = pd.to_numeric(stays["age"], errors="coerce")
    return stays


def get_micro_time(micro: pd.DataFrame) -> pd.Series:
    if "charttime" in micro.columns:
        t = parse_datetime_series(micro["charttime"])
    else:
        t = pd.Series(pd.NaT, index=micro.index)
    if t.isna().all() and "chartdate" in micro.columns:
        t = parse_datetime_series(micro["chartdate"])
    elif "chartdate" in micro.columns:
        d = parse_datetime_series(micro["chartdate"])
        t = t.fillna(d)
    return t


def build_positive_culture_cohort(stays: pd.DataFrame, micro: pd.DataFrame, top_species: int) -> pd.DataFrame:
    if "org_name" not in micro.columns:
        raise ValueError("MICROBIOLOGYEVENTS must contain org_name")

    m = micro.copy()
    m = standardize_event_ids(m)
    m["micro_time"] = get_micro_time(m)
    m["org_clean"] = m["org_name"].map(clean_text)
    m = m[m["org_name"].map(is_positive_organism)].copy()
    if m.empty:
        raise ValueError("No positive microbiology organism rows found")

    join_keys = ["subject_id", "hadm_id"]
    if "subject_id" not in m.columns or "hadm_id" not in m.columns:
        raise ValueError("MICROBIOLOGYEVENTS must contain subject_id and hadm_id")

    stay_cols = ["subject_id", "hadm_id", "icustay_id", "intime", "outtime", "age"]
    if "gender" in stays.columns:
        stay_cols.append("gender")
    joined = m.merge(stays[stay_cols], on=join_keys, how="inner")

    has_time = joined["micro_time"].notna() & joined["intime"].notna()
    in_after_intime = joined["micro_time"] >= joined["intime"]
    before_out = joined["outtime"].isna() | (joined["micro_time"] <= joined["outtime"])
    timed = joined[has_time & in_after_intime & before_out].copy()

    if timed.empty:
        warn("No timed positive cultures inside ICU stay; falling back to first positive culture per stay by available time/order")
        timed = joined.copy()

    timed = timed.sort_values(["icustay_id", "micro_time"], na_position="last")
    first_pos = timed.groupby("icustay_id", as_index=False).first()
    first_pos["label_3class"] = first_pos["org_name"].map(pathogen_class)
    first_pos["species_raw"] = first_pos["org_name"].map(normalize_species)

    vc = first_pos["species_raw"].value_counts()
    top = set(vc.head(top_species).index)
    first_pos["label_species"] = first_pos["species_raw"].where(first_pos["species_raw"].isin(top), "OTHER")

    cohort_cols = [
        "subject_id", "hadm_id", "icustay_id", "intime", "outtime", "age",
        "org_name", "spec_type_desc", "micro_time", "label_3class", "label_species",
    ]
    if "gender" in first_pos.columns:
        cohort_cols.insert(5, "gender")
    return first_pos[[c for c in cohort_cols if c in first_pos.columns]].copy()


def build_specimen_window_features(micro: pd.DataFrame, cohort: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    out = cohort[["icustay_id"]].drop_duplicates().copy()
    if "spec_type_desc" not in micro.columns:
        for w in windows:
            out[f"specimen_family_w{w}h"] = "UNKNOWN"
            out[f"specimen_count_w{w}h"] = 0
            out[f"specimen_unique_family_count_w{w}h"] = 0
        return out

    m = micro.copy()
    m["micro_time"] = get_micro_time(m)
    m = m[m["micro_time"].notna()].copy()
    if m.empty:
        for w in windows:
            out[f"specimen_family_w{w}h"] = "UNKNOWN"
            out[f"specimen_count_w{w}h"] = 0
            out[f"specimen_unique_family_count_w{w}h"] = 0
        return out

    base = cohort[["subject_id", "hadm_id", "icustay_id", "intime"]].drop_duplicates()
    m = m.merge(base, on=["subject_id", "hadm_id"], how="inner")
    m["hours_from_t0"] = (m["micro_time"] - m["intime"]).dt.total_seconds() / 3600.0
    m = m[(m["hours_from_t0"] >= 0) & (m["hours_from_t0"] <= max(windows))].copy()
    m["specimen_family"] = m["spec_type_desc"].map(normalize_specimen_family)

    for w in windows:
        mw = m[m["hours_from_t0"] <= w].sort_values(["icustay_id", "hours_from_t0"])
        first = mw.groupby("icustay_id")["specimen_family"].first().rename(f"specimen_family_w{w}h")
        count = mw.groupby("icustay_id").size().rename(f"specimen_count_w{w}h")
        uniq = mw.groupby("icustay_id")["specimen_family"].nunique().rename(f"specimen_unique_family_count_w{w}h")
        tmp = pd.concat([first, count, uniq], axis=1).reset_index()
        out = out.merge(tmp, on="icustay_id", how="left")
        out[f"specimen_family_w{w}h"] = out[f"specimen_family_w{w}h"].fillna("UNKNOWN")
        out[f"specimen_count_w{w}h"] = out[f"specimen_count_w{w}h"].fillna(0).astype(int)
        out[f"specimen_unique_family_count_w{w}h"] = out[f"specimen_unique_family_count_w{w}h"].fillna(0).astype(int)
    return out


def prepare_lab_events(labs: pd.DataFrame) -> pd.DataFrame:
    if labs.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])
    x = labs.copy()
    x = standardize_event_ids(x)
    if "itemid" not in x.columns:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])
    x["itemid"] = pd.to_numeric(x["itemid"], errors="coerce").astype("Int64")
    x = x[x["itemid"].isin(set(LAB_ITEM_MAP.keys()))].copy()
    if x.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])
    x["variable"] = x["itemid"].astype(int).map(LAB_ITEM_MAP)
    if "valuenum" in x.columns:
        x["value"] = pd.to_numeric(x["valuenum"], errors="coerce")
    elif "value" in x.columns:
        x["value"] = pd.to_numeric(x["value"], errors="coerce")
    else:
        x["value"] = np.nan
    x["charttime"] = parse_datetime_series(x["charttime"] if "charttime" in x.columns else x.get("chartdate", pd.Series(pd.NaT, index=x.index)))
    x = x.dropna(subset=["subject_id", "hadm_id", "charttime", "variable", "value"])
    return x[["subject_id", "hadm_id", "charttime", "variable", "value"]]


def prepare_chart_events(charts: pd.DataFrame) -> pd.DataFrame:
    if charts.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])
    x = charts.copy()
    x = standardize_event_ids(x)
    if "itemid" not in x.columns:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])
    x["itemid"] = pd.to_numeric(x["itemid"], errors="coerce").astype("Int64")
    x = x[x["itemid"].isin(set(CHART_ITEM_MAP.keys()))].copy()
    if x.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "variable", "value"])

    if "valuenum" in x.columns:
        x["value"] = pd.to_numeric(x["valuenum"], errors="coerce")
    elif "value" in x.columns:
        x["value"] = pd.to_numeric(x["value"], errors="coerce")
    else:
        x["value"] = np.nan

    names = []
    values = []
    for itemid, val in zip(x["itemid"].astype(int), x["value"]):
        spec = CHART_ITEM_MAP.get(int(itemid))
        if spec is None or pd.isna(val):
            names.append(None)
            values.append(np.nan)
            continue
        v = float(val)
        if spec.transform == "f_to_c":
            v = (v - 32.0) * 5.0 / 9.0
        names.append(spec.name)
        values.append(v)
    x["variable"] = names
    x["value"] = values
    x["charttime"] = parse_datetime_series(x["charttime"] if "charttime" in x.columns else pd.Series(pd.NaT, index=x.index))

    sanity = {
        "spo2": (20, 100), "temperature_c": (25, 45), "heart_rate": (20, 250),
        "resp_rate": (1, 80), "mean_bp": (20, 200), "sys_bp": (30, 300),
        "dias_bp": (10, 200), "glucose": (5, 1000),
    }
    keep = []
    for var, val in zip(x["variable"], x["value"]):
        if var is None or pd.isna(val):
            keep.append(False)
            continue
        lo, hi = sanity.get(var, (-np.inf, np.inf))
        keep.append(lo <= float(val) <= hi)
    x = x[keep].copy()
    x = x.dropna(subset=["subject_id", "hadm_id", "charttime", "variable", "value"])
    return x[["subject_id", "hadm_id", "charttime", "variable", "value"]]


def summarize_group(g: pd.DataFrame) -> Dict[str, float]:
    g = g.sort_values("hours_from_t0")
    vals = pd.to_numeric(g["value"], errors="coerce").dropna().astype(float)
    if vals.empty:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "first": np.nan, "last": np.nan, "delta": np.nan, "slope": np.nan, "count": 0}

    gg = g.loc[vals.index].sort_values("hours_from_t0")
    vals_arr = gg["value"].astype(float).values
    hours_arr = gg["hours_from_t0"].astype(float).values

    if len(vals_arr) >= 2 and len(np.unique(hours_arr)) >= 2:
        try:
            slope = float(np.polyfit(hours_arr, vals_arr, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    return {
        "mean": float(np.mean(vals_arr)),
        "std": float(np.std(vals_arr, ddof=0)) if len(vals_arr) > 1 else 0.0,
        "min": float(np.min(vals_arr)),
        "max": float(np.max(vals_arr)),
        "first": float(vals_arr[0]),
        "last": float(vals_arr[-1]),
        "delta": float(vals_arr[-1] - vals_arr[0]),
        "slope": slope,
        "count": int(len(vals_arr)),
    }


def build_summary_features(events: pd.DataFrame, cohort: pd.DataFrame, windows: Sequence[int], source_prefix: str) -> pd.DataFrame:
    out = cohort[["icustay_id"]].drop_duplicates().copy()
    if events.empty:
        return out

    base = cohort[["subject_id", "hadm_id", "icustay_id", "intime"]].drop_duplicates()
    e = events.merge(base, on=["subject_id", "hadm_id"], how="inner")
    e["hours_from_t0"] = (e["charttime"] - e["intime"]).dt.total_seconds() / 3600.0
    e = e[(e["hours_from_t0"] >= 0) & (e["hours_from_t0"] <= max(windows))].copy()
    if e.empty:
        return out

    for w in windows:
        ew = e[e["hours_from_t0"] <= w].copy()
        rows = []
        for (stay_id, var), g in ew.groupby(["icustay_id", "variable"]):
            stats = summarize_group(g)
            row = {"icustay_id": stay_id}
            for k, v in stats.items():
                row[f"{source_prefix}_{var}_w{w}h_{k}"] = v
            rows.append(row)
        if not rows:
            continue
        wide = pd.DataFrame(rows).groupby("icustay_id", as_index=False).first()
        out = out.merge(wide, on="icustay_id", how="left")
    return out


# =============================================================================
# Shared model evaluation
# =============================================================================

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = X.copy()
    categorical_cols: List[str] = []
    numeric_cols: List[str] = []

    for c in X.columns:
        if is_bool_dtype(X[c]) or is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    for c in categorical_cols:
        X[c] = X[c].where(X[c].notna(), np.nan)
        X.loc[X[c].notna(), c] = X.loc[X[c].notna(), c].astype(str)
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot_encoder())])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers), numeric_cols, categorical_cols


def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if is_bool_dtype(X[c]) or is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].where(X[c].notna(), np.nan)
            X.loc[X[c].notna(), c] = X.loc[X[c].notna(), c].astype(str)
    return X


def make_models(preprocess: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "DummyMostFreq": Pipeline([("preprocess", preprocess), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "DummyStratified": Pipeline([("preprocess", preprocess), ("clf", DummyClassifier(strategy="stratified", random_state=42))]),
        "LogRegBalanced": Pipeline([("preprocess", preprocess), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=42))]),
        "ExtraTreesBalanced": Pipeline([("preprocess", preprocess), ("clf", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))]),
        "RandomForestBalanced": Pipeline([("preprocess", preprocess), ("clf", RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("preprocess", preprocess), ("clf", HistGradientBoostingClassifier(learning_rate=0.03, max_iter=250, l2_regularization=0.1, random_state=42))]),
    }


def choose_cv(y: pd.Series, groups: Optional[pd.Series], max_splits: int):
    counts = y.value_counts()
    min_count = int(counts.min())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        return None, 0

    if groups is not None and HAVE_STRATIFIED_GROUP_KFOLD:
        if groups.nunique() >= n_splits:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits


def cv_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv, groups: Optional[pd.Series]) -> Dict[str, Tuple[float, float]]:
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}
    kwargs = {}
    if groups is not None and HAVE_STRATIFIED_GROUP_KFOLD and isinstance(cv, StratifiedGroupKFold):
        kwargs["groups"] = groups
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None, error_score="raise", **kwargs)
    return {k: (float(np.mean(scores[f"test_{k}"])), float(np.std(scores[f"test_{k}"]))) for k in scoring}


def print_score_line(name: str, scores: Dict[str, Tuple[float, float]], base: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    parts = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        mean, std = scores[metric]
        parts.append(f"{metric}={mean:.3f}+/-{std:.3f}")
    suffix = ""
    if base is not None:
        pass_macro = scores["macro_f1"][0] > base["macro_f1"][0]
        pass_bal = scores["balanced_accuracy"][0] > base["balanced_accuracy"][0]
        pass_acc = scores["accuracy"][0] > base["accuracy"][0]
        suffix = f"  [{'PASS' if (pass_macro and pass_bal) else 'BELOW'} baseline macroF1+balAcc; acc {'PASS' if pass_acc else 'BELOW'}]"
    print(f"  {name:<22} " + "  ".join(parts) + suffix)


def evaluate_label(df: pd.DataFrame, label_col: str, max_splits: int = 5, min_class_count: int = 2, id_cols: Optional[set] = None) -> None:
    banner(f"Evaluating {label_col}")
    if label_col not in df.columns:
        print(f"SKIP: {label_col} not found")
        return

    work = df[df[label_col].notna()].copy()
    counts = work[label_col].value_counts()
    print("Class counts:")
    for k, v in counts.items():
        print(f"  {str(k):<45} n={v}")

    keep_classes = counts[counts >= min_class_count].index
    dropped = counts[counts < min_class_count]
    if len(dropped):
        print("Dropping rare classes for CV stability:")
        for k, v in dropped.items():
            print(f"  {str(k):<45} n={v}")

    work = work[work[label_col].isin(keep_classes)].copy()
    if work[label_col].nunique() < 2:
        print("Not enough classes after filtering; skipping.")
        return

    default_drop_cols = {
        "subject_id", "hadm_id", "icustay_id", "stay_id", "intime", "outtime", "micro_time",
        "org_name", "spec_type_desc", "organism", "culture_sites", "label_3class", "label_species", "species_raw",
        "patientunitstayid", "group_id", "patienthealthsystemstayid", "uniquepid", "source_dataset",
    }
    if id_cols:
        default_drop_cols.update(id_cols)

    feature_cols = [c for c in work.columns if c not in default_drop_cols]
    X = sanitize_feature_matrix(work[feature_cols].copy())
    y = work[label_col].astype(str)

    groups = None
    for gcol in ["subject_id", "uniquepid", "group_id", "patientunitstayid"]:
        if gcol in work.columns:
            groups = work[gcol]
            break

    preprocess, numeric_cols, categorical_cols = build_preprocessor(X)
    print(f"\nN={len(work)} classes={y.nunique()} numeric_features={len(numeric_cols)} categorical_features={len(categorical_cols)}")
    if categorical_cols:
        print("Categorical:", categorical_cols[:30])

    cv, n_splits = choose_cv(y, groups, max_splits=max_splits)
    if cv is None:
        print("Not enough samples per class for CV; skipping.")
        return
    print(f"CV: {type(cv).__name__}, n_splits={n_splits}")

    models = make_models(preprocess)
    results = {}
    baseline = None
    for name, model in models.items():
        try:
            scores = cv_scores(model, X, y, cv, groups)
            results[name] = scores
            if name == "DummyMostFreq":
                baseline = scores
                print_score_line(name, scores)
            else:
                print_score_line(name, scores, baseline)
        except Exception as exc:
            print(f"  {name:<22} ERROR: {exc}")

    candidate_names = [n for n in results if not n.startswith("Dummy")]
    if candidate_names:
        best_name = max(candidate_names, key=lambda n: results[n]["macro_f1"][0])
        print(f"\nBest non-dummy by macroF1: {best_name}")


# =============================================================================
# MIMIC direct evaluator mode
# =============================================================================

def load_all_mimic(args):
    patients = load_table(args.dir, "patients")
    admissions = load_table(args.dir, "admissions")
    icustays = load_table(args.dir, "icustays")

    lab_usecols = ["subject_id", "hadm_id", "itemid", "charttime", "chartdate", "valuenum", "value"]
    chart_usecols = ["subject_id", "hadm_id", "stay_id", "icustay_id", "itemid", "charttime", "valuenum", "value"]
    micro_usecols = ["subject_id", "hadm_id", "microevent_id", "chartdate", "charttime", "spec_itemid", "spec_type_desc", "org_itemid", "org_name", "isolate_num", "test_name", "comments", "interpretation"]

    labs = load_event_table_filtered(args.dir, "labevents", LAB_ITEM_MAP.keys(), lab_usecols, args.chunksize)
    charts = load_event_table_filtered(args.dir, "chartevents", CHART_ITEM_MAP.keys(), chart_usecols, args.chunksize)
    micro = load_table(args.dir, "microbiologyevents", usecols_lower=micro_usecols)
    return patients, admissions, icustays, labs, charts, micro


def run_mimic_eval(args) -> int:
    windows = sorted(set(int(x.strip()) for x in args.windows.split(",") if x.strip()))

    banner("PenuX — Direct MIMIC Leakage-Aware Evaluation")
    print(f"MIMIC dir : {args.dir}")
    print(f"Windows   : {windows} hours")
    print("Text input: specimen family inside window only; no interpretation field")

    print("\n[1/6] Loading tables...")
    patients, admissions, icustays, labs_raw, charts_raw, micro = load_all_mimic(args)

    print("\n[2/6] Building ICU stay table...")
    stays = build_stays(patients, admissions, icustays)
    print(f"  ICU stays loaded: {len(stays)}")

    print("\n[3/6] Building positive-culture cohort and labels...")
    cohort = build_positive_culture_cohort(stays, micro, top_species=args.top_species)
    print(f"  Cohort: {len(cohort)} ICU stays with positive cultures")
    for label_col in ["label_3class", "label_species"]:
        print(f"\n  {label_col}:")
        for k, v in cohort[label_col].value_counts().items():
            print(f"    {k:<45} n={v}")

    print("\n[4/6] Preparing event streams...")
    labs = prepare_lab_events(labs_raw)
    charts = prepare_chart_events(charts_raw)
    print(f"  Lab event rows after itemid filter     : {len(labs)}")
    print(f"  Chart event rows after itemid filter   : {len(charts)}")

    print("\n[5/6] Building summary features for each window...")
    lab_feats = build_summary_features(labs, cohort, windows, source_prefix="lab")
    chart_feats = build_summary_features(charts, cohort, windows, source_prefix="chart")
    spec_feats = build_specimen_window_features(micro, cohort, windows)

    df = cohort.copy()
    df = df.merge(lab_feats, on="icustay_id", how="left")
    df = df.merge(chart_feats, on="icustay_id", how="left")
    df = df.merge(spec_feats, on="icustay_id", how="left")

    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("UNKNOWN").astype(str)

    count_cols = [c for c in df.columns if c.endswith("_count") or "_count_w" in c]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    summary_cols = [c for c in df.columns if any(c.endswith(s) for s in ["_mean", "_last", "_max", "_min"])]
    for c in summary_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    feature_cols = [c for c in df.columns if c not in {"subject_id", "hadm_id", "icustay_id", "stay_id", "intime", "outtime", "micro_time", "org_name", "spec_type_desc", "label_3class", "label_species", "species_raw"}]
    print(f"  Final matrix: N={len(df)} rows, feature columns={len(feature_cols)}")
    print("  Example feature columns:")
    for c in feature_cols[:40]:
        print(f"    {c}")
    if len(feature_cols) > 40:
        print(f"    ... {len(feature_cols) - 40} more")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Wrote feature matrix: {out_path}")

    print("\n[6/6] Evaluating labels...")
    evaluate_label(df, "label_3class", args.max_splits, args.min_class_count)
    evaluate_label(df, "label_species", args.max_splits, args.min_class_count)

    banner("DONE")
    print("Interpretation rule: prefer macroF1 and balanced accuracy over majority-class accuracy.")
    return 0


# =============================================================================
# MIMIC -> eICU-like conversion
# =============================================================================

def build_mimiciii_stays(root: Path) -> pd.DataFrame:
    patients = lower_columns(read_csv_any(find_file(root, ["PATIENTS", "patients"])))
    admissions = lower_columns(read_csv_any(find_file(root, ["ADMISSIONS", "admissions"])))
    icu = lower_columns(read_csv_any(find_file(root, ["ICUSTAYS", "icustays"])))

    stays = icu.merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
    stays = stays.merge(patients[["subject_id", "gender", "dob"]], on="subject_id", how="left")
    stays["intime"] = to_dt(stays["intime"])
    stays["outtime"] = to_dt(stays["outtime"])
    stays["dob"] = to_dt(stays["dob"])
    stays["age"] = ((stays["intime"] - stays["dob"]).dt.days / 365.25).clip(lower=0, upper=90)

    return pd.DataFrame({
        "patientunitstayid": stays["icustay_id"],
        "patienthealthsystemstayid": stays["hadm_id"],
        "uniquepid": stays["subject_id"],
        "subject_id": stays["subject_id"],
        "hadm_id": stays["hadm_id"],
        "stay_id": stays["icustay_id"],
        "gender": stays["gender"].fillna("UNKNOWN"),
        "age": stays["age"].round(1),
        "admissionheight": np.nan,
        "admissionweight": np.nan,
        "hospitaladmitoffset": minute_offset(stays["admittime"], stays["intime"]) if "admittime" in stays.columns else np.nan,
        "unitdischargeoffset": minute_offset(stays["outtime"], stays["intime"]),
        "apacheadmissiondx": stays["diagnosis"].fillna("UNKNOWN") if "diagnosis" in stays.columns else "UNKNOWN",
        "source_dataset": "MIMIC-III-DEMO",
        "intime": stays["intime"],
        "outtime": stays["outtime"],
    }).drop_duplicates("patientunitstayid")


def build_mimiciv_stays(root: Path) -> pd.DataFrame:
    patients = lower_columns(read_csv_any(find_file(root, ["patients"])))
    admissions = lower_columns(read_csv_any(find_file(root, ["admissions"])))
    icu = lower_columns(read_csv_any(find_file(root, ["icustays"])))

    stays = icu.merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
    stays = stays.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    stays["intime"] = to_dt(stays["intime"])
    stays["outtime"] = to_dt(stays["outtime"])

    return pd.DataFrame({
        "patientunitstayid": stays["stay_id"],
        "patienthealthsystemstayid": stays["hadm_id"],
        "uniquepid": stays["subject_id"],
        "subject_id": stays["subject_id"],
        "hadm_id": stays["hadm_id"],
        "stay_id": stays["stay_id"],
        "gender": stays["gender"].fillna("UNKNOWN"),
        "age": clean_age_years(stays["anchor_age"]),
        "admissionheight": np.nan,
        "admissionweight": np.nan,
        "hospitaladmitoffset": minute_offset(stays["admittime"], stays["intime"]) if "admittime" in stays.columns else np.nan,
        "unitdischargeoffset": minute_offset(stays["outtime"], stays["intime"]),
        "apacheadmissiondx": stays["admission_type"].fillna("UNKNOWN") if "admission_type" in stays.columns else "UNKNOWN",
        "source_dataset": "MIMIC-IV-DEMO",
        "intime": stays["intime"],
        "outtime": stays["outtime"],
    }).drop_duplicates("patientunitstayid")


def convert_mimic_labs(root: Path, stays: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    p = find_file(root, ["LABEVENTS", "labevents"])
    if p is None:
        return pd.DataFrame()

    labs = lower_columns(read_csv_any(p))
    dlab_path = find_file(root, ["D_LABITEMS", "d_labitems"])
    if dlab_path:
        dlab = lower_columns(read_csv_any(dlab_path))
        label_col = "label" if "label" in dlab.columns else pick_col(dlab, ["fluid", "category"])
        labs = labs.merge(dlab[["itemid", label_col]], on="itemid", how="left")
        labs["labname"] = labs[label_col].apply(normalize_lab_label)
    else:
        labs["labname"] = labs["itemid"].astype(str)

    m = labs.merge(stays[["subject_id", "hadm_id", "patientunitstayid", "intime"]], on=["subject_id", "hadm_id"], how="inner")
    m["charttime"] = to_dt(m["charttime"])
    m["labresultoffset"] = minute_offset(m["charttime"], m["intime"])
    m["labresult"] = to_num(m["valuenum"])

    return pd.DataFrame({
        "patientunitstayid": m["patientunitstayid"],
        "labresultoffset": m["labresultoffset"],
        "labname": m["labname"],
        "labresult": m["labresult"],
        "labmeasurenamesystem": m["valueuom"] if "valueuom" in m.columns else "",
        "labmeasurenameinterface": m["valueuom"] if "valueuom" in m.columns else "",
        "source_dataset": source_dataset,
        "itemid": m["itemid"],
    }).dropna(subset=["patientunitstayid", "labresultoffset", "labresult"])


def convert_mimic_vitals(root: Path, stays: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    p = find_file(root, ["CHARTEVENTS", "chartevents"])
    if p is None:
        return pd.DataFrame()

    charts = lower_columns(read_csv_any(p))
    charts["itemid"] = to_num(charts["itemid"])
    charts = charts[charts["itemid"].isin(MIMIC_CHARTITEMS_EICU.keys())].copy()
    if charts.empty:
        return pd.DataFrame()

    charts["vital_name"] = charts["itemid"].astype(int).map(MIMIC_CHARTITEMS_EICU)
    charts["value"] = to_num(charts["valuenum"])
    join_keys = ["subject_id", "hadm_id"]
    if "stay_id" in charts.columns and "stay_id" in stays.columns and source_dataset == "MIMIC-IV-DEMO":
        join_keys = ["subject_id", "hadm_id", "stay_id"]
    m = charts.merge(stays[join_keys + ["patientunitstayid", "intime"]], on=join_keys, how="inner")
    m["charttime"] = to_dt(m["charttime"])
    m["observationoffset"] = minute_offset(m["charttime"], m["intime"])

    temp_f = m["vital_name"].eq("temperature_f")
    m.loc[temp_f, "value"] = (m.loc[temp_f, "value"] - 32.0) * 5.0 / 9.0
    m.loc[temp_f, "vital_name"] = "temperature"

    wide = m.pivot_table(index=["patientunitstayid", "observationoffset"], columns="vital_name", values="value", aggfunc="mean").reset_index()
    for col in ["temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["source_dataset"] = source_dataset
    return wide[["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "source_dataset"]]


def convert_mimic_micro(root: Path, stays: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    p = find_file(root, ["MICROBIOLOGYEVENTS", "microbiologyevents"])
    if p is None:
        return pd.DataFrame()

    micro = lower_columns(read_csv_any(p))
    m = micro.merge(stays[["subject_id", "hadm_id", "patientunitstayid", "intime"]], on=["subject_id", "hadm_id"], how="inner")
    time_col = "charttime" if "charttime" in m.columns else "chartdate"
    m[time_col] = to_dt(m[time_col])
    m["culturetakenoffset"] = minute_offset(m[time_col], m["intime"])

    org = m["org_name"] if "org_name" in m.columns else np.nan
    spec = m["spec_type_desc"] if "spec_type_desc" in m.columns else "UNKNOWN"

    return pd.DataFrame({
        "patientunitstayid": m["patientunitstayid"],
        "culturetakenoffset": m["culturetakenoffset"],
        "culturesite": spec,
        "organism": org,
        "source_dataset": source_dataset,
        "spec_type_desc": spec,
        "org_name": org,
    }).dropna(subset=["patientunitstayid", "culturetakenoffset"])


def convert_mimic_medication(root: Path, stays: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    p = find_file(root, ["PRESCRIPTIONS", "prescriptions", "INPUTEVENTS_MV", "inputevents_mv", "inputevents"])
    if p is None:
        return pd.DataFrame()

    meds = lower_columns(read_csv_any(p))
    time_col = "starttime" if "starttime" in meds.columns else ("startdate" if "startdate" in meds.columns else None)
    if time_col is None:
        return pd.DataFrame()

    drug_col = "drug" if "drug" in meds.columns else ("label" if "label" in meds.columns else None)
    if drug_col is None:
        return pd.DataFrame()

    join_keys = ["subject_id", "hadm_id"]
    if "stay_id" in meds.columns and "stay_id" in stays.columns and source_dataset == "MIMIC-IV-DEMO":
        join_keys = ["subject_id", "hadm_id", "stay_id"]

    m = meds.merge(stays[join_keys + ["patientunitstayid", "intime"]], on=join_keys, how="inner")
    m[time_col] = to_dt(m[time_col])
    start_offset = minute_offset(m[time_col], m["intime"])

    stop_col = "stoptime" if "stoptime" in m.columns else ("endtime" if "endtime" in m.columns else ("enddate" if "enddate" in m.columns else None))
    stop_offset = minute_offset(m[stop_col], m["intime"]) if stop_col else pd.Series([pd.NA] * len(m))

    return pd.DataFrame({
        "patientunitstayid": m["patientunitstayid"],
        "drugstartoffset": start_offset,
        "drugstopoffset": stop_offset,
        "drugname": m[drug_col].astype(str),
        "dosage": m["dose_val_rx"].astype(str) if "dose_val_rx" in m.columns else "",
        "routeadmin": m["route"].astype(str) if "route" in m.columns else "",
        "source_dataset": source_dataset,
    }).dropna(subset=["patientunitstayid", "drugstartoffset"])


def concat_nonempty(parts: List[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def write_csv(df: pd.DataFrame, outdir: Path, name: str, columns: List[str]) -> None:
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    path = outdir / name
    print(f"Writing {path}: {df.shape}")
    df[columns].to_csv(path, index=False)


def run_mimic_to_eicu(args) -> int:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    patient_parts, lab_parts, vital_parts, med_parts, micro_parts = [], [], [], [], []

    if args.mimiciii_dir:
        root = Path(args.mimiciii_dir)
        banner(f"Converting MIMIC-III Demo -> eICU-like schema: {root}")
        stays = build_mimiciii_stays(root)
        patient_parts.append(stays.drop(columns=["intime", "outtime"], errors="ignore"))
        lab_parts.append(convert_mimic_labs(root, stays, "MIMIC-III-DEMO"))
        vital_parts.append(convert_mimic_vitals(root, stays, "MIMIC-III-DEMO"))
        med_parts.append(convert_mimic_medication(root, stays, "MIMIC-III-DEMO"))
        micro_parts.append(convert_mimic_micro(root, stays, "MIMIC-III-DEMO"))

    if args.mimiciv_dir:
        root = Path(args.mimiciv_dir)
        banner(f"Converting MIMIC-IV Demo -> eICU-like schema: {root}")
        stays = build_mimiciv_stays(root)
        patient_parts.append(stays.drop(columns=["intime", "outtime"], errors="ignore"))
        lab_parts.append(convert_mimic_labs(root, stays, "MIMIC-IV-DEMO"))
        vital_parts.append(convert_mimic_vitals(root, stays, "MIMIC-IV-DEMO"))
        med_parts.append(convert_mimic_medication(root, stays, "MIMIC-IV-DEMO"))
        micro_parts.append(convert_mimic_micro(root, stays, "MIMIC-IV-DEMO"))

    patient = concat_nonempty(patient_parts)
    lab = concat_nonempty(lab_parts)
    vital = concat_nonempty(vital_parts)
    med = concat_nonempty(med_parts)
    micro = concat_nonempty(micro_parts)

    patient_cols = ["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "gender", "age", "admissionheight", "admissionweight", "hospitaladmitoffset", "unitdischargeoffset", "apacheadmissiondx", "source_dataset", "subject_id", "hadm_id", "stay_id"]
    lab_cols = ["patientunitstayid", "labresultoffset", "labname", "labresult", "labmeasurenamesystem", "labmeasurenameinterface", "source_dataset", "itemid"]
    vital_cols = ["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "source_dataset"]
    med_cols = ["patientunitstayid", "drugstartoffset", "drugstopoffset", "drugname", "dosage", "routeadmin", "source_dataset"]
    micro_cols = ["patientunitstayid", "culturetakenoffset", "culturesite", "organism", "source_dataset", "spec_type_desc", "org_name"]

    # Normalize eICU-like join keys in generated CSVs so re-loading does not infer conflicting dtypes.
    for _df in [patient, lab, vital, med, micro]:
        if _df is not None and not _df.empty and "patientunitstayid" in _df.columns:
            _df["patientunitstayid"] = normalize_join_id_series(_df["patientunitstayid"])

    write_csv(patient, outdir, "patient.csv", patient_cols)
    write_csv(lab, outdir, "lab.csv", lab_cols)
    write_csv(vital, outdir, "vitalPeriodic.csv", vital_cols)
    write_csv(med, outdir, "medication.csv", med_cols)
    write_csv(micro, outdir, "microLab.csv", micro_cols)

    (outdir / "README_schema_mapping.txt").write_text(
        "MIMIC Demo to eICU-like schema mapping\n\n"
        "MIMIC icustay_id/stay_id -> patientunitstayid\n"
        "MIMIC hadm_id -> patienthealthsystemstayid\n"
        "MIMIC subject_id -> uniquepid\n"
        "MIMIC ICU intime -> T0\n"
        "event time - ICU intime -> offset in minutes\n\n"
        "microLab.organism/spec_type_desc should be used for labels only, not features.\n",
        encoding="utf-8",
    )

    banner("MIMIC -> eICU-like conversion DONE")
    print("Output:", outdir.resolve())
    print("patient:", patient.shape)
    print("lab    :", lab.shape)
    print("vital  :", vital.shape)
    print("med    :", med.shape)
    print("micro  :", micro.shape)
    return 0


# =============================================================================
# eICU/eICU-like evaluation mode
# =============================================================================

def find_eicu_table(root: Path, names: Iterable[str]) -> Optional[Path]:
    return find_file(root, names)


def build_eicu_micro_labels(root: Path, patient_ids) -> pd.DataFrame:
    p = find_eicu_table(root, ["microLab", "microlab", "microbiology", "culture"])
    if p is None:
        print("  WARNING: No microLab/microbiology table found. Labels cannot be built.")
        out = pd.DataFrame({"patientunitstayid": list(patient_ids)})
        out["patientunitstayid"] = normalize_join_id_series(out["patientunitstayid"])
        return out

    micro = lower_columns(read_csv_any(p))
    id_col = pick_col(micro, ["patientunitstayid"], required=True, table="microLab")
    org_col = pick_col(micro, ["organism", "organismname", "organism_name", "culturedorganism", "resultorganism", "microorganism", "org_name"], required=False, table="microLab")
    offset_col = pick_col(micro, ["culturetakenoffset", "culturepositiveoffset", "resultoffset", "microresultoffset", "labresultoffset", "observationoffset"], required=False, table="microLab")
    site_col = pick_col(micro, ["culturesite", "site", "specimen", "specimentype", "specimen_type", "culture_site", "spec_type_desc"], required=False, table="microLab")

    if org_col is None:
        print("  WARNING: micro table found, but no organism column found.")
        out = pd.DataFrame({"patientunitstayid": list(patient_ids)})
        out["patientunitstayid"] = normalize_join_id_series(out["patientunitstayid"])
        return out

    use_cols = [id_col, org_col] + ([offset_col] if offset_col else []) + ([site_col] if site_col else [])
    micro = micro[use_cols].copy().rename(columns={id_col: "patientunitstayid", org_col: "organism"})
    micro["patientunitstayid"] = normalize_join_id_series(micro["patientunitstayid"])
    if offset_col:
        micro = micro.rename(columns={offset_col: "culture_offset_min"})
        micro["culture_offset_min"] = safe_num(micro["culture_offset_min"])
    else:
        micro["culture_offset_min"] = np.nan
    if site_col:
        micro = micro.rename(columns={site_col: "culture_site"})
    else:
        micro["culture_site"] = "UNKNOWN"

    micro = micro[micro["organism"].apply(is_positive_organism)].copy()
    if micro.empty:
        print("  WARNING: no positive organisms found.")
        out = pd.DataFrame({"patientunitstayid": list(patient_ids)})
        out["patientunitstayid"] = normalize_join_id_series(out["patientunitstayid"])
        return out

    micro["label_group_raw"] = micro["organism"].apply(pathogen_class)
    micro["label_species_raw"] = micro["organism"].apply(normalize_species)

    rows = []
    for pid, g in micro.groupby("patientunitstayid"):
        groups = sorted(set(g["label_group_raw"].dropna()))
        species = g["label_species_raw"].dropna().astype(str)
        label_3 = np.nan if not groups else (groups[0] if len(groups) == 1 else "MIXED")
        label_sp = np.nan if species.empty else species.value_counts().index[0]
        rows.append({
            "patientunitstayid": pid,
            "label_3class": label_3,
            "label_species": label_sp,
            "micro_positive_count": int(len(g)),
            "first_culture_offset_min": float(np.nanmin(g["culture_offset_min"])) if g["culture_offset_min"].notna().any() else np.nan,
            "culture_sites": "|".join(sorted(set(g["culture_site"].astype(str).head(20)))),
        })
    labels = pd.DataFrame(rows)
    if "patientunitstayid" in labels.columns:
        labels["patientunitstayid"] = normalize_join_id_series(labels["patientunitstayid"])
    return labels


def eicu_patient_features(root: Path) -> pd.DataFrame:
    p = find_eicu_table(root, ["patient"])
    if p is None:
        raise FileNotFoundError("patient.csv or patient.csv.gz not found")
    patient = lower_columns(read_csv_any(p))

    id_col = pick_col(patient, ["patientunitstayid"], required=True, table="patient")
    sys_col = pick_col(patient, ["patienthealthsystemstayid"], required=False, table="patient")
    uniq_col = pick_col(patient, ["uniquepid"], required=False, table="patient")
    gender_col = pick_col(patient, ["gender"], required=False, table="patient")
    age_col = pick_col(patient, ["age"], required=False, table="patient")
    ethnicity_col = pick_col(patient, ["ethnicity"], required=False, table="patient")
    height_col = pick_col(patient, ["admissionheight"], required=False, table="patient")
    weight_col = pick_col(patient, ["admissionweight"], required=False, table="patient")
    dx_col = pick_col(patient, ["apacheadmissiondx", "admissiondx"], required=False, table="patient")
    source_col = pick_col(patient, ["source_dataset"], required=False, table="patient")

    F = pd.DataFrame()
    F["patientunitstayid"] = patient[id_col]
    F["group_id"] = patient[sys_col] if sys_col else patient[id_col]
    F["uniquepid"] = patient[uniq_col] if uniq_col else patient[id_col]
    F["gender"] = patient[gender_col].astype(str).fillna("UNKNOWN") if gender_col else "UNKNOWN"
    F["age"] = patient[age_col].apply(lambda x: float(re.sub(r"[^0-9.]", "", str(x))) if str(x).startswith(">") else pd.to_numeric(x, errors="coerce")) if age_col else np.nan
    F["ethnicity"] = patient[ethnicity_col].astype(str).fillna("UNKNOWN") if ethnicity_col else "UNKNOWN"
    F["admissionheight"] = safe_num(patient[height_col]) if height_col else np.nan
    F["admissionweight"] = safe_num(patient[weight_col]) if weight_col else np.nan
    F["apacheadmissiondx"] = patient[dx_col].astype(str).fillna("UNKNOWN") if dx_col else "UNKNOWN"
    F["source_dataset"] = patient[source_col].astype(str).fillna("UNKNOWN") if source_col else "UNKNOWN"

    # Normalize join keys before any downstream merge.
    F["patientunitstayid"] = normalize_join_id_series(F["patientunitstayid"])
    F["group_id"] = normalize_join_id_series(F["group_id"])
    F["uniquepid"] = normalize_join_id_series(F["uniquepid"])

    return F.dropna(subset=["patientunitstayid"]).drop_duplicates("patientunitstayid")


def eicu_summarize_series(g, value_col="value", offset_col="hours_from_t0"):
    g = g[[value_col, offset_col]].copy()
    g[value_col] = safe_num(g[value_col])
    g[offset_col] = safe_num(g[offset_col])
    g = g.dropna().sort_values(offset_col)

    if g.empty:
        return pd.Series({"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "first": np.nan, "last": np.nan, "delta": np.nan, "slope": np.nan, "count": 0})

    y = g[value_col].astype(float).to_numpy()
    t = g[offset_col].astype(float).to_numpy()
    if len(y) >= 2 and np.nanstd(t) > 0:
        try:
            slope = float(np.polyfit(t, y, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    return pd.Series({"mean": float(np.nanmean(y)), "std": float(np.nanstd(y)), "min": float(np.nanmin(y)), "max": float(np.nanmax(y)), "first": float(y[0]), "last": float(y[-1]), "delta": float(y[-1] - y[0]) if len(y) >= 2 else 0.0, "slope": slope, "count": int(len(y))})


def add_eicu_long_table_summary(F, root, table_names, id_candidates, offset_candidates, variable_candidates, value_candidates, prefix, windows):
    p = find_eicu_table(root, table_names)
    if p is None:
        print(f"  SKIP {prefix}: table not found")
        return F

    df = lower_columns(read_csv_any(p))
    id_col = pick_col(df, id_candidates, required=True, table=prefix)
    offset_col = pick_col(df, offset_candidates, required=True, table=prefix)
    var_col = pick_col(df, variable_candidates, required=True, table=prefix)
    val_col = pick_col(df, value_candidates, required=True, table=prefix)

    df = df[[id_col, offset_col, var_col, val_col]].copy().rename(columns={id_col: "patientunitstayid", offset_col: "offset_min", var_col: "var_name", val_col: "value"})
    df["patientunitstayid"] = normalize_join_id_series(df["patientunitstayid"])
    df["offset_min"] = safe_num(df["offset_min"])
    df["hours_from_t0"] = df["offset_min"] / 60.0
    df["value"] = safe_num(df["value"])
    df["var_name"] = df["var_name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    df = df.dropna(subset=["patientunitstayid", "hours_from_t0", "var_name", "value"])
    df = df[(df["hours_from_t0"] >= 0) & (df["hours_from_t0"] <= max(windows))]
    if df.empty:
        print(f"  SKIP {prefix}: no early numeric rows")
        return F

    print(f"  {prefix}: early numeric rows={len(df)} variables={df['var_name'].nunique()}")
    for wh in windows:
        sub = df[df["hours_from_t0"] <= wh].copy()
        if sub.empty:
            continue
        summary = sub.groupby(["patientunitstayid", "var_name"], observed=True).apply(eicu_summarize_series, value_col="value", offset_col="hours_from_t0").reset_index()
        wide = summary.pivot(index="patientunitstayid", columns="var_name")
        wide.columns = [f"{prefix}_{v}_w{wh}h_{stat}" for stat, v in wide.columns]
        F = safe_merge_patientunitstayid(F, wide.reset_index(), how="left")
    return F


def add_eicu_wide_vitals_summary(F, root, table_names, prefix, windows):
    p = find_eicu_table(root, table_names)
    if p is None:
        print(f"  SKIP {prefix}: table not found")
        return F

    df = lower_columns(read_csv_any(p))
    id_col = pick_col(df, ["patientunitstayid"], required=True, table=prefix)
    offset_col = pick_col(df, ["observationoffset", "offset", "chartoffset"], required=True, table=prefix)
    ignore = {id_col, offset_col, "observationyear", "observationtime24", "observationtime"}

    numeric_cols = []
    for c in df.columns:
        if c in ignore:
            continue
        test = pd.to_numeric(df[c], errors="coerce")
        if test.notna().sum() > 0:
            numeric_cols.append(c)
    if not numeric_cols:
        print(f"  SKIP {prefix}: no numeric columns")
        return F

    df = df[[id_col, offset_col] + numeric_cols].copy().rename(columns={id_col: "patientunitstayid", offset_col: "offset_min"})
    df["patientunitstayid"] = normalize_join_id_series(df["patientunitstayid"])
    df["offset_min"] = safe_num(df["offset_min"])
    df["hours_from_t0"] = df["offset_min"] / 60.0
    df = df[(df["hours_from_t0"] >= 0) & (df["hours_from_t0"] <= max(windows))]
    print(f"  {prefix}: early rows={len(df)} numeric_cols={len(numeric_cols)}")

    long_parts = []
    for c in numeric_cols:
        tmp = df[["patientunitstayid", "hours_from_t0", c]].copy().rename(columns={c: "value"})
        tmp["value"] = safe_num(tmp["value"])
        tmp["var_name"] = c.lower()
        tmp = tmp.dropna(subset=["value"])
        if len(tmp):
            long_parts.append(tmp)
    if not long_parts:
        return F

    long = pd.concat(long_parts, ignore_index=True)
    for wh in windows:
        sub = long[long["hours_from_t0"] <= wh].copy()
        if sub.empty:
            continue
        summary = sub.groupby(["patientunitstayid", "var_name"], observed=True).apply(eicu_summarize_series, value_col="value", offset_col="hours_from_t0").reset_index()
        wide = summary.pivot(index="patientunitstayid", columns="var_name")
        wide.columns = [f"{prefix}_{v}_w{wh}h_{stat}" for stat, v in wide.columns]
        F = safe_merge_patientunitstayid(F, wide.reset_index(), how="left")
    return F


def add_eicu_medication_counts(F, root, windows):
    p = find_eicu_table(root, ["medication"])
    if p is None:
        print("  SKIP medication: table not found")
        return F

    df = lower_columns(read_csv_any(p))
    id_col = pick_col(df, ["patientunitstayid"], required=True, table="medication")
    offset_col = pick_col(df, ["drugstartoffset", "medicationoffset", "treatmentoffset"], required=False, table="medication")
    name_col = pick_col(df, ["drugname", "medicationname"], required=False, table="medication")
    if offset_col is None or name_col is None:
        print("  SKIP medication: missing offset or drug name")
        return F

    df = df[[id_col, offset_col, name_col]].copy().rename(columns={id_col: "patientunitstayid", offset_col: "offset_min", name_col: "drug"})
    df["patientunitstayid"] = normalize_join_id_series(df["patientunitstayid"])
    df["offset_min"] = safe_num(df["offset_min"])
    df["hours_from_t0"] = df["offset_min"] / 60.0
    df["drug"] = df["drug"].astype(str).str.upper()
    df = df[(df["hours_from_t0"] >= 0) & (df["hours_from_t0"] <= max(windows))]
    if df.empty:
        print("  SKIP medication: no early rows")
        return F

    def category(drug):
        drug = "" if pd.isna(drug) else str(drug).upper()
        if any(k in drug for k in ["NOREPINEPHRINE", "LEVOPHED", "EPINEPHRINE", "VASOPRESSIN", "PHENYLEPHRINE", "DOPAMINE", "DOBUTAMINE"]):
            return "vasopressor"
        if any(k in drug for k in ["VANCOMYCIN", "CEF", "PIPERACILLIN", "TAZOBACTAM", "MEROPENEM", "IMIPENEM", "CIPRO", "LEVO", "METRONIDAZOLE", "GENTAMICIN", "AZITHRO", "PENICILLIN"]):
            return "antibiotic"
        if any(k in drug for k in ["FLUCONAZOLE", "MICAFUNGIN", "ANIDULAFUNGIN", "CASPOFUNGIN", "AMPHOTERICIN", "VORICONAZOLE"]):
            return "antifungal"
        return "other_med"

    df["drug"] = df["drug"].fillna("").astype(str).str.upper()
    df["med_cat"] = df["drug"].apply(category)
    for wh in windows:
        sub = df[df["hours_from_t0"] <= wh].copy()
        if sub.empty:
            continue
        cnt = sub.groupby(["patientunitstayid", "med_cat"]).size().reset_index(name="count")
        wide = cnt.pivot(index="patientunitstayid", columns="med_cat", values="count").fillna(0)
        wide.columns = [f"med_{c}_w{wh}h_count" for c in wide.columns]
        F = safe_merge_patientunitstayid(F, wide.reset_index(), how="left")
    return F


def run_eicu_eval(args) -> int:
    root = Path(args.dir)
    windows = sorted(set(int(x.strip()) for x in args.windows.split(",") if x.strip()))

    banner("PenuX — eICU/eICU-like Leakage-Aware Evaluation")
    print("eICU-like dir:", root)
    print("Windows      :", windows)
    print("Rule         : organism/microbiology text is labels only, not features")

    print("\n[1/6] Patient table")
    F = eicu_patient_features(root)
    print("  Base features:", F.shape)

    print("\n[2/6] Microbiology labels")
    labels = build_eicu_micro_labels(root, F["patientunitstayid"].unique())
    F["patientunitstayid"] = normalize_join_id_series(F["patientunitstayid"])
    if "patientunitstayid" in labels.columns:
        labels["patientunitstayid"] = normalize_join_id_series(labels["patientunitstayid"])
    F = safe_merge_patientunitstayid(F, labels, how="left")
    for label_col in ["label_3class", "label_species"]:
        if label_col in F.columns:
            print(f"\n  {label_col}:")
            print(F[label_col].value_counts(dropna=False).head(30).to_string())

    print("\n[3/6] Adding lab summary features")
    F = add_eicu_long_table_summary(
        F, root, ["lab"], ["patientunitstayid"], ["labresultoffset", "laboffset", "resultoffset"],
        ["labname"], ["labresult", "labresulttext", "labvalue"], "lab", windows
    )

    print("\n[4/6] Adding vital summary features")
    F = add_eicu_wide_vitals_summary(F, root, ["vitalPeriodic", "vitalperiodic"], "vital_periodic", windows)
    F = add_eicu_wide_vitals_summary(F, root, ["vitalAperiodic", "vitalaperiodic"], "vital_aperiodic", windows)

    print("\n[5/6] Adding optional nurseCharting and medication features")
    F = add_eicu_long_table_summary(
        F, root, ["nurseCharting", "nursecharting"], ["patientunitstayid"],
        ["nursingchartoffset", "chartoffset", "observationoffset"],
        ["nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartcelltypecat"],
        ["nursingchartvalue", "nursingchartvaluefloat"], "nurse", windows
    )
    F = add_eicu_medication_counts(F, root, windows)

    print("\n[6/6] Saving feature matrix")
    out = Path(args.output_csv)
    if str(out):
        out.parent.mkdir(parents=True, exist_ok=True)
        compression = "gzip" if str(out).endswith(".gz") else None
        F.to_csv(out, index=False, compression=compression)
        print("  Saved CSV:", out)

    print("  Final matrix:", F.shape)
    print("  First columns:")
    for c in list(F.columns[:40]):
        print("   ", c)
    if len(F.columns) > 40:
        print(f"    ... {len(F.columns) - 40} more")

    if "label_3class" in F.columns and F["label_3class"].notna().sum() >= 5:
        evaluate_label(F, "label_3class", args.max_splits, args.min_class_count)
    else:
        print("\nSKIP label_3class evaluation: no usable microbiology labels")

    if "label_species" in F.columns and F["label_species"].notna().sum() >= 5:
        evaluate_label(F, "label_species", args.max_splits, args.min_class_count)
    else:
        print("\nSKIP label_species evaluation: no usable microbiology labels")

    banner("DONE")
    return 0


def run_all(args) -> int:
    convert_args = argparse.Namespace(mimiciii_dir=args.mimiciii_dir, mimiciv_dir=args.mimiciv_dir, out=args.out)
    run_mimic_to_eicu(convert_args)

    eval_args = argparse.Namespace(
        dir=args.out,
        windows=args.windows,
        output_csv=args.output_csv,
        max_splits=args.max_splits,
        min_class_count=args.min_class_count,
    )
    return run_eicu_eval(eval_args)


# =============================================================================
# CLI dispatcher
# =============================================================================

def build_cli():
    p = argparse.ArgumentParser(description="PenuX all-in-one MIMIC/eICU schema adapter and leakage-aware evaluator")
    sub = p.add_subparsers(dest="mode", required=True)

    p_mimic = sub.add_parser("mimic-eval", help="Run direct MIMIC-III/IV leakage-aware evaluation")
    p_mimic.add_argument("--dir", required=True, help="MIMIC-III/IV directory containing CSV/CSV.GZ tables")
    p_mimic.add_argument("--windows", default="6,12")
    p_mimic.add_argument("--chunksize", type=int, default=500_000)
    p_mimic.add_argument("--top-species", type=int, default=9)
    p_mimic.add_argument("--max-splits", type=int, default=5)
    p_mimic.add_argument("--min-class-count", type=int, default=2)
    p_mimic.add_argument("--output-csv", default="")
    p_mimic.set_defaults(func=run_mimic_eval)

    p_conv = sub.add_parser("mimic-to-eicu", help="Convert MIMIC-III/IV Demo to eICU-like schema")
    p_conv.add_argument("--mimiciii-dir", default=None)
    p_conv.add_argument("--mimiciv-dir", default=None)
    p_conv.add_argument("--out", default="dataset/eicu_like_from_mimic_demo")
    p_conv.set_defaults(func=run_mimic_to_eicu)

    p_eicu = sub.add_parser("eicu-eval", help="Run eICU/eICU-like leakage-aware evaluation")
    p_eicu.add_argument("--dir", required=True)
    p_eicu.add_argument("--windows", default="6,12")
    p_eicu.add_argument("--output-csv", default="data/processed/eicu_penux_features.csv.gz")
    p_eicu.add_argument("--max-splits", type=int, default=5)
    p_eicu.add_argument("--min-class-count", type=int, default=2)
    p_eicu.set_defaults(func=run_eicu_eval)

    p_all = sub.add_parser("all", help="Convert MIMIC Demo to eICU-like schema, then evaluate")
    p_all.add_argument("--mimiciii-dir", default=None)
    p_all.add_argument("--mimiciv-dir", default=None)
    p_all.add_argument("--out", default="dataset/eicu_like_from_mimic_demo")
    p_all.add_argument("--windows", default="6,12")
    p_all.add_argument("--output-csv", default="data/processed/eicu_like_from_mimic_features.csv.gz")
    p_all.add_argument("--max-splits", type=int, default=5)
    p_all.add_argument("--min-class-count", type=int, default=2)
    p_all.set_defaults(func=run_all)

    return p


def main(argv=None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
