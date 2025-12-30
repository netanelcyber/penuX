#!/usr/bin/env python3
# mimic_resistance_pipeline_onefile_torch.py
#
# ✅ ONE FILE
# ✅ NO PANDAS
# ✅ AUTO DATASET DISCOVERY (MIMIC-III / MIMIC-IV demo layouts)
# ✅ FULLY WITHOUT TENSORFLOW (PyTorch only)
# ✅ HYBRID NN: DNN + Conv1D (ConvNN) + RNN + BiLSTM (LSTM) in ONE model
# ✅ STOP when TARGET_ACC_KIND >= TARGET_ACC (default 0.95) OR use F1 stop
# ✅ MAX_EPOCHS default 100000 (upper bound; will stop early)
# ✅ OPTIONAL retrain on FULL TRAIN (TRAIN+VAL) for best epoch
#
# ✅ EXTRA EVAL ADDED:
#    - Confusion matrix + TP/FP/FN/TN + Sensitivity/Specificity/PPV/F1 (OvR)
#    - ROC-AUC / PR-AUC (OvR per-class + macro/weighted)
#    - ROC & PR CURVES PNG (OvR per-class + macro curve)   <-- ADDED
#    - Calibration: ECE + Brier + reliability table (+ PNG if matplotlib)
#    - Bias checks by subgroups: gender / age bins / admission_type / admission_location (best-effort)
#
# IMPORTANT: Research/demo only. Not for clinical use.

import csv
import gzip
import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None  # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,                # <-- ADDED
    precision_recall_curve,   # <-- ADDED
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ===============================
# FIXED CLASS ORDER (CAPITALS)
# ===============================
CLASSES = [
    "B:PSEUDOMONAS AERUGINOSA",
    "B:STAPH AUREUS COAG +",
    "B:SERRATIA MARCESCENS",
    "B:MORGANELLA MORGANII",
    "B:ESCHERICHIA COLI",
    "B:PROTEUS MIRABILIS",
    "B:PROVIDENCIA STUARTII",
    "B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS",
    "B:YEAST",
    "B:GRAM POSITIVE COCCUS(COCCI)",
    "B:OTHER",
    "V:OTHER",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
INDEX_TO_CLASS = {i: c for c, i in CLASS_TO_INDEX.items()}

MRSA_LABEL = "B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS"
MSSA_LABEL = "B:STAPH AUREUS COAG +"


# ===============================
# PATHS (resolved per dataset run)
# ===============================
MICRO_PATH: Path
PRESC_PATH: Path
ADMISSIONS_PATH: Path
PATIENTS_PATH: Path
D_ITEMS_PATH: Path
CHARTEVENTS_PATH: Path
D_LABITEMS_PATH: Path
LABEVENTS_PATH: Path


# ===============================
# SETTINGS
# ===============================
SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)

HOURS_WINDOW = int(os.environ.get("HOURS_WINDOW", "24"))

ANTIBIOTICS = ["VANCOMYCIN", "CIPROFLOXACIN", "MEROPENEM", "PIPERACILLIN", "CEFTRIAXONE"]
ABX_ORDER = [a.lower() for a in ANTIBIOTICS]  # lowercase feature names (columns)

VITAL_ORDER = ["temperature_c", "wbc", "spo2", "age"]  # required order
NUMERIC_ORDER = VITAL_ORDER + ABX_ORDER

USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "1").strip() == "1"
MAX_CLASS_WEIGHT = float(os.environ.get("MAX_CLASS_WEIGHT", "15.0"))
BOTHER_EXTRA_DOWNWEIGHT = float(os.environ.get("BOTHER_EXTRA_DOWNWEIGHT", "0.5"))

DROPOUT = float(os.environ.get("DROPOUT", "0.25"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

# 100000 epoch upper bound (still stops early)
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "100000"))
EARLY_PATIENCE = int(os.environ.get("EARLY_PATIENCE", "360"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-7"))

WBC_SAMPLE_MAX = 200_000
MIN_TEST_UNIQUE_HADM = int(os.environ.get("MIN_TEST_UNIQUE_HADM", "2"))

# ---- Stop metric ----
TARGET_STOP_METRIC = os.environ.get("TARGET_STOP_METRIC", "acc").strip().lower()  # "acc" | "f1"

TARGET_ACC = float(os.environ.get("TARGET_ACC", "0.95"))
TARGET_ACC_KIND = os.environ.get("TARGET_ACC_KIND", "overall").strip().lower()   # "overall" | "mrsa_mssa"

TARGET_F1 = float(os.environ.get("TARGET_F1", "0.925"))
TARGET_F1_KIND = os.environ.get("TARGET_F1_KIND", "macro").strip().lower()      # "macro" | "weighted" | "mrsa_mssa"

MAX_TRAIN_RESTARTS = int(os.environ.get("MAX_TRAIN_RESTARTS", "5"))
RETRAIN_ON_FULL_TRAIN = os.environ.get("RETRAIN_ON_FULL_TRAIN", "1").strip() == "1"

# ---- Activation flexibility (DNN parts) ----
_env_acts = os.environ.get("MIMIC_ACTIVATIONS", "").strip()
if _env_acts:
    ACTIVATION_CANDIDATES = [a.strip().lower() for a in _env_acts.split(",") if a.strip()]
else:
    ACTIVATION_CANDIDATES = ["gelu", "swish", "elu", "relu"]

# ---- Text / sequence knobs ----
MAX_TEXT_TOKENS = int(os.environ.get("MAX_TEXT_TOKENS", "20000"))  # includes PAD/UNK
TEXT_SEQ_LEN = int(os.environ.get("TEXT_SEQ_LEN", "64"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "96"))
CNN_FILTERS = int(os.environ.get("CNN_FILTERS", "128"))
CNN_KERNEL = int(os.environ.get("CNN_KERNEL", "5"))
RNN_UNITS = int(os.environ.get("RNN_UNITS", "64"))
LSTM_UNITS = int(os.environ.get("LSTM_UNITS", "64"))

# ---- Optimizer ----
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))

# ---- Loss function (FIX) ----
# Options: "ce" | "wce" | "focal" | "cb_focal"
LOSS_NAME = os.environ.get("LOSS_NAME", "cb_focal").strip().lower()
LABEL_SMOOTHING = float(os.environ.get("LOSS_LABEL_SMOOTHING", "0.05"))
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
CB_BETA = float(os.environ.get("CB_BETA", "0.9999"))
FOCAL_USE_ALPHA = os.environ.get("FOCAL_USE_ALPHA", "1").strip() == "1"

DEVICE = os.environ.get("DEVICE", "").strip().lower()
if DEVICE not in {"", "cpu", "cuda"}:
    DEVICE = ""


# ===============================
# CSV helpers (no pandas) + .csv.gz support
# ===============================
def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _norm_text(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _open_text(path: Path, encoding: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding=encoding)
    return open(path, "r", newline="", encoding=encoding)


def _read_header(path: Path) -> List[str]:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with _open_text(path, enc) as f:
                r = csv.reader(f)
                header = next(r)
                return [h.strip() for h in header]
        except Exception:
            continue
    raise RuntimeError(f"Could not read header for {path}")


def _find_col_index(actual_cols: List[str], candidates: List[str]) -> int:
    canon_map = {_canon(c): i for i, c in enumerate(actual_cols)}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_map:
            return canon_map[key]
    raise ValueError(f"Missing columns {candidates}. Found sample: {actual_cols[:50]} ...")


def _resolve_usecols_idx(path: Path, wanted: Dict[str, List[str]]) -> Dict[str, int]:
    header = _read_header(path)
    resolved: Dict[str, int] = {}
    for std_name, cands in wanted.items():
        resolved[std_name] = _find_col_index(header, cands)
    return resolved


def _iter_csv_std(path: Path, wanted: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
    idx = _resolve_usecols_idx(path, wanted)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with _open_text(path, enc) as f:
                r = csv.reader(f)
                _ = next(r)  # header
                for row in r:
                    if not row:
                        continue
                    out = {}
                    for k, j in idx.items():
                        out[k] = row[j].strip() if j < len(row) else ""
                    yield out
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not read CSV {path}")


# ---- Optional columns (best-effort) for Bias meta ----
def _resolve_usecols_idx_optional(path: Path, wanted: Dict[str, List[str]]) -> Dict[str, int]:
    header = _read_header(path)
    canon_map = {_canon(c): i for i, c in enumerate(header)}
    resolved: Dict[str, int] = {}
    for std_name, cands in wanted.items():
        found = None
        for cand in cands:
            key = _canon(cand)
            if key in canon_map:
                found = canon_map[key]
                break
        if found is not None:
            resolved[std_name] = int(found)
    return resolved


def _iter_csv_optional(path: Path, wanted: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
    idx = _resolve_usecols_idx_optional(path, wanted)
    if not idx:
        return
        yield  # pragma: no cover
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with _open_text(path, enc) as f:
                r = csv.reader(f)
                _ = next(r)  # header
                for row in r:
                    if not row:
                        continue
                    out = {}
                    for k, j in idx.items():
                        out[k] = row[j].strip() if j < len(row) else ""
                    yield out
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not read CSV {path}")


def _parse_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


_DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S.%f",
)


def _safe_parse_datetime_str(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "nat"}:
        return None

    m = re.match(r"^\s*(\d{4})", s)
    if m:
        try:
            year = int(m.group(1))
            if year >= 3000:
                return None
        except Exception:
            pass

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


# ===============================
# Dataset discovery / path resolution
# ===============================
def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {[str(x) for x in paths]}")


def _looks_like_mimic4_root(root: Path) -> bool:
    return (root / "hosp").exists() and (root / "icu").exists()


def _looks_like_mimic3_root(root: Path) -> bool:
    candidates = [
        root / "ADMISSIONS.csv",
        root / "ADMISSIONS.csv.gz",
        root / "admissions.csv",
        root / "admissions.csv.gz",
    ]
    return any(p.exists() for p in candidates)


def _sanitize_tag(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def resolve_paths(data_root: Path) -> Tuple[Dict[str, Path], str]:
    if _looks_like_mimic4_root(data_root):
        hosp = data_root / "hosp"
        icu = data_root / "icu"

        def pick(dirp: Path, base: str) -> Path:
            return _first_existing(
                [
                    dirp / f"{base}.csv",
                    dirp / f"{base}.csv.gz",
                    dirp / f"{base.upper()}.csv",
                    dirp / f"{base.upper()}.csv.gz",
                ]
            )

        return (
            {
                "micro": pick(hosp, "microbiologyevents"),
                "presc": pick(hosp, "prescriptions"),
                "admissions": pick(hosp, "admissions"),
                "patients": pick(hosp, "patients"),
                "d_items": pick(icu, "d_items"),
                "chartevents": pick(icu, "chartevents"),
                "d_labitems": pick(hosp, "d_labitems"),
                "labevents": pick(hosp, "labevents"),
            },
            "mimic4",
        )

    def pick(base: str) -> Path:
        return _first_existing(
            [
                data_root / f"{base}.csv",
                data_root / f"{base}.csv.gz",
                data_root / f"{base.upper()}.csv",
                data_root / f"{base.upper()}.csv.gz",
            ]
        )

    return (
        {
            "micro": pick("microbiologyevents"),
            "presc": pick("prescriptions"),
            "admissions": pick("admissions"),
            "patients": pick("patients"),
            "d_items": pick("d_items"),
            "chartevents": pick("chartevents"),
            "d_labitems": pick("d_labitems"),
            "labevents": pick("labevents"),
        },
        "mimic3",
    )


def discover_dataset_roots() -> List[Path]:
    roots: List[Path] = []

    env = os.environ.get("MIMIC_AUTOROOTS", "").strip()
    if env:
        for part in env.split(","):
            p = Path(part.strip())
            if p.exists() and p.is_dir():
                roots.append(p)

    p_m4 = Path(
        "datasets/datasets/montassarba/mimic-iv-clinical-database-demo-2-2/versions/1/mimic-iv-clinical-database-demo-2.2"
    )
    if p_m4.exists() and p_m4.is_dir():
        roots.append(p_m4)

    p_base = Path("dataset/mimic")
    if p_base.exists() and p_base.is_dir():
        roots.append(p_base)
        for cand in list(p_base.glob("*")) + list(p_base.glob("*/*")):
            if cand.is_dir():
                if _looks_like_mimic4_root(cand) or _looks_like_mimic3_root(cand):
                    roots.append(cand)

    seen = set()
    out: List[Path] = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(r)
    return out


# ===============================
# MAP ORGANISM -> CLASS (RETURNS CAPITALS)
# ===============================
def map_org(org: str) -> str:
    o = str(org).upper()
    viral_keys = ["VIRUS", "INFLUENZA", "RSV", "ADENOVIRUS", "PARAINFLUENZA", "CORONAVIRUS", "SARS", "COV"]
    if any(k in o for k in viral_keys):
        return "V:OTHER"

    if "PSEUDOMONAS AERUGINOSA" in o:
        return "B:PSEUDOMONAS AERUGINOSA"
    if "STAPH AUREUS" in o and "METHICILLIN" not in o and "MRSA" not in o:
        return "B:STAPH AUREUS COAG +"
    if "SERRATIA MARCESCENS" in o:
        return "B:SERRATIA MARCESCENS"
    if "MORGANELLA MORGANII" in o:
        return "B:MORGANELLA MORGANII"
    if "ESCHERICHIA COLI" in o or "E. COLI" in o:
        return "B:ESCHERICHIA COLI"
    if "PROTEUS MIRABILIS" in o:
        return "B:PROTEUS MIRABILIS"
    if "PROVIDENCIA STUARTII" in o:
        return "B:PROVIDENCIA STUARTII"
    if "MRSA" in o or "METHICILLIN" in o:
        return "B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS"
    if "YEAST" in o or "CANDIDA" in o:
        return "B:YEAST"
    if "COCCUS" in o or "COCCI" in o:
        return "B:GRAM POSITIVE COCCUS(COCCI)"
    return "B:OTHER"


# ===============================
# ITEMID SETS: Temperature(C/F), SpO2, WBC
# ===============================
def build_itemid_sets() -> Tuple[set[int], set[int], set[int], set[int]]:
    d_items_wanted = {"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]}
    temp_c_ids: set[int] = set()
    temp_f_ids: set[int] = set()
    spo2_ids: set[int] = set()

    for row in _iter_csv_std(D_ITEMS_PATH, d_items_wanted):
        itemid = _parse_int(row["ITEMID"])
        label = row["LABEL"]
        if itemid is None:
            continue
        ll = str(label).lower()
        if "temperature" in ll and "c" in ll:
            temp_c_ids.add(itemid)
        if "temperature" in ll and "f" in ll:
            temp_f_ids.add(itemid)
        if "spo2" in ll:
            spo2_ids.add(itemid)

    if len(spo2_ids) == 0:
        for row in _iter_csv_std(D_ITEMS_PATH, d_items_wanted):
            itemid = _parse_int(row["ITEMID"])
            label = row["LABEL"]
            if itemid is None:
                continue
            ll = str(label).lower()
            if ("o2" in ll) and ("saturation" in ll):
                spo2_ids.add(itemid)

    d_lab_wanted = {"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]}
    wbc_ids: set[int] = set()
    wbc_re = re.compile(r"\bwbc\b", re.IGNORECASE)
    for row in _iter_csv_std(D_LABITEMS_PATH, d_lab_wanted):
        itemid = _parse_int(row["ITEMID"])
        label = row["LABEL"]
        if itemid is None:
            continue
        ll = str(label).lower()
        if wbc_re.search(ll) or ("white blood" in ll):
            wbc_ids.add(itemid)

    return temp_c_ids, temp_f_ids, spo2_ids, wbc_ids


# ===============================
# Build admission windows + age
# ===============================
def build_adm_windows_for_hadm_set(hadm_set: set[int]) -> Tuple[Dict[int, Tuple[datetime, datetime]], Dict[int, float]]:
    adm_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
        "ADMITTIME": ["ADMITTIME", "ADMIT_TIME", "ADMIT TIME"],
    }
    admissions: Dict[int, Tuple[int, datetime]] = {}
    for row in _iter_csv_std(ADMISSIONS_PATH, adm_wanted):
        hadm = _parse_int(row["HADM_ID"])
        sid = _parse_int(row["SUBJECT_ID"])
        adt = _safe_parse_datetime_str(row["ADMITTIME"])
        if hadm is None or sid is None or adt is None:
            continue
        if hadm in hadm_set:
            admissions[hadm] = (sid, adt)

    patients_header = [h.lower() for h in _read_header(PATIENTS_PATH)]
    has_dob = any(_canon(h) == _canon("dob") for h in patients_header)
    has_anchor = any(_canon(h) == _canon("anchor_age") for h in patients_header) and any(
        _canon(h) == _canon("anchor_year") for h in patients_header
    )

    dob_by_subject: Dict[int, datetime] = {}
    anchor_by_subject: Dict[int, Tuple[float, int]] = {}

    if has_dob:
        pat_wanted = {"SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"], "DOB": ["DOB", "DATE_OF_BIRTH", "DATE OF BIRTH"]}
        for row in _iter_csv_std(PATIENTS_PATH, pat_wanted):
            sid = _parse_int(row["SUBJECT_ID"])
            dob = _safe_parse_datetime_str(row["DOB"])
            if sid is None or dob is None:
                continue
            dob_by_subject[sid] = dob
    elif has_anchor:
        pat_wanted = {
            "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
            "ANCHOR_AGE": ["ANCHOR_AGE", "ANCHORAGE"],
            "ANCHOR_YEAR": ["ANCHOR_YEAR", "ANCHORYEAR"],
        }
        for row in _iter_csv_std(PATIENTS_PATH, pat_wanted):
            sid = _parse_int(row["SUBJECT_ID"])
            aa = _parse_float(row["ANCHOR_AGE"])
            ay = _parse_int(row["ANCHOR_YEAR"])
            if sid is None or aa is None or ay is None:
                continue
            anchor_by_subject[sid] = (float(aa), int(ay))
    else:
        raise RuntimeError("PATIENTS schema not recognized (no DOB and no anchor_age/anchor_year).")

    windows: Dict[int, Tuple[datetime, datetime]] = {}
    ages: Dict[int, float] = {}
    for hadm, (sid, adt) in admissions.items():
        if has_dob:
            dob = dob_by_subject.get(sid)
            if dob is None:
                continue
            age = float((adt - dob).days) / 365.2425
        else:
            anc = anchor_by_subject.get(sid)
            if anc is None:
                continue
            anchor_age, anchor_year = anc
            age = float(anchor_age) + float(adt.year - int(anchor_year))

        if not np.isfinite(age):
            continue
        if age > 120.0:
            age = 90.0
        age = float(np.clip(age, 0.0, 110.0))
        windows[hadm] = (adt, adt + timedelta(hours=int(HOURS_WINDOW)))
        ages[hadm] = age

    return windows, ages


# ===============================
# Compute vitals/labs per HADM_ID (no pandas)
# ===============================
def compute_vitals_features(hadm_set: set[int]) -> Dict[int, Dict[str, float]]:
    windows, ages = build_adm_windows_for_hadm_set(hadm_set)
    if len(windows) == 0:
        raise RuntimeError("No admission windows found for hadm_set.")

    temp_c_ids, temp_f_ids, spo2_ids, wbc_ids = build_itemid_sets()
    want_chart_itemids = temp_c_ids | temp_f_ids | spo2_ids

    ce_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "ITEMID": ["ITEMID", "ITEM_ID"],
        "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
        "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
    }

    temp_sum: Dict[int, float] = {}
    temp_n: Dict[int, int] = {}
    spo2_min: Dict[int, float] = {}

    for row in _iter_csv_std(CHARTEVENTS_PATH, ce_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or hadm not in windows or itemid not in want_chart_itemids:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None:
            continue
        t0, t1 = windows[hadm]
        if ct < t0 or ct > t1:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        if itemid in temp_c_ids or itemid in temp_f_ids:
            temp_c = float(v) if itemid in temp_c_ids else (float(v) - 32.0) / 1.8
            if temp_c < 30.0 or temp_c > 45.0:
                continue
            temp_sum[hadm] = temp_sum.get(hadm, 0.0) + temp_c
            temp_n[hadm] = temp_n.get(hadm, 0) + 1
        elif itemid in spo2_ids:
            spo2 = float(v)
            if spo2 < 50.0 or spo2 > 100.0:
                continue
            prev = spo2_min.get(hadm)
            spo2_min[hadm] = spo2 if (prev is None or spo2 < prev) else prev

    temp_mean: Dict[int, float] = {}
    for hadm, s in temp_sum.items():
        n = temp_n.get(hadm, 0)
        if n > 0:
            temp_mean[hadm] = s / float(n)

    le_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "ITEMID": ["ITEMID", "ITEM_ID"],
        "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
        "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
    }

    # pass1: reservoir sample for scaling check
    sample: List[float] = []
    rng = np.random.default_rng(SEED)
    seen = 0

    for row in _iter_csv_std(LABEVENTS_PATH, le_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or hadm not in windows or itemid not in wbc_ids:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None:
            continue
        t0, t1 = windows[hadm]
        if ct < t0 or ct > t1:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        seen += 1
        if len(sample) < WBC_SAMPLE_MAX:
            sample.append(float(v))
        else:
            j = int(rng.integers(0, seen))
            if j < WBC_SAMPLE_MAX:
                sample[j] = float(v)

    scale_by_1000 = False
    if len(sample) > 0:
        med = float(np.median(np.asarray(sample, dtype=np.float64)))
        if med < 200.0:
            scale_by_1000 = True

    # pass2: max WBC in window
    wbc_max: Dict[int, float] = {}
    for row in _iter_csv_std(LABEVENTS_PATH, le_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or hadm not in windows or itemid not in wbc_ids:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None:
            continue
        t0, t1 = windows[hadm]
        if ct < t0 or ct > t1:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        w = float(v) * (1000.0 if scale_by_1000 else 1.0)
        if w < 2000.0 or w > 40000.0:
            continue
        prev = wbc_max.get(hadm)
        wbc_max[hadm] = w if (prev is None or w > prev) else prev

    out: Dict[int, Dict[str, float]] = {}
    for hadm in hadm_set:
        tm = temp_mean.get(hadm)
        sp = spo2_min.get(hadm)
        wb = wbc_max.get(hadm)
        ag = ages.get(hadm)
        if tm is None or sp is None or wb is None or ag is None:
            continue
        out[hadm] = {"temperature_c": float(tm), "wbc": float(wb), "spo2": float(sp), "age": float(ag)}
    return out


# ===============================
# Load MICROBIOLOGYEVENTS -> rows (no pandas)
# ===============================
def load_micro_rows() -> List[Dict[str, Any]]:
    micro_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "SPEC_TYPE_DESC": ["SPEC_TYPE_DESC", "SPECIMEN", "SPECIMEN_TYPE", "SPEC_TYPE"],
        "ORG_NAME": ["ORG_NAME", "ORGANISM", "ORGNAME", "ORG NAME"],
        "INTERPRETATION": ["INTERPRETATION", "RESULT", "COMMENTS", "COMMENT"],
    }
    rows: List[Dict[str, Any]] = []
    for row in _iter_csv_std(MICRO_PATH, micro_wanted):
        hadm = _parse_int(row["HADM_ID"])
        org = row["ORG_NAME"]
        if hadm is None or str(org).strip() == "":
            continue

        label = map_org(org)
        if label not in CLASS_TO_INDEX:
            continue

        spec = _norm_text(row["SPEC_TYPE_DESC"])
        interp = _norm_text(row["INTERPRETATION"])
        if spec == "" or interp == "":
            continue

        rows.append({"hadm_id": int(hadm), "spec_type_desc": spec, "interpretation": interp, "label": label})
    return rows


# ===============================
# Load PRESCRIPTIONS -> hadm_id -> antibiotics binary (no pandas)
# ===============================
def load_abx_features(hadm_set: set[int]) -> Dict[int, Dict[str, float]]:
    presc_wanted = {"HADM_ID": ["HADM_ID", "HADMID"], "DRUG": ["DRUG", "DRUG_NAME", "MEDICATION"]}
    wanted_upper = set(a.upper() for a in ANTIBIOTICS)
    hadm_to_drugs: Dict[int, set[str]] = defaultdict(set)

    for row in _iter_csv_std(PRESC_PATH, presc_wanted):
        hadm = _parse_int(row["HADM_ID"])
        if hadm is None or hadm not in hadm_set:
            continue
        drug = str(row["DRUG"]).strip()
        if drug == "":
            continue
        if drug.upper() in wanted_upper:
            hadm_to_drugs[int(hadm)].add(drug.upper())

    out: Dict[int, Dict[str, float]] = {}
    for hadm in hadm_set:
        feats = {abx: 0.0 for abx in ABX_ORDER}
        for abx_u in hadm_to_drugs.get(hadm, set()):
            feats[abx_u.lower()] = 1.0
        out[hadm] = feats
    return out


# ===============================
# Tokenization (simple whitespace vocab)
# ===============================
def _tokenize(text: str) -> List[str]:
    return [t for t in str(text).strip().lower().split() if t]


def build_vocab(texts: np.ndarray, max_tokens: int) -> Dict[str, int]:
    # reserve: 0=PAD, 1=UNK
    cnt = Counter()
    for s in texts.tolist():
        for t in _tokenize(str(s)):
            cnt[t] += 1
    keep = max(2, int(max_tokens))
    vocab_items = cnt.most_common(max(0, keep - 2))
    vocab = {}
    idx = 2
    for tok, _ in vocab_items:
        vocab[tok] = idx
        idx += 1
    return vocab


def texts_to_ids(texts: np.ndarray, vocab: Dict[str, int], seq_len: int) -> np.ndarray:
    out = np.zeros((texts.shape[0], int(seq_len)), dtype=np.int64)
    for i, s in enumerate(texts.tolist()):
        toks = _tokenize(str(s))
        ids = [vocab.get(t, 1) for t in toks]  # 1=UNK
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        out[i, :len(ids)] = np.asarray(ids, dtype=np.int64)
    return out


# ===============================
# Metrics (stop metrics + base reporting)
# ===============================
def _compute_target_f1(kind: str, y_true_int: np.ndarray, y_pred_int: np.ndarray) -> Optional[float]:
    kind = str(kind).strip().lower()
    if kind in {"macro", "weighted"}:
        avg = "macro" if kind == "macro" else "weighted"
        f1 = precision_recall_fscore_support(
            y_true_int,
            y_pred_int,
            labels=np.arange(len(CLASSES)),
            average=avg,
            zero_division=0,
        )[2]
        return float(f1)

    if kind == "mrsa_mssa":
        mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
        mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
        if mrsa_idx is None or mssa_idx is None:
            return None
        mask = np.isin(y_true_int, [mrsa_idx, mssa_idx])
        if int(np.sum(mask)) == 0:
            return None
        yt = y_true_int[mask]
        yp = y_pred_int[mask]
        ytb = (yt == mrsa_idx).astype(np.int32)
        ypb = (yp == mrsa_idx).astype(np.int32)
        f1 = precision_recall_fscore_support(ytb, ypb, labels=[0, 1], average="binary", zero_division=0)[2]
        return float(f1)

    return None


def _compute_target_acc(kind: str, y_true_int: np.ndarray, y_pred_int: np.ndarray) -> Optional[float]:
    kind = str(kind).strip().lower()
    if kind in {"overall", "micro"}:
        if y_true_int.size == 0:
            return None
        return float(np.mean((y_true_int == y_pred_int).astype(np.float32)))

    if kind == "mrsa_mssa":
        mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
        mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
        if mrsa_idx is None or mssa_idx is None:
            return None
        mask = np.isin(y_true_int, [mrsa_idx, mssa_idx])
        n = int(np.sum(mask))
        if n == 0:
            return None
        yt = y_true_int[mask]
        yp = y_pred_int[mask]
        return float(np.mean((yt == yp).astype(np.float32)))

    return None


def report_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASSES)), average=None, zero_division=0
    )
    print("\n=== PER-CLASS METRICS (TEST) ===")
    print("Class                                                   |  Prec   Rec    F1   Support")
    print("-" * 85)
    for i in range(len(CLASSES)):
        print(f"{INDEX_TO_CLASS[i]:55s} | {pr[i]:6.3f} {rc[i]:6.3f} {f1[i]:6.3f} {int(sup[i]):8d}")

    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wavg = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print("\n[INFO] Macro avg:    prec={:.3f} rec={:.3f} f1={:.3f}".format(macro[0], macro[1], macro[2]))
    print("[INFO] Weighted avg: prec={:.3f} rec={:.3f} f1={:.3f}".format(wavg[0], wavg[1], wavg[2]))


def report_mrsa_vs_mssa(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
    mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
    if mrsa_idx is None or mssa_idx is None:
        print("[WARN] MRSA/MSSA class indices missing; skipping dedicated report.")
        return

    mask = np.isin(y_true, [mrsa_idx, mssa_idx])
    n = int(np.sum(mask))
    if n == 0:
        print("\n=== MRSA vs MSSA (TEST) ===")
        print("[WARN] No MRSA/MSSA samples in TEST; cannot compute.")
        return

    yt = y_true[mask]
    yp = y_pred[mask]
    ytb = (yt == mrsa_idx).astype(np.int32)
    ypb = (yp == mrsa_idx).astype(np.int32)

    pr, rc, f1, sup = precision_recall_fscore_support(ytb, ypb, labels=[0, 1], average=None, zero_division=0)
    cm = confusion_matrix(ytb, ypb, labels=[0, 1])

    print("\n=== MRSA vs MSSA (STAPH AUREUS COAG +) — TEST SUBSET ===")
    print(f"[INFO] Subset size: {n} (MSSA={int(sup[0])}, MRSA={int(sup[1])})")
    print("Label |  Prec   Rec    F1   Support")
    print(f"MSSA  | {pr[0]:6.3f} {rc[0]:6.3f} {f1[0]:6.3f} {int(sup[0]):8d}")
    print(f"MRSA  | {pr[1]:6.3f} {rc[1]:6.3f} {f1[1]:6.3f} {int(sup[1]):8d}")
    print("\nConfusion matrix (rows=true, cols=pred), [MSSA, MRSA]:")
    print(cm)


# ===============================
# EXTRA EVAL: Confusion, Sens/Spec/PPV/F1, ROC-AUC/PR-AUC, ROC/PR CURVES, Calibration, Bias
# ===============================
CALIB_BINS = int(os.environ.get("CALIB_BINS", "10"))
BIAS_MIN_GROUP_N = int(os.environ.get("BIAS_MIN_GROUP_N", "25"))


def _confusion_stats_multiclass(cm: np.ndarray) -> Dict[int, Dict[str, float]]:
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    stats: Dict[int, Dict[str, float]] = {}
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1   = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0
        sup  = int(cm[i, :].sum())

        stats[i] = {
            "TP": float(tp), "FP": float(fp), "FN": float(fn), "TN": float(tn),
            "Sensitivity/Recall": float(sens),
            "Specificity": float(spec),
            "Precision/PPV": float(ppv),
            "NPV": float(npv),
            "F1": float(f1),
            "Support": float(sup),
        }
    return stats


def report_confusion_and_rates(y_true: np.ndarray, y_pred: np.ndarray, title: str = "TEST") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    print(f"\n=== CONFUSION MATRIX ({title}) ===")
    print("[rows=true, cols=pred] shape:", cm.shape)
    print(cm)

    st = _confusion_stats_multiclass(cm)
    print(f"\n=== ONE-vs-REST RATES ({title}) ===")
    print("Class                                                   |   TP    FP    FN    TN |  Sens   Spec   PPV    F1   Support")
    print("-" * 118)
    for i in range(len(CLASSES)):
        d = st[i]
        print(
            f"{INDEX_TO_CLASS[i]:55s} |"
            f" {int(d['TP']):5d} {int(d['FP']):5d} {int(d['FN']):5d} {int(d['TN']):5d} |"
            f" {d['Sensitivity/Recall']:6.3f} {d['Specificity']:6.3f} {d['Precision/PPV']:6.3f} {d['F1']:6.3f} {int(d['Support']):8d}"
        )


def _safe_multiclass_auc_pr(y_true: np.ndarray, probs: np.ndarray) -> Tuple[Dict[int, float], Dict[int, float], float, float, float, float]:
    """
    Returns:
      per_class_roc_auc, per_class_pr_auc, macro_roc, weighted_roc, macro_pr, weighted_pr
    Computed one-vs-rest and skipping classes without both pos & neg.
    """
    y_true = y_true.astype(np.int32, copy=False)
    probs = np.asarray(probs, dtype=np.float64)
    C = probs.shape[1]

    per_roc: Dict[int, float] = {}
    per_pr: Dict[int, float] = {}
    supports = np.bincount(y_true, minlength=C).astype(np.float64)
    total = float(supports.sum()) if supports.size else 0.0

    for i in range(C):
        pos = (y_true == i).astype(np.int32)
        if pos.sum() == 0 or pos.sum() == pos.shape[0]:
            continue
        try:
            per_roc[i] = float(roc_auc_score(pos, probs[:, i]))
        except Exception:
            pass
        try:
            per_pr[i] = float(average_precision_score(pos, probs[:, i]))
        except Exception:
            pass

    def _avg(d: Dict[int, float], weighted: bool) -> float:
        if not d:
            return float("nan")
        if not weighted:
            return float(np.mean(list(d.values())))
        wsum = 0.0
        ssum = 0.0
        for i, v in d.items():
            w = float(supports[i]) / max(total, 1.0)
            wsum += w * float(v)
            ssum += w
        return float(wsum / max(ssum, 1e-12))

    macro_roc = _avg(per_roc, weighted=False)
    w_roc = _avg(per_roc, weighted=True)
    macro_pr = _avg(per_pr, weighted=False)
    w_pr = _avg(per_pr, weighted=True)
    return per_roc, per_pr, macro_roc, w_roc, macro_pr, w_pr


def report_auc_pr(y_true: np.ndarray, probs: np.ndarray, title: str = "TEST") -> None:
    per_roc, per_pr, macro_roc, w_roc, macro_pr, w_pr = _safe_multiclass_auc_pr(y_true, probs)

    print(f"\n=== ROC-AUC / PR-AUC (OvR) ({title}) ===")
    print(f"[INFO] ROC-AUC macro={macro_roc:.4f} weighted={w_roc:.4f}")
    print(f"[INFO] PR-AUC  macro={macro_pr:.4f} weighted={w_pr:.4f}")

    print("\nPer-class (skips classes missing pos/neg in this split):")
    print("Class                                                   | ROC-AUC  PR-AUC")
    print("-" * 85)
    for i in range(len(CLASSES)):
        ra = per_roc.get(i, None)
        pa = per_pr.get(i, None)
        if ra is None and pa is None:
            continue
        rs = f"{ra:.4f}" if ra is not None else "  n/a "
        ps = f"{pa:.4f}" if pa is not None else "  n/a "
        print(f"{INDEX_TO_CLASS[i]:55s} | {rs:>6s}  {ps:>6s}")


# ===============================
# ADDED: ROC/PR CURVE PLOTS (PNG)
# ===============================
def plot_auc_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    title: str,
    out_roc_png: Optional[str] = None,
    out_pr_png: Optional[str] = None,
) -> None:
    """
    Saves:
      - ROC curves (OvR per class + macro-avg curve)
      - PR curves  (OvR per class + macro-avg curve)

    Skips any class that has no positives or no negatives in this split.
    """
    if plt is None:
        print("[WARN] matplotlib not available; skipping AUC curve plots.")
        return
    if out_roc_png is None and out_pr_png is None:
        return

    y_true = y_true.astype(np.int32, copy=False)
    probs = np.asarray(probs, dtype=np.float64)
    C = probs.shape[1]

    valid: List[int] = []
    for i in range(C):
        pos = (y_true == i).astype(np.int32)
        if int(pos.sum()) == 0 or int(pos.sum()) == int(pos.shape[0]):
            continue
        valid.append(i)

    if not valid:
        print("[WARN] No valid classes with both pos/neg in this split; skipping AUC curve plots.")
        return

    # ---------- ROC ----------
    if out_roc_png is not None:
        fpr_grid = np.linspace(0.0, 1.0, 200, dtype=np.float64)
        tprs: List[np.ndarray] = []

        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

        for i in valid:
            pos = (y_true == i).astype(np.int32)
            try:
                fpr, tpr, _ = roc_curve(pos, probs[:, i])
                auc_i = float(roc_auc_score(pos, probs[:, i]))
            except Exception:
                continue

            ax.plot(fpr, tpr, linewidth=1.2, label=f"{INDEX_TO_CLASS[i][:28]}  AUC={auc_i:.3f}")
            tprs.append(np.interp(fpr_grid, fpr, tpr))

        if tprs:
            macro_tpr = np.mean(np.vstack(tprs), axis=0)
            ax.plot(fpr_grid, macro_tpr, linewidth=2.0, label="MACRO-AVG")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curves (OvR) — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_roc_png, dpi=170)
        plt.close(fig)
        print(f"[INFO] Saved ROC-AUC curves -> {out_roc_png}")

    # ---------- PR ----------
    if out_pr_png is not None:
        recall_grid = np.linspace(0.0, 1.0, 200, dtype=np.float64)
        precs: List[np.ndarray] = []

        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)

        for i in valid:
            pos = (y_true == i).astype(np.int32)
            prevalence = float(np.mean(pos))  # baseline on PR
            try:
                prec, rec, _ = precision_recall_curve(pos, probs[:, i])
                ap_i = float(average_precision_score(pos, probs[:, i]))
            except Exception:
                continue

            ax.plot(rec, prec, linewidth=1.2, label=f"{INDEX_TO_CLASS[i][:28]}  AP={ap_i:.3f}")
            ax.plot([0, 1], [prevalence, prevalence], linestyle=":", linewidth=0.8)

            # Interp precision on recall grid (sort recall ascending)
            order = np.argsort(rec)
            rec_s = rec[order]
            prec_s = prec[order]
            precs.append(np.interp(recall_grid, rec_s, prec_s, left=prec_s[0], right=prec_s[-1]))

        if precs:
            macro_prec = np.mean(np.vstack(precs), axis=0)
            ax.plot(recall_grid, macro_prec, linewidth=2.0, label="MACRO-AVG")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curves (OvR) — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_pr_png, dpi=170)
        plt.close(fig)
        print(f"[INFO] Saved PR-AUC curves -> {out_pr_png}")


def calibration_report(y_true: np.ndarray, probs: np.ndarray, title: str, out_png: Optional[str] = None) -> None:
    """
    Overall calibration using confidence=max_prob and correctness.
    Prints ECE + Brier (multiclass) + reliability table; optionally saves plot.
    """
    y_true = y_true.astype(np.int32, copy=False)
    probs = np.asarray(probs, dtype=np.float64)

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1).astype(np.int32)
    corr = (pred == y_true).astype(np.float64)

    y_oh = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))

    bins = int(max(2, CALIB_BINS))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    n = float(conf.shape[0])

    for b in range(bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        m = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
        k = int(np.sum(m))
        if k == 0:
            continue
        acc_b = float(np.mean(corr[m]))
        conf_b = float(np.mean(conf[m]))
        ece += (k / max(n, 1.0)) * abs(acc_b - conf_b)
        rows.append((lo, hi, k, acc_b, conf_b, abs(acc_b - conf_b)))

    print(f"\n=== CALIBRATION ({title}) ===")
    print(f"[INFO] ECE (bins={bins}) = {ece:.6f}")
    print(f"[INFO] Brier (multiclass) = {brier:.6f}")
    print("Bin range        |   n   |  acc   |  conf  | |acc-conf|")
    print("-" * 62)
    for lo, hi, k, acc_b, conf_b, gap in rows:
        print(f"[{lo:0.2f},{hi:0.2f}]      | {k:5d} | {acc_b:0.4f} | {conf_b:0.4f} | {gap:0.4f}")

    if plt is None or out_png is None:
        return

    xs = [0.5 * (lo + hi) for lo, hi, *_ in rows]
    ys_acc = [acc_b for (_lo, _hi, _k, acc_b, _conf_b, _gap) in rows]
    ys_conf = [conf_b for (_lo, _hi, _k, _acc_b, conf_b, _gap) in rows]

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.plot(xs, ys_acc, marker="o", label="Accuracy per bin")
    ax.plot(xs, ys_conf, marker="x", label="Confidence per bin")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram ({title})\nECE={ece:.4f}  Brier={brier:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[INFO] Saved calibration plot -> {out_png}")


def _age_bins(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=np.float64)
    out = np.empty((age.shape[0],), dtype=object)
    for i, a in enumerate(age.tolist()):
        if not np.isfinite(a):
            out[i] = "unknown"
        elif a < 18:
            out[i] = "<18"
        elif a < 40:
            out[i] = "18-39"
        elif a < 65:
            out[i] = "40-64"
        elif a < 80:
            out[i] = "65-79"
        else:
            out[i] = "80+"
    return out


def _group_report(
    group_name: str,
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    min_n: int = BIAS_MIN_GROUP_N,
) -> None:
    groups = np.asarray(groups, dtype=object)
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)
    probs = np.asarray(probs, dtype=np.float64)

    vals = [g for g in np.unique(groups) if g is not None]
    if len(vals) == 0:
        return

    print(f"\n=== BIAS CHECK: {group_name} (min_n={min_n}) ===")
    print("Group                 |   n   |  Acc  |  Macro-F1 |  Weighted-F1 |  Macro ROC-AUC |  Macro PR-AUC")
    print("-" * 98)
    for v in vals:
        m = (groups == v)
        n = int(np.sum(m))
        if n < int(min_n):
            continue
        yt = y_true[m]
        yp = y_pred[m]
        prf_macro = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        prf_w = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
        acc = float(np.mean((yt == yp).astype(np.float32)))

        _, _, macro_roc, _w_roc, macro_pr, _w_pr = _safe_multiclass_auc_pr(yt, probs[m])

        print(
            f"{str(v)[:20]:20s} | {n:5d} | {acc:0.3f} | {prf_macro[2]:0.3f}    | {prf_w[2]:0.3f}       |"
            f" {macro_roc:>12.4f} | {macro_pr:>11.4f}"
        )


def load_bias_meta_by_hadm(hadm_set: set[int]) -> Dict[int, Dict[str, str]]:
    """
    Best-effort metadata for bias checks:
      - gender/sex (from PATIENTS)
      - admission_type, admission_location (from ADMISSIONS)
    Missing fields become 'unknown'.
    """
    meta: Dict[int, Dict[str, str]] = {
        int(h): {"gender": "unknown", "admission_type": "unknown", "admission_location": "unknown"}
        for h in hadm_set
    }

    adm_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
        "ADMISSION_TYPE": ["ADMISSION_TYPE", "ADMISSIONTYPE", "admission_type"],
        "ADMISSION_LOCATION": ["ADMISSION_LOCATION", "ADMISSIONLOCATION", "admission_location"],
    }
    hadm_to_subject: Dict[int, int] = {}
    for row in _iter_csv_optional(ADMISSIONS_PATH, adm_wanted):
        hadm = _parse_int(row.get("HADM_ID", ""))
        if hadm is None or hadm not in meta:
            continue
        sid = _parse_int(row.get("SUBJECT_ID", ""))
        if sid is not None:
            hadm_to_subject[int(hadm)] = int(sid)
        at = row.get("ADMISSION_TYPE", "").strip()
        al = row.get("ADMISSION_LOCATION", "").strip()
        if at:
            meta[int(hadm)]["admission_type"] = _norm_text(at)
        if al:
            meta[int(hadm)]["admission_location"] = _norm_text(al)

    pat_wanted = {
        "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
        "GENDER": ["GENDER", "SEX", "gender", "sex"],
    }
    subject_to_gender: Dict[int, str] = {}
    if hadm_to_subject:
        for row in _iter_csv_optional(PATIENTS_PATH, pat_wanted):
            sid = _parse_int(row.get("SUBJECT_ID", ""))
            if sid is None:
                continue
            g = row.get("GENDER", "").strip()
            if g:
                subject_to_gender[int(sid)] = str(g).strip().upper()

        for hadm, sid in hadm_to_subject.items():
            g = subject_to_gender.get(int(sid))
            if g:
                meta[int(hadm)]["gender"] = g

    return meta


# ===============================
# Loss (FIX): CE / WCE / Focal / Class-balanced focal
# ===============================
def _safe_class_counts(y_int: np.ndarray) -> np.ndarray:
    return np.bincount(y_int.astype(np.int32, copy=False), minlength=len(CLASSES)).astype(np.float64)


def _inv_freq_weights(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    w = np.zeros_like(counts, dtype=np.float64)
    nz = counts > 0
    if np.any(nz):
        total = float(np.sum(counts[nz]))
        w[nz] = total / (float(np.sum(nz)) * counts[nz])
        w = w / (np.mean(w[nz]) + 1e-12)
    return w


def _class_balanced_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    w = np.zeros_like(counts, dtype=np.float64)
    nz = counts > 0
    if np.any(nz):
        eff = 1.0 - np.power(beta, counts[nz])
        eff = np.maximum(eff, 1e-12)
        w[nz] = (1.0 - beta) / eff
        w = w / (np.mean(w[nz]) + 1e-12)
    return w


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor], gamma: float, label_smoothing: float):
        super().__init__()
        self.alpha = alpha  # shape [C] or None
        self.gamma = float(gamma)
        self.ls = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        eps = self.ls
        idx = target.view(-1, 1)
        log_py = log_probs.gather(1, idx).squeeze(1)
        ce_hard = -log_py
        if eps > 0.0:
            ce_uniform = -torch.mean(log_probs, dim=-1)
            ce = (1.0 - eps) * ce_hard + eps * ce_uniform
        else:
            ce = ce_hard

        py = probs.gather(1, idx).squeeze(1).clamp(1e-9, 1.0)
        focal = torch.pow(1.0 - py, self.gamma) * ce

        if self.alpha is not None:
            a = self.alpha[target]
            focal = a * focal

        return focal.mean()


def build_loss(loss_name: str, class_counts: np.ndarray, device: torch.device) -> nn.Module:
    ln = str(loss_name).strip().lower()
    bother_idx = CLASS_TO_INDEX.get("B:OTHER", None)

    if ln == "ce":
        try:
            return nn.CrossEntropyLoss(label_smoothing=float(LABEL_SMOOTHING))
        except TypeError:
            return FocalLoss(alpha=None, gamma=0.0, label_smoothing=float(LABEL_SMOOTHING))

    if ln == "wce":
        w = _inv_freq_weights(class_counts)
        if bother_idx is not None:
            w[bother_idx] *= float(BOTHER_EXTRA_DOWNWEIGHT)
        w = np.clip(w, 0.0, float(MAX_CLASS_WEIGHT)).astype(np.float32)
        wt = torch.tensor(w, dtype=torch.float32, device=device)
        try:
            return nn.CrossEntropyLoss(weight=wt, label_smoothing=float(LABEL_SMOOTHING))
        except TypeError:
            return FocalLoss(alpha=wt, gamma=0.0, label_smoothing=float(LABEL_SMOOTHING))

    if ln == "focal":
        alpha = None
        if FOCAL_USE_ALPHA:
            w = _inv_freq_weights(class_counts)
            if bother_idx is not None:
                w[bother_idx] *= float(BOTHER_EXTRA_DOWNWEIGHT)
            w = np.clip(w, 0.0, float(MAX_CLASS_WEIGHT)).astype(np.float32)
            alpha = torch.tensor(w, dtype=torch.float32, device=device)
        return FocalLoss(alpha=alpha, gamma=float(FOCAL_GAMMA), label_smoothing=float(LABEL_SMOOTHING))

    if ln == "cb_focal":
        w = _class_balanced_weights(class_counts, beta=float(CB_BETA))
        if bother_idx is not None:
            w[bother_idx] *= float(BOTHER_EXTRA_DOWNWEIGHT)
        w = np.clip(w, 0.0, float(MAX_CLASS_WEIGHT)).astype(np.float32)
        alpha = torch.tensor(w, dtype=torch.float32, device=device)
        return FocalLoss(alpha=alpha, gamma=float(FOCAL_GAMMA), label_smoothing=float(LABEL_SMOOTHING))

    print(f"[WARN] Unknown LOSS_NAME={loss_name!r}. Falling back to CE.")
    return build_loss("ce", class_counts, device=device)


# ===============================
# Torch Dataset
# ===============================
class MIMICDataset(Dataset):
    def __init__(self, text_ids: np.ndarray, num: np.ndarray, y: np.ndarray):
        self.text_ids = text_ids.astype(np.int64, copy=False)
        self.num = num.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.text_ids[idx]),
            torch.from_numpy(self.num[idx]),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )


# ===============================
# Hybrid model: Embedding + Conv1d + RNN + BiLSTM + DNN + numeric branch
# ===============================
def _get_activation(name: str):
    a = str(name).strip().lower()
    if a == "gelu":
        return lambda x: F.gelu(x)
    if a in {"swish", "silu"}:
        return lambda x: F.silu(x)
    if a == "elu":
        return lambda x: F.elu(x)
    return lambda x: F.relu(x)


class HybridNet(nn.Module):
    def __init__(self, vocab_size: int, num_dim: int, activation_name: str):
        super().__init__()
        act = _get_activation(activation_name)
        self._act = act

        self.embed = nn.Embedding(int(vocab_size), int(EMBED_DIM), padding_idx=0)

        self.conv = nn.Conv1d(
            int(EMBED_DIM), int(CNN_FILTERS),
            kernel_size=int(CNN_KERNEL),
            padding=int(CNN_KERNEL // 2),
        )
        self.drop = nn.Dropout(float(DROPOUT))

        self.rnn = nn.RNN(input_size=int(CNN_FILTERS), hidden_size=int(RNN_UNITS), batch_first=True)
        self.lstm = nn.LSTM(
            input_size=int(RNN_UNITS),
            hidden_size=int(LSTM_UNITS),
            batch_first=True,
            bidirectional=True,
        )

        self.text_fc = nn.Linear(2 * int(LSTM_UNITS), 128)
        self.num_fc = nn.Linear(int(num_dim), 64)

        self.fc1 = nn.Linear(128 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, len(CLASSES))

    def forward(self, text_ids: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        x = self.embed(text_ids)     # [B,T,E]
        x = x.transpose(1, 2)       # [B,E,T]
        x = self.conv(x)            # [B,F,T]
        x = F.relu(x)
        x = self.drop(x)
        x = x.transpose(1, 2)       # [B,T,F]

        x, _ = self.rnn(x)          # [B,T,R]
        x, _ = self.lstm(x)         # [B,T,2H]
        x = x[:, -1, :]             # [B,2H]

        x = self._act(self.text_fc(x))
        x = self.drop(x)

        n = self._act(self.num_fc(num))
        n = self.drop(n)

        z = torch.cat([x, n], dim=1)
        z = self._act(self.fc1(z))
        z = self.drop(z)
        z = self._act(self.fc2(z))
        z = self.drop(z)
        logits = self.out(z)
        return logits


# ===============================
# Helpers: predict
# ===============================
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs_all: List[np.ndarray] = []
    for text_ids, num, _y in loader:
        text_ids = text_ids.to(device)
        num = num.to(device)
        logits = model(text_ids, num)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)


# ===============================
# Vitals signal (permute numeric cols) + Pearson r,p
# ===============================
def _mean_cce(y_true_oh: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    y = y_true_oh.astype(np.float64)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def _norm_cdf(z: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.shape[0])
    if n < 4:
        return 0.0, 1.0

    try:
        from scipy import stats as _sp_stats  # type: ignore
        r, p = _sp_stats.pearsonr(x, y)
        if not np.isfinite(r) or not np.isfinite(p):
            return 0.0, 1.0
        return float(r), float(p)
    except Exception:
        pass

    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, 1.0
    r = float(np.sum(x0 * y0) / denom)

    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = float(np.arctanh(r_clip) * np.sqrt(max(n - 3, 1)))
    p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    if not np.isfinite(p):
        p = 1.0
    return r, p


def plot_vitals_signal_torch(
    model: nn.Module,
    text_ids_te: np.ndarray,
    num_te_scaled: np.ndarray,
    y_te_oh: np.ndarray,
    num_te_raw: np.ndarray,
    out_path: str,
    device: torch.device,
    n_repeats: int = 3,
    seed: int = SEED,
) -> None:
    model.eval()
    rng = np.random.default_rng(seed)

    def _probs(text_ids: np.ndarray, num_scaled: np.ndarray, bs: int = 256) -> np.ndarray:
        probs_all = []
        with torch.no_grad():
            for i in range(0, text_ids.shape[0], bs):
                t = torch.from_numpy(text_ids[i : i + bs]).to(device)
                n = torch.from_numpy(num_scaled[i : i + bs]).to(device)
                logits = model(t, n)
                probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs_all)

    base_probs = _probs(text_ids_te, num_te_scaled)
    base_loss = _mean_cce(y_te_oh, base_probs)

    vital_cols = [0, 1, 2, 3]
    vital_names = list(VITAL_ORDER)

    imp_mean: List[float] = []
    imp_std: List[float] = []
    for j in vital_cols:
        losses: List[float] = []
        for _ in range(int(n_repeats)):
            nump = num_te_scaled.copy()
            perm = rng.permutation(nump.shape[0])
            nump[:, j] = nump[perm, j]
            p = _probs(text_ids_te, nump)
            losses.append(_mean_cce(y_te_oh, p))
        arr = np.asarray(losses, dtype=np.float64)
        imp_mean.append(float(np.mean(arr) - base_loss))
        imp_std.append(float(np.std(arr, ddof=1) if arr.size > 1 else 0.0))

    b_idxs = np.asarray([i for i, c in enumerate(CLASSES) if c.startswith("B:")], dtype=np.int32)
    bact_prob = np.sum(base_probs[:, b_idxs], axis=1).astype(np.float64) if b_idxs.size else np.zeros((base_probs.shape[0],))

    corr_r: List[float] = []
    corr_p: List[float] = []
    corr_abs: List[float] = []
    for k in range(4):
        r, p = _pearson_r_p(num_te_raw[:, k], bact_prob)
        corr_r.append(float(r))
        corr_p.append(float(p))
        corr_abs.append(float(abs(r)))

    ranked = sorted(
        zip(vital_names, imp_mean, imp_std, corr_r, corr_p, corr_abs),
        key=lambda t: t[1],
        reverse=True,
    )

    print("\n=== VITALS SIGNAL (TEST) ===")
    print(f"[INFO] Baseline test CE loss: {base_loss:.6f}")
    print("[INFO] Permutation importance on numeric branch (Δloss) + Pearson r,p vs P(bacterial):")
    for name, m, s, r, p, a in ranked:
        print(f"  {name:14s} -> Δloss={m:.6f} (±{s:.6f})  r={r:+.4f}  p={p:.3e}  |r|={a:.4f}")

    if plt is None:
        print("[WARN] matplotlib not available; skipping vitals plot PNG.")
        return

    x = np.arange(len(vital_names), dtype=np.float64)
    w = 0.38

    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(x - w / 2.0, imp_mean, width=w, label="Permutation importance (Δloss)")
    ax.bar(x + w / 2.0, corr_abs, width=w, label="|Pearson r| vs P(bacterial)")
    ax.set_xticks(x)
    ax.set_xticklabels(vital_names, rotation=0)
    ax.set_ylabel("Signal strength (scaled)")
    ax.set_title("Vitals signal (empirical)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ymax = float(max(max(imp_mean) if imp_mean else 0.0, max(corr_abs) if corr_abs else 0.0))
    for i, p in enumerate(corr_p):
        ax.text(
            x[i] + w / 2.0,
            corr_abs[i] + 0.03 * (ymax + 1e-9),
            f"p={p:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[INFO] Saved vitals feature signal plot -> {out_path}")


# ===============================
# Split helper
# ===============================
def split_with_min_unique_hadm(
    text_arr: np.ndarray,
    num_arr: np.ndarray,
    y_arr: np.ndarray,
    hadm_arr: np.ndarray,
    test_size: float,
    seed: int,
    min_unique_test_hadm: int = MIN_TEST_UNIQUE_HADM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_hadm = np.unique(hadm_arr)
    if unique_hadm.size < min_unique_test_hadm:
        raise RuntimeError(
            f"Not enough unique HADM_IDs overall ({unique_hadm.size}) to satisfy test requirement ({min_unique_test_hadm})."
        )

    for delta in range(0, 200):
        rs = int(seed + delta)
        txt_tr, txt_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te = train_test_split(
            text_arr,
            num_arr,
            y_arr,
            hadm_arr,
            test_size=float(test_size),
            random_state=rs,
            stratify=y_arr if len(np.unique(y_arr)) > 1 else None,
        )
        if len(np.unique(hadm_te)) >= min_unique_test_hadm:
            if delta > 0:
                print(f"[INFO] Adjusted split seed -> {rs} to satisfy test unique HADM_ID >= {min_unique_test_hadm}")
            return txt_tr, txt_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te

    raise RuntimeError(f"Could not find a split with >= {min_unique_test_hadm} unique HADM_IDs in TEST after many attempts.")


# ===============================
# One full run for a resolved dataset
# ===============================
def run_once(dataset_root: Path) -> None:
    global MICRO_PATH, PRESC_PATH, ADMISSIONS_PATH, PATIENTS_PATH, D_ITEMS_PATH, CHARTEVENTS_PATH, D_LABITEMS_PATH, LABEVENTS_PATH

    paths, ds_kind = resolve_paths(dataset_root)

    MICRO_PATH = paths["micro"]
    PRESC_PATH = paths["presc"]
    ADMISSIONS_PATH = paths["admissions"]
    PATIENTS_PATH = paths["patients"]
    D_ITEMS_PATH = paths["d_items"]
    CHARTEVENTS_PATH = paths["chartevents"]
    D_LABITEMS_PATH = paths["d_labitems"]
    LABEVENTS_PATH = paths["labevents"]

    ds_tag = _sanitize_tag(f"{ds_kind}_{dataset_root.name}")

    print("\n" + "=" * 98)
    print(f"[INFO] DATASET ROOT: {dataset_root.resolve()}")
    print(f"[INFO] DATASET TAG:  {ds_tag}")
    print(f"[INFO] MICRO:        {MICRO_PATH}")
    print(f"[INFO] PRESC:        {PRESC_PATH}")
    print(f"[INFO] ADMISSIONS:   {ADMISSIONS_PATH}")
    print(f"[INFO] PATIENTS:     {PATIENTS_PATH}")
    print(f"[INFO] D_ITEMS:      {D_ITEMS_PATH}")
    print(f"[INFO] CHARTEVENTS:  {CHARTEVENTS_PATH}")
    print(f"[INFO] D_LABITEMS:   {D_LABITEMS_PATH}")
    print(f"[INFO] LABEVENTS:    {LABEVENTS_PATH}")
    print("=" * 98)

    micro_rows = load_micro_rows()
    if not micro_rows:
        raise RuntimeError("No MICROBIOLOGYEVENTS rows mapped to CLASSES.")

    hadm_set = set(r["hadm_id"] for r in micro_rows)
    print(f"[INFO] MICRO rows={len(micro_rows)} unique_hadm={len(hadm_set)}")

    y_all = np.asarray([CLASS_TO_INDEX[r["label"]] for r in micro_rows], dtype=np.int32)
    dist = Counter(y_all.tolist())
    print("[INFO] Label counts:")
    for i in range(len(CLASSES)):
        print(f"  {INDEX_TO_CLASS[i]:55s} -> {dist.get(i, 0)}")

    abx_by_hadm = load_abx_features(hadm_set)
    print(f"[INFO] ABX features built for hadm={len(abx_by_hadm)} (features={len(ABX_ORDER)})")

    print(f"[INFO] Computing vitals/labs within {HOURS_WINDOW}h window ...")
    vitals_by_hadm = compute_vitals_features(hadm_set)
    print(f"[INFO] Vitals complete rows={len(vitals_by_hadm)}")

    # Join (TRAIN ON FULL TRAIN DB rows; NOT per-HADM training)
    text_data: List[str] = []
    num_data: List[List[float]] = []
    y_labels: List[int] = []
    hadm_ids: List[int] = []

    for r in micro_rows:
        hadm = int(r["hadm_id"])
        vit = vitals_by_hadm.get(hadm)
        if vit is None:
            continue
        abx = abx_by_hadm.get(hadm, {k: 0.0 for k in ABX_ORDER})

        txt = f"{r['spec_type_desc']} [sep] {r['interpretation']}"
        text_data.append(txt)

        row_num = [
            float(vit["temperature_c"]),
            float(vit["wbc"]),
            float(vit["spo2"]),
            float(vit["age"]),
        ] + [float(abx[k]) for k in ABX_ORDER]

        num_data.append(row_num)
        y_labels.append(CLASS_TO_INDEX[r["label"]])
        hadm_ids.append(hadm)

    if len(y_labels) < 20:
        raise RuntimeError(f"Too few joined rows after requiring vitals/labs: {len(y_labels)}")

    y_labels_arr = np.asarray(y_labels, dtype=np.int32)
    hadm_arr = np.asarray(hadm_ids, dtype=np.int32)
    text_arr = np.asarray(text_data, dtype=object)
    num_arr = np.asarray(num_data, dtype=np.float32)

    print(f"[INFO] Joined rows={len(y_labels_arr)} | numeric_dim={num_arr.shape[1]} | numeric_order={NUMERIC_ORDER}")

    txt_tr, txt_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te = split_with_min_unique_hadm(
        text_arr, num_arr, y_labels_arr, hadm_arr, test_size=0.2, seed=SEED, min_unique_test_hadm=MIN_TEST_UNIQUE_HADM
    )
    print(f"[INFO] TEST unique HADM_ID: {len(np.unique(hadm_te))} (requirement >= {MIN_TEST_UNIQUE_HADM})")

    txt_tr2, txt_va, num_tr2, num_va, y_tr2, y_va, hadm_tr2, hadm_va = train_test_split(
        txt_tr,
        num_tr,
        y_tr,
        hadm_tr,
        test_size=0.2,
        random_state=SEED,
        stratify=y_tr if len(np.unique(y_tr)) > 1 else None,
    )

    # Build vocab on TRAIN only
    vocab = build_vocab(txt_tr2, max_tokens=MAX_TEXT_TOKENS)
    vocab_size = min(MAX_TEXT_TOKENS, len(vocab) + 2)  # +PAD/+UNK

    X_text_tr = texts_to_ids(txt_tr2, vocab, TEXT_SEQ_LEN)
    X_text_va = texts_to_ids(txt_va, vocab, TEXT_SEQ_LEN)
    X_text_te = texts_to_ids(txt_te, vocab, TEXT_SEQ_LEN)

    # Standardize vitals on TRAIN only (first 4 numeric cols)
    mu = num_tr2[:, :4].mean(axis=0).astype(np.float32)
    sd = (num_tr2[:, :4].std(axis=0) + 1e-6).astype(np.float32)

    def apply_scaling(num_x: np.ndarray) -> np.ndarray:
        out = num_x.astype(np.float32).copy()
        out[:, :4] = (out[:, :4] - mu) / sd
        return out

    X_num_tr = apply_scaling(num_tr2)
    X_num_va = apply_scaling(num_va)
    X_num_te = apply_scaling(num_te)

    # device
    if DEVICE == "cpu":
        device = torch.device("cpu")
    elif DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Torch device: {device}")

    # DataLoaders
    tr_ds = MIMICDataset(X_text_tr, X_num_tr, y_tr2)
    va_ds = MIMICDataset(X_text_va, X_num_va, y_va)
    te_ds = MIMICDataset(X_text_te, X_num_te, y_te)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, drop_last=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False, drop_last=False)

    # class counts (TRAIN)
    class_counts_tr = _safe_class_counts(y_tr2)

    # training attempts
    best_model_state = None
    best_metric = -1.0
    best_activation = "unknown"
    best_epoch = 0

    total_attempts = max(1, int(MAX_TRAIN_RESTARTS))
    use_acc = (TARGET_STOP_METRIC == "acc")
    target_value = float(TARGET_ACC if use_acc else TARGET_F1)

    for attempt in range(total_attempts):
        act_name = ACTIVATION_CANDIDATES[attempt % max(1, len(ACTIVATION_CANDIDATES))]
        local_seed = int(SEED + 1000 * attempt)
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)

        model = HybridNet(vocab_size=vocab_size, num_dim=int(X_num_tr.shape[1]), activation_name=act_name).to(device)
        loss_fn = build_loss(LOSS_NAME, class_counts_tr, device=device)

        opt = torch.optim.Adam(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))

        best_local_metric = -1.0
        best_local_epoch = 0
        best_local_state = None
        no_improve = 0

        metric_name = f"val_target_acc({TARGET_ACC_KIND})" if use_acc else f"val_target_f1({TARGET_F1_KIND})"

        print(
            f"\n[INFO] Training attempt {attempt + 1}/{total_attempts}"
            f" | activation={act_name} | seed={local_seed}"
            f" | stop_metric={TARGET_STOP_METRIC} target={target_value:.3f}"
            f" | loss={LOSS_NAME}"
        )

        for epoch in range(1, int(MAX_EPOCHS) + 1):
            model.train()
            tr_loss_sum = 0.0
            tr_n = 0

            for text_ids, num, yb in tr_loader:
                text_ids = text_ids.to(device)
                num = num.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                logits = model(text_ids, num)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                tr_loss_sum += float(loss.item()) * int(yb.shape[0])
                tr_n += int(yb.shape[0])

            tr_loss = tr_loss_sum / max(tr_n, 1)

            # validation
            model.eval()
            va_loss_sum = 0.0
            va_n = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for text_ids, num, yb in va_loader:
                    text_ids = text_ids.to(device)
                    num = num.to(device)
                    yb = yb.to(device)

                    logits = model(text_ids, num)
                    loss = loss_fn(logits, yb)
                    va_loss_sum += float(loss.item()) * int(yb.shape[0])
                    va_n += int(yb.shape[0])

                    pred = torch.argmax(logits, dim=-1)
                    y_true.append(yb.cpu().numpy())
                    y_pred.append(pred.cpu().numpy())

            va_loss = va_loss_sum / max(va_n, 1)
            y_true_np = np.concatenate(y_true).astype(np.int32)
            y_pred_np = np.concatenate(y_pred).astype(np.int32)

            if use_acc:
                val_metric = _compute_target_acc(TARGET_ACC_KIND, y_true_np, y_pred_np)
            else:
                val_metric = _compute_target_f1(TARGET_F1_KIND, y_true_np, y_pred_np)

            if val_metric is None:
                val_metric = -1.0

            print(f"[E{epoch:05d}] tr_loss={tr_loss:.5f} va_loss={va_loss:.5f} {metric_name}={val_metric:.4f}")

            if float(val_metric) > float(best_local_metric) + float(MIN_DELTA):
                best_local_metric = float(val_metric)
                best_local_epoch = int(epoch)
                best_local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if float(val_metric) >= float(target_value):
                print(f"[INFO] Reached target {metric_name}: {val_metric:.4f} >= {target_value:.3f} -> stopping.")
                break

            if no_improve >= int(EARLY_PATIENCE):
                print(f"[INFO] Early stop: no metric improvement for {EARLY_PATIENCE} epochs.")
                break

        if best_local_state is not None:
            model.load_state_dict(best_local_state)

        print(f"[INFO] Attempt best {metric_name}={best_local_metric:.4f} at epoch={best_local_epoch}")

        if best_local_metric > best_metric:
            best_metric = best_local_metric
            best_activation = act_name
            best_epoch = best_local_epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_local_metric >= target_value:
            break

    if best_model_state is None:
        raise RuntimeError("Training failed: no model state captured.")

    # build final model and load best state
    final_model = HybridNet(vocab_size=vocab_size, num_dim=int(X_num_tr.shape[1]), activation_name=best_activation).to(device)
    final_model.load_state_dict(best_model_state)

    print(f"[INFO] BEST across attempts: metric={best_metric:.4f} | activation={best_activation} | best_epoch={best_epoch}")

    # Optional retrain on full train (TRAIN+VAL) for best_epoch
    if RETRAIN_ON_FULL_TRAIN and best_epoch > 0:
        print(f"[INFO] Retraining on FULL TRAIN (TRAIN+VAL) for {best_epoch} epochs | activation={best_activation}")

        txt_full = np.concatenate([txt_tr2, txt_va]).astype(object)
        num_full_raw = np.vstack([num_tr2, num_va]).astype(np.float32)
        y_full = np.concatenate([y_tr2, y_va]).astype(np.int32)

        X_num_full = apply_scaling(num_full_raw)
        X_text_full = texts_to_ids(txt_full, vocab, TEXT_SEQ_LEN)

        full_ds = MIMICDataset(X_text_full, X_num_full, y_full)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        class_counts_full = _safe_class_counts(y_full)
        loss_fn_full = build_loss(LOSS_NAME, class_counts_full, device=device)

        torch.manual_seed(SEED + 9999)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + 9999)

        final_model = HybridNet(vocab_size=vocab_size, num_dim=int(X_num_tr.shape[1]), activation_name=best_activation).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))

        final_model.train()
        for ep in range(1, int(best_epoch) + 1):
            loss_sum = 0.0
            n_sum = 0
            for text_ids, num, yb in full_loader:
                text_ids = text_ids.to(device)
                num = num.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                logits = final_model(text_ids, num)
                loss = loss_fn_full(logits, yb)
                loss.backward()
                opt.step()

                loss_sum += float(loss.item()) * int(yb.shape[0])
                n_sum += int(yb.shape[0])

            print(f"[FULL E{ep:04d}] loss={loss_sum / max(n_sum, 1):.5f}")

    # TEST evaluation
    probs_te = predict_proba(final_model, te_loader, device=device)
    y_pred = np.argmax(probs_te, axis=1).astype(np.int32)
    acc = float(np.mean((y_pred == y_te).astype(np.float32)))

    print(f"\n=== GENERAL ACCURACY ON TEST SET ({ds_tag}) [activation={best_activation}]: {acc:.4f} ===")
    report_multiclass_metrics(y_true=y_te, y_pred=y_pred)
    report_mrsa_vs_mssa(y_true=y_te, y_pred=y_pred)

    # --- Added: Confusion + FP/FN + rates ---
    report_confusion_and_rates(y_true=y_te, y_pred=y_pred, title=f"TEST {ds_tag}")

    # --- Added: ROC-AUC / PR-AUC ---
    report_auc_pr(y_true=y_te, probs=probs_te, title=f"TEST {ds_tag}")

    # --- Added: ROC/PR Curves (PNG) ---
    roc_png = f"roc_auc__{ds_tag}__{_sanitize_tag(best_activation)}.png"
    pr_png  = f"pr_auc__{ds_tag}__{_sanitize_tag(best_activation)}.png"
    plot_auc_curves(y_true=y_te, probs=probs_te, title=f"TEST {ds_tag}", out_roc_png=roc_png, out_pr_png=pr_png)

    # --- Added: Calibration (ECE/Brier + optional plot) ---
    calib_png = f"calibration__{ds_tag}__{_sanitize_tag(best_activation)}.png"
    calibration_report(y_true=y_te, probs=probs_te, title=f"TEST {ds_tag}", out_png=calib_png)

    # --- Added: Bias checks (gender / age bins / admission fields) ---
    meta_by_hadm = load_bias_meta_by_hadm(hadm_set)
    gender_te = np.asarray([meta_by_hadm.get(int(h), {}).get("gender", "unknown") for h in hadm_te], dtype=object)
    admtype_te = np.asarray([meta_by_hadm.get(int(h), {}).get("admission_type", "unknown") for h in hadm_te], dtype=object)
    admloc_te = np.asarray([meta_by_hadm.get(int(h), {}).get("admission_location", "unknown") for h in hadm_te], dtype=object)

    age_te = num_te[:, 3].astype(np.float64, copy=False)  # raw age column
    agebin_te = _age_bins(age_te)

    _group_report("gender", gender_te, y_te, y_pred, probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("age_bin", agebin_te, y_te, y_pred, probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("admission_type", admtype_te, y_te, y_pred, probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("admission_location", admloc_te, y_te, y_pred, probs_te, min_n=BIAS_MIN_GROUP_N)

    # vitals signal plot
    y_te_oh = np.eye(len(CLASSES), dtype=np.float32)[y_te.astype(np.int32)]
    plot_path = f"vitals_feature_signal__{ds_tag}__{_sanitize_tag(best_activation)}.png"
    plot_vitals_signal_torch(
        model=final_model,
        text_ids_te=X_text_te,
        num_te_scaled=X_num_te,
        y_te_oh=y_te_oh,
        num_te_raw=num_te,
        out_path=plot_path,
        device=device,
        n_repeats=3,
        seed=SEED,
    )

    print("\n[INFO] Done.")
    print(f"[INFO] Feature names lowercase (numeric order): {NUMERIC_ORDER}")
    print("[INFO] Text input: spec_type_desc + ' [sep] ' + interpretation")


# ===============================
# Entrypoint (AUTO)
# ===============================
def main_auto() -> int:
    if torch is None:
        print("[ERROR] PyTorch is not installed. Install torch to run this model.", file=sys.stderr)
        return 2

    roots = discover_dataset_roots()
    if not roots:
        print("[ERROR] No dataset roots found. Expected one of:", file=sys.stderr)
        print("  - datasets/datasets/montassarba/mimic-iv-clinical-database-demo-2-2/versions/1/mimic-iv-clinical-database-demo-2.2", file=sys.stderr)
        print("  - dataset/mimic (or subfolder containing MIMIC-III/MIMIC-IV demo layout)", file=sys.stderr)
        print("\nOptionally set env var MIMIC_AUTOROOTS=\"/path1,/path2\".", file=sys.stderr)
        return 2

    ran_any = False
    for root in roots:
        try:
            try:
                _ = resolve_paths(root)
            except Exception:
                continue
            ran_any = True
            run_once(root)
        except Exception as e:
            print(f"\n[ERROR] Failed on root={root}: {e}", file=sys.stderr)

    if not ran_any:
        print("[ERROR] Found candidate folders, but none matched a runnable MIMIC-III/MIMIC-IV layout.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main_auto())
