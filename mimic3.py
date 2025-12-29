RUN MODEL ON datasets/datasets/montassarba/mimic-iv-clinical-database-demo-2-2/versions/1/mimic-iv-clinical-database-demo-2.2 AND dataset/mimic

#!/usr/bin/env python3
# mimic3_resistance_pipeline_onefile.py  (NO-PANDAS)
#
# Requirements:
# - ALL FEATURE COLUMN NAMES are lowercase (keys/order)
# - CATEGORICAL VALUES are lowercased too (reduces cardinality / duplicates)
# - CLASSES remain CAPITALS (targets + printing)
# - Adds vitals/labs features in REQUIRED ORDER:
#     ["temperature_c", "wbc", "spo2", "age"]
# - Prints per-HADM predictions ORDERED by probability (descending)
#
# Bias fix for "B:OTHER -> 1.0000":
# - stratified split
# - fit OneHotEncoder on TRAIN only
# - standardize vitals on TRAIN only
# - sample_weight class balancing (+ optional extra down-weight for B:OTHER)
# - label_smoothing to reduce overconfidence
#
# Added:
# - Optimized epochs via EarlyStopping + ReduceLROnPlateau (restores best weights)
# - Feature-importance / correlation plot for vitals:
#     - Permutation importance (Δ multiclass CE loss on TEST)
#     - Pearson r + p-value (raw vitals vs P(bacterial)) — ALWAYS printed
#     - PNG saved if matplotlib is available: vitals_feature_signal.png
# - F1 focus on MRSA vs STAPH AUREUS (COAG +):
#     - Per-class precision/recall/F1 (multiclass)
#     - Dedicated MRSA-vs-MSSA report on the TEST subset containing those labels
#
# References (DOI):
# [R1] Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database.
#      Scientific Data. 2016;3:160035. doi:10.1038/sdata.2016.35
# [R2] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python.
#      JMLR. 2011;12:2825-2830. (canonical) doi:10.5555/1953048.2078195
# [R3] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition (ResNet; for general DL context).
#      CVPR 2016. doi:10.1109/CVPR.2016.308
# [R4] Friedman J, Hastie T, Tibshirani R. The Elements of Statistical Learning (One-hot / regularization background).
#      2nd ed. 2009. doi:10.1007/978-0-387-84858-7 (book DOI varies by edition; this is a common one)
#
# Clinical/process motivation refs (DOI):
# [C1] Li L, Georgiou A, Vecellio E, Toouli G, Wilson R. The effect of laboratory testing on emergency department length of stay:
#      a multihospital longitudinal study. Acad Emerg Med. 2015. doi:10.1111/acem.12565
# [C2] Vrijsen B, et al. Shorter laboratory turnaround time is associated with shorter emergency department length of stay.
#      BMC Emerg Med. 2022. doi:10.1186/s12873-022-00763-w
# [C3] Carrier ER, et al. Association between emergency department length of stay and rates of admission to inpatient and observation services.
#      JAMA Intern Med. 2014. doi:10.1001/jamainternmed.2014.3467
# [C4] Kenig A, et al. Blood cultures of adult patients discharged from the emergency department—is the safety net reliable?
#      Eur J Clin Microbiol Infect Dis. 2020. doi:10.1007/s10096-020-03838-3
# [C5] Chan J, et al. Epidemiology and outcomes of bloodstream infections in patients discharged from the emergency department.
#      CJEM. 2015. doi:10.2310/8000.2013.131349
# [C6] Dargère S, Cormier H, Verdon R. Contaminants in blood cultures: importance, implications, interpretation and prevention.
#      Clin Microbiol Infect. 2018. doi:10.1016/j.cmi.2018.03.030
# [C7] Peri AM, et al. Rapid Diagnostic Tests and Antimicrobial Stewardship Programs for the Management of Bloodstream Infection:
#      a systematic review and network meta-analysis. Clin Infect Dis. 2024. doi:10.1093/cid/ciae234
# [C8] Altun O, et al. Evaluation of the FilmArray Blood Culture Identification Panel: results from a multicenter controlled trial.
#      J Clin Microbiol. 2016. doi:10.1128/JCM.01679-15

import csv
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Optional metrics (still NO-PANDAS)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Optional plotting (still NO-PANDAS)
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
# PATHS
# ===============================
MIMIC_DIR = Path("dataset/mimic/mimic-iii-clinical-database-demo-1.4")
MICRO_PATH = MIMIC_DIR / "MICROBIOLOGYEVENTS.csv"
PRESC_PATH = MIMIC_DIR / "PRESCRIPTIONS.csv"
ADMISSIONS_PATH = MIMIC_DIR / "ADMISSIONS.csv"
PATIENTS_PATH = MIMIC_DIR / "PATIENTS.csv"
D_ITEMS_PATH = MIMIC_DIR / "D_ITEMS.csv"
CHARTEVENTS_PATH = MIMIC_DIR / "CHARTEVENTS.csv"
D_LABITEMS_PATH = MIMIC_DIR / "D_LABITEMS.csv"
LABEVENTS_PATH = MIMIC_DIR / "LABEVENTS.csv"


# ===============================
# SETTINGS
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

HOURS_WINDOW = 24

ANTIBIOTICS = ["VANCOMYCIN", "CIPROFLOXACIN", "MEROPENEM", "PIPERACILLIN", "CEFTRIAXONE"]
ABX_ORDER = [a.lower() for a in ANTIBIOTICS]  # lowercase feature names (columns)

VITAL_ORDER = ["temperature_c", "wbc", "spo2", "age"]  # required order, lowercase feature names
NUMERIC_ORDER = VITAL_ORDER + ABX_ORDER

# bias-fix knobs
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.05
MAX_CLASS_WEIGHT = 15.0
BOTHER_EXTRA_DOWNWEIGHT = 0.5

DROPOUT = 0.2
BATCH_SIZE = 64

# Epoch optimization (instead of fixed EPOCHS):
# - Train up to MAX_EPOCHS, but stop early on no val_loss improvement.
MAX_EPOCHS = 120
EARLY_PATIENCE = 6
MIN_DELTA = 1e-4

WBC_SAMPLE_MAX = 200_000

# Guarantee test split has at least this many UNIQUE HADM_IDs
MIN_TEST_UNIQUE_HADM = 2


# ===============================
# CSV helpers (no pandas)
# ===============================
def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _norm_text(x: Any) -> str:
    """Lowercase categorical values + normalize whitespace."""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _read_header(path: Path) -> List[str]:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=enc) as f:
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
            with open(path, "r", newline="", encoding=enc) as f:
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

    pat_wanted = {"SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"], "DOB": ["DOB", "DATE_OF_BIRTH", "DATE OF BIRTH"]}
    dob_by_subject: Dict[int, datetime] = {}
    for row in _iter_csv_std(PATIENTS_PATH, pat_wanted):
        sid = _parse_int(row["SUBJECT_ID"])
        dob = _safe_parse_datetime_str(row["DOB"])
        if sid is None or dob is None:
            continue
        dob_by_subject[sid] = dob

    windows: Dict[int, Tuple[datetime, datetime]] = {}
    ages: Dict[int, float] = {}
    for hadm, (sid, adt) in admissions.items():
        dob = dob_by_subject.get(sid)
        if dob is None:
            continue
        age = float((adt - dob).days) / 365.2425
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
# (categorical VALUES lowercased!)
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
# (feature keys lowercase)
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


def make_sample_weights(y_labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_labels, minlength=len(CLASSES)).astype(np.float64)
    total = float(np.sum(counts))
    w = np.zeros_like(counts, dtype=np.float64)
    for i, c in enumerate(counts):
        w[i] = (total / (len(CLASSES) * c)) if c > 0 else 0.0

    bother_idx = CLASS_TO_INDEX.get("B:OTHER", None)
    if bother_idx is not None:
        w[bother_idx] *= float(BOTHER_EXTRA_DOWNWEIGHT)

    w = np.clip(w, 0.0, float(MAX_CLASS_WEIGHT))
    return w[y_labels].astype(np.float32)


# ===============================
# Vitals signal: permutation importance + Pearson r,p
# ===============================
def _mean_cce(y_true_oh: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    y = y_true_oh.astype(np.float64)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def _norm_cdf(z: float) -> float:
    # Standard normal CDF using erf (no scipy required)
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Returns (r, p).
    Prefers scipy if available; otherwise uses Fisher z approximation:
      z = atanh(r) * sqrt(n-3),  p ~= 2*(1 - Phi(|z|))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.shape[0])
    if n < 4:
        return 0.0, 1.0

    # Try scipy first (exact p)
    try:
        from scipy import stats as _sp_stats  # type: ignore
        r, p = _sp_stats.pearsonr(x, y)
        if not np.isfinite(r) or not np.isfinite(p):
            return 0.0, 1.0
        return float(r), float(p)
    except Exception:
        pass

    # Manual r
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, 1.0
    r = float(np.sum(x0 * y0) / denom)

    # Fisher z approx p-value
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = float(np.arctanh(r_clip) * np.sqrt(max(n - 3, 1)))
    p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    if not np.isfinite(p):
        p = 1.0
    return r, p


def plot_vitals_signal(
    model: tf.keras.Model,
    X_te: np.ndarray,
    y_te_oh: np.ndarray,
    num_te_raw: np.ndarray,
    cat_dim: int,
    out_path: str = "vitals_feature_signal.png",
    n_repeats: int = 3,
    seed: int = SEED,
) -> None:
    """
    Always PRINTS:
      - baseline CE loss
      - permutation importance (Δloss) for vitals
      - Pearson r + p-value for each vital vs P(bacterial)

    Only PLOTS if matplotlib is available.
    """
    base_probs = model.predict(X_te, verbose=0)
    base_loss = _mean_cce(y_te_oh, base_probs)

    vital_cols_in_X = [cat_dim + i for i in range(4)]
    vital_names = list(VITAL_ORDER)

    rng = np.random.default_rng(seed)
    imp_mean: List[float] = []
    imp_std: List[float] = []

    # Permutation importance on TEST (Δloss)
    for j in vital_cols_in_X:
        losses: List[float] = []
        for _ in range(int(n_repeats)):
            Xp = X_te.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]
            p = model.predict(Xp, verbose=0)
            losses.append(_mean_cce(y_te_oh, p))
        arr = np.asarray(losses, dtype=np.float64)
        imp_mean.append(float(np.mean(arr) - base_loss))
        imp_std.append(float(np.std(arr, ddof=1) if arr.size > 1 else 0.0))

    # P(bacterial) = sum probs of "B:" classes
    b_idxs = np.asarray([i for i, c in enumerate(CLASSES) if c.startswith("B:")], dtype=np.int32)
    bact_prob = np.sum(base_probs[:, b_idxs], axis=1).astype(np.float64) if b_idxs.size else np.zeros((X_te.shape[0],), dtype=np.float64)

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
    print("[INFO] Permutation importance (Δloss) + Pearson r,p vs P(bacterial):")
    for name, m, s, r, p, a in ranked:
        print(f"  {name:14s} -> Δloss={m:.6f} (±{s:.6f})  r={r:+.4f}  p={p:.3e}  |r|={a:.4f}")

    if plt is None:
        print("[WARN] matplotlib not available; skipping vitals plot PNG (but p-values were printed).")
        return

    # Plot
    x = np.arange(len(vital_names), dtype=np.float64)
    w = 0.38

    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(x - w / 2.0, imp_mean, width=w, label="Permutation importance (Δloss)")
    ax.bar(x + w / 2.0, corr_abs, width=w, label="|Pearson r| vs P(bacterial)")
    ax.set_xticks(x)
    ax.set_xticklabels(vital_names, rotation=0)
    ax.set_ylabel("Signal strength (scaled)")
    ax.set_title("Vitals signal (empirical; expected Fever > WBC > SpO₂)")
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
# Metrics reporting
# ===============================
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
    print("\n[INFO] Macro avg:   prec={:.3f} rec={:.3f} f1={:.3f}".format(macro[0], macro[1], macro[2]))
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
    # map to binary: MRSA=1, MSSA=0
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
# Split helper: ensure >= 2 unique HADM in TEST
# ===============================
def split_with_min_unique_hadm(
    cat_arr: np.ndarray,
    num_arr: np.ndarray,
    y_arr: np.ndarray,
    hadm_arr: np.ndarray,
    test_size: float,
    seed: int,
    min_unique_test_hadm: int = MIN_TEST_UNIQUE_HADM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Try multiple seeds to ensure test has at least `min_unique_test_hadm` UNIQUE HADM_IDs.
    Keeps stratification on labels when possible.
    """
    unique_hadm = np.unique(hadm_arr)
    if unique_hadm.size < min_unique_test_hadm:
        raise RuntimeError(
            f"Not enough unique HADM_IDs overall ({unique_hadm.size}) to satisfy test requirement ({min_unique_test_hadm})."
        )

    for delta in range(0, 200):
        rs = int(seed + delta)
        cat_tr, cat_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te = train_test_split(
            cat_arr,
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
            return cat_tr, cat_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te

    raise RuntimeError(f"Could not find a split with >= {min_unique_test_hadm} unique HADM_IDs in TEST after many attempts.")


def main():
    for p in [MICRO_PATH, PRESC_PATH, ADMISSIONS_PATH, PATIENTS_PATH, D_ITEMS_PATH, CHARTEVENTS_PATH, D_LABITEMS_PATH, LABEVENTS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")

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

    # Join micro + vitals + abx
    cat_data: List[List[str]] = []
    num_data: List[List[float]] = []
    y_labels: List[int] = []
    hadm_ids: List[int] = []

    for r in micro_rows:
        hadm = int(r["hadm_id"])
        vit = vitals_by_hadm.get(hadm)
        if vit is None:
            continue
        abx = abx_by_hadm.get(hadm, {k: 0.0 for k in ABX_ORDER})

        cat_data.append([r["spec_type_desc"], r["interpretation"]])

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
    cat_arr = np.asarray(cat_data, dtype=object)
    num_arr = np.asarray(num_data, dtype=np.float32)

    print(f"[INFO] Joined rows={len(y_labels_arr)} | numeric_dim={num_arr.shape[1]} | numeric_order={NUMERIC_ORDER}")

    # Split test ensuring >=2 unique HADM_ID in TEST
    cat_tr, cat_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te = split_with_min_unique_hadm(
        cat_arr,
        num_arr,
        y_labels_arr,
        hadm_arr,
        test_size=0.2,
        seed=SEED,
        min_unique_test_hadm=MIN_TEST_UNIQUE_HADM,
    )
    print(f"[INFO] TEST unique HADM_ID: {len(np.unique(hadm_te))} (requirement >= {MIN_TEST_UNIQUE_HADM})")

    # Split train->val stratified
    cat_tr2, cat_va, num_tr2, num_va, y_tr2, y_va, hadm_tr2, hadm_va = train_test_split(
        cat_tr, num_tr, y_tr, hadm_tr,
        test_size=0.2, random_state=SEED,
        stratify=y_tr if len(np.unique(y_tr)) > 1 else None,
    )

    # Fit OneHotEncoder on TRAIN ONLY
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_cat_tr = ohe.fit_transform(cat_tr2)
    X_cat_va = ohe.transform(cat_va)
    X_cat_te = ohe.transform(cat_te)

    # Standardize vitals on TRAIN ONLY (first 4 numeric cols)
    mu = num_tr2[:, :4].mean(axis=0).astype(np.float32)
    sd = (num_tr2[:, :4].std(axis=0) + 1e-6).astype(np.float32)

    def apply_scaling(num_x: np.ndarray) -> np.ndarray:
        out = num_x.astype(np.float32).copy()
        out[:, :4] = (out[:, :4] - mu) / sd
        return out

    X_num_tr = apply_scaling(num_tr2)
    X_num_va = apply_scaling(num_va)
    X_num_te = apply_scaling(num_te)

    X_tr = np.hstack([X_cat_tr.astype(np.float32), X_num_tr])
    X_va = np.hstack([X_cat_va.astype(np.float32), X_num_va])
    X_te = np.hstack([X_cat_te.astype(np.float32), X_num_te])

    y_tr_oh = tf.keras.utils.to_categorical(y_tr2, num_classes=len(CLASSES))
    y_va_oh = tf.keras.utils.to_categorical(y_va, num_classes=len(CLASSES))
    y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes=len(CLASSES))

    sample_weight = make_sample_weights(y_tr2) if USE_CLASS_WEIGHTS else None

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(LABEL_SMOOTHING))
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(EARLY_PATIENCE),
            min_delta=float(MIN_DELTA),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=max(2, int(EARLY_PATIENCE // 2)),
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_tr, y_tr_oh,
        epochs=int(MAX_EPOCHS), batch_size=BATCH_SIZE,
        validation_data=(X_va, y_va_oh),
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Report best epoch by val_loss
    val_losses = history.history.get("val_loss", [])
    val_accs = history.history.get("val_accuracy", [])
    if val_losses:
        best_ep = int(np.argmin(np.asarray(val_losses, dtype=np.float64)) + 1)
        best_vl = float(np.min(np.asarray(val_losses, dtype=np.float64)))
        msg = f"[INFO] Best epoch (min val_loss): {best_ep} | val_loss={best_vl:.6f}"
        if val_accs and 0 < best_ep <= len(val_accs):
            msg += f" | val_acc={float(val_accs[best_ep - 1]):.4f}"
        print(msg)

    probs_te = model.predict(X_te, verbose=0)
    y_pred = np.argmax(probs_te, axis=1)
    acc = float((y_pred == y_te).mean())
    print(f"\n=== GENERAL ACCURACY ON TEST SET: {acc:.4f} ===")

    # Per-class metrics (includes F1)
    report_multiclass_metrics(y_true=y_te, y_pred=y_pred)

    # MRSA vs MSSA emphasis
    report_mrsa_vs_mssa(y_true=y_te, y_pred=y_pred)

    # Vitals signal report + plot (p-values always printed)
    cat_dim = int(X_cat_tr.shape[1])
    plot_vitals_signal(
        model=model,
        X_te=X_te,
        y_te_oh=y_te_oh,
        num_te_raw=num_te,  # raw (unscaled) numeric from the split
        cat_dim=cat_dim,
        out_path="vitals_feature_signal.png",
        n_repeats=3,
        seed=SEED,
    )

    print("\n=== PREDICTIONS PER HADM_ID (sorted by prob) ===")
    for hid in np.unique(hadm_te):
        idxs = np.where(hadm_te == hid)[0]
        mean_probs = probs_te[idxs].mean(axis=0).reshape(-1)
        pairs = sorted(zip(CLASSES, mean_probs.tolist()), key=lambda t: t[1], reverse=True)

        print(f"\nHADM_ID: {int(hid)}  (n_rows={len(idxs)})")
        for cls, p in pairs:
            print(f"{cls:65s} -> {p:.4f}")

    print("\n[INFO] Done.")
    print(f"[INFO] Feature 'column names' lowercase (numeric order): {NUMERIC_ORDER}")
    print("[INFO] Categorical 'values' lowercased: spec_type_desc, interpretation")


if __name__ == "__main__":
    main()
