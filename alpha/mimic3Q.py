import csv
import gzip
import re
import os
import sys
import math
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

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
random.seed(SEED)

HOURS_WINDOW = int(os.environ.get("HOURS_WINDOW", "24"))

ANTIBIOTICS = ["VANCOMYCIN", "CIPROFLOXACIN", "MEROPENEM", "PIPERACILLIN", "CEFTRIAXONE"]
ABX_ORDER = [a.lower() for a in ANTIBIOTICS]

VITAL_ORDER = ["temperature_c", "wbc", "spo2", "age"]
NUMERIC_ORDER = VITAL_ORDER + ABX_ORDER

USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "1").strip() == "1"
MAX_CLASS_WEIGHT = float(os.environ.get("MAX_CLASS_WEIGHT", "15.0"))
BOTHER_EXTRA_DOWNWEIGHT = float(os.environ.get("BOTHER_EXTRA_DOWNWEIGHT", "0.5"))

MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "80"))
EARLY_PATIENCE = int(os.environ.get("EARLY_PATIENCE", "10"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-6"))

MAX_TEXT_TOKENS = int(os.environ.get("MAX_TEXT_TOKENS", "5000"))
TEXT_BINARY = os.environ.get("TEXT_BINARY", "0").strip() == "1"

LR = float(os.environ.get("LR", "0.03"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))

CALIB_BINS = int(os.environ.get("CALIB_BINS", "10"))
BIAS_MIN_GROUP_N = int(os.environ.get("BIAS_MIN_GROUP_N", "25"))
WBC_SAMPLE_MAX = 200_000
MIN_TEST_UNIQUE_HADM = int(os.environ.get("MIN_TEST_UNIQUE_HADM", "2"))

TARGET_STOP_METRIC = os.environ.get("TARGET_STOP_METRIC", "acc").strip().lower()
TARGET_ACC = float(os.environ.get("TARGET_ACC", "0.95"))
TARGET_ACC_KIND = os.environ.get("TARGET_ACC_KIND", "overall").strip().lower()
TARGET_F1 = float(os.environ.get("TARGET_F1", "0.925"))
TARGET_F1_KIND = os.environ.get("TARGET_F1_KIND", "macro").strip().lower()

MAX_TRAIN_RESTARTS = int(os.environ.get("MAX_TRAIN_RESTARTS", "3"))
RETRAIN_ON_FULL_TRAIN = os.environ.get("RETRAIN_ON_FULL_TRAIN", "1").strip() == "1"

# ---- Pure Python MLP ----
MLP_HIDDEN_DIM = int(os.environ.get("MLP_HIDDEN_DIM", "48"))
MLP_INIT_SCALE = float(os.environ.get("MLP_INIT_SCALE", "0.05"))
ACTIVATION_NAME = os.environ.get("ACTIVATION_NAME", "pade_tanh").strip().lower()  # "tanh" | "pade_tanh"


# ===============================
# Pure Python numeric helpers
# ===============================
def isfinite(x: float) -> bool:
    return math.isfinite(float(x))


def clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / float(len(xs))


def variance(xs: List[float], ddof: int = 0) -> float:
    n = len(xs)
    if n == 0 or n - ddof <= 0:
        return 0.0
    m = mean(xs)
    return sum((x - m) * (x - m) for x in xs) / float(n - ddof)


def std(xs: List[float], ddof: int = 0) -> float:
    return math.sqrt(max(variance(xs, ddof=ddof), 0.0))


def median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def argmax(values: List[float]) -> int:
    if not values:
        raise ValueError("argmax() received empty list")
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i
    return best_i


def bincount(values: List[int], minlength: int) -> List[int]:
    out = [0] * int(minlength)
    for v in values:
        iv = int(v)
        if 0 <= iv < minlength:
            out[iv] += 1
    return out


def softmax_row(logits_row: List[float]) -> List[float]:
    if not logits_row:
        return []
    m = max(logits_row)
    exps = [math.exp(x - m) for x in logits_row]
    s = sum(exps)
    if s == 0.0:
        return [0.0 for _ in logits_row]
    return [x / s for x in exps]


def probs_from_logits(logits: List[List[float]]) -> List[List[float]]:
    return [softmax_row(row) for row in logits]


def unique(values: List[Any]) -> List[Any]:
    return sorted(set(values))


# ===============================
# Activations: tanh / pade_tanh
# ===============================
def tanh_act(x: float) -> float:
    return math.tanh(x)


def tanh_deriv_from_pre(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


def pade_tanh_act(x: float) -> float:
    """
    Padé approximation of tanh(x):
        tanh(x) ~= x * (27 + x^2) / (27 + 9x^2)
    """
    x = clip(float(x), -3.0, 3.0)
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)


def pade_tanh_deriv_from_pre(x: float) -> float:
    """
    Derivative of the clipped Padé tanh approximation.
    """
    x = clip(float(x), -3.0, 3.0)
    x2 = x * x
    num = (x2 - 9.0) * (x2 - 9.0)
    den = 9.0 * (x2 + 3.0) * (x2 + 3.0)
    return num / den


def get_activation_pair(name: str):
    n = str(name).strip().lower()
    if n == "tanh":
        return tanh_act, tanh_deriv_from_pre
    return pade_tanh_act, pade_tanh_deriv_from_pre


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
                _ = next(r)
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
        yield
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with _open_text(path, enc) as f:
                r = csv.reader(f)
                _ = next(r)
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
# MAP ORGANISM -> CLASS
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

        if not isfinite(age):
            continue
        if age > 120.0:
            age = 90.0
        age = clip(age, 0.0, 110.0)
        windows[hadm] = (adt, adt + timedelta(hours=int(HOURS_WINDOW)))
        ages[hadm] = age

    return windows, ages


# ===============================
# Compute vitals/labs per HADM_ID
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

    sample: List[float] = []
    rng = random.Random(SEED)
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
            j = rng.randrange(seen)
            if j < WBC_SAMPLE_MAX:
                sample[j] = float(v)

    scale_by_1000 = False
    if len(sample) > 0:
        med = median(sample)
        if med < 200.0:
            scale_by_1000 = True

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
# Load MICROBIOLOGYEVENTS -> rows
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
# Load PRESCRIPTIONS -> hadm_id -> antibiotics binary
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
# Tokenization and sparse features
# ===============================
def _tokenize(text: str) -> List[str]:
    return [t for t in str(text).strip().lower().split() if t]


def build_vocab(texts: List[str], max_tokens: int) -> Dict[str, int]:
    cnt = Counter()
    for s in texts:
        for t in _tokenize(str(s)):
            cnt[t] += 1
    keep = max(2, int(max_tokens))
    vocab_items = cnt.most_common(max(0, keep))
    vocab = {}
    idx = 0
    for tok, _ in vocab_items:
        vocab[tok] = idx
        idx += 1
    return vocab


def fit_numeric_scaler(num_rows: List[List[float]], n_cont: int = 4) -> Tuple[List[float], List[float]]:
    mu: List[float] = []
    sd: List[float] = []
    for j in range(n_cont):
        col = [float(row[j]) for row in num_rows]
        m = mean(col)
        s = std(col) + 1e-6
        mu.append(m)
        sd.append(s)
    return mu, sd


def apply_numeric_scaling(num_rows: List[List[float]], mu: List[float], sd: List[float], n_cont: int = 4) -> List[List[float]]:
    out: List[List[float]] = []
    for row in num_rows:
        new_row = [float(x) for x in row]
        for j in range(n_cont):
            new_row[j] = (new_row[j] - mu[j]) / sd[j]
        out.append(new_row)
    return out


def make_sparse_feature(
    num_row: List[float],
    text: str,
    vocab: Dict[str, int],
    num_offset: int = 0,
    text_offset: Optional[int] = None,
) -> Dict[int, float]:
    if text_offset is None:
        text_offset = len(num_row)

    feat: Dict[int, float] = {}

    for j, v in enumerate(num_row):
        fv = float(v)
        if abs(fv) > 1e-12:
            feat[num_offset + j] = fv

    tok_counts: Dict[int, float] = defaultdict(float)
    toks = _tokenize(text)
    if toks:
        if TEXT_BINARY:
            seen = set()
            for tok in toks:
                idx = vocab.get(tok)
                if idx is not None:
                    seen.add(idx)
            for idx in seen:
                tok_counts[idx] = 1.0
        else:
            for tok in toks:
                idx = vocab.get(tok)
                if idx is not None:
                    tok_counts[idx] += 1.0
            total = float(sum(tok_counts.values()))
            if total > 0:
                for idx in list(tok_counts.keys()):
                    tok_counts[idx] /= total

    for idx, val in tok_counts.items():
        feat[text_offset + idx] = val

    return feat


def make_sparse_dataset(texts: List[str], num_rows: List[List[float]], vocab: Dict[str, int]) -> Tuple[List[Dict[int, float]], int]:
    X: List[Dict[int, float]] = []
    num_dim = len(num_rows[0]) if num_rows else len(NUMERIC_ORDER)
    text_offset = num_dim
    total_dim = num_dim + len(vocab)
    for txt, num_row in zip(texts, num_rows):
        X.append(make_sparse_feature(num_row, txt, vocab, num_offset=0, text_offset=text_offset))
    return X, total_dim


# ===============================
# Pure Python split helpers
# ===============================
def stratified_split_indices(labels: List[int], test_size: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        label_to_indices[int(y)].append(i)

    train_idx: List[int] = []
    test_idx: List[int] = []

    for _label, idxs in label_to_indices.items():
        idxs = list(idxs)
        rng.shuffle(idxs)

        if len(idxs) == 1:
            train_part = idxs[:]
            test_part = []
        else:
            n_test = int(round(len(idxs) * float(test_size)))
            n_test = max(1, min(len(idxs) - 1, n_test))
            test_part = idxs[:n_test]
            train_part = idxs[n_test:]

        train_idx.extend(train_part)
        test_idx.extend(test_part)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def take_by_indices(xs: List[Any], idxs: List[int]) -> List[Any]:
    return [xs[i] for i in idxs]


def split_with_min_unique_hadm(
    text_arr: List[str],
    num_arr: List[List[float]],
    y_arr: List[int],
    hadm_arr: List[int],
    test_size: float,
    seed: int,
    min_unique_test_hadm: int = MIN_TEST_UNIQUE_HADM,
):
    unique_hadm_all = set(hadm_arr)
    if len(unique_hadm_all) < min_unique_test_hadm:
        raise RuntimeError(
            f"Not enough unique HADM_IDs overall ({len(unique_hadm_all)}) to satisfy test requirement ({min_unique_test_hadm})."
        )

    for delta in range(0, 200):
        rs = int(seed + delta)
        tr_idx, te_idx = stratified_split_indices(y_arr, float(test_size), rs)
        hadm_te = take_by_indices(hadm_arr, te_idx)
        if len(set(hadm_te)) >= min_unique_test_hadm:
            if delta > 0:
                print(f"[INFO] Adjusted split seed -> {rs} to satisfy test unique HADM_ID >= {min_unique_test_hadm}")
            return (
                take_by_indices(text_arr, tr_idx),
                take_by_indices(text_arr, te_idx),
                take_by_indices(num_arr, tr_idx),
                take_by_indices(num_arr, te_idx),
                take_by_indices(y_arr, tr_idx),
                take_by_indices(y_arr, te_idx),
                take_by_indices(hadm_arr, tr_idx),
                take_by_indices(hadm_arr, te_idx),
            )

    raise RuntimeError(f"Could not find a split with >= {min_unique_test_hadm} unique HADM_IDs in TEST after many attempts.")


# ===============================
# Pure Python metrics
# ===============================
def confusion_matrix_multiclass(y_true: List[int], y_pred: List[int], num_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt][yp] += 1
    return cm


def precision_recall_f1_support_per_class(y_true: List[int], y_pred: List[int], num_classes: int):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    sups: List[int] = []

    for i in range(num_classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(num_classes)) - tp
        sup = sum(cm[i])

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / float(prec + rec) if (prec + rec) > 0 else 0.0

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        sups.append(sup)

    return precs, recs, f1s, sups


def macro_average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def weighted_average(values: List[float], weights: List[int]) -> float:
    total_w = float(sum(weights))
    if total_w <= 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / float(len(y_true))


def binary_roc_curve(y_true_binary: List[int], scores: List[float]):
    paired = list(zip(scores, y_true_binary))
    paired.sort(key=lambda x: x[0], reverse=True)

    pos_total = sum(y_true_binary)
    neg_total = len(y_true_binary) - pos_total
    if pos_total == 0 or neg_total == 0:
        return [], [], []

    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]

    tp = 0
    fp = 0
    prev_score = None
    for score, y in paired:
        if prev_score is not None and score != prev_score:
            fpr.append(fp / float(neg_total))
            tpr.append(tp / float(pos_total))
            thresholds.append(prev_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    fpr.append(fp / float(neg_total))
    tpr.append(tp / float(pos_total))
    thresholds.append(prev_score if prev_score is not None else 0.0)
    return fpr, tpr, thresholds


def auc_trapezoid(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        my = 0.5 * (ys[i] + ys[i - 1])
        area += dx * my
    return area


def binary_roc_auc(y_true_binary: List[int], scores: List[float]) -> float:
    fpr, tpr, _ = binary_roc_curve(y_true_binary, scores)
    if not fpr:
        return float("nan")
    return auc_trapezoid(fpr, tpr)


def binary_pr_curve(y_true_binary: List[int], scores: List[float]):
    paired = list(zip(scores, y_true_binary))
    paired.sort(key=lambda x: x[0], reverse=True)

    total_pos = sum(y_true_binary)
    total_neg = len(y_true_binary) - total_pos
    if total_pos == 0 or total_neg == 0:
        return [], [], []

    precision = [1.0]
    recall = [0.0]
    thresholds = []

    tp = 0.0
    fp = 0.0
    for score, y in paired:
        if y == 1:
            tp += 1.0
        else:
            fp += 1.0
        p = tp / (tp + fp + 1e-12)
        r = tp / total_pos
        precision.append(p)
        recall.append(r)
        thresholds.append(score)

    return precision, recall, thresholds


def binary_pr_auc(y_true_binary: List[int], scores: List[float]) -> float:
    precision, recall, _ = binary_pr_curve(y_true_binary, scores)
    if not precision:
        return float("nan")
    return auc_trapezoid(recall, precision)


def compute_multiclass_pr_auc(logits: List[List[float]], labels: List[int], num_classes: int = 12):
    probs = probs_from_logits(logits)

    pr_aucs = []
    class_metrics = {}

    for c in range(num_classes):
        true_binary = [1 if y == c else 0 for y in labels]
        total_positives = sum(true_binary)

        if total_positives == 0:
            continue

        class_probs = [row[c] for row in probs]
        auc = binary_pr_auc(true_binary, class_probs)
        if isfinite(auc):
            pr_aucs.append(auc)
            class_metrics[c] = auc

    macro_pr_auc = sum(pr_aucs) / float(len(pr_aucs)) if pr_aucs else 0.0
    return macro_pr_auc, class_metrics


def multiclass_auc_pr_from_probs(y_true: List[int], probs: List[List[float]]):
    C = len(probs[0]) if probs else 0
    per_roc: Dict[int, float] = {}
    per_pr: Dict[int, float] = {}
    supports = bincount(y_true, minlength=C)
    total = float(sum(supports))

    for i in range(C):
        y_bin = [1 if y == i else 0 for y in y_true]
        pos = sum(y_bin)
        neg = len(y_bin) - pos
        if pos == 0 or neg == 0:
            continue

        scores = [row[i] for row in probs]
        ra = binary_roc_auc(y_bin, scores)
        pa = binary_pr_auc(y_bin, scores)
        if isfinite(ra):
            per_roc[i] = ra
        if isfinite(pa):
            per_pr[i] = pa

    def _avg(d: Dict[int, float], weighted: bool) -> float:
        if not d:
            return float("nan")
        if not weighted:
            return sum(d.values()) / float(len(d))
        wsum = 0.0
        ssum = 0.0
        for i, v in d.items():
            w = float(supports[i]) / max(total, 1.0)
            wsum += w * v
            ssum += w
        return wsum / max(ssum, 1e-12)

    macro_roc = _avg(per_roc, weighted=False)
    w_roc = _avg(per_roc, weighted=True)
    macro_pr = _avg(per_pr, weighted=False)
    w_pr = _avg(per_pr, weighted=True)
    return per_roc, per_pr, macro_roc, w_roc, macro_pr, w_pr


def _compute_target_f1(kind: str, y_true_int: List[int], y_pred_int: List[int]) -> Optional[float]:
    kind = str(kind).strip().lower()
    pr, rc, f1, sup = precision_recall_f1_support_per_class(y_true_int, y_pred_int, len(CLASSES))

    if kind == "macro":
        return macro_average(f1)
    if kind == "weighted":
        return weighted_average(f1, sup)

    if kind == "mrsa_mssa":
        mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
        mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
        if mrsa_idx is None or mssa_idx is None:
            return None
        pairs = [(yt, yp) for yt, yp in zip(y_true_int, y_pred_int) if yt in {mrsa_idx, mssa_idx}]
        if not pairs:
            return None

        yt = [1 if a == mrsa_idx else 0 for a, _ in pairs]
        yp = [1 if b == mrsa_idx else 0 for _, b in pairs]

        tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
        fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        return 2.0 * prec * rec / float(prec + rec) if (prec + rec) > 0 else 0.0

    return None


def _compute_target_acc(kind: str, y_true_int: List[int], y_pred_int: List[int]) -> Optional[float]:
    kind = str(kind).strip().lower()
    if kind in {"overall", "micro"}:
        return accuracy_score(y_true_int, y_pred_int)

    if kind == "mrsa_mssa":
        mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
        mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
        if mrsa_idx is None or mssa_idx is None:
            return None
        pairs = [(yt, yp) for yt, yp in zip(y_true_int, y_pred_int) if yt in {mrsa_idx, mssa_idx}]
        if not pairs:
            return None
        return sum(1 for a, b in pairs if a == b) / float(len(pairs))

    return None


# ===============================
# Reporting
# ===============================
def report_multiclass_metrics(y_true: List[int], y_pred: List[int]) -> None:
    pr, rc, f1, sup = precision_recall_f1_support_per_class(y_true, y_pred, len(CLASSES))
    print("\n=== PER-CLASS METRICS (TEST) ===")
    print("Class                                                   |  Prec   Rec    F1   Support")
    print("-" * 85)
    for i in range(len(CLASSES)):
        print(f"{INDEX_TO_CLASS[i]:55s} | {pr[i]:6.3f} {rc[i]:6.3f} {f1[i]:6.3f} {int(sup[i]):8d}")

    print("\n[INFO] Macro avg:    prec={:.3f} rec={:.3f} f1={:.3f}".format(
        macro_average(pr), macro_average(rc), macro_average(f1)
    ))
    print("[INFO] Weighted avg: prec={:.3f} rec={:.3f} f1={:.3f}".format(
        weighted_average(pr, sup), weighted_average(rc, sup), weighted_average(f1, sup)
    ))


def report_mrsa_vs_mssa(y_true: List[int], y_pred: List[int]) -> None:
    mrsa_idx = CLASS_TO_INDEX.get(MRSA_LABEL, None)
    mssa_idx = CLASS_TO_INDEX.get(MSSA_LABEL, None)
    if mrsa_idx is None or mssa_idx is None:
        print("[WARN] MRSA/MSSA class indices missing; skipping dedicated report.")
        return

    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt in {mrsa_idx, mssa_idx}]
    n = len(pairs)
    if n == 0:
        print("\n=== MRSA vs MSSA (TEST) ===")
        print("[WARN] No MRSA/MSSA samples in TEST; cannot compute.")
        return

    yt = [1 if a == mrsa_idx else 0 for a, _ in pairs]
    yp = [1 if b == mrsa_idx else 0 for _, b in pairs]

    cm = [[0, 0], [0, 0]]
    for a, b in zip(yt, yp):
        cm[a][b] += 1

    def prf_for_label(label: int):
        tp = cm[label][label]
        fn = sum(cm[label]) - tp
        fp = cm[0][label] + cm[1][label] - tp
        support = sum(cm[label])
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / float(prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1, support

    p0, r0, f0, s0 = prf_for_label(0)
    p1, r1, f1v, s1 = prf_for_label(1)

    print("\n=== MRSA vs MSSA (STAPH AUREUS COAG +) — TEST SUBSET ===")
    print(f"[INFO] Subset size: {n} (MSSA={int(s0)}, MRSA={int(s1)})")
    print("Label |  Prec   Rec    F1   Support")
    print(f"MSSA  | {p0:6.3f} {r0:6.3f} {f0:6.3f} {int(s0):8d}")
    print(f"MRSA  | {p1:6.3f} {r1:6.3f} {f1v:6.3f} {int(s1):8d}")
    print("\nConfusion matrix (rows=true, cols=pred), [MSSA, MRSA]:")
    print(cm)


def _confusion_stats_multiclass(cm: List[List[int]]) -> Dict[int, Dict[str, float]]:
    total = sum(sum(row) for row in cm)
    stats: Dict[int, Dict[str, float]] = {}
    n = len(cm)
    for i in range(n):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n)) - tp
        tn = total - tp - fn - fp

        sens = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * ppv * sens / float(ppv + sens)) if (ppv + sens) > 0 else 0.0
        sup = sum(cm[i])

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


def report_confusion_and_rates(y_true: List[int], y_pred: List[int], title: str = "TEST") -> None:
    cm = confusion_matrix_multiclass(y_true, y_pred, len(CLASSES))
    print(f"\n=== CONFUSION MATRIX ({title}) ===")
    print("[rows=true, cols=pred] shape:", (len(cm), len(cm[0]) if cm else 0))
    for row in cm:
        print(row)

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


def report_auc_pr(y_true: List[int], probs: List[List[float]], title: str = "TEST") -> None:
    per_roc, per_pr, macro_roc, w_roc, macro_pr, w_pr = multiclass_auc_pr_from_probs(y_true, probs)

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
# Calibration / ECE / Temperature Scaling
# ===============================
class PurePythonECE:
    def __init__(self, n_bins: int = 15):
        self.n_bins = int(n_bins)
        self.bin_lowers = [i / self.n_bins for i in range(self.n_bins)]
        self.bin_uppers = [(i + 1) / self.n_bins for i in range(self.n_bins)]

    def __call__(self, logits: List[List[float]], labels: List[int]) -> float:
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have same length")
        if not logits:
            return 0.0

        confidences = []
        accuracies = []
        for row, label in zip(logits, labels):
            probs = softmax_row(row)
            pred = argmax(probs)
            conf = probs[pred]
            confidences.append(conf)
            accuracies.append(1.0 if pred == label else 0.0)

        n = float(len(labels))
        ece = 0.0

        for lo, hi in zip(self.bin_lowers, self.bin_uppers):
            idxs = [i for i, conf in enumerate(confidences) if conf > lo and conf <= hi]
            if not idxs:
                continue

            prop = len(idxs) / n
            acc_bin = mean([accuracies[i] for i in idxs])
            conf_bin = mean([confidences[i] for i in idxs])
            ece += abs(acc_bin - conf_bin) * prop

        return ece


class TemperatureScaler:
    def __init__(self, init_temperature: float = 1.5):
        self.temperature = float(init_temperature)

    def scale_row(self, logits_row: List[float]) -> List[float]:
        t = max(self.temperature, 1e-8)
        return [x / t for x in logits_row]

    def scale_logits(self, logits: List[List[float]]) -> List[List[float]]:
        return [self.scale_row(row) for row in logits]

    def _mean_cross_entropy(self, logits: List[List[float]], labels: List[int]) -> float:
        total = 0.0
        eps = 1e-12
        for row, y in zip(logits, labels):
            probs = softmax_row(self.scale_row(row))
            py = max(probs[y], eps)
            total += -math.log(py)
        return total / float(len(labels)) if labels else 0.0

    def _gradient(self, logits: List[List[float]], labels: List[int]) -> float:
        t = max(self.temperature, 1e-8)
        grad_sum = 0.0
        for row, y in zip(logits, labels):
            scaled = [x / t for x in row]
            probs = softmax_row(scaled)
            expected_logit = sum(p * l for p, l in zip(probs, row))
            grad_sum += (row[y] - expected_logit) / (t * t)
        return grad_sum / float(len(labels)) if labels else 0.0

    def fit(self, logits: List[List[float]], labels: List[int], lr: float = 0.01, max_iter: int = 200):
        prev_loss = self._mean_cross_entropy(logits, labels)
        for _ in range(int(max_iter)):
            grad = self._gradient(logits, labels)
            new_t = self.temperature - lr * grad
            if new_t <= 1e-6:
                new_t = 1e-6
            self.temperature = new_t
            cur_loss = self._mean_cross_entropy(logits, labels)
            if abs(cur_loss - prev_loss) < 1e-7:
                break
            prev_loss = cur_loss
        return self.temperature


def calibration_report_from_probs(y_true: List[int], probs: List[List[float]], title: str, out_png: Optional[str] = None) -> None:
    if not probs:
        print(f"\n=== CALIBRATION ({title}) ===")
        print("[WARN] Empty probability table.")
        return

    conf = [max(row) for row in probs]
    pred = [argmax(row) for row in probs]
    corr = [1.0 if p == y else 0.0 for p, y in zip(pred, y_true)]

    brier_rows = []
    for row, y in zip(probs, y_true):
        yy = [0.0] * len(row)
        yy[y] = 1.0
        brier_rows.append(sum((a - b) * (a - b) for a, b in zip(row, yy)))
    brier = mean(brier_rows)

    bins = max(2, CALIB_BINS)
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    rows = []
    n = float(len(conf))

    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        idxs = [i for i, c in enumerate(conf) if (c >= lo and c < hi) or (b == bins - 1 and c <= hi and c >= lo)]
        if not idxs:
            continue
        acc_b = mean([corr[i] for i in idxs])
        conf_b = mean([conf[i] for i in idxs])
        ece += (len(idxs) / max(n, 1.0)) * abs(acc_b - conf_b)
        rows.append((lo, hi, len(idxs), acc_b, conf_b, abs(acc_b - conf_b)))

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


# ===============================
# Pure Python MLP
# ===============================
class PurePythonMLP:
    """
    Pure Python MLP:
        sparse input -> hidden(tanh / pade_tanh) -> logits -> softmax
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        activation_name: str = "pade_tanh",
        seed: int = 42,
        init_scale: float = 0.05,
    ):
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.activation_name = str(activation_name).strip().lower()

        self.act, self.act_deriv = get_activation_pair(self.activation_name)

        rng = random.Random(seed)
        s = float(init_scale)

        self.W1: List[List[float]] = [
            [rng.uniform(-s, s) for _ in range(self.num_features)]
            for _ in range(self.hidden_dim)
        ]
        self.b1: List[float] = [0.0 for _ in range(self.hidden_dim)]

        self.W2: List[List[float]] = [
            [rng.uniform(-s, s) for _ in range(self.hidden_dim)]
            for _ in range(self.num_classes)
        ]
        self.b2: List[float] = [0.0 for _ in range(self.num_classes)]

    def state_dict(self):
        return {
            "W1": [row[:] for row in self.W1],
            "b1": self.b1[:],
            "W2": [row[:] for row in self.W2],
            "b2": self.b2[:],
        }

    def load_state_dict(self, state):
        self.W1 = [row[:] for row in state["W1"]]
        self.b1 = state["b1"][:]
        self.W2 = [row[:] for row in state["W2"]]
        self.b2 = state["b2"][:]

    def forward_one(self, x: Dict[int, float]):
        hidden_pre = self.b1[:]
        for j, v in x.items():
            fv = float(v)
            for h in range(self.hidden_dim):
                hidden_pre[h] += self.W1[h][j] * fv

        hidden_act = [self.act(z) for z in hidden_pre]

        logits = self.b2[:]
        for h, hv in enumerate(hidden_act):
            for c in range(self.num_classes):
                logits[c] += self.W2[c][h] * hv

        return hidden_pre, hidden_act, logits

    def logits_one(self, x: Dict[int, float]) -> List[float]:
        _pre, _hid, logits = self.forward_one(x)
        return logits

    def logits_dataset(self, X: List[Dict[int, float]]) -> List[List[float]]:
        return [self.logits_one(x) for x in X]

    def predict_proba_one(self, x: Dict[int, float]) -> List[float]:
        return softmax_row(self.logits_one(x))

    def predict_proba(self, X: List[Dict[int, float]]) -> List[List[float]]:
        return [self.predict_proba_one(x) for x in X]

    def predict(self, X: List[Dict[int, float]]) -> List[int]:
        return [argmax(self.predict_proba_one(x)) for x in X]

    def train_epoch(
        self,
        X: List[Dict[int, float]],
        y: List[int],
        lr: float,
        weight_decay: float,
        class_weights: Optional[List[float]] = None,
        seed: int = 0,
    ) -> float:
        idxs = list(range(len(y)))
        rng = random.Random(seed)
        rng.shuffle(idxs)

        total_loss = 0.0
        eps = 1e-12

        for i in idxs:
            x = X[i]
            yi = int(y[i])

            hidden_pre, hidden_act, logits = self.forward_one(x)
            probs = softmax_row(logits)

            alpha = 1.0
            if class_weights is not None:
                alpha = float(class_weights[yi])

            py = max(probs[yi], eps)
            total_loss += -math.log(py) * alpha

            err_out = [p * alpha for p in probs]
            err_out[yi] -= 1.0 * alpha

            hidden_delta = [0.0 for _ in range(self.hidden_dim)]
            for h in range(self.hidden_dim):
                s = 0.0
                for c in range(self.num_classes):
                    s += err_out[c] * self.W2[c][h]
                hidden_delta[h] = s * self.act_deriv(hidden_pre[h])

            for c in range(self.num_classes):
                self.b2[c] -= lr * err_out[c]
                for h in range(self.hidden_dim):
                    grad = err_out[c] * hidden_act[h] + weight_decay * self.W2[c][h]
                    self.W2[c][h] -= lr * grad

            for h in range(self.hidden_dim):
                self.b1[h] -= lr * hidden_delta[h]

            for j, v in x.items():
                fv = float(v)
                for h in range(self.hidden_dim):
                    grad = hidden_delta[h] * fv + weight_decay * self.W1[h][j]
                    self.W1[h][j] -= lr * grad

        return total_loss / float(max(len(y), 1))


def build_class_weights(y_train: List[int]) -> List[float]:
    counts = bincount(y_train, minlength=len(CLASSES))
    nz = [c for c in counts if c > 0]
    if not nz:
        return [1.0 for _ in range(len(CLASSES))]

    total = float(sum(nz))
    n_nonzero = float(len(nz))
    weights = [0.0 for _ in range(len(CLASSES))]
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = total / (n_nonzero * float(c))
        else:
            weights[i] = 0.0

    nz_weights = [w for w in weights if w > 0]
    if nz_weights:
        m = mean(nz_weights)
        weights = [w / m if w > 0 else 0.0 for w in weights]

    bother_idx = CLASS_TO_INDEX.get("B:OTHER", None)
    if bother_idx is not None and weights[bother_idx] > 0:
        weights[bother_idx] *= float(BOTHER_EXTRA_DOWNWEIGHT)

    weights = [clip(float(w), 0.0, float(MAX_CLASS_WEIGHT)) if w > 0 else 0.0 for w in weights]
    return [w if w > 0 else 1.0 for w in weights]


# ===============================
# ROC/PR curve plots
# ===============================
def plot_auc_curves(
    y_true: List[int],
    probs: List[List[float]],
    title: str,
    out_roc_png: Optional[str] = None,
    out_pr_png: Optional[str] = None,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping AUC curve plots.")
        return
    if out_roc_png is None and out_pr_png is None:
        return
    if not probs:
        return

    C = len(probs[0])
    valid: List[int] = []
    for i in range(C):
        y_bin = [1 if y == i else 0 for y in y_true]
        pos = sum(y_bin)
        neg = len(y_bin) - pos
        if pos == 0 or neg == 0:
            continue
        valid.append(i)

    if not valid:
        print("[WARN] No valid classes with both pos/neg in this split; skipping AUC curve plots.")
        return

    if out_roc_png is not None:
        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

        for i in valid:
            y_bin = [1 if y == i else 0 for y in y_true]
            scores = [row[i] for row in probs]
            fpr, tpr, _ = binary_roc_curve(y_bin, scores)
            auc_i = binary_roc_auc(y_bin, scores)
            ax.plot(fpr, tpr, linewidth=1.2, label=f"{INDEX_TO_CLASS[i][:28]}  AUC={auc_i:.3f}")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curves (OvR) — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_roc_png, dpi=170)
        plt.close(fig)
        print(f"[INFO] Saved ROC-AUC curves -> {out_roc_png}")

    if out_pr_png is not None:
        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)

        for i in valid:
            y_bin = [1 if y == i else 0 for y in y_true]
            prevalence = mean(y_bin)
            scores = [row[i] for row in probs]
            prec, rec, _ = binary_pr_curve(y_bin, scores)
            ap_i = binary_pr_auc(y_bin, scores)

            ax.plot(rec, prec, linewidth=1.2, label=f"{INDEX_TO_CLASS[i][:28]}  PR-AUC={ap_i:.3f}")
            ax.plot([0, 1], [prevalence, prevalence], linestyle=":", linewidth=0.8)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curves (OvR) — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_pr_png, dpi=170)
        plt.close(fig)
        print(f"[INFO] Saved PR-AUC curves -> {out_pr_png}")


# ===============================
# Bias meta
# ===============================
def _age_bins(age: List[float]) -> List[str]:
    out = []
    for a in age:
        if not isfinite(a):
            out.append("unknown")
        elif a < 18:
            out.append("<18")
        elif a < 40:
            out.append("18-39")
        elif a < 65:
            out.append("40-64")
        elif a < 80:
            out.append("65-79")
        else:
            out.append("80+")
    return out


def _group_report(
    group_name: str,
    groups: List[str],
    y_true: List[int],
    y_pred: List[int],
    probs: List[List[float]],
    min_n: int = BIAS_MIN_GROUP_N,
) -> None:
    vals = [g for g in unique(groups) if g is not None]
    if len(vals) == 0:
        return

    print(f"\n=== BIAS CHECK: {group_name} (min_n={min_n}) ===")
    print("Group                 |   n   |  Acc  |  Macro-F1 |  Weighted-F1 |  Macro ROC-AUC |  Macro PR-AUC")
    print("-" * 98)
    for v in vals:
        idxs = [i for i, g in enumerate(groups) if g == v]
        n = len(idxs)
        if n < int(min_n):
            continue

        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        pp = [probs[i] for i in idxs]

        prf_macro = precision_recall_f1_support_per_class(yt, yp, len(CLASSES))
        macro_f1 = macro_average(prf_macro[2])
        weighted_f1 = weighted_average(prf_macro[2], prf_macro[3])
        acc = accuracy_score(yt, yp)

        _, _, macro_roc, _w_roc, macro_pr, _w_pr = multiclass_auc_pr_from_probs(yt, pp)

        print(
            f"{str(v)[:20]:20s} | {n:5d} | {acc:0.3f} | {macro_f1:0.3f}    | {weighted_f1:0.3f}       |"
            f" {macro_roc:>12.4f} | {macro_pr:>11.4f}"
        )


def load_bias_meta_by_hadm(hadm_set: set[int]) -> Dict[int, Dict[str, str]]:
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

    ds_tag = _sanitize_tag(f"{ds_kind}_{dataset_root.name}_pure_python_mlp")

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

    y_all = [CLASS_TO_INDEX[r["label"]] for r in micro_rows]
    dist = Counter(y_all)
    print("[INFO] Label counts:")
    for i in range(len(CLASSES)):
        print(f"  {INDEX_TO_CLASS[i]:55s} -> {dist.get(i, 0)}")

    abx_by_hadm = load_abx_features(hadm_set)
    print(f"[INFO] ABX features built for hadm={len(abx_by_hadm)} (features={len(ABX_ORDER)})")

    print(f"[INFO] Computing vitals/labs within {HOURS_WINDOW}h window ...")
    vitals_by_hadm = compute_vitals_features(hadm_set)
    print(f"[INFO] Vitals complete rows={len(vitals_by_hadm)}")

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

    print(f"[INFO] Joined rows={len(y_labels)} | numeric_dim={len(num_data[0])} | numeric_order={NUMERIC_ORDER}")

    txt_tr, txt_te, num_tr, num_te, y_tr, y_te, hadm_tr, hadm_te = split_with_min_unique_hadm(
        text_data, num_data, y_labels, hadm_ids, test_size=0.2, seed=SEED, min_unique_test_hadm=MIN_TEST_UNIQUE_HADM
    )
    print(f"[INFO] TEST unique HADM_ID: {len(set(hadm_te))} (requirement >= {MIN_TEST_UNIQUE_HADM})")

    txt_tr2, txt_va, num_tr2, num_va, y_tr2, y_va, hadm_tr2, hadm_va = split_with_min_unique_hadm(
        txt_tr, num_tr, y_tr, hadm_tr, test_size=0.2, seed=SEED + 17, min_unique_test_hadm=1
    )

    vocab = build_vocab(txt_tr2, max_tokens=MAX_TEXT_TOKENS)
    print(f"[INFO] Vocabulary size={len(vocab)}")

    mu, sd = fit_numeric_scaler(num_tr2, n_cont=4)
    X_num_tr = apply_numeric_scaling(num_tr2, mu, sd, n_cont=4)
    X_num_va = apply_numeric_scaling(num_va, mu, sd, n_cont=4)
    X_num_te = apply_numeric_scaling(num_te, mu, sd, n_cont=4)

    X_tr, total_dim = make_sparse_dataset(txt_tr2, X_num_tr, vocab)
    X_va, _ = make_sparse_dataset(txt_va, X_num_va, vocab)
    X_te, _ = make_sparse_dataset(txt_te, X_num_te, vocab)

    print(f"[INFO] Total feature dimension={total_dim}")

    class_weights = build_class_weights(y_tr2) if USE_CLASS_WEIGHTS else [1.0 for _ in range(len(CLASSES))]
    print("[INFO] Class weights:")
    for i, w in enumerate(class_weights):
        print(f"  {INDEX_TO_CLASS[i]:55s} -> {w:.4f}")

    total_attempts = max(1, int(MAX_TRAIN_RESTARTS))
    use_acc = (TARGET_STOP_METRIC == "acc")
    target_value = float(TARGET_ACC if use_acc else TARGET_F1)

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    best_seed = SEED

    for attempt in range(total_attempts):
        local_seed = SEED + 1000 * attempt
        model = PurePythonMLP(
            num_features=total_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            num_classes=len(CLASSES),
            activation_name=ACTIVATION_NAME,
            seed=local_seed,
            init_scale=MLP_INIT_SCALE,
        )

        best_local_metric = -1.0
        best_local_epoch = 0
        best_local_state = None
        no_improve = 0

        metric_name = f"val_target_acc({TARGET_ACC_KIND})" if use_acc else f"val_target_f1({TARGET_F1_KIND})"
        print(
            f"\n[INFO] Training attempt {attempt + 1}/{total_attempts}"
            f" | seed={local_seed}"
            f" | hidden_dim={MLP_HIDDEN_DIM}"
            f" | activation={ACTIVATION_NAME}"
            f" | stop_metric={TARGET_STOP_METRIC} target={target_value:.3f}"
        )

        for epoch in range(1, int(MAX_EPOCHS) + 1):
            tr_loss = model.train_epoch(
                X_tr,
                y_tr2,
                lr=float(LR),
                weight_decay=float(WEIGHT_DECAY),
                class_weights=class_weights,
                seed=local_seed + epoch,
            )

            y_pred_va = model.predict(X_va)

            if use_acc:
                val_metric = _compute_target_acc(TARGET_ACC_KIND, y_va, y_pred_va)
            else:
                val_metric = _compute_target_f1(TARGET_F1_KIND, y_va, y_pred_va)
            if val_metric is None:
                val_metric = -1.0

            print(f"[E{epoch:05d}] tr_loss={tr_loss:.5f} {metric_name}={val_metric:.4f}")

            if float(val_metric) > float(best_local_metric) + float(MIN_DELTA):
                best_local_metric = float(val_metric)
                best_local_epoch = int(epoch)
                best_local_state = model.state_dict()
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
            best_epoch = best_local_epoch
            best_seed = local_seed
            best_state = best_local_state

        if best_local_metric >= target_value:
            break

    if best_state is None:
        raise RuntimeError("Training failed: no model state captured.")

    print(f"[INFO] BEST across attempts: metric={best_metric:.4f} | seed={best_seed} | best_epoch={best_epoch}")

    if RETRAIN_ON_FULL_TRAIN and best_epoch > 0:
        print(f"[INFO] Retraining on FULL TRAIN (TRAIN+VAL) for {best_epoch} epochs")
        txt_full = txt_tr2 + txt_va
        num_full_raw = num_tr2 + num_va
        y_full = y_tr2 + y_va

        X_num_full = apply_numeric_scaling(num_full_raw, mu, sd, n_cont=4)
        X_full, _ = make_sparse_dataset(txt_full, X_num_full, vocab)

        class_weights_full = build_class_weights(y_full) if USE_CLASS_WEIGHTS else [1.0 for _ in range(len(CLASSES))]
        final_model = PurePythonMLP(
            num_features=total_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            num_classes=len(CLASSES),
            activation_name=ACTIVATION_NAME,
            seed=best_seed + 9999,
            init_scale=MLP_INIT_SCALE,
        )
        for ep in range(1, int(best_epoch) + 1):
            loss = final_model.train_epoch(
                X_full,
                y_full,
                lr=float(LR),
                weight_decay=float(WEIGHT_DECAY),
                class_weights=class_weights_full,
                seed=best_seed + 50000 + ep,
            )
            print(f"[FULL E{ep:04d}] loss={loss:.5f}")
    else:
        final_model = PurePythonMLP(
            num_features=total_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            num_classes=len(CLASSES),
            activation_name=ACTIVATION_NAME,
            seed=best_seed + 123,
            init_scale=MLP_INIT_SCALE,
        )
        final_model.load_state_dict(best_state)

    logits_te = final_model.logits_dataset(X_te)
    probs_te = probs_from_logits(logits_te)
    y_pred = [argmax(row) for row in probs_te]
    acc = accuracy_score(y_te, y_pred)

    print(f"\n=== GENERAL ACCURACY ON TEST SET ({ds_tag}): {acc:.4f} ===")
    report_multiclass_metrics(y_true=y_te, y_pred=y_pred)
    report_mrsa_vs_mssa(y_true=y_te, y_pred=y_pred)
    report_confusion_and_rates(y_true=y_te, y_pred=y_pred, title=f"TEST {ds_tag}")
    report_auc_pr(y_true=y_te, probs=probs_te, title=f"TEST {ds_tag}")

    val_logits = final_model.logits_dataset(X_va)
    ece_metric = PurePythonECE(n_bins=15)
    pre_ece_test = ece_metric(logits_te, y_te)

    scaler = TemperatureScaler(init_temperature=1.5)
    optimal_t = scaler.fit(val_logits, y_va, lr=0.01, max_iter=200)
    scaled_logits_te = scaler.scale_logits(logits_te)
    scaled_probs_te = probs_from_logits(scaled_logits_te)
    post_ece_test = ece_metric(scaled_logits_te, y_te)

    print(f"\n=== TEMPERATURE SCALING ({ds_tag}) ===")
    print(f"[*] Pre-Calibration ECE:  {pre_ece_test:.6f}")
    print(f"[*] Optimal Temperature:  {optimal_t:.6f}")
    print(f"[*] Post-Calibration ECE: {post_ece_test:.6f}")

    roc_png = f"roc_auc__{ds_tag}.png"
    pr_png = f"pr_auc__{ds_tag}.png"
    plot_auc_curves(y_true=y_te, probs=scaled_probs_te, title=f"TEST {ds_tag}", out_roc_png=roc_png, out_pr_png=pr_png)

    calib_png = f"calibration__{ds_tag}.png"
    calibration_report_from_probs(y_true=y_te, probs=scaled_probs_te, title=f"TEST {ds_tag}", out_png=calib_png)

    meta_by_hadm = load_bias_meta_by_hadm(hadm_set)
    gender_te = [meta_by_hadm.get(int(h), {}).get("gender", "unknown") for h in hadm_te]
    admtype_te = [meta_by_hadm.get(int(h), {}).get("admission_type", "unknown") for h in hadm_te]
    admloc_te = [meta_by_hadm.get(int(h), {}).get("admission_location", "unknown") for h in hadm_te]
    age_te = [float(row[3]) for row in num_te]
    agebin_te = _age_bins(age_te)

    _group_report("gender", gender_te, y_te, y_pred, scaled_probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("age_bin", agebin_te, y_te, y_pred, scaled_probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("admission_type", admtype_te, y_te, y_pred, scaled_probs_te, min_n=BIAS_MIN_GROUP_N)
    _group_report("admission_location", admloc_te, y_te, y_pred, scaled_probs_te, min_n=BIAS_MIN_GROUP_N)

    print("\n[INFO] Done.")
    print(f"[INFO] Feature names lowercase (numeric order): {NUMERIC_ORDER}")
    print("[INFO] Text input: spec_type_desc + ' [sep] ' + interpretation")
    print(f"[INFO] Model: pure Python MLP | hidden_dim={MLP_HIDDEN_DIM} | activation={ACTIVATION_NAME}")


# ===============================
# Entrypoint (AUTO)
# ===============================
def main_auto() -> int:
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
