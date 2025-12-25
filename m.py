# dual_coral_mooney_mimic.py  (NO-PANDAS version)
import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Iterable, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# =========================
# SEED
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# PATHS / CONFIG
# =========================
MOONEY_ROOT = Path("dataset")  # expects dataset/train/{Normal,Bacterial,Viral} and dataset/test/...
MIMIC_DIR_DEFAULT = Path("dataset/mimic/mimic-iii-clinical-database-demo-1.4")

IMG_SIZE = 128
BATCH_IMG = 32
BATCH_CLIN = 64
EMB_DIM = 128

SCALER_FILE = Path("clin_scaler.npz")
PATHOGEN_VOCAB_JSON = Path("pathogen_vocab.json")

_BAD_ORG_TOKENS = [
    "no growth",
    "negative",
    "not detected",
    "none",
    "normal flora",
    "mixed flora",
    "contaminant",
    "contamination",
    "see comment",
    "test not performed",
]

# =========================
# CSV helpers (no pandas)
# =========================
def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


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
    raise ValueError(
        f"Could not find any of columns {candidates} in CSV header. "
        f"Found sample: {actual_cols[:50]} ..."
    )


def _resolve_usecols_idx(path: Path, wanted: Dict[str, List[str]]) -> Dict[str, int]:
    header = _read_header(path)
    resolved: Dict[str, int] = {}
    for std_name, cands in wanted.items():
        resolved[std_name] = _find_col_index(header, cands)
    return resolved


def _iter_csv_std(path: Path, wanted: Dict[str, List[str]]) -> Iterable[Dict[str, str]]:
    """
    Yield dict rows with standardized keys (wanted's keys) -> raw string cell values.
    Streaming (good for huge MIMIC tables).
    """
    idx = _resolve_usecols_idx(path, wanted)
    # Use csv.reader for speed and to avoid DictReader header-key mismatch due to stripping.
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=enc) as f:
                r = csv.reader(f)
                header = next(r)  # consume
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


def _parse_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "" or ss.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return int(float(ss))
    except Exception:
        return None


def _parse_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "" or ss.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return float(ss)
    except Exception:
        return None


_DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S.%f",
)


def _safe_parse_datetime_str(x: Any) -> Optional[datetime]:
    """
    Safe datetime parser for MIMIC weird dates (e.g. year 3000 for de-identified DOB).
    Returns datetime or None.
    """
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
    # last resort: ISO parser
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


# =========================
# Mooney utilities
# =========================
def list_mooney_files(split_dir: Path, task: str) -> Tuple[List[str], List[int]]:
    """
    task:
      - 'binary'   : Normal=0, (Bacterial+Viral)=1
      - '3class'   : Normal=0, Bacterial=1, Viral=2
      - 'pathogen' : Bacterial=0, Viral=1 (EXCLUDES Normal)
      - 'specpath' : image side uses Bacterial=0, Viral=1 (EXCLUDES Normal)
    """
    files, labels = [], []
    exts = {".jpg", ".jpeg", ".png"}

    def add_from_class(cls: str, y: int):
        for p in (split_dir / cls).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p.resolve()))
                labels.append(y)

    if task == "binary":
        add_from_class("Normal", 0)
        add_from_class("Bacterial", 1)
        add_from_class("Viral", 1)
    elif task in ("pathogen", "specpath"):
        add_from_class("Bacterial", 0)
        add_from_class("Viral", 1)
    else:  # 3class
        add_from_class("Normal", 0)
        add_from_class("Bacterial", 1)
        add_from_class("Viral", 2)

    return files, labels


def decode_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def make_img_ds(files: List[str], labels: List[int], batch: int, shuffle: bool) -> tf.data.Dataset:
    x = tf.constant(files)
    y = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(files), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, yy: (decode_image(p), yy), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# MIMIC labeling helpers (no pandas)
# =========================
def is_pneumonia_icd9(code: Any) -> bool:
    if code is None:
        return False
    c = str(code).strip()
    if c == "" or c.lower() in {"nan", "none"}:
        return False
    c = c.replace(".", "")
    return c.startswith(("480", "481", "482", "483", "484", "485", "486", "4870"))


def infer_bact_viral_from_org(org: Optional[str]) -> str:
    """
    Heuristic: if organism name contains viral keywords => Viral else Bacterial.
    """
    if org is None:
        return "Unknown"
    s = str(org).lower()
    viral_keys = [
        "virus",
        "influenza",
        "rsv",
        "adenovirus",
        "parainfluenza",
        "coronavirus",
        "metapneumovirus",
        "cov",
        "sars",
    ]
    if any(k in s for k in viral_keys):
        return "Viral"
    return "Bacterial"


def normalize_org_name(org: Optional[str]) -> Optional[str]:
    if org is None:
        return None
    s = str(org).strip()
    if not s:
        return None
    sl = s.lower()
    if any(tok in sl for tok in _BAD_ORG_TOKENS):
        return None

    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("*", "").strip()
    s = re.sub(r"\s+\(.*?\)\s*$", "", s).strip()
    return s if s else None


def build_pathogen_vocab_from_counts(
    bact_counts: Counter, viral_counts: Counter, top_k_bact: int, top_k_viral: int
) -> List[str]:
    top_b = [k for (k, _) in bact_counts.most_common(top_k_bact)]
    top_v = [k for (k, _) in viral_counts.most_common(top_k_viral)]
    classes = [f"B:{x}" for x in top_b] + ["B:OTHER"] + [f"V:{x}" for x in top_v] + ["V:OTHER"]
    return classes


def assign_specific_pathogen_label(org_norm: Optional[str], classes: List[str]) -> Optional[int]:
    if org_norm is None:
        return None
    grp = infer_bact_viral_from_org(org_norm)
    if grp not in ("Bacterial", "Viral"):
        return None

    prefix = "B:" if grp == "Bacterial" else "V:"
    exact = prefix + org_norm
    if exact in classes:
        return classes.index(exact)

    other = prefix + "OTHER"
    if other in classes:
        return classes.index(other)

    return None


def classes_to_group_map(classes: List[str]) -> np.ndarray:
    """
    For specpath alignment: map each clinical class -> group label
      B:* => 0 (Bacterial)
      V:* => 1 (Viral)
    """
    g = []
    for c in classes:
        if c.startswith("B:"):
            g.append(0)
        elif c.startswith("V:"):
            g.append(1)
        else:
            g.append(-1)
    return np.array(g, dtype=np.int32)


def _label_contains_all(label: str, must_contain: List[str]) -> bool:
    l = str(label).lower()
    return all(m.lower() in l for m in must_contain)


# =========================
# MIMIC feature building (streaming, no pandas)
# =========================
@dataclass(frozen=True)
class AdmWindow:
    subject_id: int
    admittime: datetime
    t_end: datetime
    age: float
    label_bin: int


def build_mimic_features_csv(
    mimic_dir: Path,
    out_csv: Path,
    task: str,
    hours: int = 24,
    top_k_bact: int = 10,
    top_k_viral: int = 10,
) -> Path:
    mimic_dir = Path(mimic_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building MIMIC features from:", mimic_dir.resolve())

    # ---- ADMISSIONS
    admissions_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
        "ADMITTIME": ["ADMITTIME", "ADMIT_TIME", "ADMIT TIME", "ADMIT_TIME "],
    }
    admissions: Dict[int, Tuple[int, datetime]] = {}
    for row in _iter_csv_std(mimic_dir / "ADMISSIONS.csv", admissions_wanted):
        hadm = _parse_int(row["HADM_ID"])
        sid = _parse_int(row["SUBJECT_ID"])
        adt = _safe_parse_datetime_str(row["ADMITTIME"])
        if hadm is None or sid is None or adt is None:
            continue
        admissions[hadm] = (sid, adt)

    # ---- PATIENTS
    patients_wanted = {
        "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
        "DOB": ["DOB", "DATE_OF_BIRTH", "DATE OF BIRTH"],
    }
    dob_by_subject: Dict[int, datetime] = {}
    for row in _iter_csv_std(mimic_dir / "PATIENTS.csv", patients_wanted):
        sid = _parse_int(row["SUBJECT_ID"])
        dob = _safe_parse_datetime_str(row["DOB"])
        if sid is None or dob is None:
            continue
        dob_by_subject[sid] = dob

    # ---- DIAGNOSES_ICD
    dx_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "ICD9_CODE": ["ICD9_CODE", "ICD9", "ICD9CODE", "ICD9 CODE"],
    }
    pneu_hadm: set[int] = set()
    viral_hadm: set[int] = set()
    bact_hadm: set[int] = set()

    for row in _iter_csv_std(mimic_dir / "DIAGNOSES_ICD.csv", dx_wanted):
        hadm = _parse_int(row["HADM_ID"])
        code = row["ICD9_CODE"]
        if hadm is None:
            continue
        if is_pneumonia_icd9(code):
            pneu_hadm.add(hadm)

        c = str(code).strip().replace(".", "")
        if c.startswith(("480", "4870")):
            viral_hadm.add(hadm)
        if c.startswith(("481", "482")):
            bact_hadm.add(hadm)

    # ---- D_ITEMS (for temp/spo2)
    d_items_wanted = {"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]}
    temp_c_ids: set[int] = set()
    temp_f_ids: set[int] = set()
    spo2_ids: set[int] = set()

    for row in _iter_csv_std(mimic_dir / "D_ITEMS.csv", d_items_wanted):
        itemid = _parse_int(row["ITEMID"])
        label = row["LABEL"]
        if itemid is None:
            continue
        ll = str(label).lower()
        # mirror your old heuristic: "temperature"+"c" / "temperature"+"f"
        if "temperature" in ll and "c" in ll:
            temp_c_ids.add(itemid)
        if "temperature" in ll and "f" in ll:
            temp_f_ids.add(itemid)
        if "spo2" in ll:
            spo2_ids.add(itemid)

    if len(spo2_ids) == 0:
        # fallback: contains "o2" and "saturation"
        for row in _iter_csv_std(mimic_dir / "D_ITEMS.csv", d_items_wanted):
            itemid = _parse_int(row["ITEMID"])
            label = row["LABEL"]
            if itemid is None:
                continue
            ll = str(label).lower()
            if ("o2" in ll) and ("saturation" in ll):
                spo2_ids.add(itemid)

    # ---- D_LABITEMS (WBC)
    d_lab_wanted = {"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]}
    wbc_ids: set[int] = set()
    wbc_re = re.compile(r"\bwbc\b", re.IGNORECASE)
    for row in _iter_csv_std(mimic_dir / "D_LABITEMS.csv", d_lab_wanted):
        itemid = _parse_int(row["ITEMID"])
        label = row["LABEL"]
        if itemid is None:
            continue
        ll = str(label).lower()
        if wbc_re.search(ll) or ("white blood" in ll):
            wbc_ids.add(itemid)

    # ---- build admission windows + age
    adm_windows: Dict[int, AdmWindow] = {}
    for hadm, (sid, adt) in admissions.items():
        dob = dob_by_subject.get(sid)
        if dob is None:
            continue
        # age in years
        delta_days = (adt - dob).days
        age = float(delta_days) / 365.2425
        if not np.isfinite(age):
            continue
        if age > 120.0:
            age = 90.0
        age = float(np.clip(age, 0.0, 110.0))
        t_end = adt + timedelta(hours=int(hours))
        label_bin = 1 if hadm in pneu_hadm else 0
        adm_windows[hadm] = AdmWindow(subject_id=sid, admittime=adt, t_end=t_end, age=age, label_bin=label_bin)

    # ---- CHARTEVENTS: aggregate temperature(mean) and spo2(min)
    ce_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "ITEMID": ["ITEMID", "ITEM_ID"],
        "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
        "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
    }
    temp_sum: Dict[int, float] = {}
    temp_n: Dict[int, int] = {}
    spo2_min: Dict[int, float] = {}

    want_itemids = temp_c_ids | temp_f_ids | spo2_ids
    ce_path = mimic_dir / "CHARTEVENTS.csv"
    for row in _iter_csv_std(ce_path, ce_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or itemid not in want_itemids:
            continue
        win = adm_windows.get(hadm)
        if win is None:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None or ct < win.admittime or ct > win.t_end:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        if itemid in temp_c_ids or itemid in temp_f_ids:
            temp_c = v
            if itemid in temp_f_ids:
                temp_c = (float(v) - 32.0) / 1.8
            if temp_c < 30.0 or temp_c > 45.0:
                continue
            temp_sum[hadm] = temp_sum.get(hadm, 0.0) + float(temp_c)
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

    # ---- LABEVENTS: WBC max with possible unit scaling
    le_wanted = {
        "HADM_ID": ["HADM_ID", "HADMID"],
        "ITEMID": ["ITEMID", "ITEM_ID"],
        "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
        "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
    }
    le_path = mimic_dir / "LABEVENTS.csv"

    # pass 1: sample for median to decide scaling
    sample: List[float] = []
    max_sample = 200_000
    rng = np.random.default_rng(SEED)
    seen = 0

    for row in _iter_csv_std(le_path, le_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or itemid not in wbc_ids:
            continue
        win = adm_windows.get(hadm)
        if win is None:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None or ct < win.admittime or ct > win.t_end:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        seen += 1
        if len(sample) < max_sample:
            sample.append(float(v))
        else:
            # reservoir sampling
            j = int(rng.integers(0, seen))
            if j < max_sample:
                sample[j] = float(v)

    scale_wbc_by_1000 = False
    if len(sample) > 0:
        med = float(np.median(np.array(sample, dtype=np.float64)))
        if med < 200.0:
            scale_wbc_by_1000 = True

    # pass 2: compute per-admission max WBC after scaling + filters
    wbc_max: Dict[int, float] = {}
    for row in _iter_csv_std(le_path, le_wanted):
        hadm = _parse_int(row["HADM_ID"])
        itemid = _parse_int(row["ITEMID"])
        if hadm is None or itemid is None or itemid not in wbc_ids:
            continue
        win = adm_windows.get(hadm)
        if win is None:
            continue
        ct = _safe_parse_datetime_str(row["CHARTTIME"])
        if ct is None or ct < win.admittime or ct > win.t_end:
            continue
        v = _parse_float(row["VALUENUM"])
        if v is None:
            continue

        w = float(v) * (1000.0 if scale_wbc_by_1000 else 1.0)
        if w < 2000.0 or w > 40000.0:
            continue
        prev = wbc_max.get(hadm)
        wbc_max[hadm] = w if (prev is None or w > prev) else prev

    # ---- Build base feature rows (require all 4 features)
    rows_base: List[Dict[str, Any]] = []
    for hadm, win in adm_windows.items():
        tm = temp_mean.get(hadm)
        sp = spo2_min.get(hadm)
        wb = wbc_max.get(hadm)
        if tm is None or sp is None or wb is None:
            continue
        rows_base.append(
            {
                "HADM_ID": hadm,
                "temperature_c": float(tm),
                "spo2": float(sp),
                "wbc": float(wb),
                "age": float(win.age),
                "label_bin": int(win.label_bin),
            }
        )

    if len(rows_base) == 0:
        raise RuntimeError("No complete feature rows found (temperature/spo2/wbc/age). Try increasing --hours.")

    # ---- Apply task labeling
    out_rows: List[Dict[str, Any]] = []

    if task == "binary":
        for r in rows_base:
            out_rows.append({**r, "label": int(r["label_bin"])})

    elif task == "pathogen":
        # pneumonia only, label 0=B,1=V using ICD9 buckets
        for r in rows_base:
            if int(r["label_bin"]) != 1:
                continue
            hadm = int(r["HADM_ID"])
            if hadm not in (viral_hadm | bact_hadm):
                continue
            lab = 0 if hadm in bact_hadm else 1
            out_rows.append({**r, "label": int(lab)})

    elif task == "specpath":
        # pneumonia only, label by specific pathogen from MICROBIOLOGYEVENTS within window
        pneu_hadm_base = {int(r["HADM_ID"]) for r in rows_base if int(r["label_bin"]) == 1}
        if len(pneu_hadm_base) == 0:
            raise RuntimeError("No pneumonia admissions with complete vitals/labs found; cannot do --task specpath.")

        micro_wanted = {
            "HADM_ID": ["HADM_ID", "HADMID"],
            "ORG_NAME": ["ORG_NAME", "ORGANISM", "ORGNAME", "ORG NAME"],
            "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME", "CHARTDATE", "CHART_DATE", "CHART DATE"],
        }

        # per-hadm organism counts
        org_counts_by_hadm: Dict[int, Counter] = {}
        bact_counts: Counter = Counter()
        viral_counts: Counter = Counter()

        micro_path = mimic_dir / "MICROBIOLOGYEVENTS.csv"
        any_micro_in_window = 0
        for row in _iter_csv_std(micro_path, micro_wanted):
            hadm = _parse_int(row["HADM_ID"])
            if hadm is None or hadm not in pneu_hadm_base:
                continue
            win = adm_windows.get(hadm)
            if win is None:
                continue
            ct = _safe_parse_datetime_str(row["CHARTTIME"])
            if ct is None or ct < win.admittime or ct > win.t_end:
                continue
            org = normalize_org_name(row["ORG_NAME"])
            if org is None:
                continue

            any_micro_in_window += 1
            grp = infer_bact_viral_from_org(org)
            if grp == "Bacterial":
                bact_counts[org] += 1
            elif grp == "Viral":
                viral_counts[org] += 1

            if hadm not in org_counts_by_hadm:
                org_counts_by_hadm[hadm] = Counter()
            org_counts_by_hadm[hadm][org] += 1

        if any_micro_in_window == 0:
            raise RuntimeError(
                "No MICROBIOLOGYEVENTS rows in the selected time window for pneumonia admissions. "
                "Try increasing --hours."
            )

        classes = build_pathogen_vocab_from_counts(bact_counts, viral_counts, top_k_bact=top_k_bact, top_k_viral=top_k_viral)
        PATHOGEN_VOCAB_JSON.write_text(json.dumps({"classes": classes}, indent=2))
        print(f"[INFO] Saved pathogen vocab: {PATHOGEN_VOCAB_JSON.resolve()} classes={len(classes)}")

        # choose one organism per admission: most frequent ORG_NORM within window
        hadm_to_org: Dict[int, str] = {}
        for hadm, ctr in org_counts_by_hadm.items():
            if len(ctr) == 0:
                continue
            hadm_to_org[hadm] = ctr.most_common(1)[0][0]

        for r in rows_base:
            if int(r["label_bin"]) != 1:
                continue
            hadm = int(r["HADM_ID"])
            org = hadm_to_org.get(hadm)
            lab = assign_specific_pathogen_label(org, classes)
            if lab is None:
                continue
            out_rows.append({**r, "label": int(lab)})

    else:  # 3class
        for r in rows_base:
            hadm = int(r["HADM_ID"])
            if hadm not in pneu_hadm:
                lab = 0
            elif hadm in bact_hadm:
                lab = 1
            elif hadm in viral_hadm:
                lab = 2
            else:
                continue
            out_rows.append({**r, "label": int(lab)})

    # ---- write output csv
    def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]):
        cols = ["HADM_ID", "temperature_c", "spo2", "wbc", "age", "label"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for rr in rows:
                w.writerow([rr[c] for c in cols])

    _write_rows_csv(out_csv, out_rows)

    # stats
    counts = Counter(int(r["label"]) for r in out_rows)
    print("[INFO] Wrote:", out_csv.resolve(), "rows=", len(out_rows))
    print("[INFO] Label counts:", dict(sorted(counts.items())))
    return out_csv


def make_clin_ds(features_csv: Path, batch: int, shuffle: bool) -> tf.data.Dataset:
    X_list: List[List[float]] = []
    y_list: List[int] = []
    with open(features_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                X_list.append(
                    [
                        float(row["temperature_c"]),
                        float(row["wbc"]),
                        float(row["spo2"]),
                        float(row["age"]),
                    ]
                )
                y_list.append(int(float(row["label"])))
            except Exception:
                continue

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# Models (tanh encoders, sigmoid/softmax heads)
# =========================
def build_img_encoder(emb_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(emb_dim, activation=None)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("tanh")(x)  # bounded embedding
    return tf.keras.Model(inp, x, name="img_encoder")


def build_clin_encoder(emb_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(4,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(emb_dim, activation=None)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("tanh")(x)  # bounded embedding
    return tf.keras.Model(inp, x, name="clin_encoder")


def build_head(num_classes: int, name: str) -> tf.keras.Model:
    inp = layers.Input(shape=(EMB_DIM,))
    x = layers.Dense(64, activation="tanh")(inp)

    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)  # P(class=1)
    else:
        out = layers.Dense(num_classes, activation="softmax")(x)  # probabilities

    return tf.keras.Model(inp, out, name=name)


# =========================
# Double-correlation losses
# =========================
def _corr_matrix(z: tf.Tensor) -> tf.Tensor:
    z = tf.cast(z, tf.float32)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.shape(z)[0]
    n_f = tf.cast(tf.maximum(n - 1, 1), tf.float32)
    cov = tf.matmul(z, z, transpose_a=True) / n_f
    std = tf.sqrt(tf.maximum(tf.linalg.diag_part(cov), 1e-6))
    denom = tf.tensordot(std, std, axes=0) + 1e-6
    return cov / denom


def mean_loss(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    ma = tf.reduce_mean(a, axis=0)
    mb = tf.reduce_mean(b, axis=0)
    return tf.reduce_mean(tf.square(ma - mb))


def coral_corr_loss(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    ca = _corr_matrix(a)
    cb = _corr_matrix(b)
    return tf.reduce_mean(tf.square(ca - cb))


# =========================
# Training loop
# =========================
def train_dual(
    img_train: tf.data.Dataset,
    clin_train: tf.data.Dataset,
    img_val: tf.data.Dataset,
    clin_val: tf.data.Dataset,
    num_classes_img: int,
    num_classes_clin: int,
    epochs: int,
    lam_mean: float,
    lam_corr: float,
    steps_per_epoch: int,
    clin_label_to_group: Optional[np.ndarray] = None,
):
    img_enc = build_img_encoder(EMB_DIM)
    clin_enc = build_clin_encoder(EMB_DIM)
    head_img = build_head(num_classes_img, name="img_head")
    head_clin = build_head(num_classes_clin, name="clin_head")

    opt = tf.keras.optimizers.Adam(1e-3)

    # heads output probabilities (sigmoid/softmax), so from_logits=False
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    img_it = iter(img_train.repeat())
    clin_it = iter(clin_train.repeat())

    if clin_label_to_group is not None:
        clin_label_to_group_tf = tf.constant(clin_label_to_group, dtype=tf.int32)
    else:
        clin_label_to_group_tf = None

    def loss_for(y_true_int: tf.Tensor, y_pred_probs: tf.Tensor, num_classes: int) -> tf.Tensor:
        if num_classes == 2:
            y_true = tf.cast(tf.expand_dims(y_true_int, -1), tf.float32)  # (B,1)
            return bce(y_true, y_pred_probs)  # (B,1) sigmoid
        return sce(y_true_int, y_pred_probs)  # (B,C) softmax

    def eval_accuracy(img_ds, clin_ds):
        # image accuracy
        if num_classes_img == 2:
            acc_img = tf.keras.metrics.BinaryAccuracy()
            for xb, yb in img_ds:
                z = img_enc(xb, training=False)
                p = head_img(z, training=False)  # (B,1)
                acc_img.update_state(tf.cast(tf.expand_dims(yb, -1), tf.float32), p)
            acc_img_val = float(acc_img.result().numpy())
        else:
            acc_img = tf.keras.metrics.SparseCategoricalAccuracy()
            for xb, yb in img_ds:
                z = img_enc(xb, training=False)
                p = head_img(z, training=False)  # (B,C)
                acc_img.update_state(yb, p)
            acc_img_val = float(acc_img.result().numpy())

        # clinical accuracy
        if num_classes_clin == 2:
            acc_cl = tf.keras.metrics.BinaryAccuracy()
            for xc, yc in clin_ds:
                z = clin_enc(xc, training=False)
                p = head_clin(z, training=False)  # (B,1)
                acc_cl.update_state(tf.cast(tf.expand_dims(yc, -1), tf.float32), p)
            acc_cl_val = float(acc_cl.result().numpy())
        else:
            acc_cl = tf.keras.metrics.SparseCategoricalAccuracy()
            for xc, yc in clin_ds:
                z = clin_enc(xc, training=False)
                p = head_clin(z, training=False)  # (B,C)
                acc_cl.update_state(yc, p)
            acc_cl_val = float(acc_cl.result().numpy())

        return acc_img_val, acc_cl_val

    for ep in range(1, epochs + 1):
        loss_meter = tf.keras.metrics.Mean()
        cls_meter = tf.keras.metrics.Mean()
        align_meter = tf.keras.metrics.Mean()

        for _ in range(steps_per_epoch):
            xb, yb = next(img_it)
            xc, yc = next(clin_it)

            with tf.GradientTape() as tape:
                zi = img_enc(xb, training=True)
                zc = clin_enc(xc, training=True)

                pi = head_img(zi, training=True)   # probs
                pc = head_clin(zc, training=True)  # probs

                loss_cls = loss_for(yb, pi, num_classes_img) + loss_for(yc, pc, num_classes_clin)

                loss_align = tf.constant(0.0, dtype=tf.float32)

                # Alignment strategy:
                # - if same label space (and not using group map): align per class index
                # - specpath: align by group (Bacterial/Viral) using clin_label_to_group map
                if clin_label_to_group_tf is None and num_classes_img == num_classes_clin:
                    for k in range(num_classes_img):
                        zi_k = tf.boolean_mask(zi, tf.equal(yb, k))
                        zc_k = tf.boolean_mask(zc, tf.equal(yc, k))
                        if tf.shape(zi_k)[0] >= 4 and tf.shape(zc_k)[0] >= 4:
                            loss_align += lam_mean * mean_loss(zi_k, zc_k)
                            loss_align += lam_corr * coral_corr_loss(zi_k, zc_k)
                else:
                    yc_group = tf.gather(clin_label_to_group_tf, tf.cast(yc, tf.int32))
                    for g in range(num_classes_img):  # for specpath img side is 2 groups (B/V)
                        zi_g = tf.boolean_mask(zi, tf.equal(yb, g))
                        zc_g = tf.boolean_mask(zc, tf.equal(yc_group, g))
                        if tf.shape(zi_g)[0] >= 4 and tf.shape(zc_g)[0] >= 4:
                            loss_align += lam_mean * mean_loss(zi_g, zc_g)
                            loss_align += lam_corr * coral_corr_loss(zi_g, zc_g)

                loss_total = loss_cls + loss_align

            vars_all = (
                img_enc.trainable_variables
                + clin_enc.trainable_variables
                + head_img.trainable_variables
                + head_clin.trainable_variables
            )
            grads = tape.gradient(loss_total, vars_all)
            opt.apply_gradients(zip(grads, vars_all))

            loss_meter.update_state(loss_total)
            cls_meter.update_state(loss_cls)
            align_meter.update_state(loss_align)

        acc_i, acc_c = eval_accuracy(img_val, clin_val)
        print(
            f"[EPOCH {ep:03d}] total={loss_meter.result().numpy():.4f} "
            f"cls={cls_meter.result().numpy():.4f} align={align_meter.result().numpy():.4f} | "
            f"val_acc_img={acc_i:.3f} val_acc_clin={acc_c:.3f}"
        )

    img_enc.save("img_encoder.keras")
    clin_enc.save("clin_encoder.keras")
    head_img.save("img_head.keras")
    head_clin.save("clin_head.keras")
    print("[INFO] Saved: img_encoder.keras, clin_encoder.keras, img_head.keras, clin_head.keras")


# =========================
# Clinical predictor
# =========================
def _class_names(task: str) -> List[str]:
    if task == "binary":
        return ["Normal", "Pneumonia"]
    if task == "pathogen":
        return ["Bacterial", "Viral"]
    if task == "3class":
        return ["Normal", "Bacterial", "Viral"]
    if task == "specpath":
        if not PATHOGEN_VOCAB_JSON.exists():
            raise FileNotFoundError(f"Missing {PATHOGEN_VOCAB_JSON}. Train/build features first.")
        return json.loads(PATHOGEN_VOCAB_JSON.read_text())["classes"]
    raise ValueError(f"Unknown task: {task}")


def predict_from_vitals(task: str, temperature_c: float, wbc: float, spo2: float, age: float, top: int = 15):
    if not Path("clin_encoder.keras").exists() or not Path("clin_head.keras").exists():
        raise FileNotFoundError("Missing clin_encoder.keras / clin_head.keras. Train first.")

    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Missing {SCALER_FILE}. Train first (it is saved during training).")

    scaler = np.load(SCALER_FILE)
    mu = scaler["mu"].astype(np.float32)
    sd = scaler["sd"].astype(np.float32)

    x = np.array([[temperature_c, wbc, spo2, age]], dtype=np.float32)
    x = (x - mu) / (sd + 1e-6)

    clin_enc = tf.keras.models.load_model("clin_encoder.keras")
    clin_head = tf.keras.models.load_model("clin_head.keras")

    z = clin_enc(x, training=False)
    out = clin_head(z, training=False).numpy().reshape(-1)

    names = _class_names(task)
    if len(names) == 2:
        # binary head outputs sigmoid P(class=1)
        p1 = float(out[0])
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        probs = out.astype(np.float32)  # already softmax probs

    pairs = sorted(zip(names, probs.tolist()), key=lambda t: t[1], reverse=True)

    print("\n[CLINICAL PREDICTION] (ordered by probability)")
    for name, p in pairs[:top]:
        print(f"  {name:>35s}: {p:.4f}")

    if task == "specpath":
        bact = [(n, p) for (n, p) in pairs if n.startswith("B:")]
        viral = [(n, p) for (n, p) in pairs if n.startswith("V:")]

        print("\n[BACTERIAL pathogens] (ordered)")
        for n, p in bact[: min(10, len(bact))]:
            print(f"  {n:>35s}: {p:.4f}")

        print("\n[VIRAL pathogens] (ordered)")
        for n, p in viral[: min(10, len(viral))]:
            print(f"  {n:>35s}: {p:.4f}")

    print()


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "3class", "pathogen", "specpath"], default="specpath")
    ap.add_argument("--mimic_dir", type=str, default=str(MIMIC_DIR_DEFAULT))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lam_mean", type=float, default=0.05)
    ap.add_argument("--lam_corr", type=float, default=0.10)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--rebuild_mimic_features", action="store_true")

    ap.add_argument("--top_k_bact", type=int, default=10)
    ap.add_argument("--top_k_viral", type=int, default=10)

    # clinical inference
    ap.add_argument("--predict_clin", action="store_true")
    ap.add_argument("--temperature_c", type=float, default=None)
    ap.add_argument("--wbc", type=float, default=None)
    ap.add_argument("--spo2", type=float, default=None)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--top", type=int, default=15)

    args = ap.parse_args()

    if args.predict_clin:
        for nm in ["temperature_c", "wbc", "spo2", "age"]:
            if getattr(args, nm) is None:
                raise ValueError(f"--predict_clin requires --{nm}")
        predict_from_vitals(args.task, args.temperature_c, args.wbc, args.spo2, args.age, top=args.top)
        return

    # ---- Mooney
    train_dir = MOONEY_ROOT / "train"
    test_dir = MOONEY_ROOT / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Mooney folders not found.\nExpected:\n  {train_dir.resolve()}\n  {test_dir.resolve()}"
        )

    mo_tr_f, mo_tr_y = list_mooney_files(train_dir, args.task)
    mo_te_f, mo_te_y = list_mooney_files(test_dir, args.task)
    if len(mo_tr_f) == 0 or len(mo_te_f) == 0:
        raise RuntimeError("No Mooney images found. Check dataset/train and dataset/test structure.")

    num_classes_img = 2 if args.task in ("binary", "pathogen", "specpath") else 3
    img_train = make_img_ds(mo_tr_f, mo_tr_y, batch=BATCH_IMG, shuffle=True)
    img_val = make_img_ds(mo_te_f, mo_te_y, batch=BATCH_IMG, shuffle=False)

    # ---- MIMIC features auto-create
    mimic_dir = Path(args.mimic_dir)
    if not mimic_dir.exists():
        raise FileNotFoundError(f"MIMIC dir not found: {mimic_dir.resolve()}")

    mimic_features_csv = mimic_dir / f"mimic_features_{args.task}.csv"
    if args.rebuild_mimic_features or (not mimic_features_csv.exists()):
        build_mimic_features_csv(
            mimic_dir,
            mimic_features_csv,
            task=args.task,
            hours=args.hours,
            top_k_bact=args.top_k_bact,
            top_k_viral=args.top_k_viral,
        )
    else:
        print("[INFO] Using existing MIMIC features:", mimic_features_csv.resolve())
        if args.task == "specpath" and not PATHOGEN_VOCAB_JSON.exists():
            raise FileNotFoundError(
                f"Missing {PATHOGEN_VOCAB_JSON}. Re-run with --rebuild_mimic_features to generate it."
            )

    # ---- Load features (no pandas)
    feats: List[Dict[str, Any]] = []
    with open(mimic_features_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                feats.append(
                    {
                        "HADM_ID": int(float(row["HADM_ID"])),
                        "temperature_c": float(row["temperature_c"]),
                        "spo2": float(row["spo2"]),
                        "wbc": float(row["wbc"]),
                        "age": float(row["age"]),
                        "label": int(float(row["label"])),
                    }
                )
            except Exception:
                continue

    if len(feats) < 20:
        print("[WARN] Very small MIMIC feature set. Consider increasing --hours.")

    # Determine clinical output dimension + alignment map
    if args.task == "specpath":
        classes = json.loads(PATHOGEN_VOCAB_JSON.read_text())["classes"]
        num_classes_clin = len(classes)
        clin_label_to_group = classes_to_group_map(classes)  # align to img bacterial/viral
    else:
        num_classes_clin = 2 if args.task in ("binary", "pathogen") else 3
        clin_label_to_group = None

    # ---- shuffle + split
    n = len(feats)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    feats = [feats[i] for i in idx]

    n_tr = max(int(0.8 * n), 1)
    feats_tr = feats[:n_tr]
    feats_va = feats[n_tr:] if n_tr < n else feats[:]

    # ---- standardize clinical (fit on train)
    cols = ["temperature_c", "wbc", "spo2", "age"]
    Xtr = np.array([[r[c] for c in cols] for r in feats_tr], dtype=np.float32)
    mu = Xtr.mean(axis=0).astype(np.float32)
    sd = (Xtr.std(axis=0) + 1e-6).astype(np.float32)
    np.savez(SCALER_FILE, mu=mu, sd=sd)
    print(f"[INFO] Saved clinical scaler: {SCALER_FILE.resolve()}")

    def _apply_z(rows: List[Dict[str, Any]]):
        for r in rows:
            x = np.array([r[c] for c in cols], dtype=np.float32)
            x = (x - mu) / (sd + 1e-6)
            for j, c in enumerate(cols):
                r[c] = float(x[j])

    _apply_z(feats_tr)
    _apply_z(feats_va)

    # ---- write train/val csvs (keeps your pipeline)
    train_csv = Path("mimic_train.csv")
    val_csv = Path("mimic_val.csv")

    def _write_feats(path: Path, rows: List[Dict[str, Any]]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["HADM_ID", "temperature_c", "spo2", "wbc", "age", "label"])
            for r in rows:
                w.writerow([r["HADM_ID"], r["temperature_c"], r["spo2"], r["wbc"], r["age"], r["label"]])

    _write_feats(train_csv, feats_tr)
    _write_feats(val_csv, feats_va)

    clin_train = make_clin_ds(train_csv, batch=BATCH_CLIN, shuffle=True)
    clin_val = make_clin_ds(val_csv, batch=BATCH_CLIN, shuffle=False)

    # ---- Train dual
    train_dual(
        img_train=img_train,
        clin_train=clin_train,
        img_val=img_val,
        clin_val=clin_val,
        num_classes_img=num_classes_img,
        num_classes_clin=num_classes_clin,
        epochs=args.epochs,
        lam_mean=args.lam_mean,
        lam_corr=args.lam_corr,
        steps_per_epoch=args.steps_per_epoch,
        clin_label_to_group=clin_label_to_group,
    )


if __name__ == "__main__":
    main()

# Example:
# python dual_coral_mooney_mimic.py --task specpath --top_k_bact 10 --top_k_viral 10 \
#   --rebuild_mimic_features --hours 24 --epochs 20 --steps_per_epoch 200
