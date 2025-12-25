# dual_coral_mooney_mimic.py
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
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


# =========================
# Robust CSV column helpers
# =========================
def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _find_col(actual_cols, candidates):
    canon_map = {_canon(c): c for c in actual_cols}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_map:
            return canon_map[key]
    raise ValueError(
        f"Could not find any of columns {candidates} in CSV header. "
        f"Found sample: {list(actual_cols)[:50]} ..."
    )


def _resolve_usecols(path: Path, wanted: dict) -> dict:
    header = pd.read_csv(path, nrows=0)
    header.columns = [c.strip() for c in header.columns]
    actual = header.columns
    resolved = {}
    for std_name, cands in wanted.items():
        resolved[std_name] = _find_col(actual, cands)
    return resolved


def _read_csv_std(path: Path, wanted: dict, low_memory=True) -> pd.DataFrame:
    resolved = _resolve_usecols(path, wanted)
    df = pd.read_csv(path, usecols=list(resolved.values()), low_memory=low_memory)
    df.columns = [c.strip() for c in df.columns]
    inv = {v: k for k, v in resolved.items()}
    return df.rename(columns=inv)


def _safe_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Safe datetime parser for MIMIC weird dates (e.g. year 3000 for de-identified DOB).
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan})

    # If looks like YYYY-..., kill years >= 3000 (common de-id pattern)
    year = pd.to_numeric(s.str.slice(0, 4), errors="coerce")
    s = s.mask(year >= 3000)

    return pd.to_datetime(s, errors="coerce")


# =========================
# Mooney utilities
# =========================
def list_mooney_files(split_dir: Path, task: str) -> Tuple[List[str], List[int]]:
    """
    task:
      - 'binary' : Normal=0, (Bacterial+Viral)=1
      - '3class' : Normal=0, Bacterial=1, Viral=2
    """
    files, labels = [], []
    exts = {".jpg", ".jpeg", ".png"}

    if task == "binary":
        for p in (split_dir / "Normal").rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p.resolve()))
                labels.append(0)
        for cls in ["Bacterial", "Viral"]:
            for p in (split_dir / cls).rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    files.append(str(p.resolve()))
                    labels.append(1)
    else:
        for cls, y in [("Normal", 0), ("Bacterial", 1), ("Viral", 2)]:
            for p in (split_dir / cls).rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    files.append(str(p.resolve()))
                    labels.append(y)

    return files, labels


def decode_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
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
# MIMIC feature building
# =========================
def is_pneumonia_icd9(code: str) -> bool:
    if pd.isna(code):
        return False
    c = str(code).strip().replace(".", "")
    return c.startswith(("480", "481", "482", "483", "484", "485", "486", "4870"))


def infer_bact_viral_from_org(org: Optional[str]) -> str:
    if org is None or (isinstance(org, float) and np.isnan(org)):
        return "Unknown"
    s = str(org).lower()
    viral_keys = ["virus", "influenza", "rsv", "adenovirus", "parainfluenza", "coronavirus", "metapneumovirus"]
    if any(k in s for k in viral_keys):
        return "Viral"
    return "Bacterial"


def find_itemids_by_label(d_items: pd.DataFrame, must_contain: List[str]) -> set:
    lab = d_items["LABEL"].astype(str).str.lower()
    mask = np.ones(len(d_items), dtype=bool)
    for m in must_contain:
        mask &= lab.str.contains(m.lower(), na=False)
    return set(d_items.loc[mask, "ITEMID"].astype(int).tolist())


def build_mimic_features_csv(mimic_dir: Path, out_csv: Path, task: str, hours: int = 24) -> Path:
    mimic_dir = Path(mimic_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building MIMIC features from:", mimic_dir.resolve())

    # ---- core tables (robust columns)
    admissions = _read_csv_std(
        mimic_dir / "ADMISSIONS.csv",
        wanted={
            "HADM_ID": ["HADM_ID", "HADMID"],
            "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
            "ADMITTIME": ["ADMITTIME", "ADMIT_TIME", "ADMIT TIME", "ADMIT_TIME "],
        },
        low_memory=False,
    )
    admissions["ADMITTIME"] = _safe_parse_datetime(admissions["ADMITTIME"])
    admissions = admissions.dropna(subset=["ADMITTIME"]).reset_index(drop=True)
    admissions["HADM_ID"] = admissions["HADM_ID"].astype(int)
    admissions["SUBJECT_ID"] = admissions["SUBJECT_ID"].astype(int)

    patients = _read_csv_std(
        mimic_dir / "PATIENTS.csv",
        wanted={
            "SUBJECT_ID": ["SUBJECT_ID", "SUBJECTID"],
            "DOB": ["DOB", "DATE_OF_BIRTH", "DATE OF BIRTH"],
        },
        low_memory=False,
    )
    patients["DOB"] = _safe_parse_datetime(patients["DOB"])
    patients = patients.dropna(subset=["DOB"]).reset_index(drop=True)
    patients["SUBJECT_ID"] = patients["SUBJECT_ID"].astype(int)

    dx = _read_csv_std(
        mimic_dir / "DIAGNOSES_ICD.csv",
        wanted={
            "HADM_ID": ["HADM_ID", "HADMID"],
            "ICD9_CODE": ["ICD9_CODE", "ICD9", "ICD9CODE", "ICD9 CODE"],
        },
        low_memory=False,
    )
    dx["HADM_ID"] = dx["HADM_ID"].astype(int)

    d_items = _read_csv_std(
        mimic_dir / "D_ITEMS.csv",
        wanted={"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]},
        low_memory=False,
    )
    d_items["ITEMID"] = d_items["ITEMID"].astype(int)

    d_lab = _read_csv_std(
        mimic_dir / "D_LABITEMS.csv",
        wanted={"ITEMID": ["ITEMID", "ITEM_ID"], "LABEL": ["LABEL", "NAME"]},
        low_memory=False,
    )
    d_lab["ITEMID"] = d_lab["ITEMID"].astype(int)

    # ---- labels (ICD9 pneumonia)
    pneu_hadm = set(dx.loc[dx["ICD9_CODE"].astype(str).apply(is_pneumonia_icd9), "HADM_ID"].astype(int).tolist())
    admissions["label_bin"] = admissions["HADM_ID"].isin(pneu_hadm).astype(int)

    # ---- join for age & window
    adm_pat = admissions.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="left")
    adm_pat = adm_pat.dropna(subset=["DOB"]).reset_index(drop=True)

    # SAFE age compute using days (prevents int64 ns overflow)
    delta_days = (adm_pat["ADMITTIME"] - adm_pat["DOB"]).dt.days.astype("float64")
    age = delta_days / 365.2425
    age = age.where(np.isfinite(age), np.nan)
    age = np.where(age > 120.0, 90.0, age)  # de-id cap (age>89 often grouped)
    age = np.clip(age, 0.0, 110.0)
    adm_pat["age"] = age.astype("float64")

    adm_pat["t_end"] = adm_pat["ADMITTIME"] + pd.to_timedelta(hours, unit="h")

    # ---- ITEMIDs
    temp_c_ids = find_itemids_by_label(d_items, ["temperature", "c"])
    temp_f_ids = find_itemids_by_label(d_items, ["temperature", "f"])
    spo2_ids = find_itemids_by_label(d_items, ["spo2"])
    if len(spo2_ids) == 0:
        spo2_ids = find_itemids_by_label(d_items, ["o2", "saturation"])

    wbc_ids = set(
        d_lab.loc[
            d_lab["LABEL"].astype(str).str.lower().str.contains(r"\bwbc\b", na=False, regex=True)
            | d_lab["LABEL"].astype(str).str.lower().str.contains("white blood", na=False, regex=False),
            "ITEMID",
        ].astype(int).tolist()
    )

    # ---- CHARTEVENTS (temp/spo2) robust columns
    ce = _read_csv_std(
        mimic_dir / "CHARTEVENTS.csv",
        wanted={
            "HADM_ID": ["HADM_ID", "HADMID"],
            "ITEMID": ["ITEMID", "ITEM_ID"],
            "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
            "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
        },
        low_memory=True,
    )
    ce["HADM_ID"] = pd.to_numeric(ce["HADM_ID"], errors="coerce").astype("Int64")
    ce["ITEMID"] = pd.to_numeric(ce["ITEMID"], errors="coerce").astype("Int64")
    ce["VALUENUM"] = pd.to_numeric(ce["VALUENUM"], errors="coerce")
    ce["CHARTTIME"] = _safe_parse_datetime(ce["CHARTTIME"])
    ce = ce.dropna(subset=["HADM_ID", "ITEMID", "VALUENUM", "CHARTTIME"]).copy()
    ce["HADM_ID"] = ce["HADM_ID"].astype(int)
    ce["ITEMID"] = ce["ITEMID"].astype(int)

    ce = ce[ce["ITEMID"].isin(temp_c_ids | temp_f_ids | spo2_ids)].copy()
    ce = ce.merge(adm_pat[["HADM_ID", "ADMITTIME", "t_end"]], on="HADM_ID", how="inner")
    ce = ce[(ce["CHARTTIME"] >= ce["ADMITTIME"]) & (ce["CHARTTIME"] <= ce["t_end"])].copy()

    ce_temp_c = ce[ce["ITEMID"].isin(temp_c_ids)].copy()
    ce_temp_f = ce[ce["ITEMID"].isin(temp_f_ids)].copy()
    ce_temp_f["VALUENUM"] = (ce_temp_f["VALUENUM"].astype(float) - 32.0) / 1.8
    ce_temp = pd.concat([ce_temp_c, ce_temp_f], ignore_index=True)
    ce_temp = ce_temp[(ce_temp["VALUENUM"] >= 30) & (ce_temp["VALUENUM"] <= 45)]

    ce_spo2 = ce[ce["ITEMID"].isin(spo2_ids)].copy()
    ce_spo2 = ce_spo2[(ce_spo2["VALUENUM"] >= 50) & (ce_spo2["VALUENUM"] <= 100)]

    # ---- LABEVENTS (WBC) robust columns
    le = _read_csv_std(
        mimic_dir / "LABEVENTS.csv",
        wanted={
            "HADM_ID": ["HADM_ID", "HADMID"],
            "ITEMID": ["ITEMID", "ITEM_ID"],
            "CHARTTIME": ["CHARTTIME", "CHART_TIME", "CHART TIME"],
            "VALUENUM": ["VALUENUM", "VALUE", "VALUE_NUM", "VALUE NUM"],
        },
        low_memory=True,
    )
    le["HADM_ID"] = pd.to_numeric(le["HADM_ID"], errors="coerce").astype("Int64")
    le["ITEMID"] = pd.to_numeric(le["ITEMID"], errors="coerce").astype("Int64")
    le["VALUENUM"] = pd.to_numeric(le["VALUENUM"], errors="coerce")
    le["CHARTTIME"] = _safe_parse_datetime(le["CHARTTIME"])
    le = le.dropna(subset=["HADM_ID", "ITEMID", "VALUENUM", "CHARTTIME"]).copy()
    le["HADM_ID"] = le["HADM_ID"].astype(int)
    le["ITEMID"] = le["ITEMID"].astype(int)

    le = le[le["ITEMID"].isin(wbc_ids)].copy()
    le = le.merge(adm_pat[["HADM_ID", "ADMITTIME", "t_end"]], on="HADM_ID", how="inner")
    le = le[(le["CHARTTIME"] >= le["ADMITTIME"]) & (le["CHARTTIME"] <= le["t_end"])].copy()

    wbc = le.copy()
    if len(wbc) > 0 and wbc["VALUENUM"].median() < 200:
        wbc["VALUENUM"] = wbc["VALUENUM"] * 1000.0
    wbc = wbc[(wbc["VALUENUM"] >= 2000) & (wbc["VALUENUM"] <= 40000)]

    # ---- aggregate per admission
    feat = adm_pat[["HADM_ID", "label_bin", "age"]].copy()
    feat = feat.merge(ce_temp.groupby("HADM_ID")["VALUENUM"].mean().rename("temperature_c"), on="HADM_ID", how="left")
    feat = feat.merge(ce_spo2.groupby("HADM_ID")["VALUENUM"].min().rename("spo2"), on="HADM_ID", how="left")
    feat = feat.merge(wbc.groupby("HADM_ID")["VALUENUM"].max().rename("wbc"), on="HADM_ID", how="left")
    feat = feat.dropna(subset=["temperature_c", "spo2", "wbc", "age"]).reset_index(drop=True)

    # ---- labels for task
    if task == "binary":
        feat["label"] = feat["label_bin"].astype(int)
    else:
        micro = _read_csv_std(
            mimic_dir / "MICROBIOLOGYEVENTS.csv",
            wanted={
                "HADM_ID": ["HADM_ID", "HADMID"],
                "ORG_NAME": ["ORG_NAME", "ORGANISM", "ORGNAME", "ORG NAME"],
            },
            low_memory=True,
        )
        micro["HADM_ID"] = pd.to_numeric(micro["HADM_ID"], errors="coerce").astype("Int64")
        micro = micro.dropna(subset=["HADM_ID"]).copy()
        micro["HADM_ID"] = micro["HADM_ID"].astype(int)
        micro["bv"] = micro["ORG_NAME"].apply(infer_bact_viral_from_org)

        hadm_to_bv = micro.groupby("HADM_ID")["bv"].agg(
            lambda x: "Viral" if ("Viral" in set(x)) else ("Bacterial" if ("Bacterial" in set(x)) else "Unknown")
        )

        label3 = []
        for hadm, is_p in zip(feat["HADM_ID"].astype(int).tolist(), feat["label_bin"].astype(int).tolist()):
            if is_p == 0:
                label3.append(0)
                continue
            bv = hadm_to_bv.get(hadm, "Unknown")
            if bv == "Bacterial":
                label3.append(1)
            elif bv == "Viral":
                label3.append(2)
            else:
                label3.append(-1)

        feat["label"] = label3
        feat = feat[feat["label"].isin([0, 1, 2])].reset_index(drop=True)

    feat[["HADM_ID", "temperature_c", "spo2", "wbc", "age", "label"]].to_csv(out_csv, index=False)
    print("[INFO] Wrote:", out_csv.resolve(), "rows=", len(feat))
    print("[INFO] Label counts:", feat["label"].value_counts().to_dict())
    return out_csv


def make_clin_ds(features_csv: Path, batch: int, shuffle: bool) -> tf.data.Dataset:
    df = pd.read_csv(features_csv)
    X = df[["temperature_c", "wbc", "spo2", "age"]].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# Models
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
    return tf.keras.Model(inp, x, name="img_encoder")


def build_clin_encoder(emb_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(4,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(emb_dim, activation=None)(x)
    x = layers.LayerNormalization()(x)
    return tf.keras.Model(inp, x, name="clin_encoder")


def build_head(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(EMB_DIM,))
    x = layers.Dense(64, activation="relu")(inp)
    out = layers.Dense(num_classes, activation=None)(x)  # logits
    return tf.keras.Model(inp, out, name="head")


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
    num_classes: int,
    epochs: int,
    lam_mean: float,
    lam_corr: float,
    steps_per_epoch: int,
):
    img_enc = build_img_encoder(EMB_DIM)
    clin_enc = build_clin_encoder(EMB_DIM)
    head_img = build_head(num_classes)
    head_clin = build_head(num_classes)

    opt = tf.keras.optimizers.Adam(1e-3)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    img_it = iter(img_train.repeat())
    clin_it = iter(clin_train.repeat())

    def eval_accuracy(img_ds, clin_ds):
        acc_img = tf.keras.metrics.SparseCategoricalAccuracy()
        for xb, yb in img_ds:
            z = img_enc(xb, training=False)
            logits = head_img(z, training=False)
            acc_img.update_state(yb, tf.nn.softmax(logits))

        acc_cl = tf.keras.metrics.SparseCategoricalAccuracy()
        for xc, yc in clin_ds:
            z = clin_enc(xc, training=False)
            logits = head_clin(z, training=False)
            acc_cl.update_state(yc, tf.nn.softmax(logits))

        return float(acc_img.result().numpy()), float(acc_cl.result().numpy())

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

                li = head_img(zi, training=True)
                lc = head_clin(zc, training=True)

                loss_cls = ce(yb, li) + ce(yc, lc)

                loss_align = tf.constant(0.0, dtype=tf.float32)
                for k in range(num_classes):
                    zi_k = tf.boolean_mask(zi, tf.equal(yb, k))
                    zc_k = tf.boolean_mask(zc, tf.equal(yc, k))
                    if tf.shape(zi_k)[0] >= 4 and tf.shape(zc_k)[0] >= 4:
                        loss_align += lam_mean * mean_loss(zi_k, zc_k)
                        loss_align += lam_corr * coral_corr_loss(zi_k, zc_k)

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
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "3class"], default="3class")
    ap.add_argument("--mimic_dir", type=str, default=str(MIMIC_DIR_DEFAULT))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lam_mean", type=float, default=0.05)
    ap.add_argument("--lam_corr", type=float, default=0.10)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--rebuild_mimic_features", action="store_true")
    args = ap.parse_args()

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

    num_classes = 2 if args.task == "binary" else 3
    img_train = make_img_ds(mo_tr_f, mo_tr_y, batch=BATCH_IMG, shuffle=True)
    img_val = make_img_ds(mo_te_f, mo_te_y, batch=BATCH_IMG, shuffle=False)

    # ---- MIMIC features auto-create
    mimic_dir = Path(args.mimic_dir)
    if not mimic_dir.exists():
        raise FileNotFoundError(f"MIMIC dir not found: {mimic_dir.resolve()}")

    mimic_features_csv = mimic_dir / f"mimic_features_{args.task}.csv"
    if args.rebuild_mimic_features or (not mimic_features_csv.exists()):
        build_mimic_features_csv(mimic_dir, mimic_features_csv, task=args.task, hours=args.hours)
    else:
        print("[INFO] Using existing MIMIC features:", mimic_features_csv.resolve())

    df = pd.read_csv(mimic_features_csv)
    if len(df) < 20:
        print("[WARN] Very small MIMIC feature set. Consider --task binary or increase --hours.")
    if df["label"].nunique() < (2 if args.task == "binary" else 3):
        print("[WARN] Not all classes exist in MIMIC features for this task; alignment for missing classes won't apply.")

    # split train/val for clinical
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_tr = max(int(0.8 * n), 1)
    df_tr = df.iloc[:n_tr].copy()
    df_va = df.iloc[n_tr:].copy()
    if len(df_va) == 0:
        df_va = df_tr.copy()

    # standardize clinical (fit on train)
    mu = df_tr[["temperature_c", "wbc", "spo2", "age"]].mean()
    sd = df_tr[["temperature_c", "wbc", "spo2", "age"]].std() + 1e-6
    for dfx in (df_tr, df_va):
        X = (dfx[["temperature_c", "wbc", "spo2", "age"]] - mu) / sd
        dfx.loc[:, ["temperature_c", "wbc", "spo2", "age"]] = X

    train_csv = Path("mimic_train.csv")
    val_csv = Path("mimic_val.csv")
    df_tr.to_csv(train_csv, index=False)
    df_va.to_csv(val_csv, index=False)

    clin_train = make_clin_ds(train_csv, batch=BATCH_CLIN, shuffle=True)
    clin_val = make_clin_ds(val_csv, batch=BATCH_CLIN, shuffle=False)

    # ---- Train dual
    train_dual(
        img_train=img_train,
        clin_train=clin_train,
        img_val=img_val,
        clin_val=clin_val,
        num_classes=num_classes,
        epochs=args.epochs,
        lam_mean=args.lam_mean,
        lam_corr=args.lam_corr,
        steps_per_epoch=args.steps_per_epoch,
    )


if __name__ == "__main__":
    main()
