"""
FULL MULTI-MODAL (X-ray + Clinical/Blood/Vitals e.g., SpO2) AI PIPELINE — ONE PY FILE
Classes: Normal, Bacterial, Viral
Image dataset: paultimothymooney/chest-xray-pneumonia

✅ NEW: "AUTO-DOWNLOAD / AUTO-CREATE clinical.csv"
- The Mooney chest-xray dataset does NOT include clinical/blood/vitals.
- So this script will:
  (A) If you provide --clinical_csv_url, it will download that CSV automatically.
  (B) If clinical.csv is missing and no URL is provided, it will AUTO-GENERATE a clinical.csv
      aligned to every prepared image filename (synthetic-but-plausible ranges), so training runs end-to-end.

Expected clinical.csv columns (default):
  filename, temperature_c, wbc, spo2, age

Run:
  python multimodal_lung.py
  python multimodal_lung.py --clinical_csv_url "https://.../clinical.csv"
  python multimodal_lung.py --features temperature_c,wbc,spo2,age

Note:
- Auto-generated clinical.csv is synthetic. Replace it with real patient labs/vitals when available.
"""

import os
import sys
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# optional dependency (recommended for CSV)
try:
    import pandas as pd
except Exception:
    pd = None


# ============================================
# CONFIGURATION
# ============================================
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 3
SEED = 42

CLASS_NAMES = ["Normal", "Bacterial", "Viral"]  # enforced order everywhere

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# KAGGLE DATASET (DOWNLOAD + PREP)
# ============================================
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

RAW_DIR = Path("kaggle_raw")
OUT_DIR = Path("dataset")
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"

# Clinical CSV (user-provided OR auto-downloaded OR auto-created)
CLINICAL_CSV = Path("clinical.csv")

# Default clinical feature columns
CLINICAL_FEATURES = ["temperature_c", "wbc", "spo2", "age"]


# ============================================
# HELPERS: OPTIONAL INSTALLS
# ============================================
def _pip_install(pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs))


def ensure_kaggle_download_tools():
    try:
        import kagglehub  # noqa: F401
        return "kagglehub"
    except Exception:
        pass

    try:
        import kaggle  # noqa: F401
        return "kaggle"
    except Exception:
        pass

    _pip_install(["kagglehub", "kaggle"])
    try:
        import kagglehub  # noqa: F401
        return "kagglehub"
    except Exception:
        import kaggle  # noqa: F401
        return "kaggle"


# ============================================
# HELPERS: DOWNLOAD FROM KAGGLE
# ============================================
def download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tool = ensure_kaggle_download_tools()

    if tool == "kagglehub":
        import kagglehub
        try:
            path = kagglehub.dataset_download(dataset_slug)
            return Path(path)
        except Exception as e:
            print(f"[WARN] kagglehub failed ({e}). Falling back to Kaggle API...")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError(
            "Kaggle authentication failed.\n\n"
            "Fix (ONE of these):\n"
            "  A) Put kaggle.json at: ~/.kaggle/kaggle.json  (chmod 600)\n"
            "  B) Or set env vars: KAGGLE_USERNAME and KAGGLE_KEY\n\n"
            f"Original error: {e}"
        )

    print(f"[INFO] Downloading Kaggle dataset: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
    return dest_dir


def find_chest_xray_root(download_root: Path) -> Path:
    direct = download_root / "chest_xray"
    if direct.exists():
        return direct

    for p in download_root.rglob("chest_xray"):
        if p.is_dir():
            return p

    raise FileNotFoundError(f"Could not locate 'chest_xray' folder under: {download_root}")


def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def prepare_three_class_structure(chest_xray_root: Path, out_dir: Path):
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        for cls in CLASS_NAMES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def subtype_from_filename(name: str):
        low = name.lower()
        if "bacteria" in low:
            return "Bacterial"
        if "virus" in low or "viral" in low:
            return "Viral"
        return None

    split_map = {"train": "train", "val": "train", "test": "test"}

    for src_split, dst_split in split_map.items():
        src_normal = chest_xray_root / src_split / "NORMAL"
        src_pneu = chest_xray_root / src_split / "PNEUMONIA"

        if not src_normal.exists() or not src_pneu.exists():
            raise FileNotFoundError(
                f"Expected folders missing under {chest_xray_root / src_split}.\n"
                f"Found NORMAL? {src_normal.exists()} | PNEUMONIA? {src_pneu.exists()}"
            )

        for img in src_normal.glob("*"):
            if img.is_file():
                dst = out_dir / dst_split / "Normal" / f"{src_split}_{img.name}"
                safe_link_or_copy(img, dst)

        for img in src_pneu.glob("*"):
            if not img.is_file():
                continue
            subtype = subtype_from_filename(img.name)
            if subtype is None:
                continue
            dst = out_dir / dst_split / subtype / f"{src_split}_{img.name}"
            safe_link_or_copy(img, dst)


def has_images(p: Path) -> bool:
    if not p.exists():
        return False
    return any(p.rglob("*.jpeg")) or any(p.rglob("*.jpg")) or any(p.rglob("*.png"))


def ensure_dataset_ready():
    prepared = (
        (TRAIN_DIR / "Normal").exists()
        and (TRAIN_DIR / "Bacterial").exists()
        and (TRAIN_DIR / "Viral").exists()
        and (TEST_DIR / "Normal").exists()
        and (TEST_DIR / "Bacterial").exists()
        and (TEST_DIR / "Viral").exists()
        and has_images(TRAIN_DIR)
        and has_images(TEST_DIR)
    )

    if prepared:
        print("[INFO] Prepared dataset already exists. Skipping download/prep.")
        return

    print("[INFO] Preparing dataset (download + restructure)...")
    download_root = download_kaggle_dataset(KAGGLE_DATASET, RAW_DIR)
    chest_xray_root = find_chest_xray_root(download_root)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prepare_three_class_structure(chest_xray_root, OUT_DIR)
    print("[INFO] Dataset ready at:", OUT_DIR.resolve())


# ============================================
# FILE LISTING + SPLIT
# ============================================
def list_files_and_labels(root_dir: Path) -> Tuple[List[str], List[int]]:
    filepaths = []
    labels = []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = root_dir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                filepaths.append(str(p.resolve()))
                labels.append(label_idx)
    return filepaths, labels


def stratified_split(filepaths: List[str], labels: List[int], val_frac=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    filepaths = np.array(filepaths)
    labels = np.array(labels)

    train_idx = []
    val_idx = []

    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (filepaths[train_idx].tolist(), labels[train_idx].tolist(),
            filepaths[val_idx].tolist(), labels[val_idx].tolist())


# ============================================
# CLINICAL CSV: AUTO-DOWNLOAD / AUTO-CREATE
# ============================================
def ensure_requests():
    try:
        import requests  # noqa: F401
    except Exception:
        _pip_install(["requests"])


def maybe_download_clinical_csv(csv_path: Path, csv_url: Optional[str]) -> bool:
    """
    If csv_url provided and file missing, download it.
    Returns True if downloaded, False otherwise.
    """
    if csv_url is None:
        return False
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return False

    ensure_requests()
    import requests

    print(f"[INFO] Downloading clinical CSV from URL -> {csv_path.resolve()}")
    r = requests.get(csv_url, timeout=60)
    r.raise_for_status()
    csv_path.write_bytes(r.content)
    print("[INFO] Clinical CSV downloaded.")
    return True


def ensure_pandas():
    global pd
    if pd is None:
        _pip_install(["pandas"])
        import pandas as _pd
        pd = _pd


def make_synthetic_row_for_label(label_idx: int, rng: np.random.RandomState) -> Dict[str, float]:
    """
    Slightly different distributions by class (purely synthetic).
    """
    if label_idx == 0:  # Normal
        temp = rng.normal(37.0, 0.4)
        wbc = rng.normal(7000, 1200)
        spo2 = rng.normal(97.0, 1.5)
        age = rng.randint(18, 85)
    elif label_idx == 1:  # Bacterial
        temp = rng.normal(39.0, 0.7)
        wbc = rng.normal(15000, 3000)
        spo2 = rng.normal(92.0, 3.0)
        age = rng.randint(18, 90)
    else:  # Viral
        temp = rng.normal(38.2, 0.6)
        wbc = rng.normal(9500, 2200)
        spo2 = rng.normal(93.5, 3.0)
        age = rng.randint(18, 90)

    # clamp plausibly
    temp = float(np.clip(temp, 35.0, 42.0))
    wbc = float(np.clip(wbc, 2000, 40000))
    spo2 = float(np.clip(spo2, 70.0, 100.0))
    age = float(np.clip(age, 0, 110))

    return {
        "temperature_c": temp,
        "wbc": wbc,
        "spo2": spo2,
        "age": age,
    }


def auto_create_clinical_csv(csv_path: Path,
                             all_files: List[str],
                             all_labels: List[int],
                             feature_cols: List[str],
                             seed: int = SEED):
    """
    Creates a clinical.csv aligned to *every* image basename in the prepared dataset.
    Uses synthetic values (because the Kaggle image dataset has no clinical data).
    """
    ensure_pandas()
    rng = np.random.RandomState(seed)

    rows = []
    for fp, lab in zip(all_files, all_labels):
        base = os.path.basename(fp)
        synth = make_synthetic_row_for_label(lab, rng)

        row = {"filename": base}
        for c in feature_cols:
            if c not in synth:
                # If user adds extra columns, fill with standard normal
                row[c] = float(rng.normal(0, 1))
            else:
                row[c] = synth[c]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[INFO] Auto-created synthetic clinical CSV at: {csv_path.resolve()}")
    print("[INFO] Replace this CSV with real labs/vitals when available.")


def ensure_clinical_csv_exists(csv_path: Path,
                               csv_url: Optional[str],
                               feature_cols: List[str]):
    """
    Ensures clinical.csv exists.
    Priority:
      1) If missing and url provided -> download.
      2) If still missing -> auto-create synthetic CSV based on dataset files/labels.
    """
    maybe_download_clinical_csv(csv_path, csv_url)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    print("[WARN] clinical.csv not found (and no valid download). Auto-generating synthetic clinical.csv...")
    # Need dataset prepared to enumerate images
    ensure_dataset_ready()
    train_all_files, train_all_labels = list_files_and_labels(TRAIN_DIR)
    test_files, test_labels = list_files_and_labels(TEST_DIR)
    all_files = train_all_files + test_files
    all_labels = train_all_labels + test_labels
    if len(all_files) == 0:
        raise RuntimeError("No images found to build clinical.csv. Dataset prep failed?")
    auto_create_clinical_csv(csv_path, all_files, all_labels, feature_cols, seed=SEED)


def load_clinical_map(csv_path: Path, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    ensure_pandas()
    if not csv_path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("clinical.csv must include a 'filename' column (basename of image file).")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"clinical.csv missing columns: {missing_cols}")

    df["filename"] = df["filename"].astype(str)

    clin_map = {}
    for _, row in df.iterrows():
        fname = row["filename"]
        clin_map[fname] = row[feature_cols].to_numpy(dtype=np.float32)

    return clin_map


def build_aligned_clinical_array(filepaths: List[str], clin_map: Dict[str, np.ndarray], d: int):
    Xc = np.zeros((len(filepaths), d), dtype=np.float32)
    missing = 0
    examples = []
    for i, fp in enumerate(filepaths):
        base = os.path.basename(fp)
        v = clin_map.get(base, None)
        if v is None:
            missing += 1
            if len(examples) < 5:
                examples.append(base)
            continue
        if v.shape[0] != d:
            raise ValueError(f"Clinical feature length mismatch for {base}: got {v.shape[0]}, expected {d}")
        Xc[i] = v

    if missing > 0:
        print(f"[WARN] Missing clinical rows for {missing}/{len(filepaths)} images. Example: {examples}")
        print("[WARN] These will be filled with zeros. Better: fix your clinical.csv mapping.")

    return Xc


def standardize_clinical(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sigma, (val - mu) / sigma, (test - mu) / sigma


# ============================================
# TF.DATA PIPELINE
# ============================================
def decode_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(filepaths: List[str], labels: List[int], clinical: np.ndarray,
                 shuffle: bool, batch_size: int) -> tf.data.Dataset:
    y = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, clinical, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 4096), seed=SEED, reshuffle_each_iteration=True)

    def _map(fp, xc, y1):
        img = decode_image(fp)
        return (xc, img), y1

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================
# MODEL
# ============================================
def build_multimodal_model(num_clin_features: int):
    clinical_input = layers.Input(shape=(num_clin_features,), name="Clinical_Input")
    c = layers.Dense(64, activation="relu")(clinical_input)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(32, activation="relu")(c)

    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="Image_Input")
    x = layers.Conv2D(32, 3, activation="relu")(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    merged = layers.concatenate([c, x])
    m = layers.Dense(128, activation="relu")(merged)
    m = layers.Dropout(0.4)(m)

    output = layers.Dense(NUM_CLASSES, activation="softmax")(m)

    model = models.Model(inputs=[clinical_input, image_input], outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc", multi_label=True, num_labels=NUM_CLASSES),
        ]
    )
    return model


# ============================================
# EVALUATION + PLOTS
# ============================================
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show()
    print("[INFO] Saved confusion matrix to confusion_matrix.png")


def plot_training_curves(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history.get("accuracy", []), label="acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history.get("auc", []), label="auc")
    plt.plot(history.history.get("val_auc", []), label="val_auc")
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.show()
    print("[INFO] Saved training curves to training_curves.png")


def evaluate_model(model, ds_test: tf.data.Dataset, y_true_int: np.ndarray):
    preds = model.predict(ds_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true_int, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true_int, y_pred)
    plot_confusion_matrix(cm)

    y_true_oh = tf.one_hot(y_true_int, depth=NUM_CLASSES).numpy()
    auc = roc_auc_score(y_true_oh, preds, multi_class="ovr")
    print(f"Overall AUC (OvR): {auc:.4f}")


# ============================================
# TRAINING
# ============================================
def train_model(clinical_csv: Path,
                clinical_csv_url: Optional[str],
                clinical_features: List[str]):
    ensure_dataset_ready()

    # Ensure clinical CSV exists (download or auto-create)
    ensure_clinical_csv_exists(clinical_csv, clinical_csv_url, clinical_features)

    # Load images
    train_all_files, train_all_labels = list_files_and_labels(TRAIN_DIR)
    test_files, test_labels = list_files_and_labels(TEST_DIR)

    if len(train_all_files) == 0 or len(test_files) == 0:
        raise RuntimeError("No images found. Check dataset preparation.")

    train_files, train_labels, val_files, val_labels = stratified_split(
        train_all_files, train_all_labels, val_frac=0.2, seed=SEED
    )

    d = len(clinical_features)

    # Load clinical map from CSV and align
    clin_map = load_clinical_map(clinical_csv, clinical_features)
    Xc_train = build_aligned_clinical_array(train_files, clin_map, d)
    Xc_val = build_aligned_clinical_array(val_files, clin_map, d)
    Xc_test = build_aligned_clinical_array(test_files, clin_map, d)

    # Standardize
    Xc_train, Xc_val, Xc_test = standardize_clinical(Xc_train, Xc_val, Xc_test)

    # tf.data datasets (shuffle only train)
    ds_train = make_dataset(train_files, train_labels, Xc_train, shuffle=True, batch_size=BATCH_SIZE)
    ds_val = make_dataset(val_files, val_labels, Xc_val, shuffle=False, batch_size=BATCH_SIZE)
    ds_test = make_dataset(test_files, test_labels, Xc_test, shuffle=False, batch_size=BATCH_SIZE)

    model = build_multimodal_model(num_clin_features=d)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_multimodal_model.keras", monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    evaluate_model(model, ds_test, np.array(test_labels, dtype=int))
    model.save("MultiModal_Lung_Disease_AI.h5")
    print("\n[INFO] Model saved as MultiModal_Lung_Disease_AI.h5")
    plot_training_curves(history)


# ============================================
# CLI
# ============================================
def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", type=str, default=str(CLINICAL_CSV),
                    help="Path to clinical.csv (downloaded or auto-created if missing).")
    ap.add_argument("--clinical_csv_url", type=str, default=None,
                    help="Optional URL to download clinical.csv if missing.")
    ap.add_argument("--features", type=str, default=",".join(CLINICAL_FEATURES),
                    help="Comma-separated clinical feature column names in the CSV.")
    return ap.parse_args(argv)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    feats = [f.strip() for f in args.features.split(",") if f.strip()]

    train_model(
        clinical_csv=Path(args.clinical_csv),
        clinical_csv_url=args.clinical_csv_url,
        clinical_features=feats
    )
