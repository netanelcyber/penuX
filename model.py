# ============================================
# FULL MULTI-MODAL AI PIPELINE FOR LUNG DISEASE
# (Auto-download + prepare Kaggle dataset)
# Classes: Normal, Bacterial, Viral
# ============================================

import os
import sys
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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

RAW_DIR = Path("kaggle_raw")         # where Kaggle download/extract lives
OUT_DIR = Path("dataset")           # final structure used by ImageDataGenerator
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"

# ============================================
# HELPERS: OPTIONAL INSTALLS
# ============================================
def _pip_install(pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs))

def ensure_kaggle_download_tools():
    """
    Tries kagglehub first (often simplest). If not available, tries Kaggle API package.
    """
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

    # Install both (quietly) to maximize chance of success.
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
    """
    Returns a local folder path containing the dataset files.
    Strategy:
      1) Try kagglehub.dataset_download (downloads into cache; returns cache path).
      2) Fallback to Kaggle API (downloads zip into dest_dir and unzips).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    tool = ensure_kaggle_download_tools()

    # ---- Option A: kagglehub
    if tool == "kagglehub":
        import kagglehub
        try:
            path = kagglehub.dataset_download(dataset_slug)
            return Path(path)
        except Exception as e:
            print(f"[WARN] kagglehub failed ({e}). Falling back to Kaggle API...")

    # ---- Option B: Kaggle API (needs ~/.kaggle/kaggle.json OR env vars)
    # If kaggle.json is missing, this will error with a clear message.
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

    # Download + unzip
    print(f"[INFO] Downloading Kaggle dataset: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
    return dest_dir
def multimodal_generator(image_gen, clinical_data):
    i = 0
    n = len(clinical_data)

    while True:
        images, labels = next(image_gen)
        b = images.shape[0]

        # wraparound-safe clinical batch
        if i + b <= n:
            clinical_batch = clinical_data[i:i+b]
            i += b
        else:
            part1 = clinical_data[i:n]
            part2 = clinical_data[0:(i+b) % n]
            clinical_batch = np.concatenate([part1, part2], axis=0)
            i = (i + b) % n

        clinical_batch = clinical_batch.astype(np.float32)
        images = images.astype(np.float32)
        yield (clinical_batch, images), labels

def find_chest_xray_root(download_root: Path) -> Path:
    """
    Finds the 'chest_xray' folder inside the downloaded content.
    """
    # common direct
    direct = download_root / "chest_xray"
    if direct.exists():
        return direct

    # search
    for p in download_root.rglob("chest_xray"):
        if p.is_dir():
            return p

    raise FileNotFoundError(f"Could not locate 'chest_xray' folder under: {download_root}")

def safe_link_or_copy(src: Path, dst: Path):
    """
    Create a hardlink if possible (fast, saves space). Fallback to copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def prepare_three_class_structure(chest_xray_root: Path, out_dir: Path):
    """
    Converts Kaggle 2-class structure:
      chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
    into 3-class folder structure:
      dataset/{train,test}/{Normal,Bacterial,Viral}

    We MERGE the original 'val' into our 'train' because the training code uses
    validation_split=0.2 on TRAIN_DIR.
    """
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # Ensure class folders exist
    for split in ["train", "test"]:
        for cls in CLASS_NAMES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def subtype_from_filename(name: str) -> str | None:
        low = name.lower()
        if "bacteria" in low:
            return "Bacterial"
        if "virus" in low or "viral" in low:
            return "Viral"
        return None

    split_map = {
        "train": "train",
        "val": "train",   # merge into train (we'll do our own val split via ImageDataGenerator)
        "test": "test",
    }

    for src_split, dst_split in split_map.items():
        src_normal = chest_xray_root / src_split / "NORMAL"
        src_pneu = chest_xray_root / src_split / "PNEUMONIA"

        if not src_normal.exists() or not src_pneu.exists():
            # Some mirrors may differ slightly; fail fast with context.
            raise FileNotFoundError(
                f"Expected folders missing under {chest_xray_root / src_split}.\n"
                f"Found NORMAL? {src_normal.exists()} | PNEUMONIA? {src_pneu.exists()}"
            )

        # NORMAL -> Normal
        for img in src_normal.glob("*"):
            if img.is_file():
                dst = out_dir / dst_split / "Normal" / f"{src_split}_{img.name}"
                safe_link_or_copy(img, dst)

        # PNEUMONIA -> Bacterial/Viral by filename
        for img in src_pneu.glob("*"):
            if not img.is_file():
                continue
            subtype = subtype_from_filename(img.name)
            if subtype is None:
                # skip unknown patterns
                continue
            dst = out_dir / dst_split / subtype / f"{src_split}_{img.name}"
            safe_link_or_copy(img, dst)

def ensure_dataset_ready():
    """
    If dataset/train and dataset/test already look prepared, skip.
    Otherwise download from Kaggle and prepare.
    """
    def has_images(p: Path) -> bool:
        return p.exists() and any(p.rglob("*.jpeg")) or any(p.rglob("*.jpg")) or any(p.rglob("*.png"))

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

    # Rebuild clean output folders (avoid mixing old files)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prepare_three_class_structure(chest_xray_root, OUT_DIR)
    print("[INFO] Dataset ready at:", OUT_DIR.resolve())

# ============================================
# IMAGE DATA LOADER
# ============================================
def load_image_generators():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_NAMES,          # enforce label order
        shuffle=True,
        seed=SEED
    )

    val_gen = datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_NAMES,          # enforce label order
        shuffle=False
    )

    test_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,          # enforce label order
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# ============================================
# SYNTHETIC CLINICAL DATA GENERATION
# ============================================
def generate_clinical_features(n):
    """
    Features:
      Temperature (°C)
      WBC (cells/µL)
      SpO2 (%)
      Age (years)
    """
    temp = np.random.normal(38, 1.0, n)
    wbc = np.random.normal(12000, 3500, n)
    spo2 = np.random.normal(93, 3, n)
    age = np.random.randint(18, 90, n)
    return np.column_stack([temp, wbc, spo2, age]).astype(np.float32)

def standardize_clinical(train, val, test):
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sigma, (val - mu) / sigma, (test - mu) / sigma

# ============================================
# MULTI-MODAL MODEL
# ============================================
def build_multimodal_model():
    # ---- Clinical branch ----
    clinical_input = layers.Input(shape=(4,), name="Clinical_Input")
    c = layers.Dense(64, activation="relu")(clinical_input)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(32, activation="relu")(c)

    # ---- Image branch ----
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="Image_Input")
    x = layers.Conv2D(32, 3, activation="relu")(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # ---- Fusion ----
    merged = layers.concatenate([c, x])
    m = layers.Dense(128, activation="relu")(merged)
    m = layers.Dropout(0.4)(m)

    output = layers.Dense(NUM_CLASSES, activation="softmax")(m)

    model = models.Model(
        inputs=[clinical_input, image_input],
        outputs=output
    )

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
# MULTI-MODAL GENERATOR (ROBUST WRAPAROUND)
# ============================================
def multi1modal_generator(image_gen, clinical_data):
    """
    Yields ([clinical_batch, image_batch], labels).
    Ensures clinical_batch always matches image_batch size (wrap-around safe).
    """
    i = 0
    n = len(clinical_data)

    while True:
        images, labels = next(image_gen)
        b = images.shape[0]

        if i + b <= n:
            clinical_batch = clinical_data[i:i + b]
            i = i + b
        else:
            part1 = clinical_data[i:n]
            part2 = clinical_data[0:(i + b) % n]
            clinical_batch = np.concatenate([part1, part2], axis=0)
            i = (i + b) % n

        yield [clinical_batch, images], labels

# ============================================
# TRAINING FUNCTION
# ============================================
def train_model():
    ensure_dataset_ready()
    train_gen, val_gen, test_gen = load_image_generators()

    Xc_train = generate_clinical_features(train_gen.samples)
    Xc_val = generate_clinical_features(val_gen.samples)
    Xc_test = generate_clinical_features(test_gen.samples)

    # Standardize clinical features for stability
    Xc_train, Xc_val, Xc_test = standardize_clinical(Xc_train, Xc_val, Xc_test)

    model = build_multimodal_model()
    model.summary()

    steps_per_epoch = math.ceil(train_gen.samples / BATCH_SIZE)
    validation_steps = math.ceil(val_gen.samples / BATCH_SIZE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_multimodal_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True
        ),
    ]

    history = model.fit(
        multimodal_generator(train_gen, Xc_train),
        validation_data=multimodal_generator(val_gen, Xc_val),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    evaluate_model(model, test_gen, Xc_test)
    model.save("MultiModal_Lung_Disease_AI.h5")
    print("\n[INFO] Model saved as MultiModal_Lung_Disease_AI.h5")
    plot_training_curves(history)

# ============================================
# EVALUATION
# ============================================
def evaluate_model(model, test_gen, clinical_data):
    preds = []
    labels = []

    steps = math.ceil(test_gen.samples / BATCH_SIZE)
    gen = multimodal_generator(test_gen, clinical_data)

    for _ in range(steps):
        (Xc, Xi), y = next(gen)
        p = model.predict([Xc, Xi], verbose=0)
        preds.append(p)
        labels.append(y)

    preds = np.vstack(preds)[:test_gen.samples]
    labels = np.vstack(labels)[:test_gen.samples]

    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(preds, axis=1)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

    auc = roc_auc_score(labels, preds, multi_class="ovr")
    print(f"Overall AUC (OvR): {auc:.4f}")

# ============================================
# CONFUSION MATRIX PLOT
# ============================================
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show()
    print("[INFO] Saved confusion matrix to confusion_matrix.png")

def plot_training_curves(history):
    # Make a quick plot for loss/acc/auc (optional)
    keys = history.history.keys()
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history.get("accuracy", []), label="acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    # AUC
    plt.subplot(1, 3, 3)
    plt.plot(history.history.get("auc", []), label="auc")
    plt.plot(history.history.get("val_auc", []), label="val_auc")
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.show()
    print("[INFO] Saved training curves to training_curves.png")

# ============================================
# GRAD-CAM PLACEHOLDER (EXPLAINABILITY)
# ============================================
def grad_cam_placeholder():
    """
    Grad-CAM can be added for the image branch
    to visualize regions contributing to prediction.
    """
    pass

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    train_model()
