"""Train 5 LSTM-based models on the Chinese AP dataset and append to eval_results.json.

Features are reshaped to (n_features, 1) so each lab value is treated as one
time-step in a sequence — a standard approach for applying RNNs to tabular data.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
tf.random.set_seed(42)
np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(np.float32)
N_FEAT = X.shape[1]   # 106

# ── Helpers ────────────────────────────────────────────────────────────────
def best_threshold(y_true, y_prob):
    _, _, ths = roc_curve(y_true, y_prob)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ths]
    return float(ths[np.argmax(f1s)])

def sweep_thresholds(y_true, y_prob):
    rows = []
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        ppv  = tp / (tp + fp) if (tp + fp) else 0
        f1   = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": round(t, 2), "tp": int(tp), "fp": int(fp),
                     "fn": int(fn), "tn": int(tn),
                     "sensitivity": round(sens * 100, 1),
                     "specificity": round(spec * 100, 1),
                     "ppv": round(ppv * 100, 1), "f1": round(f1, 3)})
    return rows

def roc_points(y_true, y_prob, n=40):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.round(np.linspace(0, len(fpr) - 1, n)).astype(int)
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]

def cross_val_proba(model_fn, X, y, lr, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_epochs = []
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr).reshape(-1, N_FEAT, 1)
        Xva = sc.transform(Xva).reshape(-1, N_FEAT, 1)
        model = model_fn()
        es = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                           monitor="val_auc", mode="max")
        hist = model.fit(Xtr, ytr, epochs=80, batch_size=32, verbose=0,
                         validation_data=(Xva, yva), callbacks=[es])
        best_ep = int(np.argmax(hist.history["val_auc"])) + 1
        fold_epochs.append(best_ep)
        oof[va] = model.predict(Xva, verbose=0).ravel()
    return oof, fold_epochs

def compile_model(m, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
    return m

# ── 1. Vanilla LSTM ────────────────────────────────────────────────────────
def make_lstm(lr=8e-4):
    inp = keras.Input(shape=(N_FEAT, 1))
    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out), lr)

# ── 2. Stacked LSTM (2-layer) ──────────────────────────────────────────────
def make_stacked_lstm(lr=5e-4):
    inp = keras.Input(shape=(N_FEAT, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out), lr)

# ── 3. Bidirectional LSTM ─────────────────────────────────────────────────
def make_bilstm(lr=8e-4):
    inp = keras.Input(shape=(N_FEAT, 1))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out), lr)

# ── 4. LSTM + Self-Attention ──────────────────────────────────────────────
def make_lstm_attention(lr=8e-4):
    inp = keras.Input(shape=(N_FEAT, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)          # (batch, N_FEAT, 64)
    # Bahdanau-style additive attention
    score = layers.Dense(1, activation="tanh")(x)            # (batch, N_FEAT, 1)
    weights = layers.Softmax(axis=1)(score)                  # (batch, N_FEAT, 1)
    context = layers.Multiply()([x, weights])                # (batch, N_FEAT, 64)
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)  # (batch, 64)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out), lr)

# ── 5. CNN + LSTM ──────────────────────────────────────────────────────────
def make_cnn_lstm(lr=8e-4):
    inp = keras.Input(shape=(N_FEAT, 1))
    x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(keras.Model(inp, out), lr)

# ── Train all 5 ───────────────────────────────────────────────────────────
configs = [
    ("LSTM",              make_lstm,           8e-4),
    ("Stacked LSTM",      make_stacked_lstm,   5e-4),
    ("Bidirectional LSTM",make_bilstm,         8e-4),
    ("LSTM + Attention",  make_lstm_attention, 8e-4),
    ("CNN-LSTM",          make_cnn_lstm,       8e-4),
]

with open("PenuX-AP-Severity/models/eval_results.json") as f:
    results = json.load(f)

for name, fn, lr in configs:
    print(f"\n=== {name} ===")
    oof, fold_epochs = cross_val_proba(fn, X, y, lr)
    auc  = roc_auc_score(y, oof)
    thr  = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn_) if (tp + fn_) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv  = tp / (tp + fp) if (tp + fp) else 0
    f1   = f1_score(y, pred)
    opt_ep = int(round(np.mean(fold_epochs)))
    print(f"AUC={auc:.4f}  F1={f1:.3f}  T={thr:.3f}  "
          f"Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  "
          f"epochs/fold={fold_epochs}  mean={opt_ep}  lr={lr}")
    results[name] = {
        "auc": round(auc, 4),
        "f1": round(f1, 3),
        "threshold": round(thr, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
        "sens": round(sens * 100, 1),
        "spec": round(spec * 100, 1),
        "ppv":  round(ppv  * 100, 1),
        "optimal_epochs":   opt_ep,
        "epochs_per_fold":  fold_epochs,
        "learning_rate":    lr,
        "roc":     roc_points(y, oof),
        "sweep":   sweep_thresholds(y, oof),
        "features": [],
    }

with open("PenuX-AP-Severity/models/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone — eval_results.json updated.")
