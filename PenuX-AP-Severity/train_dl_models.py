"""Train 3 deep-learning models on the Chinese AP dataset and append results to eval_results.json."""
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
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)
N_FEAT = X.shape[1]

# ── Helpers ────────────────────────────────────────────────────────────────
def best_threshold(y_true, y_prob):
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ths]
    return float(ths[np.argmax(f1s)])

def sweep_thresholds(y_true, y_prob):
    rows = []
    for t in [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80]:
        pred = (y_prob >= t).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true, pred).ravel()
        sens = tp/(tp+fn) if (tp+fn) else 0
        spec = tn/(tn+fp) if (tn+fp) else 0
        ppv  = tp/(tp+fp) if (tp+fp) else 0
        f1   = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold":round(t,3),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),
                     "sensitivity":round(sens*100,1),"specificity":round(spec*100,1),"ppv":round(ppv*100,1),"f1":round(f1,3)})
    return rows

def roc_points(y_true, y_prob, n=40):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.round(np.linspace(0, len(fpr)-1, n)).astype(int)
    return [[round(float(fpr[i]),4), round(float(tpr[i]),4)] for i in idx]

def cross_val_proba(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    importances = np.zeros(N_FEAT)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        model = model_fn()
        model.fit(Xtr, ytr, epochs=60, batch_size=32, verbose=0,
                  validation_data=(Xva, yva),
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                                           monitor='val_auc', mode='max')])
        oof[va] = model.predict(Xva, verbose=0).ravel()
    return oof

# ── Model 1: MLP (3-hidden-layer) ─────────────────────────────────────────
def make_mlp():
    inp = keras.Input(shape=(N_FEAT,))
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[keras.metrics.AUC(name='auc')])
    return m

# ── Model 2: Residual MLP (skip connections) ──────────────────────────────
def make_residual():
    inp = keras.Input(shape=(N_FEAT,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    # residual block 1
    r = x
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Add()([x, r])
    x = layers.BatchNormalization()(x)
    # residual block 2
    r2 = x
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Add()([x, r2])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(8e-4),
              loss='binary_crossentropy',
              metrics=[keras.metrics.AUC(name='auc')])
    return m

# ── Model 3: Attention MLP (feature-wise soft attention) ──────────────────
def make_attention_mlp():
    inp = keras.Input(shape=(N_FEAT,))
    # attention gate
    att = layers.Dense(N_FEAT, activation='sigmoid', name='att_gate')(inp)
    x = layers.Multiply()([inp, att])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[keras.metrics.AUC(name='auc')])
    return m

# ── Train all 3 ───────────────────────────────────────────────────────────
configs = [
    ("MLP", make_mlp),
    ("Residual MLP", make_residual),
    ("Attention MLP", make_attention_mlp),
]

results = {}
for name, fn in configs:
    print(f"\n=== {name} ===")
    oof = cross_val_proba(fn, X, y)
    auc = roc_auc_score(y, oof)
    thr = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn,fp,fn_,tp = confusion_matrix(y, pred).ravel()
    sens = tp/(tp+fn_) if (tp+fn_) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    ppv  = tp/(tp+fp) if (tp+fp) else 0
    f1   = f1_score(y, pred)
    print(f"AUC={auc:.4f}  F1={f1:.3f}  T={thr:.3f}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%")
    results[name] = {
        "auc": round(auc, 4),
        "f1": round(f1, 3),
        "threshold": round(thr, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
        "sens": round(sens*100, 1),
        "spec": round(spec*100, 1),
        "ppv": round(ppv*100, 1),
        "roc": roc_points(y, oof),
        "sweep": sweep_thresholds(y, oof),
        "features": [],  # DL models don't have simple feature importance
    }

# ── Merge into eval_results.json ──────────────────────────────────────────
with open("PenuX-AP-Severity/models/eval_results.json") as f:
    existing = json.load(f)

existing.update(results)

with open("PenuX-AP-Severity/models/eval_results.json", "w") as f:
    json.dump(existing, f, indent=2)

print("\nDone — eval_results.json updated.")
