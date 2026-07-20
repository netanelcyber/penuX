"""Train XGBoost, LightGBM, and CatBoost on the Chinese AP dataset — replacing the
LSTM/recurrent family, which is a poor methodological fit for tabular (non-sequential)
lab data. Appends results to eval_results.json using the same schema as
train_dl_models.py / export_rf_model.py so models.html can render them identically."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(np.float32)
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)
N_FEAT = X.shape[1]

# ── Helpers (identical to train_dl_models.py) ───────────────────────────────
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
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model = model_fn()
        model.fit(Xtr, ytr)
        oof[va] = model.predict_proba(Xva)[:, 1]
    return oof

def top_features(model, names, n=5):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:n]
    total = float(imp.sum()) or 1.0
    return [{"name": names[i], "importance": round(float(imp[i]) / total, 4)} for i in order]

# ── Model configs ────────────────────────────────────────────────────────────
def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_weight=3,
        eval_metric='auc', random_state=42, n_jobs=-1,
    )

def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1,
    )

def make_catboost():
    return CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42, verbose=False,
    )

configs = [
    ("XGBoost", make_xgb),
    ("LightGBM", make_lgbm),
    ("CatBoost", make_catboost),
]

results = {}
for name, fn in configs:
    print(f"\n=== {name} ===")
    oof = cross_val_proba(fn, X, y)
    auc = roc_auc_score(y, oof)
    thr = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
    sens = tp/(tp+fn_) if (tp+fn_) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    ppv  = tp/(tp+fp) if (tp+fp) else 0
    f1   = f1_score(y, pred)
    print(f"AUC={auc:.4f}  F1={f1:.3f}  T={thr:.3f}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%")

    # Refit once on the full dataset to extract feature importances for display.
    full_model = fn()
    full_model.fit(X, y)
    feats = top_features(full_model, FEATURE_NAMES)

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
        "features": feats,
    }

# ── Merge into eval_results.json ──────────────────────────────────────────
with open("PenuX-AP-Severity/models/eval_results.json") as f:
    existing = json.load(f)

# Remove LSTM-family entries being replaced.
for old_key in ["LSTM", "Stacked LSTM", "Bidirectional LSTM", "LSTM + Attention", "CNN-LSTM"]:
    existing.pop(old_key, None)

existing.update(results)

tmp_path = "PenuX-AP-Severity/models/eval_results.json.tmp"
with open(tmp_path, "w") as f:
    json.dump(existing, f, indent=2)
import os
os.replace(tmp_path, "PenuX-AP-Severity/models/eval_results.json")

print("\nDone — eval_results.json updated (LSTM family removed, GBDT family added).")
