"""Attempt to beat CatBoost (current best, AUC=0.882/F1=0.926) with two approaches:
  1. A stacking ensemble (meta-learner over the 5 strongest base models)
  2. A hyperparameter-tuned CatBoost (randomized search over the same CV protocol)
Both use the identical 5-fold stratified CV / threshold-sweep / ROC-extraction
methodology as every other model in eval_results.json, so results are directly
comparable — no shortcuts, no leakage (base-model OOF predictions for the
stacking ensemble are generated via a proper nested/out-of-fold scheme)."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(int)
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)
N_FEAT = X.shape[1]

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def summarize(name, y, oof):
    auc = roc_auc_score(y, oof)
    thr = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
    sens = tp/(tp+fn_) if (tp+fn_) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    ppv  = tp/(tp+fp) if (tp+fp) else 0
    f1   = f1_score(y, pred)
    print(f"{name}: AUC={auc:.4f}  F1={f1:.3f}  T={thr:.3f}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%")
    return {
        "auc": round(auc, 4), "f1": round(f1, 3), "threshold": round(thr, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
        "sens": round(sens*100, 1), "spec": round(spec*100, 1), "ppv": round(ppv*100, 1),
        "roc": roc_points(y, oof), "sweep": sweep_thresholds(y, oof),
    }

# ── 1. Stacking ensemble ────────────────────────────────────────────────────
# Base learners: the 5 strongest models so far (RF, GB, XGBoost, LightGBM, CatBoost).
# Proper nested CV: for each outer fold, base-model OOF predictions on the outer
# training set are generated via an INNER 5-fold CV (no leakage), then a
# Logistic Regression meta-learner is trained on those OOF predictions and
# applied to the outer validation fold's base-model predictions (trained on
# the full outer-training set).
def make_base_models():
    return {
        "rf":  RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42),
        "gb":  GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
        "xgb": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3,
                                  eval_metric='auc', random_state=42, n_jobs=-1),
        "lgb": lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                   min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
        "cat": CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                                   random_seed=42, verbose=False),
    }

print("\n=== Stacking Ensemble (RF+GB+XGBoost+LightGBM+CatBoost -> LogisticRegression) ===")
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_oof = np.zeros(len(y), dtype=np.float32)

for tr_idx, va_idx in outer.split(X, y):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Inner CV to build leakage-free meta-features for training the meta-learner.
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    meta_train = np.zeros((len(ytr), 5))
    base_names = ["rf", "gb", "xgb", "lgb", "cat"]

    for inner_tr, inner_va in inner.split(Xtr, ytr):
        models = make_base_models()
        for j, name in enumerate(base_names):
            m = models[name]
            m.fit(Xtr[inner_tr], ytr[inner_tr])
            meta_train[inner_va, j] = m.predict_proba(Xtr[inner_va])[:, 1]

    # Refit base models on the full outer-training fold, predict outer-validation fold.
    final_models = make_base_models()
    meta_val = np.zeros((len(yva), 5))
    for j, name in enumerate(base_names):
        m = final_models[name]
        m.fit(Xtr, ytr)
        meta_val[:, j] = m.predict_proba(Xva)[:, 1]

    meta_learner = LogisticRegression(max_iter=1000)
    meta_learner.fit(meta_train, ytr)
    stack_oof[va_idx] = meta_learner.predict_proba(meta_val)[:, 1]

stack_result = summarize("Stacking Ensemble", y, stack_oof)
stack_result["features"] = []  # meta-learner has no direct per-lab-value importance

# ── 2. Hyperparameter-tuned CatBoost ────────────────────────────────────────
print("\n=== Tuned CatBoost (RandomizedSearchCV, inner 3-fold) ===")
param_dist = {
    "iterations": [200, 300, 500, 800],
    "depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "bagging_temperature": [0, 0.5, 1, 2],
}

tuned_oof = np.zeros(len(y), dtype=np.float32)
best_params_per_fold = []

for tr_idx, va_idx in outer.split(X, y):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    base = CatBoostClassifier(random_seed=42, verbose=False)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=15, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1),
        random_state=42, n_jobs=-1,
    )
    search.fit(Xtr, ytr)
    best_params_per_fold.append(search.best_params_)
    tuned_oof[va_idx] = search.best_estimator_.predict_proba(Xva)[:, 1]

tuned_result = summarize("Tuned CatBoost", y, tuned_oof)
print("Best params per fold:", best_params_per_fold)

# Refit on full data for feature importances + a representative hyperparameter set.
full_search = RandomizedSearchCV(
    CatBoostClassifier(random_seed=42, verbose=False), param_dist, n_iter=15,
    scoring="roc_auc", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1),
    random_state=42, n_jobs=-1,
)
full_search.fit(X, y)
imp = np.asarray(full_search.best_estimator_.feature_importances_)
order = np.argsort(imp)[::-1][:5]
total = float(imp.sum()) or 1.0
tuned_result["features"] = [[FEATURE_NAMES[i], round(float(imp[i]) / total, 4)] for i in order]
tuned_result["best_params"] = full_search.best_params_

# ── Merge into eval_results.json (atomic write) ────────────────────────────
with open("PenuX-AP-Severity/models/eval_results.json") as f:
    existing = json.load(f)

existing["Stacking Ensemble"] = stack_result
existing["Tuned CatBoost"] = tuned_result

tmp_path = "PenuX-AP-Severity/models/eval_results.json.tmp"
with open(tmp_path, "w") as f:
    json.dump(existing, f, indent=2)
import os
os.replace(tmp_path, "PenuX-AP-Severity/models/eval_results.json")

print("\nDone — eval_results.json updated (Stacking Ensemble, Tuned CatBoost added).")
print(f"\nCatBoost baseline: AUC=0.882 F1=0.926")
print(f"Stacking Ensemble: AUC={stack_result['auc']} F1={stack_result['f1']}")
print(f"Tuned CatBoost:    AUC={tuned_result['auc']} F1={tuned_result['f1']}")
