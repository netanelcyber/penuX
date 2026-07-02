"""Recompute F2 (recall-weighted F-beta, beta=2) optimal thresholds for the
strongest models. F2 weights recall 4x more than precision — appropriate for
SAP triage, where missing a severe case (FN) is clinically worse than a false
alarm (FP). Uses the same 5-fold stratified CV / OOF-prediction methodology
as every other script in this repo, but selects the threshold that maximizes
fbeta_score(beta=2) instead of F1, via exhaustive search over all unique
predicted probabilities (not just the 8-point coarse grid in eval_results.json)."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, f1_score, confusion_matrix, roc_curve

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from interpret.glassbox import ExplainableBoostingClassifier

np.random.seed(42)
BETA = 2

df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(int)
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)

def best_fbeta_threshold(y_true, y_prob, beta=BETA):
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0) for t in ths]
    idx = int(np.argmax(scores))
    return float(ths[idx]), scores[idx]

def cross_val_proba(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in skf.split(X, y):
        model = model_fn()
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof

configs = {
    "Logistic Regression": lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3,
                                          eval_metric='auc', random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
                                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
    "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                                            random_seed=42, verbose=False),
    "NGBoost": lambda: NGBClassifier(Dist=Bernoulli, n_estimators=200, learning_rate=0.03,
                                      minibatch_frac=0.8, col_sample=0.8, random_state=42, verbose=False),
    "EBM": lambda: ExplainableBoostingClassifier(feature_names=FEATURE_NAMES, interactions=10,
                                                  learning_rate=0.02, max_bins=256, random_state=42, n_jobs=-1),
}

results = []
for name, fn in configs.items():
    oof = cross_val_proba(fn, X, y)
    auc = roc_auc_score(y, oof)
    thr, f2 = best_fbeta_threshold(y, oof, beta=BETA)
    pred = (oof >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
    sens = tp/(tp+fn_) if (tp+fn_) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    ppv  = tp/(tp+fp) if (tp+fp) else 0
    f1_at_f2_thr = f1_score(y, pred)
    print(f"{name:20s}  AUC={auc:.4f}  F2={f2:.4f}  T={thr:.3f}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  PPV={ppv*100:.1f}%  F1@thisT={f1_at_f2_thr:.3f}  TP={tp} FP={fp} FN={fn_} TN={tn}")
    results.append({
        "model": name, "auc": round(auc,4), "f2": round(f2,4), "threshold": round(thr,3),
        "sens": round(sens*100,1), "spec": round(spec*100,1), "ppv": round(ppv*100,1),
        "f1_at_this_threshold": round(f1_at_f2_thr,3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
    })

results.sort(key=lambda r: -r["f2"])
print("\n=== Ranked by F2 (recall-weighted, beta=2) ===")
for r in results:
    print(f"{r['f2']:.4f}  {r['model']}  (T={r['threshold']}, Sens={r['sens']}%, FN={r['fn']})")

with open("PenuX-AP-Severity/models/f2_optimal_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to PenuX-AP-Severity/models/f2_optimal_results.json")
