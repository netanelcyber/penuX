"""Sweep F-beta across several beta values, using the SAME beta consistently
across all models within each sweep point (never a different beta per model —
that would make cross-model comparison meaningless). For each beta, find each
model's beta-optimal threshold and report which model wins at that beta.

OOF predictions are computed once per model (5-fold stratified CV) and then
re-swept cheaply across all beta values — no need to retrain per beta."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix, roc_curve

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from interpret.glassbox import ExplainableBoostingClassifier

np.random.seed(42)
BETAS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(int)
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)

def cross_val_proba(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in skf.split(X, y):
        model = model_fn()
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof

def best_fbeta(y_true, y_prob, beta):
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0) for t in ths]
    idx = int(np.argmax(scores))
    return float(ths[idx]), scores[idx]

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

# ── Train once per model, cache OOF ────────────────────────────────────────
print("Computing OOF predictions (one CV run per model)...")
oof_cache = {}
auc_cache = {}
for name, fn in configs.items():
    oof = cross_val_proba(fn, X, y)
    oof_cache[name] = oof
    auc_cache[name] = roc_auc_score(y, oof)
    print(f"  {name}: AUC={auc_cache[name]:.4f}")

# ── Sweep beta, same beta applied to every model at each sweep point ───────
all_results = {}
print("\n" + "="*90)
for beta in BETAS:
    print(f"\n=== beta={beta} ===")
    rows = []
    for name in configs:
        oof = oof_cache[name]
        thr, score = best_fbeta(y, oof, beta)
        pred = (oof >= thr).astype(int)
        tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
        sens = tp/(tp+fn_) if (tp+fn_) else 0
        spec = tn/(tn+fp) if (tn+fp) else 0
        ppv  = tp/(tp+fp) if (tp+fp) else 0
        degenerate = spec < 0.01  # threshold collapsed to "predict everyone positive"
        rows.append({
            "model": name, "beta": beta, "fbeta": round(score,4), "threshold": round(thr,3),
            "sens": round(sens*100,1), "spec": round(spec*100,1), "ppv": round(ppv*100,1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn),
            "degenerate": degenerate,
        })
        flag = " [DEGENERATE: predicts everyone positive]" if degenerate else ""
        print(f"  {name:20s}  F{beta}={score:.4f}  T={thr:.3f}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  FN={fn_}{flag}")

    rows.sort(key=lambda r: -r["fbeta"])
    non_degenerate = [r for r in rows if not r["degenerate"]]
    winner = non_degenerate[0] if non_degenerate else rows[0]
    print(f"  --> Best non-degenerate: {winner['model']} (F{beta}={winner['fbeta']})")
    all_results[str(beta)] = rows

with open("PenuX-AP-Severity/models/fbeta_sweep_results.json", "w") as f:
    json.dump({"betas": BETAS, "results": all_results, "auc": {k: round(v,4) for k,v in auc_cache.items()}}, f, indent=2)

print("\nSaved to PenuX-AP-Severity/models/fbeta_sweep_results.json")
