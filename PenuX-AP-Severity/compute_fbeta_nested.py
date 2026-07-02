"""Optimal F-beta selection WITHOUT overfitting the threshold choice.

Two safeguards against overfitting, both applied together:

1. NESTED CV for threshold selection. Every earlier script in this repo
   picked the beta-optimal threshold using the exact same out-of-fold (OOF)
   predictions it then reported the score on — mildly optimistic, since the
   threshold was tuned to maximize performance on the data it's evaluated on.
   Here, for each outer fold: an INNER 5-fold CV on the outer-training data
   selects the threshold (using only inner-fold-held-out predictions), then
   that threshold is applied to a model trained on the FULL outer-training
   fold and scored on the outer-validation fold, which never influenced the
   threshold choice.

2. REPEATED CV for stability. The whole nested procedure is repeated across
   5 different random seeds for the outer/inner splits. If the selected
   threshold and resulting F-beta score are stable across repeats, the
   "optimal" operating point is real signal. If they swing wildly, that's
   direct evidence the earlier single-split threshold selection was fitting
   noise in a 722-patient dataset.

Only beta=2 is evaluated (the highest beta that didn't produce degenerate
"predict everyone positive" results in the earlier sweep), on the four
models that won at some beta in that sweep: Gradient Boosting, CatBoost,
Random Forest, LightGBM.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve

import lightgbm as lgb
from catboost import CatBoostClassifier

np.random.seed(42)
BETA = 2
N_REPEATS = 5
SEEDS = [42, 1, 7, 123, 2024]

df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(int)

configs = {
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
                                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
    "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                                            random_seed=42, verbose=False),
}

def best_fbeta_threshold(y_true, y_prob, beta):
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0) for t in ths]
    idx = int(np.argmax(scores))
    return float(ths[idx])

def nested_fbeta_once(model_fn, X, y, beta, outer_seed, inner_seed):
    """One repeat of the full nested procedure. Returns per-outer-fold thresholds
    and the pooled true-held-out F-beta score."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    pooled_true = []
    pooled_pred = []
    fold_thresholds = []

    for tr_idx, va_idx in outer.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Inner CV: generate leakage-free OOF predictions on the outer-training
        # fold only, to select the threshold — outer-validation labels/predictions
        # never touch this step.
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_seed)
        inner_oof = np.zeros(len(ytr), dtype=np.float32)
        for itr, iva in inner.split(Xtr, ytr):
            m = model_fn()
            m.fit(Xtr[itr], ytr[itr])
            inner_oof[iva] = m.predict_proba(Xtr[iva])[:, 1]
        thr = best_fbeta_threshold(ytr, inner_oof, beta)
        fold_thresholds.append(thr)

        # Refit on the FULL outer-training fold, apply the inner-selected
        # threshold to the untouched outer-validation fold.
        final_model = model_fn()
        final_model.fit(Xtr, ytr)
        va_prob = final_model.predict_proba(Xva)[:, 1]
        va_pred = (va_prob >= thr).astype(int)

        pooled_true.extend(yva.tolist())
        pooled_pred.extend(va_pred.tolist())

    pooled_true = np.array(pooled_true)
    pooled_pred = np.array(pooled_pred)
    score = fbeta_score(pooled_true, pooled_pred, beta=beta, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(pooled_true, pooled_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    return score, fold_thresholds, sens, spec, int(fn)

results = {}
for name, fn in configs.items():
    print(f"\n=== {name} (nested, beta={BETA}, {N_REPEATS} repeats) ===")
    repeat_scores = []
    repeat_thresholds = []
    for rep, seed in enumerate(SEEDS):
        score, fold_thrs, sens, spec, fn_count = nested_fbeta_once(fn, X, y, BETA, outer_seed=seed, inner_seed=seed + 1000)
        mean_thr = float(np.mean(fold_thrs))
        repeat_scores.append(score)
        repeat_thresholds.append(mean_thr)
        print(f"  repeat {rep+1} (seed={seed}): F{BETA}={score:.4f}  mean_threshold={mean_thr:.3f}  "
              f"fold_thresholds={[round(t,3) for t in fold_thrs]}  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  FN={fn_count}")

    score_mean, score_std = np.mean(repeat_scores), np.std(repeat_scores)
    thr_mean, thr_std = np.mean(repeat_thresholds), np.std(repeat_thresholds)
    print(f"  --> F{BETA} = {score_mean:.4f} +/- {score_std:.4f}   threshold = {thr_mean:.3f} +/- {thr_std:.3f}")

    results[name] = {
        "beta": BETA,
        "fbeta_mean": round(float(score_mean), 4),
        "fbeta_std": round(float(score_std), 4),
        "threshold_mean": round(float(thr_mean), 3),
        "threshold_std": round(float(thr_std), 3),
        "repeat_scores": [round(float(s), 4) for s in repeat_scores],
        "repeat_thresholds": [round(float(t), 3) for t in repeat_thresholds],
    }

print("\n" + "="*80)
print(f"RANKED by mean nested F{BETA} (honest, non-overfit estimate):")
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["fbeta_mean"]):
    stability = "STABLE" if r["threshold_std"] < 0.05 else "UNSTABLE (threshold varies a lot across repeats)"
    print(f"  {name:20s}  F{BETA}={r['fbeta_mean']:.4f} (+/-{r['fbeta_std']:.4f})  "
          f"T={r['threshold_mean']:.3f} (+/-{r['threshold_std']:.3f})  [{stability}]")

with open("PenuX-AP-Severity/models/fbeta_nested_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to PenuX-AP-Severity/models/fbeta_nested_results.json")
