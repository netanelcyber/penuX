"""Finer-grained version of compute_fbeta_optimal_beta.py focused on the
beta in [1.5, 2.0] range (step 0.1), using the same overfitting-resistant
nested + repeated CV procedure, for the 4 models that led at some threshold
in earlier sweeps. 5 repeats per (beta, model) since this covers fewer beta
points than the coarse 1.0-2.0 sweep."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve

import lightgbm as lgb
from catboost import CatBoostClassifier

np.random.seed(42)
BETAS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
SEEDS = [42, 1, 7, 123, 2024]  # 5 repeats — fewer beta points than the coarse sweep, can afford more repeats

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
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    pooled_true, pooled_pred, fold_thresholds = [], [], []

    for tr_idx, va_idx in outer.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_seed)
        inner_oof = np.zeros(len(ytr), dtype=np.float32)
        for itr, iva in inner.split(Xtr, ytr):
            m = model_fn()
            m.fit(Xtr[itr], ytr[itr])
            inner_oof[iva] = m.predict_proba(Xtr[iva])[:, 1]
        thr = best_fbeta_threshold(ytr, inner_oof, beta)
        fold_thresholds.append(thr)

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
    degenerate = bool(spec < 0.01)
    return score, fold_thresholds, sens, spec, int(fn), degenerate

all_results = {}
for beta in BETAS:
    print(f"\n{'='*80}\nbeta={beta}\n{'='*80}")
    beta_results = {}
    for name, fn in configs.items():
        repeat_scores, repeat_thresholds, degenerate_flags = [], [], []
        for seed in SEEDS:
            score, fold_thrs, sens, spec, fn_count, degenerate = nested_fbeta_once(
                fn, X, y, beta, outer_seed=seed, inner_seed=seed + 1000)
            repeat_scores.append(score)
            repeat_thresholds.append(float(np.mean(fold_thrs)))
            degenerate_flags.append(degenerate)

        score_mean, score_std = float(np.mean(repeat_scores)), float(np.std(repeat_scores))
        thr_mean, thr_std = float(np.mean(repeat_thresholds)), float(np.std(repeat_thresholds))
        any_degenerate = any(degenerate_flags)
        print(f"  {name:20s}  F{beta}={score_mean:.4f} (+/-{score_std:.4f})  T={thr_mean:.3f} (+/-{thr_std:.3f})"
              f"{'  [DEGENERATE in >=1 repeat]' if any_degenerate else ''}")

        beta_results[name] = {
            "fbeta_mean": round(score_mean, 4), "fbeta_std": round(score_std, 4),
            "threshold_mean": round(thr_mean, 3), "threshold_std": round(thr_std, 3),
            "any_degenerate": any_degenerate,
        }
    all_results[str(beta)] = beta_results

print(f"\n{'='*80}\nSUMMARY: best non-degenerate model at each beta\n{'='*80}")
for beta in BETAS:
    br = all_results[str(beta)]
    candidates = {k: v for k, v in br.items() if not v["any_degenerate"]}
    winner_name = max(candidates, key=lambda k: candidates[k]["fbeta_mean"])
    winner = candidates[winner_name]
    # Check how close the field is (max - min among non-degenerate)
    vals = [v["fbeta_mean"] for v in candidates.values()]
    spread = max(vals) - min(vals)
    print(f"beta={beta}: {winner_name} F{beta}={winner['fbeta_mean']:.4f} (+/-{winner['fbeta_std']:.4f})  "
          f"[field spread across models: {spread:.4f}]")

with open("PenuX-AP-Severity/models/fbeta_fine_1_5_to_2_results.json", "w") as f:
    json.dump({"betas": BETAS, "results": all_results}, f, indent=2)
print("\nSaved to PenuX-AP-Severity/models/fbeta_fine_1_5_to_2_results.json")
