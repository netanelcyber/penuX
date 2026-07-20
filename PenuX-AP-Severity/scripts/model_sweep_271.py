"""Large model sweep: trains and evaluates ~270+ real model configurations
on the sanitized AP severity dataset (data/public_sanitized/ap_multiml_sanitized.csv),
via 5-fold stratified cross-validation. Every result is a genuinely trained
and evaluated model — no fabricated numbers.

Output: PenuX-AP-Severity/models/model_sweep_271_results.json
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "public_sanitized" / "ap_multiml_sanitized.csv"
OUT_PATH = ROOT / "models" / "model_sweep_271_results.json"
TARGET_COL = "Diagnostic Result"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET_COL])
y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print(f"Loaded {len(X)} rows, {X.shape[1]} features, positive rate {y.mean():.3f}")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"auroc": "roc_auc", "auprc": "average_precision", "f1": "f1", "accuracy": "accuracy"}


def build_configs():
    configs = []

    for C in [0.01, 0.1, 1, 10, 100]:
        for solver in ["liblinear", "lbfgs"]:
            configs.append((f"LogisticRegression(C={C},solver={solver})",
                             LogisticRegression(C=C, solver=solver, max_iter=2000)))

    for alpha in [0.01, 0.1, 1, 10, 100]:
        configs.append((f"RidgeClassifier(alpha={alpha})", RidgeClassifier(alpha=alpha)))

    for loss in ["hinge", "log_loss", "modified_huber"]:
        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            configs.append((f"SGDClassifier(loss={loss},alpha={alpha})",
                             SGDClassifier(loss=loss, alpha=alpha, max_iter=2000, random_state=42)))

    for k in [3, 5, 7, 9, 11, 15, 21]:
        for weights in ["uniform", "distance"]:
            configs.append((f"KNN(k={k},weights={weights})",
                             KNeighborsClassifier(n_neighbors=k, weights=weights)))

    for depth in [2, 3, 4, 5, 6, 8, 10, None]:
        for criterion in ["gini", "entropy"]:
            configs.append((f"DecisionTree(depth={depth},criterion={criterion})",
                             DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=42)))

    for n_est in [50, 100, 200, 300]:
        for depth in [None, 5, 10, 15]:
            for criterion in ["gini", "entropy"]:
                configs.append((f"RandomForest(n={n_est},depth={depth},criterion={criterion})",
                                 RandomForestClassifier(n_estimators=n_est, max_depth=depth, criterion=criterion,
                                                         random_state=42, n_jobs=-1)))

    for n_est in [50, 100, 200, 300]:
        for depth in [None, 5, 10, 15]:
            for criterion in ["gini", "entropy"]:
                configs.append((f"ExtraTrees(n={n_est},depth={depth},criterion={criterion})",
                                 ExtraTreesClassifier(n_estimators=n_est, max_depth=depth, criterion=criterion,
                                                       random_state=42, n_jobs=-1)))

    for n_est in [50, 100, 200]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for depth in [2, 3, 4]:
                configs.append((f"GradBoost(n={n_est},lr={lr},depth={depth})",
                                 GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth,
                                                             random_state=42)))

    for n_est in [50, 100, 200]:
        for lr in [0.1, 0.5, 1.0]:
            configs.append((f"AdaBoost(n={n_est},lr={lr})",
                             AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)))

    configs.append(("GaussianNB", GaussianNB()))
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        configs.append((f"BernoulliNB(alpha={alpha})", BernoulliNB(alpha=alpha)))

    for hidden in [(50,), (100,), (50, 50), (100, 50)]:
        for alpha in [0.0001, 0.001, 0.01]:
            configs.append((f"MLP(hidden={hidden},alpha={alpha})",
                             MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=1000, random_state=42)))

    for kernel in ["linear", "rbf", "poly"]:
        for C in [0.1, 1, 10, 100]:
            configs.append((f"SVC(kernel={kernel},C={C})",
                             SVC(kernel=kernel, C=C, probability=True, random_state=42)))

    for n_est in [50, 100, 200]:
        for depth in [3, 4, 5, 6]:
            for lr in [0.01, 0.05, 0.1]:
                configs.append((f"XGBoost(n={n_est},depth={depth},lr={lr})",
                                 XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                                use_label_encoder=False, eval_metric="logloss",
                                                random_state=42, verbosity=0)))

    for n_est in [50, 100, 200]:
        for depth in [3, 5, 7, -1]:
            for lr in [0.01, 0.05, 0.1]:
                configs.append((f"LightGBM(n={n_est},depth={depth},lr={lr})",
                                 LGBMClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                                 random_state=42, verbosity=-1)))

    for iters in [100, 200, 300]:
        for depth in [4, 6, 8]:
            for lr in [0.01, 0.05, 0.1]:
                configs.append((f"CatBoost(iter={iters},depth={depth},lr={lr})",
                                 CatBoostClassifier(iterations=iters, depth=depth, learning_rate=lr,
                                                     random_state=42, verbose=False)))

    return configs


configs = build_configs()
print(f"Built {len(configs)} model configurations")

results = []
t_start = time.time()
for i, (name, model) in enumerate(configs, 1):
    needs_scaling = isinstance(model, (LogisticRegression, RidgeClassifier, SGDClassifier,
                                        KNeighborsClassifier, MLPClassifier, SVC))
    estimator = make_pipeline(StandardScaler(), model) if needs_scaling else model
    try:
        t0 = time.time()
        scores = cross_validate(estimator, X, y, cv=CV, scoring=SCORING, n_jobs=1, error_score="raise")
        elapsed = time.time() - t0
        result = {
            "name": name,
            "auroc_mean": round(float(np.mean(scores["test_auroc"])), 4),
            "auroc_std": round(float(np.std(scores["test_auroc"])), 4),
            "auprc_mean": round(float(np.mean(scores["test_auprc"])), 4),
            "f1_mean": round(float(np.mean(scores["test_f1"])), 4),
            "accuracy_mean": round(float(np.mean(scores["test_accuracy"])), 4),
            "fit_seconds": round(elapsed, 2),
            "status": "ok",
        }
    except Exception as e:
        result = {"name": name, "status": "failed", "error": str(e)[:200]}
    results.append(result)
    if i % 20 == 0 or i == len(configs):
        print(f"[{i}/{len(configs)}] {name} -> "
              f"{result.get('auroc_mean', 'FAILED')} AUROC ({time.time()-t_start:.0f}s elapsed)")

ok_results = [r for r in results if r["status"] == "ok"]
ok_results.sort(key=lambda r: r["auroc_mean"], reverse=True)

output = {
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "n_samples": len(X),
    "n_features": X.shape[1],
    "positive_rate": round(float(y.mean()), 4),
    "cv_folds": 5,
    "n_configs_attempted": len(configs),
    "n_configs_succeeded": len(ok_results),
    "n_configs_failed": len(configs) - len(ok_results),
    "total_runtime_seconds": round(time.time() - t_start, 1),
    "results_ranked_by_auroc": ok_results,
    "failed": [r for r in results if r["status"] == "failed"],
}

OUT_PATH.parent.mkdir(exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone. {len(ok_results)}/{len(configs)} succeeded. Best: {ok_results[0]['name']} "
      f"(AUROC={ok_results[0]['auroc_mean']}). Saved to {OUT_PATH}")
