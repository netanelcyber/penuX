"""Builds new combination/ensemble models out of the top performers from the
~12,000-model zoo, on the SAME clearly-labeled SIMULATED dataset used
throughout this project (see docs/dataset_landscape.md -- no real data).

Strategies tested:
  1. Simple averaging of the top-K models' out-of-fold probabilities
  2. AUROC-weighted averaging
  3. Stacking: a logistic-regression meta-learner trained on the base
     models' out-of-fold probabilities, itself evaluated via an outer
     5-fold CV to avoid the meta-learner "seeing" its own training rows.

Selected base models are diverse by family (not just the global top-15,
which would otherwise be dominated by near-duplicate hyperparameter
neighbors of the single best configuration).
"""
import sys
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_zoo_psychosis import build_model_zoo
from penux_psychosis.simulate_data import simulate_dataset

N_SPLITS = 5
RANDOM_SEED = 42

# Diverse top performers selected from outputs/psychosis_model_zoo_ranked.csv,
# one or two per major algorithm family (not just the global top-N, which
# would be dominated by near-duplicate hyperparameter neighbors).
SELECTED_MODELS = [
    "gbdt_sklearn_n90_lr0.05_sub0.7",
    "extra_trees_huge_n100_leafnodes50",
    "xgboost_dart_n170_d2_lr0.1_drop0.05",
    "knn_k25_distance",
    "catboost_n30_d4_lr0.3",
    "extra_trees_n130_d8",
    "xgboost_n550_d2_lr0.01",
    "bagging_n800_ms0.4",
    "rf_n180_d5_None",
    "gaussian_process_matern",
    "svc_rbf_C0.001",
    "gaussian_nb",
    "lightgbm_goss_n170_leaves7_lr0.01",
    "torch_dnn_(32, 16, 8)_drop0.1_lr0.0003_wd0.001",
    "lightgbm_n100_leaves3_lr0.1",
]


def get_oof_probability(estimator, X, y, seed):
    """Return out-of-fold predicted probabilities for one estimator."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_id = np.zeros(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            proba = pipe.decision_function(X[test_idx])
        oof[test_idx] = proba
        fold_id[test_idx] = fold
    return oof, fold_id


def main():
    df = simulate_dataset()
    assert df["is_simulated"].all(), "Refusing to run on non-simulated data"
    feature_cols = ["Arg", "TP", "ALP", "HDL", "UA", "LDL"]
    X = df[feature_cols].values
    y = df["label"].values

    zoo = dict(build_model_zoo())
    missing = [n for n in SELECTED_MODELS if n not in zoo]
    if missing:
        raise SystemExit(f"Selected models not found in zoo: {missing}")

    print(f"Computing out-of-fold predictions for {len(SELECTED_MODELS)} base models...")
    base_oof = {}
    fold_ids = None
    for name in SELECTED_MODELS:
        oof, fids = get_oof_probability(zoo[name], X, y, RANDOM_SEED)
        base_oof[name] = oof
        fold_ids = fids
        auroc = roc_auc_score(y, oof)
        print(f"  {name:55s} AUROC={auroc:.4f}")

    P = np.column_stack([base_oof[n] for n in SELECTED_MODELS])  # (n_samples, n_models)
    aurocs = np.array([roc_auc_score(y, P[:, i]) for i in range(P.shape[1])])
    order = np.argsort(-aurocs)

    print("\n=== Combination strategies ===")
    results = {}

    for k in [3, 5, 10, 15]:
        top_k_idx = order[:k]
        avg = P[:, top_k_idx].mean(axis=1)
        auroc = roc_auc_score(y, avg)
        results[f"simple_average_top{k}"] = auroc
        print(f"Simple average of top {k:2d} models: AUROC={auroc:.4f}")

    for k in [5, 10, 15]:
        top_k_idx = order[:k]
        w = aurocs[top_k_idx]
        w = (w - w.min() + 1e-6)  # keep weights positive
        w = w / w.sum()
        weighted = (P[:, top_k_idx] * w).sum(axis=1)
        auroc = roc_auc_score(y, weighted)
        results[f"auroc_weighted_top{k}"] = auroc
        print(f"AUROC-weighted average of top {k:2d} models: AUROC={auroc:.4f}")

    # Stacking: meta-logistic-regression on the base OOF predictions,
    # evaluated via an outer 5-fold CV using the SAME fold assignment as the
    # base models (so the meta-learner never trains on a row whose own base
    # predictions it is being scored on).
    outer_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    stack_oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in outer_skf.split(P, y):
        meta = LogisticRegression(max_iter=2000)
        meta.fit(P[train_idx], y[train_idx])
        stack_oof[test_idx] = meta.predict_proba(P[test_idx])[:, 1]
    stack_auroc = roc_auc_score(y, stack_oof)
    results["stacking_logreg_all15"] = stack_auroc
    print(f"Stacking (logistic-regression meta-learner, all 15 base models): AUROC={stack_auroc:.4f}")

    best_single = aurocs.max()
    best_single_name = SELECTED_MODELS[order[0]]
    print(f"\nBest single base model: {best_single_name} AUROC={best_single:.4f}")
    best_combo = max(results, key=results.get)
    print(f"Best combination strategy: {best_combo} AUROC={results[best_combo]:.4f}")
    print(f"Improvement over best single model: {results[best_combo] - best_single:+.4f}")

    import csv
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ensemble_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "auroc"])
        for name in SELECTED_MODELS:
            writer.writerow([f"single:{name}", roc_auc_score(y, base_oof[name])])
        for strategy, auroc in results.items():
            writer.writerow([strategy, auroc])
    print("\nSaved outputs/ensemble_results.csv")


if __name__ == "__main__":
    main()
