"""Research-grade XGBoost utilities for early SAP prediction.

The utilities in this module are deliberately development-set centric:
hyperparameters are tuned inside cross-validation and the final held-out test
set is never used for model or threshold selection.

RESEARCH USE ONLY. Atlanta-defined SAP remains the reference outcome; this
module predicts that outcome and does not replace the Revised Atlanta
Classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from penux_ap.config import RANDOM_SEED


@dataclass
class NestedXGBResult:
    """Outputs from leakage-resistant nested XGBoost development validation."""

    oof_probabilities: np.ndarray
    fold_summaries: list[dict[str, Any]]
    outer_folds: int
    inner_folds_requested: int
    calibration: str | None


def class_imbalance_ratio(y: pd.Series | np.ndarray) -> float:
    """Return negatives / positives for XGBoost ``scale_pos_weight``."""
    arr = np.asarray(y, dtype=int).reshape(-1)
    positives = int(np.sum(arr == 1))
    negatives = int(np.sum(arr == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Both outcome classes are required")
    return float(negatives / positives)


def xgboost_parameter_space(scale_pos_weight: float) -> dict[str, list[Any]]:
    """Conservative search space for medium-sized tabular AP cohorts.

    The space emphasizes regularization and shallow trees to reduce overfitting
    on cohorts with hundreds to low-thousands of observations.
    """
    if scale_pos_weight <= 0:
        raise ValueError("scale_pos_weight must be > 0")
    weight_candidates = sorted(
        {
            round(max(0.25, scale_pos_weight * multiplier), 6)
            for multiplier in (0.75, 1.0, 1.25, 1.5)
        }
    )
    return {
        "classifier__n_estimators": [200, 350, 500, 700],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.015, 0.03, 0.05, 0.08, 0.12],
        "classifier__min_child_weight": [1, 3, 5, 8],
        "classifier__subsample": [0.65, 0.8, 0.95, 1.0],
        "classifier__colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "classifier__gamma": [0.0, 0.1, 0.3, 0.7],
        "classifier__reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__max_delta_step": [0, 1, 3],
        "classifier__scale_pos_weight": weight_candidates,
    }


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for this workflow. Install with: "
            "pip install -e '.[xgboost]'"
        ) from exc
    return XGBClassifier


def build_research_xgboost_pipeline(preprocessor, scale_pos_weight: float) -> Pipeline:
    """Build a regularized XGBoost pipeline with safe research defaults."""
    XGBClassifier = _require_xgboost()
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
    )
    return Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]
    )


def _valid_folds(y: pd.Series | np.ndarray, requested: int) -> int:
    if requested < 2:
        raise ValueError("Cross-validation requires at least two folds")
    arr = np.asarray(y, dtype=int).reshape(-1)
    counts = np.bincount(arr, minlength=2)
    minority = int(counts.min())
    if minority < 2:
        raise ValueError("At least two examples are required in each class")
    return min(int(requested), minority)


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    cv_folds: int = 4,
    n_iter: int = 30,
    scoring: str = "average_precision",
    random_state: int = RANDOM_SEED,
    search_jobs: int = 1,
) -> RandomizedSearchCV:
    """Tune XGBoost inside a development partition only."""
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    folds = _valid_folds(y, cv_folds)
    ratio = class_imbalance_ratio(y)
    pipe = build_research_xgboost_pipeline(preprocessor, ratio)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=xgboost_parameter_space(ratio),
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=search_jobs,
        cv=cv,
        refit=True,
        random_state=random_state,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def nested_oof_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    outer_folds: int = 5,
    inner_folds: int = 4,
    n_iter: int = 25,
    scoring: str = "average_precision",
    calibration: str | None = "sigmoid",
    calibration_folds: int = 3,
    random_state: int = RANDOM_SEED,
    search_jobs: int = 1,
) -> NestedXGBResult:
    """Generate nested-CV out-of-fold XGBoost probabilities.

    Each outer validation fold is predicted by a model whose hyperparameters
    were selected only inside that fold's outer-training partition. Optional
    probability calibration is also fit only on the outer-training partition.
    These OOF probabilities are therefore suitable for development-stage
    threshold locking before a final untouched hold-out test evaluation.
    """
    if calibration not in {None, "sigmoid", "isotonic"}:
        raise ValueError("calibration must be None, 'sigmoid', or 'isotonic'")

    outer_n = _valid_folds(y, outer_folds)
    outer_cv = StratifiedKFold(
        n_splits=outer_n,
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(y), np.nan, dtype=float)
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        inner_n = _valid_folds(y_train, inner_folds)
        search = tune_xgboost(
            X_train,
            y_train,
            preprocessor,
            cv_folds=inner_n,
            n_iter=n_iter,
            scoring=scoring,
            random_state=random_state + fold_index,
            search_jobs=search_jobs,
        )

        estimator = search.best_estimator_
        calibration_used = calibration
        if calibration is not None:
            cal_n = _valid_folds(y_train, calibration_folds)
            cal_cv = StratifiedKFold(
                n_splits=cal_n,
                shuffle=True,
                random_state=random_state + 1000 + fold_index,
            )
            calibrated = CalibratedClassifierCV(
                estimator=clone(estimator),
                method=calibration,
                cv=cal_cv,
            )
            calibrated.fit(X_train, y_train)
            probabilities = calibrated.predict_proba(X_valid)[:, 1]
        else:
            probabilities = estimator.predict_proba(X_valid)[:, 1]

        oof[valid_idx] = probabilities
        fold_summaries.append(
            {
                "outer_fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "train_prevalence": float(y_train.mean()),
                "inner_folds": int(inner_n),
                "inner_best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "calibration": calibration_used,
            }
        )

    if np.isnan(oof).any():
        raise RuntimeError("Nested OOF generation left unpredicted rows")

    return NestedXGBResult(
        oof_probabilities=oof,
        fold_summaries=fold_summaries,
        outer_folds=outer_n,
        inner_folds_requested=inner_folds,
        calibration=calibration,
    )


def fit_final_xgboost(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    preprocessor,
    inner_folds: int = 5,
    n_iter: int = 40,
    scoring: str = "average_precision",
    calibration: str | None = "sigmoid",
    calibration_folds: int = 5,
    random_state: int = RANDOM_SEED,
    search_jobs: int = 1,
):
    """Tune on all development data and fit the model used on final hold-out test.

    Returns ``(prediction_model, fitted_uncalibrated_model, search_summary)``.
    The uncalibrated fitted model is returned separately for native XGBoost
    feature-importance extraction.
    """
    search = tune_xgboost(
        X_dev,
        y_dev,
        preprocessor,
        cv_folds=inner_folds,
        n_iter=n_iter,
        scoring=scoring,
        random_state=random_state,
        search_jobs=search_jobs,
    )
    uncalibrated = search.best_estimator_

    if calibration is None:
        prediction_model = uncalibrated
    else:
        if calibration not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration must be None, 'sigmoid', or 'isotonic'")
        cal_n = _valid_folds(y_dev, calibration_folds)
        cal_cv = StratifiedKFold(
            n_splits=cal_n,
            shuffle=True,
            random_state=random_state + 2000,
        )
        prediction_model = CalibratedClassifierCV(
            estimator=clone(uncalibrated),
            method=calibration,
            cv=cal_cv,
        )
        prediction_model.fit(X_dev, y_dev)

    summary = {
        "best_cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "scoring": scoring,
        "n_iter": int(n_iter),
        "inner_folds": int(search.cv.n_splits),
        "calibration": calibration,
        "development_scale_pos_weight": class_imbalance_ratio(y_dev),
    }
    return prediction_model, uncalibrated, summary
