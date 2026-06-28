"""ML model registry and training utilities."""
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from penux_ap.config import RANDOM_SEED
from penux_ap.preprocessing import ColumnTransformer

log = logging.getLogger(__name__)


def get_model_registry() -> dict:
    """Return available model estimators. Optional deps skipped gracefully."""
    registry = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_SEED
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_SEED
        ),
    }

    try:
        from xgboost import XGBClassifier
        registry["xgboost"] = XGBClassifier(
            n_estimators=200, scale_pos_weight=1, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        log.info("xgboost not installed; skipping.")

    try:
        from lightgbm import LGBMClassifier
        registry["lightgbm"] = LGBMClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, verbose=-1
        )
    except ImportError:
        log.info("lightgbm not installed; skipping.")

    return registry


def build_pipeline(model_name: str, preprocessor: ColumnTransformer) -> Pipeline:
    """Wrap a model in a sklearn Pipeline with the given preprocessor."""
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(registry)}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", registry[model_name])])


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Fit a named model pipeline on training data."""
    pipe = build_pipeline(model_name, preprocessor)
    log.info("Training '%s'...", model_name)
    pipe.fit(X_train, y_train)
    log.info("Training complete for '%s'.", model_name)
    return pipe


def predict_proba_safe(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities, falling back to decision_function if needed."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Sigmoid normalisation
        return 1.0 / (1.0 + np.exp(-scores))
    raise RuntimeError(f"Model {type(model)} has neither predict_proba nor decision_function.")
