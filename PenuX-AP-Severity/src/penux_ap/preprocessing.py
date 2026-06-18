"""Preprocessing pipeline: imputation, encoding, scaling, splitting."""
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from penux_ap.config import RANDOM_SEED, TEST_SIZE

log = logging.getLogger(__name__)


def infer_feature_types(df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    """Split columns into numeric and categorical, excluding the target."""
    feature_cols = [c for c in df.columns if c != target_column]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    log.info("Numeric features: %d, Categorical features: %d", len(numeric), len(categorical))
    return {"numeric": numeric, "categorical": categorical}


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that must be fit only on training data."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(
        "Train: %d rows (positives: %d), Test: %d rows (positives: %d)",
        len(X_train), int(y_train.sum()), len(X_test), int(y_test.sum()),
    )
    return X_train, X_test, y_train, y_test
