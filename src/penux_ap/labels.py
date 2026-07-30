"""Target column detection and label binarization."""
import logging
import warnings

import pandas as pd

from penux_ap.config import LIKELY_TARGET_COLUMNS

log = logging.getLogger(__name__)

_POSITIVE_LABELS = {"1", "yes", "true", "sap", "severe", "1.0"}
_NEGATIVE_LABELS = {"0", "no", "false", "non-sap", "non_sap", "nonsap", "non-severe", "non_severe", "0.0"}


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the most likely target column in a DataFrame."""
    cols = list(df.columns)
    for candidate in LIKELY_TARGET_COLUMNS:
        if candidate in cols:
            return candidate
    lower_map = {c.lower(): c for c in cols}
    for candidate in LIKELY_TARGET_COLUMNS:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for candidate in LIKELY_TARGET_COLUMNS:
        for col in cols:
            if candidate in col.lower():
                return col
    raise ValueError(
        f"Cannot detect target column. Columns: {cols}. "
        "Use --target-column to specify."
    )


def binarize_target(series: pd.Series) -> pd.Series:
    """Convert common label formats to 0/1 integers.

    Handles: 0/1, yes/no, true/false, SAP/non-SAP, severe/non-severe.
    """
    unique = series.dropna().unique()
    n_unique = len(unique)

    if n_unique > 2:
        warnings.warn(
            f"Target has {n_unique} unique values: {unique[:10]}. "
            "Binarization may not be meaningful. Proceeding with numeric cast.",
            UserWarning,
            stacklevel=2,
        )

    # Already numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())
        if vals <= {0, 1, 0.0, 1.0}:
            return series.astype(float).astype("Int64")
        warnings.warn(f"Numeric target has values outside {{0,1}}: {vals}", UserWarning, stacklevel=2)
        return series

    str_series = series.astype(str).str.strip().str.lower()
    result = pd.Series(index=series.index, dtype="Int64")
    for idx, val in str_series.items():
        if val in _POSITIVE_LABELS:
            result[idx] = 1
        elif val in _NEGATIVE_LABELS:
            result[idx] = 0
        else:
            result[idx] = pd.NA
    n_na = result.isna().sum()
    if n_na > 0:
        log.warning("Binarization produced %d NA values from unrecognized labels.", n_na)
    return result


def describe_target(series: pd.Series) -> dict:
    """Return a summary of the target distribution."""
    counts = series.value_counts(dropna=False).to_dict()
    n_total = len(series)
    n_missing = int(series.isna().sum())
    return {
        "n_total": n_total,
        "n_missing": n_missing,
        "n_classes": int(series.nunique(dropna=True)),
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": float((series == 1).mean()) if pd.api.types.is_numeric_dtype(series) else None,
    }
