"""Target column detection and label binarization."""
import logging
import warnings
from pathlib import Path

import pandas as pd

from penux_ap.config import LIKELY_TARGET_COLUMNS

log = logging.getLogger(__name__)

_POSITIVE_LABELS = {"1", "yes", "true", "sap", "severe", "1.0"}
_NEGATIVE_LABELS = {"0", "no", "false", "non-sap", "non_sap", "nonsap", "non-severe", "non_severe", "0.0"}

# The two public AP datasets used in this repository encode SAP as the raw value
# 0 and non-SAP as the raw value 1.  This is the reverse of the common sklearn
# convention where 1 is treated as the positive class.  Keep the rule explicit so
# threshold-dependent metrics (F1/F2/Fbeta, sensitivity, specificity, PPV, NPV)
# are always computed for SAP rather than for the non-SAP majority class.
_REVERSED_PUBLIC_AP_FILENAMES = {
    "ap_multiml_sanitized.csv",
    "ap_lnn_sanitized.csv",
    "ap_lnn_sanitized_en.csv",
}
_REVERSED_PUBLIC_AP_TARGET_COLUMNS = {
    "diagnostic result",
    "严重程度",
}


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


def infer_positive_value(target_column: str | None = None, data_path: str | Path | None = None, requested: str | int | None = "auto") -> int:
    """Resolve which raw binarized value denotes SAP.

    Parameters
    ----------
    target_column:
        Name of the target column, after detection if possible.
    data_path:
        Optional dataset path. Known public AP filenames are detected here.
    requested:
        "auto", 0, or 1.  In "auto" mode, the known public AP datasets are
        flipped so raw 0 becomes SAP=1.  Unknown datasets keep the standard
        convention raw 1 = positive.
    """
    if requested is None:
        requested = "auto"
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value in {"0", "1"}:
            return int(value)
        if value != "auto":
            raise ValueError("positive value must be 'auto', 0, or 1")
    elif requested in {0, 1}:
        return int(requested)
    else:
        raise ValueError("positive value must be 'auto', 0, or 1")

    if data_path is not None:
        name = Path(str(data_path)).name
        if name in _REVERSED_PUBLIC_AP_FILENAMES:
            return 0

    if target_column is not None:
        col = str(target_column).strip().lower()
        if col in _REVERSED_PUBLIC_AP_TARGET_COLUMNS:
            return 0

    return 1


def apply_positive_value(y: pd.Series, positive_value: int) -> pd.Series:
    """Return labels with SAP encoded as 1.

    ``binarize_target`` preserves the raw 0/1 coding.  For the public AP
    datasets, raw 0 means SAP; this helper flips those labels so downstream
    metrics and predicted probabilities consistently use 1=SAP.
    """
    if positive_value not in {0, 1}:
        raise ValueError("positive_value must be 0 or 1")
    y = y.astype(int)
    if positive_value == 0:
        return 1 - y
    return y


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
