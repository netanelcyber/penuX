"""Data loading, target detection, and sanitization utilities."""
import logging
import re
from pathlib import Path

import pandas as pd

from penux_ap.config import (
    IDENTIFIER_COLUMN_PATTERNS,
    LIKELY_TARGET_COLUMNS,
    SUPPORTED_EXTENSIONS,
)

log = logging.getLogger(__name__)


def load_dataset(path: str | Path, target_column: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Args:
        path: Path to the data file.
        target_column: Optional explicit target column name.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    log.info("Loaded %s: shape=%s", path.name, df.shape)
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {list(df.columns)}")
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the most likely target column.

    Priority: exact match > case-insensitive match > common variant.
    """
    cols = list(df.columns)
    # Exact match
    for candidate in LIKELY_TARGET_COLUMNS:
        if candidate in cols:
            log.info("Detected target column (exact): '%s'", candidate)
            return candidate
    # Case-insensitive
    lower_cols = {c.lower(): c for c in cols}
    for candidate in LIKELY_TARGET_COLUMNS:
        if candidate.lower() in lower_cols:
            found = lower_cols[candidate.lower()]
            log.info("Detected target column (case-insensitive): '%s'", found)
            return found
    # Partial match
    for candidate in LIKELY_TARGET_COLUMNS:
        for col in cols:
            if candidate in col.lower():
                log.info("Detected target column (partial): '%s'", col)
                return col
    raise ValueError(
        f"Could not detect target column. Columns: {cols}. "
        "Provide --target-column explicitly."
    )


def _is_identifier_column(col: str) -> bool:
    col_lower = col.lower().strip()
    for pattern in IDENTIFIER_COLUMN_PATTERNS:
        if pattern == col_lower:
            return True
        # avoid matching clinical cols that contain "id" as substring of a word
        if re.search(rf"\b{re.escape(pattern)}\b", col_lower):
            return True
    return False


def sanitize_identifiers(
    df: pd.DataFrame,
    keep_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop likely identifier columns from a DataFrame.

    Args:
        df: Input DataFrame.
        keep_columns: Columns to preserve even if they match identifier patterns.

    Returns:
        (sanitized DataFrame, list of removed column names)
    """
    keep = set(keep_columns or [])
    to_drop = [
        col for col in df.columns
        if col not in keep and _is_identifier_column(col)
    ]
    if to_drop:
        log.warning("Removing identifier columns: %s", to_drop)
    return df.drop(columns=to_drop), to_drop
