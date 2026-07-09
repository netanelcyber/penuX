"""Tests for label detection and binarization using in-memory DataFrames only."""
import pandas as pd
import pytest

from penux_ap.labels import (
    apply_positive_value,
    binarize_target,
    describe_target,
    detect_target_column,
    infer_positive_value,
)


def test_detect_target_column_exact():
    df = pd.DataFrame({"age": [1], "severe": [0]})
    assert detect_target_column(df) == "severe"


def test_detect_target_column_case_insensitive():
    df = pd.DataFrame({"age": [1], "Severe": [0]})
    assert detect_target_column(df) == "Severe"


def test_detect_target_column_raises():
    df = pd.DataFrame({"age": [1], "weight": [70]})
    with pytest.raises(ValueError):
        detect_target_column(df)


def test_binarize_binary_numeric():
    s = pd.Series([0, 1, 0, 1])
    result = binarize_target(s)
    assert list(result) == [0, 1, 0, 1]


def test_binarize_yes_no():
    s = pd.Series(["yes", "no", "yes", "no"])
    result = binarize_target(s)
    assert list(result.dropna()) == [1, 0, 1, 0]


def test_binarize_sap_labels():
    s = pd.Series(["SAP", "non-SAP", "SAP"])
    result = binarize_target(s)
    assert list(result.dropna()) == [1, 0, 1]


def test_infer_positive_value_for_multiml_dataset():
    assert infer_positive_value("Diagnostic Result", "data/public_sanitized/ap_multiml_sanitized.csv") == 0


def test_infer_positive_value_for_lnn_dataset():
    assert infer_positive_value("严重程度", "data/public_sanitized/ap_lnn_sanitized.csv") == 0


def test_infer_positive_value_unknown_defaults_to_one():
    assert infer_positive_value("severe", "data/local/hospital_dataset.csv") == 1


def test_apply_positive_value_flips_raw_zero_positive():
    raw = pd.Series([0, 1, 0, 1])
    result = apply_positive_value(raw, positive_value=0)
    assert list(result) == [1, 0, 1, 0]


def test_describe_target():
    s = pd.Series([0, 1, 0, 1, 1])
    summary = describe_target(s)
    assert summary["n_total"] == 5
    assert summary["n_classes"] == 2
    assert summary["n_missing"] == 0
