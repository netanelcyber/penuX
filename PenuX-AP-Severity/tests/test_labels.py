"""Tests for label detection and binarization using in-memory DataFrames only."""
import pandas as pd
import pytest

from penux_ap.labels import binarize_target, describe_target, detect_target_column


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


def test_describe_target():
    s = pd.Series([0, 1, 0, 1, 1])
    summary = describe_target(s)
    assert summary["n_total"] == 5
    assert summary["n_classes"] == 2
    assert summary["n_missing"] == 0
