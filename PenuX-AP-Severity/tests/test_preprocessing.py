"""Tests for preprocessing utilities using in-memory DataFrames only."""
import numpy as np
import pandas as pd
import pytest

from penux_ap.preprocessing import (
    build_preprocessor,
    infer_feature_types,
    make_train_test_split,
    summarize_missingness,
)


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "age": [45.0, 60.0, 35.0, 72.0, 50.0, 40.0],
        "wbc": [12.0, 15.0, 8.0, 20.0, 10.0, 9.0],
        "sex": ["M", "F", "M", "F", "M", "F"],
        "severe": [0, 1, 0, 1, 0, 1],
    })


def test_infer_feature_types(small_df):
    types = infer_feature_types(small_df, "severe")
    assert "age" in types["numeric"]
    assert "wbc" in types["numeric"]
    assert "sex" in types["categorical"]
    assert "severe" not in types["numeric"]
    assert "severe" not in types["categorical"]


def test_summarize_missingness():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1, 2, 3]})
    miss = summarize_missingness(df)
    assert "a" in miss.index
    assert "b" not in miss.index


def test_build_preprocessor_fits(small_df):
    types = infer_feature_types(small_df, "severe")
    X = small_df.drop(columns=["severe"])
    preprocessor = build_preprocessor(types["numeric"], types["categorical"])
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(small_df)


def test_make_train_test_split(small_df):
    X = small_df.drop(columns=["severe"])
    y = small_df["severe"]
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.33, random_state=42)
    assert len(X_train) + len(X_test) == len(small_df)
    assert len(y_train) == len(X_train)
