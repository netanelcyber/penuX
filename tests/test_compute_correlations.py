import sys
from pathlib import Path
import pandas as pd
import tempfile
import os

# Ensure project root is on sys.path so imports work consistently during pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compute_correlations import compute_correlations, run


def test_compute_correlations_numeric_only():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
        "c": [2, 3, 2, 3],
    })
    pearson, spearman = compute_correlations(df)
    assert "a" in pearson.columns
    assert pearson.loc["a", "b"] < 0


def test_run_writes_outputs(tmp_path):
    df = pd.DataFrame({
        "filename": ["x", "y", "z"],
        "temp": [36.6, 37.0, 38.1],
        "wbc": [5000, 7000, 6000],
        "age": [20, 30, 40],
    })
    csv_path = tmp_path / "test_clinical.csv"
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    out = run(csv_path, outdir, prefix="test")
    # Check files exist
    for k, p in out.items():
        assert Path(p).exists()
    # check csvs readable
    p_df = pd.read_csv(out["pearson_csv"])
    assert "temp" in p_df.columns
