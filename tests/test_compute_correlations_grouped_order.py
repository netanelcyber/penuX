import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_correlations import run


def write_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)


def test_group_order_respected(tmp_path):
    # Create a small dataset with group labels
    df = pd.DataFrame({
        "temperature_c": [37.0, 37.5, 36.8, 38.0, 36.6, 37.2],
        "wbc": [7000, 12000, 5000, 15000, 4800, 7500],
        "spo2": [96, 92, 98, 90, 99, 95],
        "age": [60, 45, 80, 55, 30, 70],
        "label": [0, 1, 0, 1, 2, 2],
    })
    csv_in = tmp_path / "in.csv"
    write_csv(csv_in, df)

    # group map (label -> name)
    gm = {"0": "Bacterial", "1": "Viral", "2": "Other"}
    gm_path = tmp_path / "gm.json"
    gm_path.write_text(json.dumps(gm))

    outdir = tmp_path / "out"
    # specify an explicit order
    res = run(csv_in, outdir, prefix="test", columns=None, groupby="label", compute_per_group=True, grouped_heatmap=True, group_map=str(gm_path), sample_limit=2, seed=1, group_order=["Viral", "Bacterial", "Other"])

    # confirm per-group outputs exist and the order used is reflected
    assert (outdir / "test_pearson_Viral.csv").exists()
    assert (outdir / "test_pearson_Bacterial.csv").exists()
    assert (outdir / "test_pearson_Other.csv").exists()
    # check that the run returned the order used
    assert res.get("group_order_used") == ["Viral", "Bacterial", "Other"]
