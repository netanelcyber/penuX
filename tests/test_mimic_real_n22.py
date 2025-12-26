import sys
from pathlib import Path
import pandas as pd

# Use non-interactive backend to avoid display issues in CI
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compute_correlations import run


def test_mimic_real_n22(tmp_path):
    """Use real rows from `dataset/clinical.csv` (MIMIC-derived) and run pipeline.

    This takes the first 22 rows from the provided `dataset/clinical.csv`, adds
    a simple derived `group` column parsed from `filename`, and runs `run()`
    to ensure outputs are produced on real data.
    """
    src = Path(__file__).resolve().parents[1] / "dataset" / "clinical.csv"
    assert src.exists(), f"Expected dataset file at {src}"

    df = pd.read_csv(src)
    # Take the first 22 real rows
    df = df.head(22).copy()

    # Derive a simple group: filenames containing 'NORMAL' -> 'Normal', else 'Abnormal'
    df["group"] = df["filename"].astype(str).apply(lambda s: "Normal" if "NORMAL" in s.upper() else "Abnormal")

    csv_path = tmp_path / "mimic_real_22.csv"
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out_real"
    res = run(
        csv_path,
        outdir,
        prefix="mimic_real22",
        groupby="group",
        compute_per_group=True,
        grouped_heatmap=True,
        sample_limit=10,
        seed=1,
        group_order=["Normal", "Abnormal"],
    )

    # Top-level outputs exist
    for k in ["pearson_csv", "spearman_csv", "pearson_png", "spearman_png", "grouped_heatmap"]:
        assert k in res
        assert Path(res[k]).exists()

    # Ensure per-group CSVs were written for groups present in the data
    present = sorted(df["group"].dropna().unique().tolist())
    for g in present:
        key = f"pearson_csv_{g}"
        assert key in res
        assert Path(res[key]).exists()

    # Check that the saved Pearson matrix is square and non-empty
    p_df = pd.read_csv(res["pearson_csv"], index_col=0)
    assert p_df.shape[0] == p_df.shape[1]
    assert p_df.shape[0] > 0
