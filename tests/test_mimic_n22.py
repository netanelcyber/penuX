import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Use non-interactive backend to avoid display issues in CI
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compute_correlations import run


def test_mimic_n22(tmp_path):
    """Construct a small MIMIC-like dataframe with N=22 and run the pipeline.

    This ensures the top-level correlation files and per-group outputs are produced.
    """
    n = 22
    rng = np.random.RandomState(0)

    df = pd.DataFrame(
        {
            "filename": [f"p{i}" for i in range(n)],
            "temperature_c": rng.normal(37.0, 1.0, size=n),
            "wbc": rng.randint(4000, 20000, size=n),
            "spo2": rng.randint(85, 100, size=n),
            "age": rng.randint(18, 90, size=n),
            # assign three groups to exercise per-group computations
            "pathogen": [rng.choice(["Bacterial", "Viral", "Normal"], p=[0.4, 0.4, 0.2]) for _ in range(n)],
        }
    )

    csv_path = tmp_path / "mimic22.csv"
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    res = run(
        csv_path,
        outdir,
        prefix="mimic22",
        groupby="pathogen",
        compute_per_group=True,
        grouped_heatmap=True,
        sample_limit=10,
        seed=1,
        group_order=["Bacterial", "Viral", "Normal"],
    )

    # Top-level outputs exist
    for k in ["pearson_csv", "spearman_csv", "pearson_png", "spearman_png", "grouped_heatmap"]:
        assert k in res
        assert Path(res[k]).exists()

    # Ensure per-group CSVs were written for groups present in the data
    present = sorted(df["pathogen"].dropna().unique().tolist())
    for g in present:
        key = f"pearson_csv_{g}"
        assert key in res
        assert Path(res[key]).exists()

    # Check that the saved Pearson matrix is square and non-empty
    p_df = pd.read_csv(res["pearson_csv"], index_col=0)
    assert p_df.shape[0] == p_df.shape[1]
    assert p_df.shape[0] > 0
