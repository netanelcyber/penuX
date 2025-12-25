"""Compute and save correlation matrices (Pearson and Spearman) and heatmap images.

Usage:
    python -m scripts.compute_correlations --input clinical.csv --outdir outputs/correlations

Outputs created per run:
    - <prefix>_pearson.csv
    - <prefix>_spearman.csv
    - <prefix>_pearson.png
    - <prefix>_spearman.png

"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_correlations(df: pd.DataFrame, columns: list | None = None):
    """Return Pearson and Spearman correlation DataFrames for numeric columns.

    Args:
        df: input DataFrame
        columns: optional list of columns to restrict to
    Returns: (pearson_df, spearman_df)
    """
    if columns is not None:
        df = df.loc[:, columns]
    # select numeric
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found to compute correlations.")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    return pearson, spearman


def save_matrix_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    logger.info(f"Saved CSV: {path}")


def save_heatmap(df: pd.DataFrame, path: Path, title: str = "Correlation"):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(4, df.shape[0] * 0.6), max(4, df.shape[1] * 0.6)))
    sns.set(style="white")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1, square=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved heatmap: {path}")


def run(input_csv: Path, outdir: Path, prefix: str = "clinical", columns: list | None = None):
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    pearson, spearman = compute_correlations(df, columns)

    # Save CSVs
    pearson_csv = outdir / f"{prefix}_pearson.csv"
    spearman_csv = outdir / f"{prefix}_spearman.csv"
    save_matrix_csv(pearson, pearson_csv)
    save_matrix_csv(spearman, spearman_csv)

    # Save heatmaps
    pearson_png = outdir / f"{prefix}_pearson.png"
    spearman_png = outdir / f"{prefix}_spearman.png"
    save_heatmap(pearson, pearson_png, title=f"{prefix} Pearson correlation")
    save_heatmap(spearman, spearman_png, title=f"{prefix} Spearman correlation")

    return {
        "pearson_csv": pearson_csv,
        "spearman_csv": spearman_csv,
        "pearson_png": pearson_png,
        "spearman_png": spearman_png,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute and save correlation matrices and heatmaps")
    parser.add_argument("--input", "-i", type=Path, default=Path("clinical.csv"), help="Input CSV file")
    parser.add_argument("--outdir", "-o", type=Path, default=Path("outputs/correlations"), help="Output directory")
    parser.add_argument("--prefix", "-p", type=str, default="clinical", help="Prefix for output files")
    parser.add_argument("--columns", "-c", type=str, default=None, help="Comma-separated list of columns to include")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        cols = None
    return run(args.input, args.outdir, args.prefix, cols)


if __name__ == "__main__":
    main()
