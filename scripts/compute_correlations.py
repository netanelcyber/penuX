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
import json
import numpy as np
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


def save_sample_feature_heatmap(df_samples: pd.DataFrame, numeric_cols: list, group_col: str | None, path: Path, title: str = "Samples by group"):
    """Create a heatmap of samples x features ordered/grouped by group_col.

    - df_samples: original DataFrame (must contain numeric_cols and optionally group_col)
    - numeric_cols: list of numeric columns to plot
    - group_col: name of group column (or None)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = df_samples.loc[:, numeric_cols].astype(float).to_numpy()
    # z-score normalize columns for visualization
    col_mean = np.nanmean(data, axis=0, keepdims=True)
    col_std = np.nanstd(data, axis=0, keepdims=True) + 1e-6
    data_z = (data - col_mean) / col_std

    # ordering
    if group_col is not None and group_col in df_samples.columns:
        groups = df_samples[group_col].astype(str).tolist()
        order = np.argsort(groups)
        data_z = data_z[order, :]
        ordered_groups = [groups[i] for i in order]
    else:
        ordered_groups = None

    plt.figure(figsize=(max(6, len(numeric_cols) * 1.2), max(6, data_z.shape[0] * 0.02)))
    sns.set(style="white")
    ax = sns.heatmap(data_z, cmap="vlag", center=0, cbar_kws={"label": "z-score"}, xticklabels=numeric_cols, yticklabels=False)
    ax.set_title(title)

    # add horizontal lines between groups
    if ordered_groups is not None:
        boundaries = []
        prev = ordered_groups[0]
        for i, g in enumerate(ordered_groups):
            if g != prev:
                boundaries.append(i)
                prev = g
        for b in boundaries:
            ax.hlines(b, *ax.get_xlim(), colors="black", linewidth=1.0)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved sample-feature heatmap: {path}")

def run(
    input_csv: Path,
    outdir: Path,
    prefix: str = "clinical",
    columns: list | None = None,
    groupby: str | None = None,
    compute_per_group: bool = False,
    grouped_heatmap: bool = False,
    group_map: Path | None = None,
    sample_limit: int | None = None,
    seed: int = 42,
    group_order: list | None = None,
):
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    if group_map is not None:
        try:
            gm = json.loads(Path(group_map).read_text())
        except Exception:
            gm = None
    else:
        gm = None

    # If groupby exists and maps are provided (e.g., numeric label -> name), apply mapping
    if groupby is not None and groupby in df.columns and gm is not None:
        # map numeric labels to names when possible
        df[groupby] = df[groupby].apply(lambda x: gm.get(str(int(x)), str(x)) if (not isinstance(x, str) and not np.isnan(x)) else (gm.get(str(x), x) if gm else x))

    # base correlations on numeric columns (or restricted)
    pearson, spearman = compute_correlations(df, columns)

    # Save top-level CSVs
    pearson_csv = outdir / f"{prefix}_pearson.csv"
    spearman_csv = outdir / f"{prefix}_spearman.csv"
    save_matrix_csv(pearson, pearson_csv)
    save_matrix_csv(spearman, spearman_csv)

    # Save top-level heatmaps
    pearson_png = outdir / f"{prefix}_pearson.png"
    spearman_png = outdir / f"{prefix}_spearman.png"
    save_heatmap(pearson, pearson_png, title=f"{prefix} Pearson correlation")
    save_heatmap(spearman, spearman_png, title=f"{prefix} Spearman correlation")

    results = {
        "pearson_csv": pearson_csv,
        "spearman_csv": spearman_csv,
        "pearson_png": pearson_png,
        "spearman_png": spearman_png,
    }
    # record group ordering used (if relevant)
    results["group_order_used"] = group_order if group_order is not None else None

    # Per-group correlations
    if compute_per_group:
        if groupby is None or groupby not in df.columns:
            raise ValueError("--compute_per_group requires --groupby <column> that exists in the CSV")
        # Determine group ordering: explicit group_order (if provided) then remaining groups sorted
        present_groups = [g for g in df[groupby].dropna().unique().tolist()]
        if group_order is not None:
            # Keep only groups actually present, preserve requested order, append any missing
            ordered = [g for g in group_order if g in present_groups]
            ordered += [g for g in sorted(present_groups, key=lambda x: str(x)) if g not in ordered]
            groups = ordered
        else:
            groups = sorted(present_groups, key=lambda x: str(x))

        rng = np.random.RandomState(seed)
        for g in groups:
            gsan = str(g).replace(" ", "_").replace("/", "_")
            sub = df[df[groupby] == g]
            if sample_limit is not None and sub.shape[0] > sample_limit:
                sub = sub.sample(sample_limit, random_state=rng)
            p, s = compute_correlations(sub, columns)
            pcsv = outdir / f"{prefix}_pearson_{gsan}.csv"
            scsv = outdir / f"{prefix}_spearman_{gsan}.csv"
            save_matrix_csv(p, pcsv)
            save_matrix_csv(s, scsv)
            ppng = outdir / f"{prefix}_pearson_{gsan}.png"
            spng = outdir / f"{prefix}_spearman_{gsan}.png"
            save_heatmap(p, ppng, title=f"{prefix} Pearson ({g})")
            save_heatmap(s, spng, title=f"{prefix} Spearman ({g})")
            results.update({f"pearson_csv_{gsan}": pcsv, f"spearman_csv_{gsan}": scsv, f"pearson_png_{gsan}": ppng, f"spearman_png_{gsan}": spng})

    # grouped sample-feature heatmap
    if grouped_heatmap:
        if groupby is None or groupby not in df.columns:
            raise ValueError("--grouped_heatmap requires --groupby <column> that exists in the CSV")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if columns is not None:
            numeric_cols = [c for c in numeric_cols if c in columns]
        # optionally subsample per group to limit size
        if sample_limit is not None:
            rng = np.random.RandomState(seed)
            rows = []
            # Use group_order if provided
            group_iter = group_order if group_order is not None else sorted(df[groupby].dropna().unique().tolist())
            for g in group_iter:
                sub = df[df[groupby] == g]
                if sub.shape[0] == 0:
                    continue
                if sub.shape[0] > sample_limit:
                    sel = sub.sample(sample_limit, random_state=rng)
                else:
                    sel = sub
                rows.append(sel)
            df_plot = pd.concat(rows, axis=0)
        else:
            # If no sampling limit but user requested group_order, reorder accordingly
            if group_order is not None:
                rows = []
                for g in group_order:
                    sub = df[df[groupby] == g]
                    if sub.shape[0] > 0:
                        rows.append(sub)
                # append any remaining groups not listed in group_order
                remaining = [g for g in df[groupby].dropna().unique().tolist() if g not in group_order]
                for g in remaining:
                    rows.append(df[df[groupby] == g])
                df_plot = pd.concat(rows, axis=0)
            else:
                df_plot = df.copy()
        gh_png = outdir / f"{prefix}_grouped_by_{groupby}.png"
        save_sample_feature_heatmap(df_plot, numeric_cols, groupby, gh_png, title=f"{prefix} samples ordered by {groupby}")
        results.update({"grouped_heatmap": gh_png})

    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute and save correlation matrices and heatmaps")
    parser.add_argument("--input", "-i", type=Path, default=Path("clinical.csv"), help="Input CSV file")
    parser.add_argument("--outdir", "-o", type=Path, default=Path("outputs/correlations"), help="Output directory")
    parser.add_argument("--prefix", "-p", type=str, default="clinical", help="Prefix for output files")
    parser.add_argument("--columns", "-c", type=str, default=None, help="Comma-separated list of columns to include")
    parser.add_argument("--groupby", type=str, default=None, help="Optional column name to group by (e.g., label or pathogen)")
    parser.add_argument("--compute_per_group", action="store_true", help="Compute per-group correlation matrices (requires --groupby)")
    parser.add_argument("--grouped_heatmap", action="store_true", help="Create a sample-feature heatmap with samples ordered by group (requires --groupby)")
    parser.add_argument("--group_map", type=str, default=None, help="Optional JSON file mapping group ids to names (e.g., pathogen_vocab.json)")
    parser.add_argument("--sample_limit", type=int, default=None, help="Max samples per group to include in grouped heatmap / group-specific computations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling when limiting per-group samples")
    parser.add_argument("--group_order", type=str, default=None, help="Optional comma-separated group names to order groups by (e.g., 'Bacterial,Viral,Other')")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        cols = None
    group_order = [g.strip() for g in args.group_order.split(",") if g.strip()] if args.group_order else None
    return run(args.input, args.outdir, args.prefix, cols, groupby=args.groupby, compute_per_group=args.compute_per_group, grouped_heatmap=args.grouped_heatmap, group_map=args.group_map, sample_limit=args.sample_limit, seed=args.seed, group_order=group_order)


if __name__ == "__main__":
    main()
