"""Filter a model_zoo_benchmark.csv to models statistically significantly
better than a baseline model's AUC, using the Hanley-McNeil (1982)
approximate standard error for AUC plus an unpaired two-sample z-test.

This is an UNPAIRED approximation: it treats each model's AUC as an
independent estimate, ignoring that every model in the zoo was evaluated
on the exact same out-of-fold CV splits (same y_true, correlated
predictions). A paired test (e.g. DeLong's test) would have more
statistical power here, because the positive correlation between
different models' predictions on the same folds reduces the variance of
the AUC *difference* -- but that requires the raw per-sample out-of-fold
probability vectors for every model, which scripts/benchmark_model_zoo.py
does not currently persist (only summary metrics). This script is
therefore a conservative screen: models that fail it might still be
significant under a proper paired test, but models that pass it are
significant even under the more conservative assumption.

Usage:
    python scripts/significance_vs_baseline.py \\
        --benchmark outputs/multiml/model_zoo_benchmark.csv \\
        --n-pos 204 --n-neg 1085 \\
        --baseline logreg_C1_None_lbfgs \\
        --outdir outputs/multiml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Approximate standard error of an AUC estimate (Hanley & McNeil, 1982)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Path to a model_zoo_benchmark.csv.")
    parser.add_argument("--n-pos", type=int, required=True, help="Number of positive (SAP) cases in the dataset.")
    parser.add_argument("--n-neg", type=int, required=True, help="Number of negative (non-SAP) cases.")
    parser.add_argument("--baseline", default="logreg_C1_None_lbfgs", help="Baseline model name in the 'model' column.")
    parser.add_argument("--alpha", type=float, default=0.05, help="One-tailed significance threshold.")
    parser.add_argument("--outdir", required=True, help="Directory to write significant_vs_<baseline>.csv into.")
    args = parser.parse_args()

    df = pd.read_csv(args.benchmark)
    ok = df[df["status"] == "ok"].copy()

    baseline_row = ok[ok["model"] == args.baseline]
    if len(baseline_row) == 0:
        raise SystemExit(f"Baseline model '{args.baseline}' not found in {args.benchmark}")
    auc_base = float(baseline_row.iloc[0]["auroc"])
    se_base = hanley_mcneil_se(auc_base, args.n_pos, args.n_neg)

    ok["se"] = ok["auroc"].apply(lambda a: hanley_mcneil_se(a, args.n_pos, args.n_neg))
    ok["z_vs_baseline"] = (ok["auroc"] - auc_base) / np.sqrt(ok["se"] ** 2 + se_base**2)
    ok["p_one_tailed"] = 1 - norm.cdf(ok["z_vs_baseline"])

    significant = ok[(ok["auroc"] > auc_base) & (ok["p_one_tailed"] < args.alpha)].copy()
    significant = significant.sort_values("auroc", ascending=False).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"significant_vs_{args.baseline}.csv"
    significant.to_csv(out_path, index=False)

    print(f"Baseline: {args.baseline}  AUC={auc_base:.4f}  SE={se_base:.4f}")
    print(f"Total models evaluated: {len(ok)}")
    print(f"Significantly better at p<{args.alpha} (one-tailed, unpaired Hanley-McNeil): {len(significant)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
