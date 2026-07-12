"""Decision curve analysis (Vickers & Elkin, 2006) for the best single model
and best ensemble on each dataset, using the real cached out-of-fold
predictions from threshold_sweep.py.

Net benefit at a threshold probability p_t (the probability above which a
clinician would act on the prediction) is:

    NB(p_t) = TP/n - FP/n * (p_t / (1 - p_t))

The (p_t/(1-p_t)) term is the "exchange rate" implied by p_t: a clinician
who would act at threshold p_t is implicitly saying they are willing to
accept that many false positives per true positive found. This is compared
against two reference strategies at the same p_t:
  - "treat all": NB_all(p_t) = prevalence - (1-prevalence) * p_t/(1-p_t)
  - "treat none": NB_none(p_t) = 0 (always)

A model is only clinically useful over the range of p_t where its net
benefit exceeds BOTH reference strategies. Unlike DOR-maximization, net
benefit is bounded and does not reward degenerate operating points: a
threshold with near-zero sensitivity has TP/n near zero, so NB(p_t) is
close to (and never much above) NB_none; a threshold with near-zero
specificity has huge FP/n, which is punished increasingly as p_t rises.

Usage:
    python scripts/decision_curve_analysis.py \\
        --data data/public_sanitized/ap_multiml_sanitized.csv \\
        --target-column "Diagnostic Result" \\
        --checkpoint outputs/multiml/model_zoo_checkpoint.csv \\
        --outdir outputs/multiml --label multiml --n-select 15 --top-k-combo 15
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import apply_positive_value, binarize_target, describe_target, infer_positive_value
from penux_ap.preprocessing import infer_feature_types
from penux_ap.utils import setup_logging, ensure_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import build_model_zoo
from ensemble_model_zoo import family
from threshold_sweep import load_or_compute_oof

log = setup_logging()
RANDOM_SEED = 42


def net_benefit(y, proba, pt_grid):
    n = len(y)
    nb = np.empty_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        y_pred = (proba >= pt).astype(int)
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        nb[i] = tp / n - fp / n * (pt / (1 - pt))
    return nb


def net_benefit_treat_all(prevalence, pt_grid):
    return prevalence - (1 - prevalence) * (pt_grid / (1 - pt_grid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-select", type=int, default=15)
    parser.add_argument("--top-k-combo", type=int, default=None)
    parser.add_argument("--pt-max", type=float, default=0.6, help="max threshold probability to plot")
    parser.add_argument("--positive-value", default="auto", choices=["auto", "0", "1"])
    args = parser.parse_args()
    top_k_combo = args.top_k_combo or args.n_select

    outdir = ensure_dir(args.outdir)
    df = load_dataset(args.data)
    target_col = args.target_column or detect_target_column(df)
    y = binarize_target(df[target_col]).dropna().astype(int)
    positive_value = infer_positive_value(target_col, args.data, args.positive_value)
    y = apply_positive_value(y, positive_value)
    df = df.loc[y.index]
    log.info("Target distribution (1=SAP): %s", describe_target(y))
    feature_types = infer_feature_types(df, target_col)
    X = df.drop(columns=[target_col])
    y_arr = y.values
    prevalence = y_arr.mean()

    ckpt = pd.read_csv(args.checkpoint)
    ckpt = ckpt[ckpt["status"] == "ok"].copy()
    ckpt["family"] = ckpt["model"].apply(family)
    diverse = ckpt.sort_values("auroc", ascending=False).groupby("family").head(2)
    diverse = diverse.sort_values("auroc", ascending=False).head(args.n_select)
    selected_names = diverse["model"].tolist()

    zoo = dict(build_model_zoo())
    cache_path = Path(outdir) / f"oof_cache_{args.label}.npz"
    base_oof = load_or_compute_oof(cache_path, selected_names, zoo, X, y, feature_types)

    P = np.column_stack([base_oof[n] for n in selected_names])
    aurocs = np.array([roc_auc_score(y_arr, P[:, i]) for i in range(P.shape[1])])
    order = np.argsort(-aurocs)
    best_single_name = selected_names[order[0]]
    best_single_oof = P[:, order[0]]
    combo_oof = P[:, order[:top_k_combo]].mean(axis=1)

    pt_grid = np.linspace(0.01, args.pt_max, 300)
    nb_all = net_benefit_treat_all(prevalence, pt_grid)
    nb_none = np.zeros_like(pt_grid)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(pt_grid, nb_all, label="treat all", color="gray", linestyle="--")
    ax.plot(pt_grid, nb_none, label="treat none", color="black", linestyle=":")

    rows = []
    for name, proba, color in [
        (f"single best ({best_single_name})", best_single_oof, "crimson"),
        (f"ensemble (top {top_k_combo})", combo_oof, "navy"),
    ]:
        nb_model = net_benefit(y_arr, proba, pt_grid)
        beats_both = nb_model > np.maximum(nb_all, nb_none)
        useful_range = pt_grid[beats_both]
        if len(useful_range) > 0:
            log.info(
                "  %s: beats both 'treat all' and 'treat none' for p_t in [%.3f, %.3f] (%d/%d grid points)",
                name, useful_range.min(), useful_range.max(), beats_both.sum(), len(pt_grid),
            )
        else:
            log.info("  %s: NEVER beats both reference strategies in [0.01, %.2f]", name, args.pt_max)
        ax.plot(pt_grid, nb_model, label=name, color=color)
        for pt, nb_v, nb_a in zip(pt_grid, nb_model, nb_all):
            rows.append({"strategy": name, "pt": pt, "net_benefit": nb_v,
                         "net_benefit_treat_all": nb_a, "beats_both": bool(nb_v > max(nb_a, 0))})

    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.set_xlabel("threshold probability $p_t$")
    ax.set_ylabel("net benefit")
    ax.set_ylim(bottom=-0.05)
    ax.set_title(f"{args.label}: decision curve analysis (Vickers & Elkin 2006)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = Path(outdir) / f"decision_curve_{args.label}.png"
    fig.savefig(fig_path, dpi=150)
    log.info("Saved %s", fig_path)

    out_path = Path(outdir) / f"decision_curve_{args.label}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
