"""Converts LR+/LR- into post-test probability of SAP, using Bayes' theorem
via odds (the same arithmetic a Fagan nomogram encodes graphically):

    pre-test odds  = pre-test probability / (1 - pre-test probability)
    post-test odds = pre-test odds * LR
    post-test prob = post-test odds / (1 + post-test odds)

This is applied two ways:
  1. At the SAMPLE level: pre-test probability = observed SAP prevalence in
     the dataset. This reproduces PPV (post-test prob given a positive
     test) and 1-NPV (post-test prob given a negative test) as a sanity
     check -- they must match the values already in threshold_sweep_*.csv.
  2. At the INDIVIDUAL-PATIENT level: pre-test probability is whatever a
     clinician assigns a specific patient from their own judgement (other
     risk factors, exam findings, etc.), not the sample average. The same
     model LR+/LR- is then applied to that patient's own prior. This is
     where likelihood ratios are actually useful clinically -- two patients
     with very different pre-test probabilities get very different
     post-test probabilities from the identical test result.

Also renders a Fagan nomogram (three parallel log-odds scales connected by
straight lines) for the four model/dataset combinations at their sample
prevalence.

Usage:
    python scripts/posttest_probability.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.utils import setup_logging, ensure_dir

log = setup_logging()

PREVALENCE = {"multiml": 204 / 1289, "lnn": 137 / 722}

MODELS = [
    ("multiml", "outputs/multiml/threshold_sweep_multiml.csv", "single best (hybrid DNN+ConvNet+GBDT)",
     "single best (hybrid_dnn(64,)_conv(8, 16)_gbdt100-5-0.05_gbdt_heavy)"),
    ("multiml", "outputs/multiml/threshold_sweep_multiml.csv", "15-model ensemble",
     "ensemble (top 15)"),
    ("lnn", "outputs/lnn/threshold_sweep_lnn.csv", "single best (LightGBM)",
     "single best (lightgbm_n800_leaves15_lr0.01)"),
    ("lnn", "outputs/lnn/threshold_sweep_lnn.csv", "5-model ensemble",
     "ensemble (top 5)"),
]

ILLUSTRATIVE_PRE_TEST_PROBS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]


def post_test_prob(pre_prob, lr):
    pre_odds = pre_prob / (1 - pre_prob)
    post_odds = pre_odds * lr
    return post_odds / (1 + post_odds)


def load_default_row(csv_path, strategy_key):
    df = pd.read_csv(csv_path)
    row = df[(df["strategy"] == strategy_key) & (df["criterion"] == "default_0.5")]
    if row.empty:
        raise SystemExit(f"Could not find default_0.5 row for {strategy_key} in {csv_path}")
    return row.iloc[0]


def main():
    outdir = ensure_dir("outputs/posttest_probability")

    sample_rows = []
    individual_rows = []
    fagan_lines = []

    for label, csv_path, display_name, strategy_key in MODELS:
        row = load_default_row(csv_path, strategy_key)
        lr_plus, lr_minus = row["lr_plus"], row["lr_minus"]
        prevalence = PREVALENCE[label]

        post_pos = post_test_prob(prevalence, lr_plus)
        post_neg = post_test_prob(prevalence, lr_minus)
        log.info(
            "[%s] %s: prevalence=%.4f -> post-test prob if POSITIVE=%.4f (check vs PPV=%.4f), "
            "if NEGATIVE=%.4f (check vs 1-NPV=%.4f)",
            label, display_name, prevalence, post_pos, row["ppv"], post_neg, 1 - row["npv"],
        )
        sample_rows.append({
            "dataset": label, "model": display_name, "prevalence": prevalence,
            "lr_plus": lr_plus, "lr_minus": lr_minus,
            "post_test_prob_if_positive": post_pos, "post_test_prob_if_negative": post_neg,
            "sanity_check_ppv": row["ppv"], "sanity_check_1_minus_npv": 1 - row["npv"],
        })
        fagan_lines.append((f"{label}: {display_name}", prevalence, lr_plus, lr_minus))

        for pre_prob in ILLUSTRATIVE_PRE_TEST_PROBS:
            individual_rows.append({
                "dataset": label, "model": display_name,
                "clinician_pre_test_probability": pre_prob,
                "post_test_prob_if_positive": post_test_prob(pre_prob, lr_plus),
                "post_test_prob_if_negative": post_test_prob(pre_prob, lr_minus),
            })

    sample_df = pd.DataFrame(sample_rows)
    sample_path = Path(outdir) / "posttest_probability_sample_prevalence.csv"
    sample_df.to_csv(sample_path, index=False)
    log.info("Saved %s", sample_path)

    individual_df = pd.DataFrame(individual_rows)
    individual_path = Path(outdir) / "posttest_probability_individual_patient.csv"
    individual_df.to_csv(individual_path, index=False)
    log.info("Saved %s", individual_path)

    fig = plot_fagan_nomogram(fagan_lines)
    fig_path = Path(outdir) / "fagan_nomogram.png"
    fig.savefig(fig_path, dpi=150)
    log.info("Saved %s", fig_path)


def prob_to_logodds(p):
    return np.log(p / (1 - p))


def plot_fagan_nomogram(fagan_lines):
    """Three parallel log-odds axes (pre-test prob | LR | post-test prob),
    connected by a straight line per model -- one line for LR+, one for LR-.
    A straight line is valid because log(post-odds) = log(pre-odds) + log(LR):
    on three axes all scaled in log-odds units and equally spaced, a line
    of constant total "rise" from pre-test to LR to post-test is straight.
    """
    tick_probs = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
    tick_lrs = np.array([0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000])

    fig, ax = plt.subplots(figsize=(9, 8))
    x_pre, x_lr, x_post = 0.0, 1.0, 2.0

    for x, ticks, fmt, label in [
        (x_pre, tick_probs, lambda v: f"{v*100:g}%", "pre-test\nprobability"),
        (x_lr, tick_lrs, lambda v: f"{v:g}", "likelihood\nratio"),
        (x_post, tick_probs, lambda v: f"{v*100:g}%", "post-test\nprobability"),
    ]:
        y = prob_to_logodds(ticks) if x != x_lr else np.log(ticks)
        ax.plot([x, x], [y.min(), y.max()], color="black", linewidth=1)
        for yi, ti in zip(y, ticks):
            ax.text(x + (0.04 if x != x_post else -0.04), yi, fmt(ti), fontsize=7,
                    ha="left" if x != x_post else "right", va="center")
        ax.text(x, y.max() + 0.6, label, fontsize=9, ha="center", fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(fagan_lines)))
    for (name, prevalence, lr_plus, lr_minus), color in zip(fagan_lines, colors):
        y_pre = prob_to_logodds(prevalence)
        for lr, style, tag in [(lr_plus, "-", "LR+"), (lr_minus, "--", "LR-")]:
            y_lr = np.log(lr)
            post = post_test_prob(prevalence, lr)
            y_post = prob_to_logodds(post)
            ax.plot([x_pre, x_lr, x_post], [y_pre, y_lr, y_post], color=color, linestyle=style,
                     marker="o", markersize=3, linewidth=1.3, label=f"{name} ({tag})")

    ax.set_xlim(-0.5, 2.5)
    ax.axis("off")
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.suptitle("Fagan nomogram: pre-test probability -> post-test probability of SAP", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


if __name__ == "__main__":
    main()
