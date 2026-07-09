"""Benchmarks the ~12,000-configuration model zoo on the SIMULATED
first-episode-schizophrenia-vs-control dataset only.

See src/penux_psychosis/simulate_data.py and docs/dataset_landscape.md:
this data is not real, and is not a reproduction of any specific
published study's reported statistics. Results here describe how well
models recover a synthetic, illustrative signal -- not a clinical finding.
"""
import csv
import os
import sys
import time

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model_zoo_psychosis import build_model_zoo
from penux_psychosis.simulate_data import simulate_dataset

CHECKPOINT = "outputs/psychosis_model_zoo_checkpoint.csv"
N_SPLITS = 5
RANDOM_SEED = 42


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done[row["model_name"]] = row
    return done


def main():
    os.makedirs("outputs", exist_ok=True)
    df = simulate_dataset()
    assert df["is_simulated"].all(), "Refusing to benchmark on non-simulated data with this script"
    feature_cols = ["Arg", "TP", "ALP", "HDL", "UA", "LDL"]
    X = df[feature_cols].values
    y = df["label"].values

    zoo = build_model_zoo()
    done = load_done(CHECKPOINT)
    print(f"Total configurations: {len(zoo)}; already done: {len(done)}")

    write_header = not os.path.exists(CHECKPOINT)
    with open(CHECKPOINT, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_name", "auroc", "fit_seconds", "status"])
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        t_start = time.time()
        for i, (name, estimator) in enumerate(zoo):
            if name in done:
                continue
            t0 = time.time()
            try:
                oof = np.zeros(len(y), dtype=float)
                for train_idx, test_idx in skf.split(X, y):
                    pipe = Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("clf", estimator),
                    ])
                    pipe.fit(X[train_idx], y[train_idx])
                    if hasattr(pipe, "predict_proba"):
                        proba = pipe.predict_proba(X[test_idx])[:, 1]
                    else:
                        proba = pipe.decision_function(X[test_idx])
                    oof[test_idx] = proba
                auroc = roc_auc_score(y, oof)
                status = "ok"
            except Exception as e:
                auroc = float("nan")
                status = f"error: {type(e).__name__}"
            dt = time.time() - t0
            writer.writerow([name, auroc, f"{dt:.4f}", status])
            f.flush()
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                print(f"[{i+1}/{len(zoo)}] elapsed={elapsed:.1f}s last={name} auroc={auroc}")

    print("Done.")


if __name__ == "__main__":
    main()
