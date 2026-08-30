"""Benchmarks the ~12,000-configuration model zoo on the SIMULATED
first-episode-schizophrenia-vs-control dataset only.

See src/penux_psychosis/simulate_data.py and docs/dataset_landscape.md:
this data is not real, and is not a reproduction of any specific
published study's reported statistics. Results here describe how well
models recover a synthetic, illustrative signal -- not a clinical finding.

Parallelizes ACROSS models (one model per worker process, joblib/loky).
Every individual estimator's own internal thread pool is pinned to 1
(model_zoo_psychosis.N_JOBS = 1) specifically so this outer parallelism
does not cause the thread-oversubscription slowdown documented in the
sibling PenuX-AP-Severity project (running two levels of parallelism at
once there caused 100x+ slowdowns).
"""
import csv
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
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
N_WORKERS = os.cpu_count() or 1
CHUNK_SIZE = 200  # write to checkpoint after each chunk, not only at the end


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["model_name"])
    return done


def _fit_one(name, estimator, X, y, n_splits, seed):
    t0 = time.time()
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
    return name, auroc, dt, status


def main():
    os.makedirs("outputs", exist_ok=True)
    df = simulate_dataset()
    assert df["is_simulated"].all(), "Refusing to benchmark on non-simulated data with this script"
    feature_cols = ["Arg", "TP", "ALP", "HDL", "UA", "LDL"]
    X = df[feature_cols].values
    y = df["label"].values

    zoo = build_model_zoo()
    done = load_done(CHECKPOINT)
    todo = [(n, e) for n, e in zoo if n not in done]
    print(f"Total configurations: {len(zoo)}; already done: {len(done)}; remaining: {len(todo)}; workers: {N_WORKERS}")

    write_header = not os.path.exists(CHECKPOINT)
    t_start = time.time()
    with open(CHECKPOINT, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_name", "auroc", "fit_seconds", "status"])

        for chunk_start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[chunk_start:chunk_start + CHUNK_SIZE]
            results = Parallel(n_jobs=N_WORKERS, backend="loky")(
                delayed(_fit_one)(name, est, X, y, N_SPLITS, RANDOM_SEED) for name, est in chunk
            )
            for name, auroc, dt, status in results:
                writer.writerow([name, auroc, f"{dt:.4f}", status])
            f.flush()
            done_so_far = chunk_start + len(chunk)
            elapsed = time.time() - t_start
            rate = done_so_far / elapsed if elapsed > 0 else 0
            eta_min = (len(todo) - done_so_far) / rate / 60 if rate > 0 else float("nan")
            print(f"[{done_so_far}/{len(todo)}] elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta_min:.1f}min")

    print("Done.")


if __name__ == "__main__":
    main()
