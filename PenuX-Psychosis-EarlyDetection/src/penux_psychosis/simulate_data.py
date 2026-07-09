"""Generates a SIMULATED dataset of routine blood-panel values for a
first-episode-schizophrenia-vs-control classification task.

IMPORTANT: This is not real patient data, and it is not a reproduction of any
specific published study's reported statistics. No mean/SD/effect-size numbers
here were retrieved from Frontiers in Medicine (2026) or any other specific
paper -- an attempt was made to retrieve that paper's exact reported table but
full-text access was blocked (Cloudflare/403) on both the publisher site and
PMC. The six features (Arg, TP, ALP, HDL, UA, LDL) and the case/control sample
sizes (180/214) are real, publicly reported facts about that paper's design;
everything else here (means, standard deviations, and the direction/size of
group differences) is a generic, illustrative approximation built from
general, widely-known clinical reference ranges and commonly reported
directional associations (e.g., lower HDL and altered uric acid in psychotic
disorders), NOT the paper's actual measured values. This dataset exists solely
to let a large model-zoo benchmarking pipeline be exercised end-to-end; any
resulting numbers (AUROC, feature importance, etc.) describe how well models
recover a synthetic signal, not a real clinical finding.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Illustrative, non-paper-derived reference parameters for each routine lab
# value: (control_mean, control_sd, case_mean, case_sd), chosen only to be
# within/near normal clinical reference ranges with plausible small shifts in
# directions commonly discussed in the general literature -- not fitted to,
# or copied from, any specific study's reported statistics.
FEATURE_PARAMS = {
    # Arginine (umol/L) -- amino acid, illustrative shift only
    "Arg": (65.0, 15.0, 58.0, 16.0),
    # Total protein (g/dL)
    "TP": (7.2, 0.5, 7.0, 0.55),
    # Alkaline phosphatase (U/L)
    "ALP": (75.0, 20.0, 82.0, 24.0),
    # HDL cholesterol (mg/dL) -- illustrative: lower in the case group
    "HDL": (52.0, 12.0, 44.0, 11.0),
    # Uric acid (mg/dL) -- illustrative: modest shift, direction genuinely
    # debated in the general literature, kept small here deliberately
    "UA": (5.0, 1.3, 5.4, 1.4),
    # LDL cholesterol (mg/dL) -- illustrative: higher in the case group
    "LDL": (105.0, 25.0, 118.0, 28.0),
}

N_CONTROLS = 214
N_CASES = 180


def simulate_dataset(random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a labeled synthetic dataframe: columns are the 6 features plus
    `label` (1 = simulated first-episode case, 0 = simulated control) and
    `is_simulated` (always True, kept as a visible guardrail column)."""
    rng = np.random.default_rng(random_seed)
    rows = []
    for _ in range(N_CONTROLS):
        row = {f: rng.normal(mu_c, sd_c) for f, (mu_c, sd_c, mu_k, sd_k) in FEATURE_PARAMS.items()}
        row["label"] = 0
        rows.append(row)
    for _ in range(N_CASES):
        row = {f: rng.normal(mu_k, sd_k) for f, (mu_c, sd_c, mu_k, sd_k) in FEATURE_PARAMS.items()}
        row["label"] = 1
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_simulated"] = True
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = simulate_dataset()
    print(df.shape)
    print(df["label"].value_counts())
    df.to_csv("data/public_sanitized/simulated_fep_routine_labs.csv", index=False)
    print("wrote data/public_sanitized/simulated_fep_routine_labs.csv")
