"""Controlled simulation (binormal scores, NOT real clinical data) testing
whether the DOR-maximizing threshold's statistical fragility (Section 7.2
of docs/manuscripts/arxiv_dor_latex/degenerate_dor_optima.tex) improves with
sample size. It does not: the search always finds a boundary configuration
resting on ~1-2 misclassified examples regardless of n, because it is an
adversarial search over the achievable ROC curve, not an ordinary estimation
problem that benefits from more data.

Usage:
    python scripts/dor_boundary_simulation.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def target_mu_diff(target_auc, sigma=1.0):
    return norm.ppf(target_auc) * sigma * np.sqrt(2)

def simulate_once(n, prevalence, target_auc, rng, sigma=1.0):
    n_pos = max(1, int(round(n * prevalence)))
    n_neg = n - n_pos
    mu_diff = target_mu_diff(target_auc, sigma)
    pos_scores = rng.normal(loc=mu_diff, scale=sigma, size=n_pos)
    neg_scores = rng.normal(loc=0.0, scale=sigma, size=n_neg)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    s = np.concatenate([pos_scores, neg_scores])
    return y, s

def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2*n)
    margin = z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2))
    return (max(0.0, (center-margin)/denom), min(1.0, (center+margin)/denom))

def max_dor_threshold_fast(y, s):
    """Vectorized sweep: sort by score descending, thresholds = each distinct
    score value, use cumulative sums to get TP/FP at each cut in O(n log n)."""
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    n_pos_total = y.sum()
    n_neg_total = len(y) - n_pos_total

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    s_sorted = s[order]
    is_last_of_group = np.r_[s_sorted[:-1] != s_sorted[1:], True]
    idx = np.where(is_last_of_group)[0]

    tp = tp_cum[idx]
    fp = fp_cum[idx]
    fn = n_pos_total - tp
    tn = n_neg_total - fp

    valid = (tp > 0) & (tn > 0) & (fp > 0) & (fn > 0)
    if not valid.any():
        return None
    tp, tn, fp, fn = tp[valid], tn[valid], fp[valid], fn[valid]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    lrp = sens / (1 - spec)
    lrm = (1 - sens) / spec
    dor = lrp / lrm
    best_i = np.argmax(dor)
    return dor[best_i], int(tp[best_i]), int(tn[best_i]), int(fp[best_i]), int(fn[best_i]), sens[best_i], spec[best_i]

rng = np.random.default_rng(42)
PREVALENCE = 0.158
TARGET_AUC = 0.85
N_GRID = [100, 300, 1000, 3000, 10000, 30000]
N_REPLICATES = 100

rows = []
for n in N_GRID:
    for rep in range(N_REPLICATES):
        y, s = simulate_once(n, PREVALENCE, TARGET_AUC, rng)
        result = max_dor_threshold_fast(y, s)
        if result is None:
            continue
        dor, tp, tn, fp, fn, sens, spec = result
        n_neg = tn + fp
        n_pos = tp + fn
        spec_lo, spec_hi = wilson_ci(tn, n_neg)
        sens_lo, sens_hi = wilson_ci(tp, n_pos)
        lrp = sens / (1 - spec) if spec < 1 else np.inf
        cross_a = sens_lo / (1 - spec_hi) if spec_hi < 1 else np.inf
        cross_b = sens_hi / (1 - spec_lo) if spec_lo < 1 else np.inf
        lrp_lo, lrp_hi = min(cross_a, cross_b), max(cross_a, cross_b)
        rows.append(dict(n=n, rep=rep, sens=sens, spec=spec, dor=dor, fp=fp, n_neg=n_neg,
                          lrp=lrp, lrp_lo=lrp_lo, lrp_hi=lrp_hi,
                          lrp_ratio=(lrp_hi/lrp_lo) if lrp_lo > 0 else np.inf,
                          degenerate=(sens < 0.5 or spec < 0.5)))

df = pd.DataFrame(rows)
df.to_csv('/tmp/claude-0/-home-user-penuX/86707bdc-847f-5c82-8cb5-a903b102daa8/scratchpad/dor_simulation_results.csv', index=False)

summary = df.groupby('n').agg(
    frac_degenerate=('degenerate', 'mean'),
    median_fp=('fp', 'median'),
    median_lrp_ratio=('lrp_ratio', 'median'),
    median_dor=('dor', 'median'),
).reset_index()
print(summary.to_string(index=False))
