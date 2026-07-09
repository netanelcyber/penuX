# PenuX-Psychosis-EarlyDetection

**Exploring early prediction of a first psychotic episode from routine laboratory tests, prior to clinical presentation**

> ⚠️ **RESEARCH USE ONLY — NOT A DIAGNOSTIC OR SCREENING TOOL.**
> This project must never be used to label, flag, or make any determination about a real
> person's mental health status. Predicting risk of a first psychotic episode carries serious
> potential for harm if misused: false positives can cause unwarranted alarm, stigma, and
> discriminatory treatment (insurance, employment, social); false negatives can create false
> reassurance. Nothing produced here is validated for clinical, insurance, educational, or any
> other real-world decision about an identifiable person. Local Helsinki/IRB approval, and very
> likely a formal ethics review specific to psychiatric research, would be required before any
> use beyond this exploratory, public-data research setting.

---

## Objective (as requested)

Explore whether routine laboratory tests (the kind collected in ordinary primary-care or
general hospital blood panels — not specialized psychiatric biomarkers, imaging, or genetic
testing) carry a detectable signal for a first psychotic episode **up to 24 months before**
clinical presentation, using a large comparative model-zoo methodology (in the spirit of the
sibling `PenuX-AP-Severity` project, ultimately scaling to a comparably large — here, ~12,000
configuration — model space).

## Status: methodology demonstration on clearly-labeled simulated data only

**No real dataset was found or used.** `docs/dataset_landscape.md` documents a real search for
a dataset matching the stated question (routine labs, genuine ~24-month lead time before a
documented first psychotic episode) — none exists as an open, freely accessible download; the
closest real precedent (a 2026 Frontiers in Medicine study) uses routine blood biomarkers but
collected at/near first-episode presentation, and its exact reported numbers were inaccessible
(publisher site and PMC both blocked scraper access).

Per an explicit decision made with the corresponding researcher, this project proceeded with a
**model-zoo benchmarking exercise on a synthetic dataset that is labeled as synthetic
everywhere it appears** (an `is_simulated` column the benchmark script refuses to run without;
prominent warnings in every document) — this is different from, and does not walk back, the
original commitment never to present fabricated data as real. See
`docs/simulation_article_he.md` (Hebrew) for the full write-up:

- `src/penux_psychosis/simulate_data.py` — generates 180 "case" / 214 "control" rows across 6
  routine lab features, using generic clinical reference ranges and commonly-discussed
  directional shifts from general literature (explicitly **not** Frontiers 2026's actual
  reported statistics, which were inaccessible).
- `scripts/model_zoo_psychosis.py` — 11,903 classifier configurations (linear/SVM/KNN/NB/tree/
  ensemble/GBDT families, plus a from-scratch sklearn-compatible PyTorch DNN wrapper).
- `scripts/benchmark_psychosis.py` — 5-fold stratified CV, checkpointed, parallelized across
  models; refuses to run on any dataset without `is_simulated=True`.
- Result: best configuration reached AUROC 0.809 (`gbdt_sklearn_n90_lr0.05_sub0.7`), which did
  **not** reach conventional statistical significance against a logistic-regression baseline
  (Hanley-McNeil, p≈0.18) — reported honestly, matching `PenuX-AP-Severity`'s practice of not
  suppressing negative/null findings.

**No model or number here should be interpreted as evidence about real psychosis prediction.**
If a real dataset becomes available in the future (e.g., via UK Biobank access), this
methodology can be re-run on it — but that is a distinct, not-yet-taken step.

---

## Why this is harder than PenuX-AP-Severity's dataset situation

- Psychiatric prodrome/first-episode research datasets are overwhelmingly **access-controlled**
  (institutional data use agreements, ethics-committee-approved researcher accounts), unlike the
  MIT/Apache-licensed CSV files that existed for acute pancreatitis severity.
- A genuine "routine labs, N months before a first psychotic episode" design requires
  **prospective or registry-linked cohorts** (e.g., birth-cohort or population-registry studies
  with banked/linked lab results predating diagnosis) — these are precisely the kind of data
  that carries the highest re-identification and stigma risk, so open release is rare.
- The strongest published precedents found so far (see `docs/dataset_landscape.md`) either (a)
  use blood tests collected **at or near first-episode presentation**, not 24 months prior, or
  (b) use large-scale EHR/claims data with genuine pre-onset lead time, but that data is
  proprietary/access-controlled (e.g., insurance claims databases), not an open download.

## Repository Structure (mirrors PenuX-AP-Severity)

```
PenuX-Psychosis-EarlyDetection/
├── src/penux_psychosis/
│   └── simulate_data.py         # Generates the labeled-synthetic dataset (not real data)
├── data/public_sanitized/
│   └── simulated_fep_routine_labs.csv  # The synthetic dataset itself
├── scripts/
│   ├── model_zoo_psychosis.py   # ~12,000-configuration model zoo
│   └── benchmark_psychosis.py   # 5-fold CV benchmark, checkpointed, parallel
├── docs/
│   ├── dataset_landscape.md     # Honest account of the real-dataset search
│   └── simulation_article_he.md # Hebrew write-up of the simulation-benchmark results
├── notebooks/
└── tests/
```

## Ethical & Legal Notes

- No patient data of any kind — real or synthetic-presented-as-real — is included in this
  repository.
- Any future real dataset used here would require: documented ethics/IRB approval covering
  psychiatric risk prediction specifically, a signed data use agreement with the data
  custodian, and explicit consideration of stigma and discrimination risk in how results are
  reported.
- This software, if it is ever built, will not be validated for clinical use, and predicting a
  future psychiatric diagnosis is a fundamentally higher-stakes claim than predicting severity
  of an already-diagnosed physical illness (as in `PenuX-AP-Severity`) — extra caution in
  framing, reporting, and any public communication is required.

## Citation

```
Stern, N. (2026). PenuX-Psychosis-EarlyDetection [Software, in progress].
https://github.com/netanelcyber/penux
```
