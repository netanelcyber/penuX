# PenuX-AP-Severity

**Early prediction of Severe Acute Pancreatitis (SAP) from routine admission data**

> ⚠️ **RESEARCH USE ONLY** — This software is not validated for clinical use.
> It must not be used to guide patient-care decisions.
> It is not a medical device and provides no medical advice.
> Local Helsinki / IRB approval is required before use with hospital data.

---

## Overview

PenuX-AP-Severity is a research-grade Python repository for building and evaluating
machine-learning models that predict progression to Severe Acute Pancreatitis (SAP)
using data available within the first 24 hours of hospital admission.

**Severity follows the 2012 Revised Atlanta Classification:**
- Mild AP: no organ failure, no local/systemic complications
- Moderately severe AP: transient organ failure (<48 h) and/or local/systemic complications
- **Severe AP (primary outcome): persistent organ failure >48 h**

Models are benchmarked against classical clinical scores (BISAP, APACHE II, Ranson, Modified CTSI)
when required fields are available.

---

## ⚠️ No Bundled Dataset

**No dataset is bundled for demo execution.**
Add a legally usable, de-identified dataset to `data/public_sanitized/` before running training.

---

## Installation

```bash
git clone https://github.com/netanelcyber/PenuX-AP-Severity.git
cd PenuX-AP-Severity
pip install -e .
```

For optional dependencies:
```bash
pip install -e ".[xgboost,lightgbm,shap]"
```

---

## Quick Start

### 1. Sanitize a local dataset

```bash
python scripts/sanitize_datasets.py --input data/raw --output data/public_sanitized
```

### 2. Summarize a sanitized dataset

```bash
python scripts/summarize_datasets.py \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe
```

### 3. Run baseline models

```bash
python scripts/run_baseline.py \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe \
  --outdir outputs/demo
```

Outputs saved to `outputs/demo/`:
- `metrics.json` — AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier score
- `best_model.joblib` — best fitted model pipeline
- `threshold_table.csv` — metrics at multiple decision thresholds
- `feature_importance.csv` — permutation importance

### 4. Evaluate a saved model

```bash
python scripts/evaluate_model.py \
  --model outputs/demo/best_model.joblib \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe \
  --outdir outputs/eval
```

---

## Security

The `api/` FastAPI endpoints support API-key authentication, per-client rate
limiting, request-body size limits, restrictive CORS, and audit logging
(metadata only — never request/response bodies). See `SECURITY.md` and
`docs/hipaa_iso27799_gap_analysis_he.md` for a full, honest gap analysis
against the HIPAA Security Rule (45 CFR §164.312) and ISO/IEC 27799,
including what is and is not addressed by code alone.

To enable authentication, set `PENUX_AP_API_KEY` before starting the API —
without it, endpoints remain open (a research-only default) and a startup
warning is logged.

---

## MIMIC-IV / PhysioNet

MIMIC-IV SQL extraction scripts are in `data/mimic/sql/`.

**You must obtain PhysioNet access before using MIMIC-IV:**
1. Register at https://physionet.org
2. Complete credentialing and required training (e.g. CITI)
3. Sign the MIMIC-IV Data Use Agreement
4. Do NOT commit MIMIC patient-level data to this repository

See `docs/mimic_physionet.md` for full instructions.

---

## Repository Structure

```
PenuX-AP-Severity/
├── src/penux_ap/          # Core Python package
│   ├── config.py          # Configuration constants
│   ├── datasets.py        # Data loading & sanitization
│   ├── preprocessing.py   # Feature engineering & splitting
│   ├── features.py        # AP feature definitions & aliases
│   ├── labels.py          # Target detection & binarization
│   ├── models.py          # ML model registry
│   ├── calibration.py     # Probability calibration
│   ├── evaluation.py      # Metrics & bootstrapping
│   ├── explainability.py  # Permutation importance & SHAP
│   ├── clinical_scores.py # BISAP, APACHE II, Ranson, CTSI
│   ├── leadtime.py        # Lead-time vs confidence analysis
│   └── utils.py           # Logging, I/O helpers
├── api/                   # FastAPI research endpoint
├── scripts/               # CLI scripts
├── data/
│   ├── public_sanitized/  # Add de-identified datasets here
│   └── mimic/sql/         # MIMIC-IV extraction SQL
├── docs/                  # Documentation
├── notebooks/             # Analysis notebooks
├── tests/                 # Unit tests (in-memory fixtures only)
└── outputs/               # Model and report outputs (gitignored)
```

---

## Analysis Write-Up

- `docs/sap_severity_gbdt_analysis_he.md` — Hebrew review/analysis article covering GBDT
  (XGBoost/LightGBM/CatBoost) results on the two registered public datasets, including
  F1-optimal thresholds and a high-sensitivity (≥98%) filtering threshold analysis.
  Exploratory secondary analysis only — not clinically validated.
- `docs/sap_severity_extended_analysis_he.md` — Hebrew follow-up article: the extended
  ~1,982-model comparison (DART/GOSS/Plain boosting, DNN/ConvNet/Hybrid, a quasi-SOFA
  engineered feature, statistical significance testing). Also exploratory only.
- `docs/sap_severity_english_article.md` — English-language write-up of the full extended
  analysis (123 references), covering the same 1,982-model comparison, new boosting
  algorithms, statistical significance testing, the quasi-SOFA engineered feature, a
  structured comparison with published AP-severity ML studies, and alignment with
  TRIPOD+AI/PROBAST+AI/STROBE reporting standards. Exploratory secondary analysis only —
  not clinically validated, not peer-reviewed.
- `docs/ensemble_combination_results_he.md` — Hebrew write-up of combining 15 diverse
  top-ranked models (`scripts/ensemble_model_zoo.py`) via simple averaging, AUROC-weighted
  averaging, and stacking. On multiml, the combination (AUROC=0.8659) is a statistically
  significant improvement over the single best model (paired bootstrap p=0.034) — notable
  since the single best model alone did not reach significance against baseline. On lnn,
  the smaller improvement (AUROC=0.8930) is not significant (p=0.129) — reported honestly.
  Exploratory only, not clinically validated.
- `docs/hipaa_iso27799_gap_analysis_he.md` — Hebrew security gap analysis mapping
  the current repo (especially `api/`) against HIPAA Security Rule §164.312 and
  ISO/IEC 27799 control areas: what technical gaps were fixed (API-key auth, rate
  limiting, audit logging, request-size limits, restrictive CORS, dependency
  version bounds, `.gitignore` coverage) and what still requires organizational
  process (risk assessment, BAA/DUA, encryption-at-rest, incident response,
  independent pentest) before any real hospital data is used. Honest, not a
  compliance claim.
- `docs/security_hardening_article_he.md` — Hebrew narrative article documenting
  the security audit process, findings, and concrete `api/` hardening fixes
  (API-key auth, rate limiting, audit logging, dependency pinning), framed
  against the same before/after HIPAA §164.312 table as the gap-analysis doc,
  with an explicit section on what code alone cannot achieve.

---

## Ethical & Legal Notes

- This is a retrospective model-development study
- No patient identifiers are stored or committed
- Real hospital data requires local Helsinki / IRB approval
- MIMIC-IV requires PhysioNet credentialing and a signed DUA
- The software is not validated for clinical use
- See `docs/helsinki_irb_notes.md` for IRB submission guidance

## Limitations

- Performance depends heavily on dataset quality and cohort selection
- Atlanta 2012 SAP labels require careful operationalization from EHR data
- External validation has not been performed
- Small cohort sizes may limit generalizability
- Classical score benchmarking requires complete required fields

## Citation

If you use this software in your research, please cite it:

```
Stern, N. (2024). PenuX-AP-Severity [Software].
https://github.com/netanelcyber/penux
```

Or see `CITATION.cff`.
