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

## API (FastAPI)

Run the prediction API locally:

```bash
pip install -e .
uvicorn api.main:app --reload
```

Docs at `http://localhost:8000/docs`. Endpoints: `/health`, `/predict`,
`/predict/pathogen`, `/predict/sepsis`, `/fhir/predict`, `/camelion/predict`,
`/hl7/predict` — see `docs/tasks-api.html`-style Swagger reference at
`docs/api.html` on penux.uk for the full spec.

### Deployment (Render)

1. Create a Render **Web Service**, connect this repo.
2. **Root Directory:** `PenuX-AP-Severity`
3. **Build Command:** `pip install -e .`
4. **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

The API falls back to a logistic heuristic when no trained model is
configured — set `PENUX_AP_MODEL_PATH` to a `joblib`-serialized model to use
a real trained model instead. The Keras pathogen classifier
(`/predict/pathogen`) additionally needs `tensorflow` installed and
`clin_encoder.keras`/`clin_head.keras`/`clin_scaler.npz` present in `models/`
— both are optional; the rest of the API runs fine without them.

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
