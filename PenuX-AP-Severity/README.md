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

## Dataset policy

Only legally usable, de-identified research data should be used. Do not commit PHI or patient-level data that cannot be shared under the applicable data-use agreement.

The repository includes sanitized research datasets under `data/public_sanitized/`; provenance and permitted use should be checked in the accompanying documentation before analysis.

---

## Installation

```bash
git clone https://github.com/netanelcyber/penuX.git
cd penuX/PenuX-AP-Severity
pip install -e .
```

For optional dependencies:

```bash
pip install -e ".[xgboost,lightgbm,shap]"
```

---

## Two analysis modes

### 1. Exploratory baseline

The historical baseline workflow remains available for fast exploratory comparison:

```bash
python scripts/run_baseline.py \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe \
  --outdir outputs/demo
```

This is useful for rapid experimentation, but it should not be treated as the preferred confirmatory validation pathway because model ranking is based on the held-out split.

### 2. Leakage-resistant research validation

For serious model-development experiments, use the dedicated validation workflow:

```bash
python scripts/run_research_validation.py \
  --data data/public_sanitized/ap_multiml_sanitized.csv \
  --target-column "Diagnostic Result" \
  --selection-metric auprc \
  --target-sensitivity 0.98 \
  --beta 2.5 \
  --cv-folds 5 \
  --bootstraps 1000 \
  --outdir outputs/research_validation
```

This workflow:

- creates a stratified development/test split;
- keeps the final test set untouched during candidate-model selection;
- generates out-of-fold development probabilities;
- selects the candidate model by development OOF AUPRC by default;
- locks the probability threshold on development predictions only;
- supports a prespecified 98% sensitivity research target;
- computes F-beta with configurable beta (default 2.5);
- evaluates the locked model once on the held-out test set;
- produces stratified bootstrap confidence intervals;
- exports decision-curve net-benefit data;
- writes a validation manifest for reproducibility.

A high development sensitivity target is **not** a promise that the same sensitivity will be achieved on external patients. Held-out and external performance must be reported separately.

### Optional leakage-safe univariate filtering

A univariate filter can be enabled as a sensitivity analysis:

```bash
--univariate-alpha 0.30
```

The filter is fitted inside each cross-validation fold. Do not pre-filter the complete dataset before cross-validation.

---

## Research-validation outputs

`run_research_validation.py` writes:

- `development_model_selection.json` — OOF candidate-model comparison
- `development_missingness.csv` — development-set missingness audit
- `locked_operating_point.json` — threshold chosen before test evaluation
- `best_model.joblib` — selected pipeline refitted on the development split
- `test_metrics.json` — held-out point estimates
- `test_metric_intervals.json` — stratified bootstrap confidence intervals
- `test_threshold_table.csv` — secondary threshold sweep
- `test_confusion_matrices.json` — threshold-specific confusion matrices
- `decision_curve.csv` — model/treat-all/treat-none net benefit
- `feature_importance_test.csv` — permutation importance when available
- `validation_manifest.json` — experiment design and locked choices

---

## Sanitize and summarize data

### Sanitize a local dataset

```bash
python scripts/sanitize_datasets.py --input data/raw --output data/public_sanitized
```

### Summarize a sanitized dataset

```bash
python scripts/summarize_datasets.py \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe
```

---

## Evaluate a saved model

```bash
python scripts/evaluate_model.py \
  --model outputs/demo/best_model.joblib \
  --data data/public_sanitized/<dataset_file.csv> \
  --target-column severe \
  --outdir outputs/eval
```

For publication-oriented experiments, prefer the locked test evaluation generated by `run_research_validation.py` rather than reusing the test set for repeated threshold tuning.

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

```text
PenuX-AP-Severity/
├── src/penux_ap/
│   ├── config.py
│   ├── datasets.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── labels.py
│   ├── models.py
│   ├── calibration.py
│   ├── evaluation.py
│   ├── research_validation.py   # locked-threshold + CI + DCA utilities
│   ├── explainability.py
│   ├── clinical_scores.py
│   ├── leadtime.py
│   └── utils.py
├── api/
├── scripts/
│   ├── run_baseline.py
│   └── run_research_validation.py
├── data/
│   ├── public_sanitized/
│   └── mimic/sql/
├── docs/
│   └── validation_plan.md
├── notebooks/
├── tests/
└── outputs/
```

---

## Validation principles

The preferred research workflow follows these principles:

- prespecify the target outcome and prediction horizon;
- separate development from final test evaluation;
- keep imputation, encoding, scaling, and optional feature filtering inside the pipeline;
- select the model on development data rather than the final test set;
- lock the operating threshold before test evaluation;
- report AUPRC and AUROC with uncertainty;
- report sensitivity, specificity, PPV, NPV, F1/F-beta, calibration, and confusion matrices;
- assess clinical utility with decision-curve analysis only after checking calibration and defining a plausible action threshold;
- compare against established clinical scores only when timing and required variables make the comparison valid;
- perform temporal and external validation before any claim of transportability.

See `docs/validation_plan.md` for the detailed protocol.

---

## Reporting standards

Prediction-model reporting should use the current AI-specific extensions where applicable:

- **TRIPOD+AI** for transparent reporting of prediction-model studies using regression or machine-learning methods;
- **PROBAST+AI** for structured assessment of risk of bias and applicability.

The original TRIPOD statement remains historically important, but new AI prediction-model work should use the updated framework.

---

## Ethical & Legal Notes

- This is a retrospective model-development research project
- No patient identifiers should be stored or committed
- Real hospital data requires local Helsinki / IRB approval as applicable
- MIMIC-IV requires PhysioNet credentialing and a signed DUA
- The software is not validated for clinical use
- Retrospective model performance does not establish clinical safety or benefit
- See `docs/helsinki_irb_notes.md` for IRB submission guidance

## Limitations

- Performance depends heavily on cohort definition, case mix, prevalence, and measurement workflow
- Atlanta 2012 SAP labels require careful operationalization from EHR data
- Missingness may be clinically informative and may differ across institutions
- A 98% sensitivity target can substantially reduce specificity and PPV
- Feature importance is descriptive and should not be interpreted causally
- External validation has not yet established transportability
- Small subgroup sizes may produce unstable estimates
- Classical score benchmarking requires complete and temporally valid inputs

## Citation

If you use this software in your research, please cite it:

```text
Stern, N. PenuX-AP-Severity [Software].
https://github.com/netanelcyber/penuX
```

Or see `CITATION.cff`.
