# PenuX-AP-Severity Dataset Sources

PenuX-AP-Severity uses a **multi-cohort design** rather than treating one file as the full evidence base. The first two datasets are mirrored in sanitized form. Additional cohorts are documented with their access and redistribution constraints.

> The current planning total is **3,017 nominal AP records across four core cohorts**. This is not yet a confirmed count of unique eligible patients because the two Guilin cohorts may overlap and the eICU cohort requires PenuX-specific eligibility and outcome derivation.

---

## 1. ap_multiml_sanitized.csv

| Field | Value |
|-------|-------|
| Source repo | https://github.com/longshike/Predicting-acute-pancreatitis-severity-with-multi-machine-learning-models |
| Original file | data_V7.0-non-normalize.xlsx |
| License | MIT |
| Origin | Second Affiliated Hospital of Guilin Medical University, 2016–2024 |
| N records | 1,289 |
| N SAP | 204 (15.8%) |
| N non-SAP | 1,085 |
| Features | 60 after identifier removal |
| Raw target | `Diagnostic Result`: **0=SAP, 1=non-SAP** |
| PenuX normalized target | **0=non-SAP, 1=SAP** after explicit inversion |

**Important target-coding correction.** The source repository's `model_construction.py` explicitly performs `y = 1 - y` with the comment that mild disease is normalized to 0 and severe disease to 1. Therefore the raw `Diagnostic Result` field must be inverted before training a PenuX SAP classifier. Do not pass the raw numeric target through a generic 0/1 binarizer without this normalization.

Recommended role: primary development and internal validation cohort.

---

## 2. ap_lnn_sanitized.csv

| Field | Value |
|-------|-------|
| Source repo | https://github.com/longshike/LNN-for-SAP-Prediction |
| Original file | zhenglishuju_v1.0.xlsx |
| License | Apache-2.0 |
| Origin | Second Affiliated Hospital of Guilin Medical University, 2020–2024 |
| N records | 722 |
| N SAP | 137 (19.0%) |
| N non-SAP | 585 |
| Features | 107 after identifier removal |
| Target | `严重程度` (0=non-SAP, 1=SAP) |

Recommended role: related-cohort sensitivity analysis. Do **not** describe it as institutionally independent external validation because it originates from the same hospital as the Multi-ML cohort and the study periods overlap.

---

## 3. Han et al. 2024 / OSF

Detailed provenance: `data/external_open/han2024_osf/README.md`

| Field | Value |
|-------|-------|
| Article | https://doi.org/10.1371/journal.pone.0303684 |
| Public repository | https://osf.io/m9ckf/ |
| Origin | Hefei Third Clinical College of Anhui Medical University / Third People's Hospital of Hefei City |
| Development cohort | 200 AP patients |
| Validation cohort | 60 AP patients |
| Nominal total | 260 |
| Development labels | 135 NSAP, 65 SAP |
| Severity definition | Revised Atlanta Classification |
| Laboratory variables | WBC, RDW, neutrophil %, NLR, glucose, amylase, LDH, BUN, albumin, creatinine, D-dimer, fibrinogen |
| Additional variables | age, sex, BMI, diabetes, etiology, APACHE II, BISAP, pleural effusion, ascites, CTSI |

Recommended role: institutionally independent transportability/external validation cohort. For the primary labs-only experiment, exclude imaging-derived variables and clinical scores that are not part of the intended early prediction window.

---

## 4. eICU-CRD acute-pancreatitis cohort

Detailed governance and cohort plan: `data/credentialed/eicu/README.md`

| Field | Value |
|-------|-------|
| Dataset | eICU Collaborative Research Database v2.0 |
| Provider | PhysioNet / MIT Laboratory for Computational Physiology |
| DOI | https://doi.org/10.13026/C2WM1R |
| Setting | 208 US hospitals, ICU-enriched |
| Published AP candidate count | 746 |
| Available data | laboratory measurements, vital signs, APACHE components, admission diagnoses, time-stamped diagnoses, medications and treatments |
| Access | credentialed PhysioNet user + training + DUA |
| Raw-data redistribution | prohibited by project governance; do not commit patient-level files |

The published 746-patient count comes from an eICU acute-pancreatitis cohort study: https://pubmed.ncbi.nlm.nih.gov/33361165/ . That study used mortality as its outcome; PenuX must recompute its own analytic cohort and must **not** relabel mortality as SAP.

For the PenuX severity endpoint, derive an Atlanta-compatible persistent-organ-failure outcome (>48 h) from time-stamped organ-failure variables where feasible. Final eligible N is therefore expected to differ from 746.

Recommended role: multi-center US transportability and time-aware stress test.

---

## Core multi-cohort planning total

| Cohort | Nominal records | Primary role |
|---|---:|---|
| Guilin Multi-ML | 1,289 | Development/internal validation |
| Guilin LNN | 722 | Related-cohort sensitivity analysis |
| Hefei / Han et al. | 260 | Independent-institution validation |
| eICU AP candidate cohort | 746 | Multicenter US stress test |
| **Nominal total** | **3,017** | **Multi-cohort research program** |

### Interpretation of 3,017

`3,017` is a **source-record planning total**, not a confirmed unique-patient sample size. Publication-ready reporting must provide source records, exclusions with reasons, final analytic N per cohort, SAP/non-SAP prevalence, overlap assessment, and per-cohort performance before pooled summaries.

Because the two Guilin datasets come from the same institution and overlapping periods, their counts should never be presented as guaranteed unique patients unless overlap is resolved from source-level provenance.

---

## Auxiliary dataset: MIMIC-IV-Ext Clinical Decision Making

Source: https://physionet.org/content/mimic-iv-ext-cdm/1.1/

This derived MIMIC-IV resource contains **2,400 abdominal-pathology cases, including 538 pancreatitis cases**, and extensive laboratory results plus physician diagnoses. It is useful for diagnosis-oriented experiments, laboratory-schema mapping and domain adaptation.

It is **not included in the 3,017 core SAP total** because it does not provide a ready-made Atlanta-compatible SAP label. It may enter the severity benchmark only after a compatible outcome is derived under the source data governance rules.

---

## Cross-cohort modeling rules

- Preserve a `cohort_id` for every record.
- Harmonize units and feature names before training.
- Normalize each source's outcome coding explicitly before modeling.
- Use only predictors available before the intended prediction time.
- Never use future organ-failure measurements as early predictors when those measurements contribute to the SAP outcome.
- Tune XGBoost only in development folds/cohorts.
- Lock the high-sensitivity threshold (target ≥98%) before final external evaluation.
- Report AUROC, AUPRC, sensitivity, specificity, PPV, NPV, Brier score and calibration per cohort.
- Prefer leave-one-cohort-out or explicitly external validation over a pooled random split.
- Report pooled metrics only after cohort-specific heterogeneity is shown.

---

## Compliance notes

- Preserve the original license and attribution for every source.
- Open-access publications do not automatically grant unrestricted redistribution of linked patient-level files.
- Never commit credentialed PhysioNet/eICU/MIMIC patient-level data.
- Only code, schema mappings and aggregate non-identifying results may be committed for credentialed datasets.
- All datasets, models and outputs are for **research use only** and do not constitute a clinical tool.
