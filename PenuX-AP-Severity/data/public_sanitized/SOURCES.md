# Public Sanitized Dataset Sources

The first two datasets below are currently mirrored in sanitized form in this repository. Additional public sources are documented separately and are not mirrored until their source-file license and provenance are verified.

---

## 1. ap_multiml_sanitized.csv

| Field | Value |
|-------|-------|
| Source repo | https://github.com/longshike/Predicting-acute-pancreatitis-severity-with-multi-machine-learning-models |
| Original file | data_V7.0-non-normalize.xlsx |
| License | MIT |
| Origin | Second Affiliated Hospital of Guilin Medical University, 2016–2024 |
| N patients | 1,289 |
| N SAP (label=1) | 204 (15.8%) |
| N non-SAP (label=0) | 1,085 |
| Features | 60 (after identifier removal) |
| Target column | `Diagnostic Result` (0=non-SAP, 1=SAP) |
| Identifiers removed | `ID No.`, `Name` |
| Sanitized by | penux_ap.datasets.sanitize_identifiers |

### Usage
```bash
python scripts/run_baseline.py \
  --data data/public_sanitized/ap_multiml_sanitized.csv \
  --target-column "Diagnostic Result" \
  --outdir outputs/multiml
```

---

## 2. ap_lnn_sanitized.csv

| Field | Value |
|-------|-------|
| Source repo | https://github.com/longshike/LNN-for-SAP-Prediction |
| Original file | zhenglishuju_v1.0.xlsx |
| License | Apache-2.0 |
| Origin | Second Affiliated Hospital of Guilin Medical University, 2020–2024 |
| N patients | 722 |
| N SAP (label=1) | 137 (19.0%) |
| N non-SAP (label=0) | 585 |
| Features | 107 (after identifier removal) |
| Target column | `严重程度` (severity; 0=non-SAP, 1=SAP) |
| Identifiers removed | `序号` (serial number), `姓名` (name) |
| Sanitized by | penux_ap.datasets.sanitize_identifiers |
| Note | Column names are in Chinese. Use with `--target-column 严重程度` |

### Usage
```bash
python scripts/run_baseline.py \
  --data data/public_sanitized/ap_lnn_sanitized.csv \
  --target-column "严重程度" \
  --outdir outputs/lnn
```

---

## 3. Han et al. 2024 — OSF acute-pancreatitis severity cohort

This source is documented under `data/external_open/han2024_osf/README.md` and is intentionally not silently mirrored into the repository.

| Field | Value |
|-------|-------|
| Article | https://doi.org/10.1371/journal.pone.0303684 |
| Public data repository | https://osf.io/m9ckf/ |
| Origin | Hefei Third Clinical College of Anhui Medical University / Third People's Hospital of Hefei City |
| Development cohort | 200 AP patients |
| Development labels | 135 NSAP, 65 SAP |
| Validation cohort | 60 AP patients |
| Severity definition | Revised Atlanta Classification |
| Laboratory variables reported | WBC, RDW, neutrophil %, NLR, fasting glucose, amylase, LDH, BUN, albumin, creatinine, D-dimer, fibrinogen |
| Additional clinical variables | age, sex, BMI, diabetes, etiology, APACHE II, BISAP |
| Imaging-derived variables | pleural effusion, ascites, CTSI |
| Recommended PenuX target | `0 = NSAP`, `1 = SAP` |

### Why this source matters

Unlike the two Guilin datasets, this cohort comes from a different institution and geographic setting. It is therefore a better candidate for cross-cohort transportability experiments once the exact OSF patient-level file and its redistribution terms are verified.

For a strict admission-laboratory experiment, exclude CTSI, ascites, pleural effusion, APACHE II and BISAP and evaluate only predictors available at the intended prediction time.

---

## Additional EHR source worth supporting (not fully open)

**MIMIC-IV-Ext Clinical Decision Making** contains 2,400 abdominal-pathology cases, including 538 pancreatitis cases, with laboratory tests and physician discharge diagnoses. It is useful for diagnosis-oriented experiments, but access is credentialed and governed by the PhysioNet DUA, so its files must never be committed or redistributed from this repository.

Source: https://physionet.org/content/mimic-iv-ext-cdm/

---

## Compliance Notes

- Preserve the original license and attribution for every source.
- Do not assume that an open-access article automatically grants redistribution rights for every linked patient-level file; verify the repository/file license before mirroring.
- Direct patient identifiers must be removed before any local derivative is committed.
- Never commit credentialed MIMIC/PhysioNet patient-level data.
- Do not treat the two Guilin datasets as fully independent external validation cohorts because they originate from the same institution and overlapping calendar periods.
- All datasets and outputs are for research use only and do not constitute a clinical tool.
