# Dataset Sources

## Rules

1. Only legally redistributable or privately held datasets may be used.
2. Sanitize all datasets before use: `python scripts/sanitize_datasets.py`
3. Document provenance and license for every dataset below.
4. Do not commit restricted-access data (MIMIC, hospital EHR) to this repository.

## Documenting a Dataset

For each dataset added to `data/public_sanitized/`, add an entry here:

| Field | Value |
|-------|-------|
| Dataset name | |
| Source URL | |
| License | |
| De-identification method | |
| Columns removed | |
| Date accessed | |
| N patients | |
| N SAP cases | |

## Currently Registered Datasets

### 1. ap_multiml_sanitized.csv
- **Source**: https://github.com/longshike/Predicting-acute-pancreatitis-severity-with-multi-machine-learning-models
- **License**: MIT
- **N**: 1,289 patients | 204 SAP (15.8%)
- **Target**: `Diagnostic Result` (**raw 0=SAP, raw 1=non-SAP** — reversed vs. the usual convention; see `data/public_sanitized/SOURCES.md` for details and the required flip when computing threshold-based metrics)
- **Features**: 60 clinical variables (labs + vitals)
- **Identifiers removed**: `ID No.`, `Name`

### 2. ap_lnn_sanitized.csv
- **Source**: https://github.com/longshike/LNN-for-SAP-Prediction
- **License**: Apache-2.0
- **N**: 722 patients | 137 SAP (19.0%)
- **Target**: `严重程度` (**raw 0=SAP, raw 1=non-SAP** — reversed, same caveat as the multiml dataset)
- **Features**: 107 clinical variables (column names in Chinese)
- **Identifiers removed**: `序号` (serial no.), `姓名` (name)

## Public AP Datasets (Known)

Several AP datasets have been published on Kaggle, Zenodo, and Figshare.
Before using any public dataset:
- Confirm the license permits research use
- Confirm it is fully de-identified
- Remove any remaining identifier columns using the sanitization script
- Document provenance here
