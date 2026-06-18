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

*(None — add your dataset here after sanitization)*

## Public AP Datasets (Known)

Several AP datasets have been published on Kaggle, Zenodo, and Figshare.
Before using any public dataset:
- Confirm the license permits research use
- Confirm it is fully de-identified
- Remove any remaining identifier columns using the sanitization script
- Document provenance here
