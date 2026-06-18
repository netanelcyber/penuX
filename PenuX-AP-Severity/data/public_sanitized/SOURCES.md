# Public Sanitized Dataset Sources

All datasets here have been sanitized (identifier columns removed) and are
used under their original open-source licenses.

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

## Compliance Notes

- Both datasets are published under open-source licenses (MIT / Apache-2.0)
- Both datasets originate from published peer-reviewed research
- Direct patient identifiers (name, serial number) were removed before storage here
- No re-identification is attempted or possible from these files
- These datasets are for research use only and do not represent a clinical tool
