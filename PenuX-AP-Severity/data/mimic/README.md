# MIMIC-IV Data

This directory contains SQL extraction scripts for MIMIC-IV only.
**No patient-level data is stored here.**

## Access Requirements

MIMIC-IV is a restricted-access resource managed by PhysioNet:
1. Register at https://physionet.org
2. Complete credentialing and required training (e.g., CITI Program)
3. Sign the MIMIC-IV Data Use Agreement (DUA)
4. Download from https://physionet.org/content/mimiciv/

**Do NOT commit any MIMIC patient-level data to this repository.**

## SQL Scripts

| File | Purpose |
|------|---------|
| `sql/01_ap_cohort.sql` | Identify AP admissions (ICD-9: 577.0, ICD-10: K85.*) |
| `sql/02_first_24h_labs.sql` | Extract first-24h laboratory values |
| `sql/03_first_24h_vitals.sql` | Extract first-24h vital signs |
| `sql/04_outcomes.sql` | Extract candidate outcome variables |

## Usage

Run scripts against a local MIMIC-IV PostgreSQL instance:
```bash
psql -U your_user -d mimic -f data/mimic/sql/01_ap_cohort.sql
```

## Notes

- MIMIC-IV does not contain direct Atlanta 2012 SAP labels
- Outcome operationalization (persistent organ failure >48h) requires clinical expert review
- SQL itemids may vary between MIMIC-IV versions — validate against `d_items` / `d_labitems`
