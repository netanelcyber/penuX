# MIMIC-IV / PhysioNet Access Guide

## Access Requirements

MIMIC-IV is a restricted-access clinical database managed by MIT Laboratory for Computational Physiology via PhysioNet.

### Steps to Gain Access

1. Register at https://physionet.org/register/
2. Complete the required training:
   - CITI Program "Data or Specimens Only Research" course (recommended)
   - Or equivalent institutional human-subjects training
3. Apply for credentialed access at https://physionet.org/content/mimiciv/
4. Sign the MIMIC-IV Data Use Agreement (DUA)
5. Download the dataset after approval

**Do NOT share, redistribute, or commit MIMIC patient-level data.**

## Cohort Extraction

Use the SQL scripts in `data/mimic/sql/`:

| Script | Purpose |
|--------|---------|
| `01_ap_cohort.sql` | AP admissions via ICD-9 577.0 / ICD-10 K85.* |
| `02_first_24h_labs.sql` | First-24h lab values |
| `03_first_24h_vitals.sql` | First-24h vital signs |
| `04_outcomes.sql` | Outcome proxy variables |

## ICD Codes for Acute Pancreatitis

- **ICD-9**: `577.0` (Acute pancreatitis)
- **ICD-10**: `K85.*` (Acute pancreatitis, all subtypes)
  - K85.0 — Idiopathic
  - K85.1 — Biliary
  - K85.2 — Alcoholic
  - K85.3 — Drug-induced
  - K85.8 — Other specified
  - K85.9 — Unspecified

## Outcome Operationalization

MIMIC-IV does **not** contain direct Atlanta 2012 SAP labels. Possible proxies:
- ICU admission within 48h
- Mechanical ventilation
- Vasopressor use
- Renal replacement therapy (RRT)
- In-hospital mortality

**Clinical expert review is required to operationalize SAP (persistent organ failure >48h).**

## Data Governance

- Do not store MIMIC files in this repository
- Store locally on a compliant institutional server
- Access controlled to study team only
- Delete raw data when no longer needed per DUA terms
