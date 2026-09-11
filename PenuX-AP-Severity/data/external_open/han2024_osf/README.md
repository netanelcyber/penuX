# Han et al. 2024 Acute Pancreatitis Severity Dataset (OSF)

Public source for an independent acute-pancreatitis severity cohort with laboratory and diagnosis labels.

## Study

Han X, Hu M-n, Ji P, Liu Y-f. *Construction and validation of a severity prediction model for acute pancreatitis based on CT severity index: A retrospective case-control study.* PLOS ONE. 2024;19(5):e0303684.

- Article: https://doi.org/10.1371/journal.pone.0303684
- Public data repository: https://osf.io/m9ckf/
- Study period: June 2019 to June 2023
- Institution: Hefei Third Clinical College of Anhui Medical University / Third People's Hospital of Hefei City
- Development cohort: 200 AP patients
  - NSAP: 135
  - SAP: 65
- Validation cohort: 60 AP patients
- Severity definition: Revised Atlanta Classification
  - Mild + moderately severe AP are grouped as NSAP in the study
  - SAP is the positive outcome

## Available predictor families reported by the study

### Laboratory / blood data

- WBC
- RDW
- neutrophil percentage
- NLR
- fasting blood glucose
- amylase
- LDH
- BUN
- albumin
- creatinine
- D-dimer
- fibrinogen

### Demographics / clinical variables

- age
- sex
- BMI
- diabetes history
- etiology
- APACHE II
- BISAP

### Imaging-derived variables

- pleural effusion
- ascites
- CTSI

## PenuX use

This source is useful in two distinct experiments:

1. **Labs-only replication** — restrict predictors to admission laboratory measurements and demographics that are available in the hospital data contract.
2. **Labs + clinical/imaging replication** — additionally include ascites, pleural effusion, CTSI and clinical scores where available.

Do not use CTSI, APACHE II, BISAP or post-baseline variables in a model advertised as admission-labs-only.

The primary PenuX target should be normalized to:

```text
0 = NSAP
1 = SAP
```

## Data acquisition and redistribution

The PLOS ONE article states that all data relevant to the manuscript are available from the OSF project above. The article itself is distributed under CC BY 4.0. Before mirroring any patient-level OSF file into this repository, verify the license shown on the OSF file/project itself and retain its original attribution and provenance metadata.

For that reason, this repository records the source and provides an acquisition helper but does not silently copy or relicense the OSF files.

## Independence note

This cohort originates from Hefei, Anhui, China and is institutionally distinct from the two Guilin datasets already used by PenuX-AP-Severity. That makes it substantially more useful for transportability analysis than treating the two Guilin datasets as independent external validation cohorts.

## Research-use warning

This dataset and all PenuX analyses are for research only. Results must not be used to make patient-care decisions without appropriate prospective clinical validation, governance and regulatory review.
