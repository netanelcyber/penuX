# eICU-CRD Acute Pancreatitis Cohort

This directory documents a **credentialed public-data** extension for PenuX-AP-Severity. Raw eICU-CRD patient-level files must not be committed to this repository.

## Source

- Dataset: eICU Collaborative Research Database v2.0
- Provider: PhysioNet / MIT Laboratory for Computational Physiology
- DOI: https://doi.org/10.13026/C2WM1R
- Access: credentialed PhysioNet account, required human-research training, and signed Data Use Agreement
- Coverage: >200,000 ICU unit encounters from 208 US hospitals (2014-2015)
- Available data families: laboratory measurements, vital signs, APACHE components, admission diagnoses, time-stamped diagnoses, medications, treatments and care-plan variables

## Acute pancreatitis cohort size

A published eICU-CRD acute-pancreatitis study identified **746 AP patients** before additional completeness exclusions. This number is used only as a planning estimate for the PenuX multi-cohort target; the final PenuX analytic N must be recomputed after applying the PenuX eligibility, timing, missingness and outcome-label rules.

Published reference: https://pubmed.ncbi.nlm.nih.gov/33361165/

## Why this cohort is valuable

The existing public PenuX AP datasets are predominantly single-institution Chinese cohorts. eICU provides a geographically and institutionally different, multi-center US ICU population and is therefore useful for:

1. transportability testing;
2. stress-testing performance under a higher-acuity case mix;
3. validating common laboratory predictors across institutions;
4. testing time-aware deterioration models when timestamped features are used.

## Target definition

The primary PenuX target remains:

```text
0 = non-SAP
1 = SAP
```

For eICU, this label must **not** be inferred from mortality or ICU admission alone. A PenuX extraction should derive an Atlanta-compatible severe outcome using persistent organ failure lasting >48 h, based on the Modified Marshall organ-failure domains when the required time-stamped variables are available.

A mortality endpoint may be reported separately as a secondary outcome, but it must never be mixed with the SAP label.

## Candidate early predictors

Prefer variables available at admission or within the prespecified early window, including where available:

- WBC
- hematocrit / hemoglobin
- platelet count
- BUN / urea
- creatinine
- glucose
- calcium
- sodium / potassium
- albumin
- bilirubin
- AST / ALT
- LDH
- INR / PT / APTT
- lactate
- PaO2 / FiO2-related respiratory variables
- heart rate, respiratory rate, blood pressure, temperature, oxygen saturation
- age and sex

Do not use future organ-failure measurements, interventions, or post-outcome information as predictors for an early-warning model.

## Planned cohort accounting

Nominal source-cohort counts before overlap and eligibility checks:

| Cohort | Nominal records |
|---|---:|
| Guilin Multi-ML | 1,289 |
| Guilin LNN | 722 |
| Hefei / Han et al. OSF | 260 |
| eICU AP candidate cohort | 746 |
| **Nominal total** | **3,017** |

### Important caveats

- **3,017 is not yet a confirmed unique-patient analytic N.**
- The two Guilin cohorts may overlap because they come from the same institution and overlapping calendar periods; identifiers have been removed, so cross-file deduplication may not be possible.
- eICU is ICU-enriched and therefore has a different severity spectrum from general-hospital cohorts.
- The final multi-cohort report must publish both the nominal source count and the post-eligibility analytic count for every cohort.
- Cohort identity must be retained in validation; do not simply concatenate all rows and perform a random split.

## Recommended validation design

1. Train/tune only inside development cohorts.
2. Use grouped or leave-one-cohort-out validation where possible.
3. Lock the XGBoost hyperparameters and the 98% sensitivity threshold before external evaluation.
4. Report per-cohort AUROC, AUPRC, sensitivity, specificity, PPV, NPV, Brier score and calibration.
5. Report a pooled estimate only after the cohort-specific results, with heterogeneity clearly shown.

## Data governance

The eICU files are governed by the PhysioNet Credentialed Health Data License and DUA. Raw data must remain outside this public GitHub repository. Only code, SQL, schema mappings and aggregate non-identifying results should be committed.

**RESEARCH USE ONLY. Not for clinical decision-making.**
