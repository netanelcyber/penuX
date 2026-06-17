# Helsinki / IRB Notes

This document provides guidance for obtaining ethical approval for retrospective
clinical prediction model studies using this software.

## Study Design

- **Type**: Retrospective, non-interventional, observational cohort study
- **Data source**: Existing de-identified clinical records
- **No patient contact**: Patients are not recruited, contacted, or consented
- **No effect on treatment**: The model is not used to guide care during the study
- **Data minimization**: Only variables required for prediction and outcome definition are extracted

## Suggested Language for Helsinki / IRB Submission

> "This is a retrospective, non-interventional model-development study using existing
> de-identified clinical records. The model will not be used to guide patient care
> during the study. All data will be handled in compliance with applicable
> data-protection regulations. No identifying information will be stored or published."

## Key Points for Submission

1. **Retrospective design** — uses existing records only, no prospective data collection
2. **De-identification** — all direct identifiers removed before analysis
3. **No clinical deployment** — results are for research purposes only
4. **Secure storage** — data stored on institutional servers with access controls
5. **Data minimization** — only minimum required variables extracted
6. **Publication plan** — results may be submitted to a peer-reviewed journal; no individual data published
7. **Outcome definition** — SAP defined as persistent organ failure >48h per Atlanta 2012 classification

## Checklist

- [ ] Local IRB / Helsinki committee application submitted
- [ ] Data protection officer (DPO) notified if required
- [ ] Data Use Agreement signed for any restricted dataset (e.g. MIMIC-IV)
- [ ] De-identification verified by independent reviewer
- [ ] Secure data transfer protocol documented
- [ ] Analysis pre-registered (recommended)

## Reference Standards

- Declaration of Helsinki (2013 revision)
- GDPR Article 89 (research exemption)
- HIPAA Safe Harbor / Expert Determination
- TRIPOD reporting guidelines for prediction models
