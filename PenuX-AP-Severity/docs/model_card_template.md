# Model Card — PenuX-AP-Severity

## Intended Use

**Research use only.**
This model is intended for retrospective prediction research on Severe Acute Pancreatitis.

**NOT intended for:**
- Clinical decision-making
- Real-time patient triage
- Replacing clinical judgment
- Any deployment in a hospital information system

## Population

- Adult patients (≥18 years) admitted with acute pancreatitis
- Data available within first 24 hours of admission

## Predictors

Routine admission variables: age, sex, BMI, heart rate, blood pressure, respiratory rate,
temperature, SpO2, WBC, CRP, BUN, creatinine, calcium, glucose, hematocrit, LDH, AST,
ALT, albumin, triglycerides.

## Outcome

**Severe Acute Pancreatitis (SAP)**: persistent organ failure >48 hours,
per the 2012 Revised Atlanta Classification.

## Performance

*(To be filled in after training on a real validated dataset)*

| Metric | Value (95% CI) |
|--------|----------------|
| AUROC  | TBD |
| AUPRC  | TBD |
| Sensitivity at threshold 0.5 | TBD |
| Specificity at threshold 0.5 | TBD |
| Brier score | TBD |

## Calibration

*(To be filled in after calibration)*

## Limitations

- Performance depends on dataset quality and cohort composition
- External validation has not been performed
- Small cohort sizes may limit generalizability
- Missing features reduce prediction reliability
- Not validated for non-adult populations

## Ethical Considerations

- Model must not be used for patient care without prospective validation and regulatory approval
- Requires local Helsinki/IRB approval before use with hospital data
- Predictions may reflect biases present in training data
- Disparate performance across subgroups has not been assessed

## Data Provenance

*(Describe training dataset here: source, size, time period, institution, de-identification method)*

## Validation Status

- [ ] Internal validation complete
- [ ] Calibration complete
- [ ] External validation complete
- [ ] Clinical validation complete
- [ ] Regulatory approval obtained
