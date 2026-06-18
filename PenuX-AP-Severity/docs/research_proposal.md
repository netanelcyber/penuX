# Research Proposal

## Placeholder

The full research proposal document should be placed at:

```
docs/research_proposal.docx
```

No proposal document was found during repository setup.

## Study Summary (Template)

**Title**: Early Prediction of Severe Acute Pancreatitis Using Routine Admission Data:
A Machine Learning Approach

**Background**: Severe Acute Pancreatitis (SAP), defined as persistent organ failure >48h
per the 2012 Revised Atlanta Classification, affects approximately 10–20% of AP patients
and carries significant morbidity and mortality. Early identification of patients at risk
for SAP could improve triage, resource allocation, and clinical outcomes.

**Objective**: Develop and internally validate a machine learning model to predict SAP
within the first 24 hours of hospital admission using routinely collected clinical variables.

**Design**: Retrospective observational cohort study.

**Population**: Adult patients admitted with a primary diagnosis of acute pancreatitis.

**Outcome**: SAP (persistent organ failure >48h).

**Predictors**: Age, sex, BMI, vital signs, and routine laboratory values available
within the first 24 hours of admission.

**Methods**: Logistic regression, random forest, and gradient boosting classifiers.
Internal validation via stratified cross-validation and bootstrapping.
Calibration assessment. Benchmarking against BISAP, APACHE II, Ranson, Modified CTSI.

**Ethics**: Retrospective, non-interventional. Requires local Helsinki/IRB approval.
See `docs/helsinki_irb_notes.md`.
