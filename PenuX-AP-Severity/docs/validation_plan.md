# Validation Plan

## Internal Validation

- **Stratified train/test split**: 80/20, stratified by SAP outcome
- **Repeated stratified k-fold cross-validation**: k=5 or k=10, repeated 3×
- **Bootstrapping**: 1000 iterations for AUROC confidence intervals
- **Calibration**: reliability curves, Brier score, ECE
- **Threshold analysis**: sensitivity/specificity/PPV/NPV at multiple thresholds
- **Confusion matrices**: reported at each decision threshold (including 4-hour horizon intervals if longitudinal data available)

## Primary Metric

**AUPRC (Area Under Precision-Recall Curve)** is the primary metric, because SAP is a minority outcome (typically 10–20% of AP admissions). AUROC is reported as a secondary metric.

## Calibration

- Sigmoid (Platt) calibration on a held-out calibration set
- Isotonic calibration (larger datasets)
- Reliability curves (10-bin)
- Brier score

## Clinical Score Benchmarking

When required fields are available, benchmark against:
- BISAP (≥3 predicts severe AP)
- APACHE II (≥8 associated with severe AP)
- Ranson criteria (≥3 at 48h)
- Modified CTSI

## Confusion Matrix Reporting

Confusion matrices are generated:
- At multiple decision thresholds (0.1 to 0.9 in 0.1 steps)
- At each available temporal horizon (admission, 6h, 12h, 24h)
- Saved as `confusion_matrices.json` in the output directory

## Decision-Curve Analysis (Planned)

Decision-curve analysis (DCA) will be used to assess net clinical benefit across a range of threshold probabilities. Implementation pending.

## External Validation (Future Work)

- Prospective validation cohort at the same institution
- External validation at an independent institution
- Temporal validation (different time period)

## TRIPOD-Style Reporting

This study will be reported following the TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) checklist:
- Study design
- Participants
- Outcome
- Predictors
- Sample size
- Missing data handling
- Statistical analysis
- Model development
- Model performance
- Calibration
- Limitations

Reference: Collins GS et al. BMJ. 2015;350:g7594.
