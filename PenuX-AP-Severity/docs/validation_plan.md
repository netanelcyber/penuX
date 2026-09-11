# Validation Plan

## Purpose and scope

PenuX-AP-Severity is a **research-use-only** prediction-model development project for early identification of patients who may progress to Severe Acute Pancreatitis (SAP). The validation workflow is designed to estimate model performance without using the final held-out test set for model selection, feature filtering, probability-threshold selection, or hyperparameter decisions.

The primary outcome should follow the 2012 Revised Atlanta Classification: **persistent organ failure lasting more than 48 hours**. Any site-specific operationalization of this endpoint must be documented before model fitting.

## Core anti-leakage design

The recommended workflow is implemented in `scripts/run_research_validation.py`:

1. Perform a stratified 80/20 development/test split.
2. Keep the 20% test split untouched during model development.
3. Inside the development split, generate out-of-fold (OOF) probabilities using stratified k-fold cross-validation.
4. Compare candidate models using **development OOF AUPRC** by default.
5. Select and lock the operating threshold using **development OOF predictions only**.
6. Refit the selected pipeline on the complete development split.
7. Evaluate once on the held-out test set using the locked model identity and locked threshold.
8. Report uncertainty and exploratory clinical-utility analyses on the untouched test set.

This structure prevents the common form of optimistic bias that occurs when the final test set is repeatedly inspected while choosing the model or threshold.

## Model-selection metric

**Primary model-selection metric: AUPRC (Area Under the Precision-Recall Curve).**

AUPRC is preferred because SAP is usually a minority outcome in AP cohorts and precision-recall performance is sensitive to false-positive burden under class imbalance. AUROC remains an important secondary discrimination metric.

The script supports `--selection-metric auroc` for sensitivity analyses, but the metric should be prespecified rather than chosen after viewing results.

## High-sensitivity operating point

For a screening-oriented research operating point, the default target is:

- **Sensitivity target:** 0.98
- **F-beta:** 2.5

Use:

```bash
python scripts/run_research_validation.py \
  --data data/public_sanitized/ap_multiml_sanitized.csv \
  --target-column "Diagnostic Result" \
  --target-sensitivity 0.98 \
  --beta 2.5 \
  --outdir outputs/research_validation
```

The threshold is selected from **development OOF predictions**. Among thresholds that achieve the requested development sensitivity, the implementation chooses the threshold with the greatest specificity, then PPV, then the largest threshold as the final tie-breaker. The held-out test labels are not used to tune this operating point.

A 98% development sensitivity target is a research objective, not a guarantee of 98% sensitivity on new patients. The achieved held-out sensitivity and its confidence interval must be reported separately.

## Optional univariate pre-filter

A univariate pre-filter can be used for sensitivity analyses:

```bash
--univariate-alpha 0.30
```

This uses `SelectFpr(f_classif, alpha=0.30)` **inside the model pipeline**, so the feature filter is refit independently in each cross-validation fold. Do not pre-filter the complete dataset before splitting or cross-validation, because that would leak outcome information into validation folds.

The default workflow leaves this filter disabled.

## Candidate models

The repository currently supports, subject to installed optional dependencies:

- Logistic regression
- Random forest
- Histogram gradient boosting
- Multilayer perceptron
- XGBoost
- LightGBM

All candidate preprocessing steps must remain inside the fitted pipeline. Model comparison should use the same development folds whenever possible.

## Missing data

Current baseline preprocessing uses:

- Median imputation for numeric predictors
- Most-frequent imputation for categorical predictors
- Standardization of numeric variables
- One-hot encoding of categorical variables

Missingness must be summarized on the **development split** and reported per feature. The mechanism and clinical meaning of missingness should be reviewed with domain experts; imputation alone is not evidence that missing-at-random assumptions are valid.

For advanced analyses, consider missingness indicators, multiple imputation, and site-specific missingness audits, but these should be fitted without access to the held-out test outcomes.

## Internal validation

Recommended minimum internal validation:

- Stratified development/test split: 80/20
- Development OOF prediction: stratified 5-fold CV by default
- Repeat-CV or nested-CV sensitivity analysis when computationally feasible
- Bootstrap uncertainty on the final held-out test set
- Calibration assessment
- Prespecified subgroup analyses

For smaller cohorts, the number of CV folds is automatically limited by the development-set minority-class count.

## Uncertainty reporting

`bootstrap_metric_intervals()` reports stratified bootstrap confidence intervals for:

- AUROC
- AUPRC
- Brier score
- Accuracy
- Sensitivity
- Specificity
- PPV
- NPV
- F1
- F-beta

Positive and negative cases are resampled separately so each bootstrap replicate contains both outcome classes while preserving the observed class counts.

At minimum, publications and expert-review packages should report point estimates and 95% confidence intervals rather than point estimates alone.

## Calibration

Probability calibration should be assessed separately from discrimination.

Available tools include:

- Brier score
- Reliability-curve data
- Sigmoid/Platt calibration
- Isotonic calibration

Calibration models must be fitted using development/calibration data only. The final test set must not be used to fit or choose a calibration method.

Future deepening should add calibration slope, calibration-in-the-large, flexible calibration plots, and optimism-corrected calibration estimates.

## Decision-curve analysis

Decision-curve analysis is now implemented in `penux_ap.research_validation.decision_curve_analysis()` and exported by `run_research_validation.py` as `decision_curve.csv`.

The output reports:

- Model net benefit
- Treat-all net benefit
- Treat-none net benefit
- Threshold probability
- True-positive and false-positive counts

DCA is exploratory until probability calibration, intended clinical action, and plausible threshold-probability ranges have been reviewed with gastroenterology/acute-pancreatitis experts. Net benefit must not be presented as proof of clinical benefit without prospective clinical evaluation.

## Clinical-score benchmarking

Where all required variables and timing windows are available, compare ML models against established clinical scores, including:

- BISAP
- APACHE II
- Ranson criteria
- Modified CTSI

The comparison should use the same cohort and outcome definition. Scores that require information collected later than the ML prediction horizon must be clearly identified as non-contemporaneous comparators.

## Temporal horizons

For longitudinal datasets, evaluate predictions at prespecified horizons such as:

- Admission / earliest eligible measurement window
- 6 hours
- 12 hours
- 24 hours

Predictors collected after the specified horizon must not enter that horizon's model. Outcome-defining variables and post-outcome treatment information must be excluded from predictors.

Confusion matrices should be reported at the locked primary operating point and, secondarily, over a threshold sweep.

## External and temporal validation

Before any clinical-use claim, the project requires validation beyond the development cohort:

- Temporal validation in a later cohort from the same institution
- Geographic/external validation at an independent institution
- Performance by site and acquisition workflow
- Recalibration assessment under prevalence shift
- Subgroup performance with uncertainty

External validation should reuse the locked model and prespecified operating point whenever scientifically appropriate; re-tuning should be declared as model updating, not pure external validation.

## Prespecified subgroup analyses

Where sample size permits, evaluate at least:

- Sex
- Age bands
- Etiology of acute pancreatitis
- ICU vs non-ICU presentation context
- Admission year / temporal period
- Hospital/site for multicenter data
- Relevant comorbidity strata

Report subgroup sample size, outcome prevalence, discrimination, sensitivity, specificity, PPV/NPV, and uncertainty. Avoid overinterpreting small subgroups.

## Reproducibility outputs

`run_research_validation.py` saves:

- `development_model_selection.json`
- `development_missingness.csv`
- `locked_operating_point.json`
- `best_model.joblib`
- `test_metrics.json`
- `test_metric_intervals.json`
- `test_threshold_table.csv`
- `test_confusion_matrices.json`
- `decision_curve.csv`
- `feature_importance_test.csv` when available
- `validation_manifest.json`

The manifest records the selected model, selection metric, sample sizes, prevalence, CV folds, locked threshold, beta, optional feature filter, and confirms that the test set was not used for model/threshold selection.

## Reporting standards

Use **TRIPOD+AI** as the primary reporting framework for prediction-model studies using regression or machine-learning methods. The older 2015 TRIPOD statement is retained only as historical background.

Use **PROBAST+AI** during protocol design, internal review, and manuscript preparation to assess risk of bias and applicability across participants/data sources, predictors, outcome definition, and analysis.

For any later prospective evaluation of an AI decision-support workflow in real clinical practice, use an appropriate early-stage clinical-evaluation framework and clearly distinguish model-performance validation from clinical-effectiveness evaluation.

## Minimum publication table

For the primary held-out evaluation, report:

- Cohort N and SAP N (%)
- Prediction horizon
- Predictor availability window
- Selected model and all prespecified candidate models
- AUPRC with 95% CI
- AUROC with 95% CI
- Locked probability threshold
- Sensitivity with 95% CI
- Specificity with 95% CI
- PPV and NPV with 95% CI
- F-beta and beta value
- Brier score
- Calibration results
- Clinical-score comparators
- Missingness summary
- Subgroup results
- External-validation status

## Research-use boundary

This repository is not validated for diagnosis, triage, ICU admission, treatment allocation, or any other patient-care decision. A high retrospective sensitivity or favorable DCA result does not establish safety, efficacy, transportability, or regulatory readiness.
