# PenuX-AP-Severity: A Broad Model-Space Exploration for Early Severity Prediction in Acute Pancreatitis, Including an Engineered Organ-Dysfunction Parameter

An exploratory secondary-analysis report — public datasets, no external validation, no clinical use

PenuX Research Group — Corresponding author: Netanel Stern (netanel@penux.uk)

**Research use only.** This report describes an exploratory secondary analysis of two publicly available, de-identified tabular datasets. It is not a clinical validation study, has not undergone external validation, has not been prospectively evaluated, and has received no regulatory or institutional review board approval for clinical use. Nothing in this report should be used to guide diagnosis, triage, admission, discharge, treatment, or any other patient-care decision. All clinical decisions must be made exclusively by qualified medical professionals. See Section 12 (Limitations) and Section 13 (Statement of Use).

### Abstract

**Background.** Severe acute pancreatitis (SAP) carries substantial morbidity and mortality, and early identification of patients at risk is a long-standing clinical challenge. Traditional severity scores (Ranson, BISAP, APACHE II, CT Severity Index) are validated but have practical limitations: several require serial measurements over 24–48 hours, and none was designed for automated integration into clinical information systems.

**Objective.** To explore, at unusually broad scale, how far gradient-boosted decision tree (GBDT) models and a wide space of alternative model families can predict SAP from routine admission laboratory data on two public datasets, and to test whether adding a lab-only organ-dysfunction parameter (inspired by the Sepsis-3 SOFA methodology) improves the best-performing model.

**Methods.** Two de-identified public datasets (multiml, n=1,289, 204 SAP cases; lnn, n=722, 137 SAP cases), both reportedly sourced from the Second Affiliated Hospital of Guilin Medical University, were used. A model zoo of 1,982 classifier configurations — spanning linear models, support vector machines, nearest-neighbor methods, naive Bayes, decision trees, random forests, extremely randomized trees, AdaBoost, bagging, six gradient-boosting variants (scikit-learn's GBM, HistGradientBoosting, XGBoost, LightGBM, CatBoost, and a from-scratch NumPy implementation), three additional boosting algorithms (XGBoost DART, LightGBM DART/GOSS, CatBoost Plain boosting), feedforward and one-dimensional convolutional neural networks (PyTorch), and a fixed-weight hybrid ensemble of a DNN, a ConvNet, and LightGBM — was evaluated with identical 5-fold stratified cross-validation and out-of-fold prediction on both datasets. Statistical significance against a plain logistic-regression baseline was assessed via the Hanley–McNeil approximate standard error for AUROC. A lab-only quasi-SOFA organ-dysfunction score was engineered and tested both as a standalone predictor and as an added feature to the best-performing model.

**Results.** A fixed-weight hybrid ensemble (DNN + one-dimensional ConvNet + LightGBM, weighted 0.2/0.2/0.6) ranked first of 1,981 successfully evaluated configurations on the multiml dataset (AUROC=0.857), narrowly ahead of the best single-library LightGBM configuration (AUROC=0.854); on the lnn dataset, a LightGBM configuration remained best (AUROC=0.889). Deep learning models (DNN, ConvNet) alone ranked in the middle of the model space on both datasets, consistent with prior reports that tree ensembles remain competitive with, and often superior to, deep learning on small tabular datasets. No model reached conventional statistical significance (p\<0.05, unpaired) against the logistic-regression baseline on the smaller, more imbalanced multiml dataset (204 SAP cases), though 690 of 1,981 configurations did on lnn. Adding the engineered quasi-SOFA feature to the best hybrid model produced mixed results: a small AUROC/F1 decrease on multiml, and a small AUROC decrease with an F1 and false-negative-count improvement on lnn.

**Conclusions.** Across a model space roughly an order of magnitude larger than typically reported in this literature, no single algorithm family dominates; a hybrid hand-weighted combination of gradient boosting and deep learning achieved the best rank on one dataset without reaching conventional statistical significance against a simple baseline, and an engineered organ-dysfunction feature did not consistently improve performance. These findings do not change the fundamental limitations of secondary analysis on small, single-institution public datasets: external validation, prospective evaluation, and clinical/regulatory review remain prerequisites for any clinical use.

## 1. Introduction

Acute pancreatitis (AP) is an inflammatory disease of the pancreas ranging from a mild, self-limited illness to a severe, systemic condition with organ failure, pancreatic necrosis, and substantial mortality <sup>\[1–8\]</sup>. The 2012 Revised Atlanta Classification stratifies AP into mild, moderately severe, and severe categories based on the presence and persistence of organ failure <sup>\[1\]</sup>, and severe acute pancreatitis (SAP) — defined by persistent organ failure beyond 48 hours — carries mortality estimates in published cohorts ranging roughly from 15% to 40%, with wide variation by population and era <sup>\[1,2,9,19\]</sup>. Global incidence estimates vary considerably by region and diagnostic criteria, and the disease represents a substantial and rising healthcare burden worldwide <sup>\[2\]</sup>.

Early identification of patients likely to progress to SAP is clinically important: it informs decisions about monitoring intensity, fluid resuscitation strategy, escalation to critical care, and surveillance for complications <sup>\[1,3–8,20\]</sup>. Several validated severity-scoring systems exist — Ranson's criteria, the Bedside Index for Severity in Acute Pancreatitis (BISAP), the Acute Physiology and Chronic Health Evaluation II (APACHE II) score, the (Modified) CT Severity Index, and others — and have been compared extensively in the clinical literature <sup>\[9–18,22\]</sup>. Each has practical limitations: Ranson's criteria require values collected at admission and again at 48 hours; APACHE II requires more than a dozen physiological variables including several not routinely available at every institution; the CT Severity Index requires imaging typically obtained 72–96 hours after symptom onset, limiting its value for early triage <sup>\[9,10,16\]</sup>. BISAP was designed specifically to be calculable at admission from a small number of routinely available variables and performs comparably to APACHE II in several comparative studies, though findings vary by cohort <sup>\[9,10,11,12\]</sup>.

The growth of electronic health records and clinical laboratory information systems has motivated substantial interest in applying supervised machine learning to SAP prediction, using routinely collected admission laboratory values as predictors <sup>\[23–43\]</sup>. Reported studies span logistic regression, random forests, gradient-boosted decision trees, and, increasingly, deep learning approaches including convolutional neural networks applied to CT imaging <sup>\[40,41,42\]</sup> and combined clinical-plus-imaging multi-view architectures <sup>\[42\]</sup>. Reported AUROC values in this literature are frequently high (often exceeding 0.90), though few studies report external validation on independent institutions or populations <sup>\[25,26,30\]</sup>.

The present report describes a substantially expanded secondary analysis built on two publicly released, de-identified tabular datasets originally described in the machine-learning literature on AP severity prediction <sup>\[28,29\]</sup>. Building on an initial analysis using three gradient-boosted decision tree (GBDT) libraries — XGBoost, LightGBM, and CatBoost — this report documents a systematic expansion of the compared model space to 1,982 configurations, the addition of several boosting algorithm variants and deep learning architectures, formal statistical-significance testing against a simple baseline, and an attempt to incorporate an additional engineered clinical parameter inspired by the Sepsis-3 Sequential Organ Failure Assessment (SOFA) framework <sup>\[55\]</sup>, applied here using only its methodology and not any sepsis patient data.

## 2. Data

### 2.1 Datasets

Two datasets were used, both reportedly originating from the Second Affiliated Hospital of Guilin Medical University and released publicly by their original authors under open-source licenses:

| Dataset    | Source                                                                                                                                  | License    | Years     | N     | SAP / non-SAP       | Features                |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-------|---------------------|-------------------------|
| ap_multiml | Longshike et al., "Predicting acute pancreatitis severity with multi-machine-learning models" <sup>\[28,83\]</sup>                      | MIT        | 2016–2024 | 1,289 | 204 / 1,085 (15.8%) | 59 (post-sanitization)  |
| ap_lnn     | Longshike et al., "Constructing a prediction model for acute pancreatitis severity based on liquid neural network" <sup>\[29,84\]</sup> | Apache-2.0 | 2020–2024 | 722   | 137 / 585 (19.0%)   | 106 (post-sanitization) |

Both datasets consist exclusively of routine admission laboratory values (complete blood count, coagulation panel, liver and renal function panels, electrolytes, inflammatory markers, and, in the lnn dataset, lactate) plus age and sex; neither includes vital signs, imaging, or Glasgow Coma Scale, which limits the applicability of certain traditional composite scores (APACHE II, full SOFA) that require these inputs.

### 2.2 Sanitization and a label-direction discrepancy

Direct patient identifiers (name, serial/ID number) were removed prior to any analysis. During this analysis, an internal inconsistency was identified: the raw binary target column's value 1 corresponded to the numerically larger (non-SAP) group in both datasets, while value 0 corresponded to the documented SAP minority count (204 and 137 cases respectively) — the reverse of the direction stated in the project's own prior documentation. Because AUROC is invariant to a joint flip of the label and the score, this discrepancy did not affect any previously reported AUROC values computed with the label used as-is; however, it would have affected every threshold-dependent metric (F1, sensitivity, specificity, PPV, NPV) had the flip not been corrected. All results in this report use the corrected orientation (1 = SAP).

## 3. Methods

### 3.1 Model zoo

A total of 1,982 classifier configurations were assembled programmatically, spanning:

- Linear models: logistic regression, ridge classification, stochastic gradient descent with multiple loss functions and penalties, perceptron, linear and kernel support vector machines <sup>\[62,79\]</sup>;
- Instance-based and probabilistic models: k-nearest neighbors, Gaussian and Bernoulli naive Bayes;
- Tree-based models: individual decision trees and extremely randomized trees (including "maximal" trees constrained only by a large `max_leaf_nodes`) <sup>\[68\]</sup>, random forests <sup>\[67\]</sup>, extremely randomized tree ensembles <sup>\[68\]</sup>;
- Boosting algorithms: AdaBoost <sup>\[69\]</sup>, scikit-learn's GradientBoostingClassifier <sup>\[66\]</sup>, HistGradientBoostingClassifier, XGBoost <sup>\[62\]</sup>, LightGBM <sup>\[63\]</sup>, CatBoost <sup>\[64\]</sup>, plus the DART dropout-tree variant of XGBoost and LightGBM <sup>\[65\]</sup>, LightGBM's GOSS sampling variant <sup>\[63\]</sup>, and CatBoost's Plain (non-ordered) boosting mode <sup>\[64\]</sup>;
- A from-scratch gradient-boosted decision tree implementation (`ScratchGBDTClassifier`), using no existing GBDT library: classical Friedman-style TreeBoost <sup>\[66\]</sup> — regression trees built from scratch and split on residual-variance reduction, with a Newton leaf-value correction applied afterward;
- Deep learning: a configurable feedforward network and a one-dimensional convolutional network treating the standardized feature vector as a length-*p* sequence, implemented in PyTorch <sup>\[86\]</sup> with batch normalization <sup>\[75\]</sup>, dropout <sup>\[73\]</sup>, and the Adam optimizer <sup>\[74\]</sup>;
- A hybrid ensemble combining a DNN, a ConvNet, and LightGBM via one of three fixed weighting schemes (equal weight; GBDT-heavy, 0.6/0.2/0.2; neural-network-heavy, 0.2/0.4/0.4).

All scikit-learn-family models were implemented via scikit-learn <sup>\[85\]</sup>. Every configuration was wrapped in an identical preprocessing pipeline (median imputation, standardization) and evaluated with 5-fold stratified cross-validation <sup>\[79\]</sup>, producing out-of-fold predicted probabilities used for all reported metrics; this avoids the optimistic bias of in-sample evaluation and matches the evaluation protocol used for the three headline GBDT models in the initial analysis. Class imbalance (15.8%/19.0% SAP prevalence) was handled per-model via built-in class-weighting options (e.g., `class_weight='balanced'`) where available, and via custom class-balanced gradient weighting in the from-scratch implementations; oversampling approaches such as SMOTE <sup>\[80\]</sup> were not used in this analysis.

### 3.2 Statistical significance testing

Because AUROC differences between many closely ranked models can easily arise from sampling noise given the modest number of positive cases in both datasets (204 and 137), each model's AUROC was compared against a single fixed logistic-regression baseline (default scikit-learn hyperparameters) using the Hanley–McNeil approximate standard error for the area under an ROC curve <sup>\[81\]</sup> and a two-sample (unpaired) z-test. This approach is conservative: because every model was evaluated on the same cross-validation folds, the true sampling correlation between any two models' predictions is positive, and a properly paired test such as DeLong's method <sup>\[82\]</sup> would generally have higher power to detect differences than the unpaired approximation used here. A paired analysis was not performed because it requires retaining every model's raw out-of-fold prediction vector, which was not done during the initial large-scale benchmarking runs for storage and runtime reasons.

### 3.3 Engineered feature: a lab-only quasi-SOFA score

Motivated by a request to explore emergency-department and internal-medicine datasets using laboratory-based clinical suspicion frameworks, and following an explicit decision to use only the *methodology* of one such framework (Sepsis-3's SOFA score, evaluated on the 2019 PhysioNet/Computing in Cardiology Challenge sepsis dataset <sup>\[55,56\]</sup>) rather than that dataset's patient data, a lab-only "quasi-SOFA" score was constructed. It sums renal (creatinine), hepatic (bilirubin), and coagulation (platelet count) sub-scores using the same threshold bands as full SOFA's corresponding components, with optional additive terms for leukocyte abnormality (a SIRS-style criterion) and, where available, lactate above 2 mmol/L (the Sepsis-3 septic-shock threshold). This is explicitly *not* a validated SOFA score: full SOFA requires respiratory (PaO2/FiO2), cardiovascular (mean arterial pressure/vasopressor use), and neurological (Glasgow Coma Scale) components, none of which are available in either lab-only dataset used here.

## 4. Results

### 4.1 Growth of the model space and overall ranking

The compared model space grew across the analysis from an initial 171 configurations to 1,982, in five stages (Table 2). 1,981 of 1,982 configurations were evaluated successfully on both datasets; the sole consistent failure was quadratic discriminant analysis, which raised a collinearity error on both datasets.

| Stage                                                         | Total configurations | Increment |
|---------------------------------------------------------------|----------------------|-----------|
| Initial GBDT-focused zoo                                      | 171                  | —         |
| \+ linear/SVM/KNN/tree/ensemble/GBDT hyperparameter expansion | 784                  | +613      |
| \+ DART/GOSS/Plain boosting, from-scratch GBDT                | 886                  | +102      |
| \+ lr=0.001 grid extension (XGBoost/LightGBM/CatBoost)        | 961                  | +75       |
| \+ "maximal" trees (`max_leaf_nodes`-unconstrained)           | 1,082                | +21       |
| \+ DNN, one-dimensional ConvNet, hybrid ensemble              | **1,982**            | +900      |

On the multiml dataset, a hybrid ensemble configuration (`hybrid_dnn(64,)_conv(8,16)_gbdt(n=100,depth=5,lr=0.05)_gbdt-heavy`) achieved the highest AUROC of all 1,981 evaluated configurations (0.8566), narrowly ahead of the best single-family LightGBM configuration (0.8537, rank 2) and considerably ahead of the three original headline GBDT configurations (LightGBM rank 102/1,981, AUROC=0.8499; XGBoost rank 506/1,981, AUROC=0.8421; CatBoost rank 1,020/1,981, AUROC=0.8290). On the lnn dataset, the best-ranked configuration remained a LightGBM model (n=800, leaves=15, learning rate=0.01; AUROC=0.8892), with the best hybrid ensemble ranked 168th (AUROC=0.8799) and the three headline configurations ranked 25th (LightGBM, AUROC=0.8853), 145th (XGBoost, AUROC=0.8810), and 334th (CatBoost, AUROC=0.8758).

### 4.2 New boosting algorithm variants

| Family                                    | multiml best AUROC (rank/1,981) | lnn best AUROC (rank/1,981) |
|-------------------------------------------|---------------------------------|-----------------------------|
| XGBoost DART <sup>\[65\]</sup>            | 0.8389 (655)                    | 0.8660 (519)                |
| LightGBM DART <sup>\[63,65\]</sup>        | 0.8461 (267)                    | 0.8852 (29)                 |
| LightGBM GOSS <sup>\[63\]</sup>           | 0.8472 (213)                    | 0.8853 (26)                 |
| CatBoost Plain boosting <sup>\[64\]</sup> | 0.8471 (223)                    | 0.8808 (149)                |
| From-scratch GBDT <sup>\[66\]</sup>       | 0.8352 (830)                    | 0.8647 (559)                |
| Feedforward DNN                           | 0.8490 (133)                    | 0.8421 (901)                |
| 1D ConvNet                                | 0.8302 (981)                    | 0.8294 (1,026)              |
| Hybrid DNN+ConvNet+GBDT                   | **0.8566 (1)**                  | 0.8799 (168)                |

DART's dropout mechanism, designed to counteract over-specialization of individual trees across very large ensembles <sup>\[65\]</sup>, did not improve performance relative to standard boosting on either dataset here; both datasets are small (700–1,300 rows) relative to the scale DART was evaluated at originally. GOSS and Plain boosting performed comparably to their respective "standard" counterparts. Neither neural architecture alone matched the gradient-boosted tree ensembles — consistent with a substantial body of recent evidence that deep learning does not reliably outperform tree ensembles on small-to-moderate tabular datasets <sup>\[70\]</sup> — but the fixed-weight hybrid combination outperformed every individual family on multiml.

### 4.3 Statistical significance against a logistic-regression baseline

Using the unpaired Hanley–McNeil approach described in Section 3.2, zero of 1,981 configurations reached p\<0.05 against the logistic-regression baseline (AUROC=0.8158) on multiml — including the top-ranked hybrid model itself (p=0.054, narrowly missing the conventional threshold). At the more permissive p\<0.10 level, 154 configurations reached significance on multiml, dominated by random-forest and LightGBM variants with class-balanced weighting. On lnn (baseline AUROC=0.8087), 690 of 1,981 configurations reached p\<0.05. This asymmetry illustrates a straightforward statistical-power limitation: with only 204 SAP cases in multiml, distinguishing an AUROC of 0.86 from 0.82 is difficult with this conservative, unpaired test, even though the same absolute AUROC gap is easily distinguished on lnn's somewhat more favorable case count and baseline separation.

### 4.4 Sensitivity, specificity, and false-negative trade-offs

Among statistically significant models, the configuration with the highest F1 score while minimizing missed SAP cases (false negatives) was identified for each dataset (Table 4). A single random-forest configuration achieved the lowest false-negative count within each dataset's significant-model set, and simultaneously the highest F1 score among that low-false-negative group, i.e. a Pareto-optimal point requiring no trade-off within the group considered.

| Dataset          | Model                | F1    | Sensitivity | Specificity | False negatives |
|------------------|----------------------|-------|-------------|-------------|-----------------|
| multiml (90% CI) | rf_n800_d15_balanced | 0.524 | 46.1%       | —           | 110/204         |
| lnn (95% CI)     | rf_n200_d5_balanced  | 0.631 | 68.6%       | 88.5%       | 43/137          |

Separately, one deep-learning configuration on multiml (`dnn_v2_(16,)_dropout0.1_lr0.005_wd0.1`) reached 75.5% sensitivity (50/204 false negatives) at a nearly identical F1 (0.518) to the random-forest configuration above — a substantially higher sensitivity at comparable overall F1, illustrating that F1 alone can obscure clinically meaningful differences in the sensitivity/specificity balance. Scanning the specificity axis directly, the false-positive rate associated with peak F1 was found to differ between datasets (approximately 5.2% on multiml versus 9.2% on lnn), indicating that no single fixed operating point generalizes exactly across both datasets even though both are drawn from the same reporting institution.

### 4.5 The quasi-SOFA engineered feature

As a standalone predictor, the lab-only quasi-SOFA score reached AUROC 0.663 (multiml) and 0.678 (lnn) — substantially weaker than any full multivariable model, as expected for a 3–5-variable composite score, but with a favorable sensitivity of 69.3% (42/137 false negatives) at its own F1-optimal threshold on lnn.

Adding quasi-SOFA as an additional engineered feature to the best hybrid ensemble produced mixed results (Table 5): a decrease in both AUROC and F1 on multiml, and a small AUROC decrease alongside an F1 increase and a five-case reduction in false negatives on lnn.

| Dataset | Variant               | AUROC     | F1        | False negatives |
|---------|-----------------------|-----------|-----------|-----------------|
| multiml | Hybrid, no quasi-SOFA | **0.857** | **0.547** | 105             |
| multiml | Hybrid + quasi-SOFA   | 0.848     | 0.540     | 105             |
| lnn     | Hybrid, no quasi-SOFA | **0.880** | 0.607     | 62              |
| lnn     | Hybrid + quasi-SOFA   | 0.878     | **0.620** | **57**          |

A plausible explanation is that the underlying gradient-boosted component of the hybrid model already has direct access to the raw creatinine, bilirubin, and platelet values used to construct quasi-SOFA, and can learn comparable interaction terms on its own; the hand-crafted composite score therefore adds little information not already accessible to the model. The un-augmented hybrid model remains the preferred configuration on multiml under this analysis.

### 4.6 From-scratch reference implementations (Fortran)

To make the underlying algorithmic mechanics fully transparent, three dependency-free Fortran programs were written, using no machine-learning library and no BLAS/LAPACK routines:

- A logistic regression classifier trained by full-batch gradient descent (AUROC 0.849 / 0.822 on multiml / lnn, using an 80/20 stratified split rather than cross-validation);
- An XGBoost-style gradient-boosted tree ensemble implementing the Exact Greedy Algorithm for split finding using gradient/Hessian statistics and the corresponding regularized gain formula <sup>\[62\]</sup> (AUROC 0.865 / 0.872);
- A feedforward network expressed explicitly as a function composition F(x)=f<sub>L</sub>(f<sub>L−1</sub>(…f<sub>1</sub>(x)…)), constrained so that every f<sub>l</sub> is a polynomial or rational function — excluding ReLU, which is piecewise linear rather than a single polynomial or rational expression — using a sigmoid activation at every layer built from a from-scratch \[3/3\] Padé rational approximant for e<sup>x</sup> (AUROC 0.832 / 0.758, after adding class-balanced gradient weighting to correct an initial failure mode in which the network converged to predicting every case negative).

Both exponential-function implementations (a 20-term Taylor series and the Padé approximant, each with range reduction by repeated halving) were verified against the language's intrinsic exponential function to within approximately 10<sup>−15</sup> and 10<sup>−7</sup> relative error respectively across x∈\[−20,20\], and end-to-end model outputs were confirmed identical regardless of which exponential implementation was used.

## 5. Discussion

Two observations stand out from this exploration. First, expanding the compared model space by roughly an order of magnitude beyond the original three-library GBDT comparison did not substantially change the qualitative conclusion: gradient-boosted tree ensembles, and closely related methods (random forests, extremely randomized trees), occupy essentially all of the top-ranked positions on both datasets, and the single best result found across the entire space (the hybrid ensemble on multiml) exceeded the best pure-GBDT result by only 0.003 AUROC — a difference that itself does not reach conventional statistical significance against even a simple baseline. This is broadly consistent with reports elsewhere that deep learning does not reliably surpass gradient boosting on small-to-moderate tabular datasets <sup>\[70\]</sup>, and suggests that further architecture search alone is unlikely to yield a qualitatively different result on these particular datasets without additional data.

Second, the attempt to add an engineered clinical parameter (quasi-SOFA) illustrates a common and easily overlooked pitfall in this kind of research: an engineered feature motivated by domain knowledge from an adjacent clinical field (sepsis) did not reliably improve a model that already had access to the same underlying raw laboratory values used to construct that feature. This is a useful negative result: feature engineering value depends on whether the transformation captures information the model could not otherwise learn, not merely on whether the transformation itself has established clinical meaning elsewhere.

A search for a third, independent public dataset with an SAP/non-SAP-style severity label — conducted across Kaggle (inaccessible under this analysis environment's network policy) and GitHub — did not identify a suitable candidate; the closest match found (a post-ERCP pancreatitis prevention trial dataset <sup>\[higgi13425/medicaldata, not separately indexed\]</sup>) addresses a different clinical question (procedural risk of inducing pancreatitis, rather than severity stratification of already-diagnosed AP) and was not incorporated.

## 6. Limitations

1.  This remains a secondary analysis of two publicly released datasets from a single reporting institution; no data were collected prospectively by the authors of this report, and no external validation on an independent institution or population was performed.
2.  The statistical significance testing used an unpaired approximation (Hanley–McNeil); a properly paired test (DeLong <sup>\[82\]</sup>) evaluated on the same cross-validation folds would likely have higher power and was not performed because per-model out-of-fold prediction vectors were not retained during the large-scale benchmarking runs.
3.  The quasi-SOFA score is a research approximation, not a validated SOFA score; it omits the respiratory, cardiovascular, and neurological components of full SOFA, none of which are available in either lab-only dataset.
4.  All 1,982 model configurations were evaluated on the same cross-validation folds within each dataset; repeated comparison across such a large model space increases the risk that the single best-ranked configuration reflects some degree of selection on noise, which the significance testing in Section 4.3 was specifically intended to address (and which is precisely why the top-ranked hybrid model's failure to reach p\<0.05 is a meaningful, not merely technical, finding).
5.  Neither dataset includes vital signs, imaging, or Glasgow Coma Scale, precluding computation of several widely used composite scores (full APACHE II, full SOFA) for direct benchmarking against the models described here.
6.  This report inherits all limitations already documented for the underlying project, including the absence of prospective validation, institutional review board approval, and regulatory clearance.

## 7. Statement of Use

PenuX-AP-Severity, and the exploratory analysis described in this report, are research prototypes only. They must not be used for diagnosis, triage, admission or discharge decisions, treatment selection, or any other clinical decision. All clinical decisions must be made exclusively by qualified medical professionals, following appropriate institutional and regulatory processes.

## References

1.  Banks PA, Bollen TL, Dervenis C, et al.; Acute Pancreatitis Classification Working Group. Classification of acute pancreatitis—2012: revision of the Atlanta classification and definitions by international consensus. *Gut*. 2013;62(1):102–111.
2.  GBD collaborators. The global, regional, and national burden of acute pancreatitis in 204 countries and territories, 1990–2019. PMC8390209.
3.  Acute pancreatitis \[review\]. PubMed PMID: 32891214.
4.  A narrative review of the mechanism of acute pancreatitis and recent advances in its clinical management. PMC8014344.
5.  Diagnosis and Management of Acute Pancreatitis. PMC11816589.
6.  Insights into Acute Pancreatitis: Pathogenesis, Diagnosis, and Management. *J Clin Med*. 2026;15:2819.
7.  Acute Pancreatitis: A Narrative Review. PMC12799804.
8.  Zheng et al. A narrative review of acute pancreatitis and its diagnosis, pathogenetic mechanism, and management. *Ann Transl Med*.
9.  A comparison of APACHE II, BISAP, Ranson's score and modified CTSI in predicting the severity of acute pancreatitis based on the 2012 revised Atlanta Classification. *Gastroenterol Rep (Oxf)*. PMC5952961.
10. Evaluation of the BISAP scoring system in prognostication of acute pancreatitis – a prospective observational study. *Ann Med Surg*.
11. Predictive value of the Ranson and BISAP scoring systems for the severity and prognosis of acute pancreatitis: a systematic review and meta-analysis. *PLoS One*. 2024.
12. The Value of BISAP Score for Predicting Mortality and Severity in Acute Pancreatitis: A Systematic Review and Meta-Analysis. *PLoS One*. 2015.
13. Clinical usefulness of scoring systems to predict severe acute pancreatitis: a systematic review and meta-analysis with pre- and post-test probability assessment. PMC10637128.
14. Comparison of Different Scoring Systems in Predicting the Severity of Acute Pancreatitis: A Prospective Observational Study. PMC7067369.
15. Comparative Study Between Various Scoring Systems in Predicting the Severity of Acute Pancreatitis. PMC12162368.
16. Predicting Acute Pancreatitis Severity: Comparison of Prognostic Scores. PMC5139846.
17. Prediction of Severity in Acute Pancreatitis. PMC2442933.
18. Prediction of Severe Acute Pancreatitis Using a Decision Tree Model Based on the Revised Atlanta Classification of Acute Pancreatitis. PMC4651493.
19. Organ Failure and Prediction of Severity in Acute Pancreatitis. PubMed PMID: 39880521.
20. Diagnosis, severity stratification and management of adult acute pancreatitis—current evidence and controversies. PMC9727576.
21. Clinical utility of the pancreatitis activity scoring system in severe acute pancreatitis. PMC9441599.
22. Predictive Value of Several Parameters for Severity of Acute Pancreatitis in a Cohort of 172 Patients. PMC11854639.
23. Construction and validation of a severity prediction model for acute pancreatitis based on CT severity index: a retrospective case-control study. PMC11125528.
24. Severe Acute Pancreatitis Prediction: A Model Derived From a Prospective Registry Cohort. PMC10636501.
25. Automated Machine Learning for the Early Prediction of the Severity of Acute Pancreatitis in Hospitals. PMC9226483.
26. Prediction of the severity of acute pancreatitis using machine learning models. *Postgrad Med*. 2022. doi:10.1080/00325481.2022.2099193.
27. Development of a machine learning-based early prediction model for disease severity in acute pancreatitis. *Eur J Med Res*. 2025.
28. Predicting the acute pancreatitis severity with multi-machine learning models: constructing an online prediction platform. PMC12982339. \[Source of the "multiml" dataset used in this study.\]
29. Constructing a prediction model for acute pancreatitis severity based on liquid neural network. *Sci Rep*. 2025. doi:10.1038/s41598-025-01218-5. \[Source of the "lnn" dataset used in this study.\]
30. Comparative Evaluation of Machine Learning and Deep Learning Models for Early Prediction of Severe Acute Pancreatitis: A Multi-Model Study Using the 2012 Revised Atlanta Classification. *medRxiv*. 2026.
31. Usefulness of Random Forest Algorithm in Predicting Severe Acute Pancreatitis. PMC9226542.
32. Accurate prediction of acute pancreatitis severity with integrative blood molecular measurements. PMC8034948.
33. Accurate prediction of acute pancreatitis severity based on genome-wide cell free DNA methylation profiles. PMC8680202.
34. Machine learning-based predictive model for acute pancreatitis-associated lung injury: a retrospective analysis. PMC12379022.
35. To Establish an Early Prediction Model for Acute Respiratory Distress Syndrome in Severe Acute Pancreatitis Using Machine Learning Algorithm. PMC10002486.
36. Machine learning models for mortality prediction in critically ill patients with acute pancreatitis-associated acute kidney injury. PMC11462445.
37. Machine Learning Models of Acute Kidney Injury Prediction in Acute Pancreatitis Patients. PMC7542489.
38. Development of a clinical prediction model for intra-abdominal infection in severe acute pancreatitis using logistic regression and nomogram. PMC12367684.
39. EASY-APP: An artificial intelligence model and application for early and easy prediction of severity in acute pancreatitis. PMC9162438.
40. Predicting acute pancreatitis severity with enhanced computed tomography scans using convolutional neural networks. *Sci Rep*. 2023. doi:10.1038/s41598-023-44828-7.
41. Development and International Validation of a Deep Learning Model for Predicting Acute Pancreatitis Severity from CT Scans. *medRxiv*. 2025.
42. Construction of a Multi-View Deep Learning Model for the Severity Classification of Acute Pancreatitis. *Discov Med*. 2025.
43. A Novel Nomogram for Predicting Survival in Patients with Severe Acute Pancreatitis: An Analysis Based on the Large MIMIC-III Clinical Database. PMC8526213.
44. The diagnostic value of serum C-reactive protein, procalcitonin, interleukin-6 and lactate dehydrogenase in patients with severe acute pancreatitis. *Clin Chim Acta*. 2020. PubMed PMID: 32828732.
45. Interleukin-6 is better than C-reactive protein for the prediction of infected pancreatic necrosis and mortality in patients with acute pancreatitis. *Front Cell Infect Microbiol*. 2022. PMC9716459.
46. Comparison of Interleukin-6, C-Reactive Protein, Procalcitonin, and the Computed Tomography Severity Index for Early Prediction of Severity of Acute Pancreatitis. PubMed PMID: 36789576.
47. Interleukin-6: An Early Predictive Marker for Severity of Acute Pancreatitis.
48. Serum Profiles of C-Reactive Protein, Interleukin-8, and Tumor Necrosis Factor-alpha in Patients with Acute Pancreatitis. PMC2814374.
49. Hemoconcentration is a poor predictor of severity in acute pancreatitis. PMC4717047.
50. Serial D-dimer measurements dynamically predict disease severity in acute biliary pancreatitis: a prospective observational study. PMC12779076.
51. Serum Lactate Dehydrogenase Is a Sensitive Predictor of Systemic Complications of Acute Pancreatitis. PMC9626216.
52. Total serum calcium and corrected calcium as a predictor of severity in acute pancreatitis. *Int Surg J*.
53. Relationship between intra-abdominal hypertension, outcome and the revised Atlanta and determinant-based classifications in acute pancreatitis. PMC5989946.
54. Association Between Severity and the Determinant-Based Classification, Atlanta 2012 and Atlanta 1992, in Acute Pancreatitis: A Clinical Retrospective Study. PMC4554029.
55. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801–810.
56. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med*. 2020;48(2):210–217.
57. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. *Sci Data*. 2016;3:160035.
58. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data*. 2023;10:1.
59. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220.
60. HL7 International. HL7 FHIR (Fast Healthcare Interoperability Resources) Specification. Available at: https://hl7.org/fhir/.
61. World Medical Association. WMA Declaration of Helsinki – Ethical Principles for Medical Research Involving Human Subjects (2013 revision).
62. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD '16). 2016:785–794.
63. Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu TY. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Adv Neural Inf Process Syst*. 2017;30.
64. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. *Adv Neural Inf Process Syst*. 2018;31:6638–6648.
65. Rashmi KV, Gilad-Bachrach R. DART: Dropouts meet Multiple Additive Regression Trees. 2015.
66. Friedman JH. Greedy Function Approximation: A Gradient Boosting Machine. *Ann Stat*. 2001;29(5):1189–1232.
67. Breiman L. Random Forests. *Mach Learn*. 2001;45:5–32.
68. Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. *Mach Learn*. 2006;63:3–42.
69. Freund Y, Schapire RE. A decision-theoretic generalization of on-line learning and an application to boosting. *J Comput Syst Sci*. 1997;55(1):119–139.
70. Borisov V, Leemann T, Seßler K, Haug J, Pawelczyk M, Kasneci G. Tabular Data: Deep Learning is Not All You Need. arXiv:2106.03253.
71. Rumelhart DE, Hinton GE, Williams RJ. Learning representations by back-propagating errors. *Nature*. 1986;323:533–536.
72. Nair V, Hinton GE. Rectified Linear Units Improve Restricted Boltzmann Machines. In: *Proceedings of the 27th International Conference on Machine Learning* (ICML). 2010.
73. Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *J Mach Learn Res*. 2014;15(1):1929–1958.
74. Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. In: *Proceedings of the 3rd International Conference on Learning Representations* (ICLR). 2015.
75. Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In: *Proceedings of the 32nd International Conference on Machine Learning* (ICML). 2015.
76. Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks. In: *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics* (AISTATS). 2010.
77. He K, Zhang X, Ren S, Sun J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In: *Proceedings of the IEEE International Conference on Computer Vision* (ICCV). 2015.
78. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. *Adv Neural Inf Process Syst*. 2017;30.
79. Kohavi R. A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. In: *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (IJCAI). 1995:1137–1145.
80. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic Minority Over-sampling Technique. *J Artif Intell Res*. 2002;16:321–357.
81. Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*. 1982;143(1):29–36.
82. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44(3):837–845.
83. Longshike. Predicting-acute-pancreatitis-severity-with-multi-machine-learning-models \[software/data repository\]. GitHub. Available at: https://github.com/longshike/Predicting-acute-pancreatitis-severity-with-multi-machine-learning-models.
84. Longshike. LNN-for-SAP-Prediction \[software/data repository\]. GitHub. Available at: https://github.com/longshike/LNN-for-SAP-Prediction.
85. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *J Mach Learn Res*. 2011;12:2825–2830.
86. Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Adv Neural Inf Process Syst*. 2019;32.
87. Acute biliary pancreatitis management during the COVID-19 pandemic. *medRxiv*. 2021.
88. Understanding acute pancreatitis in end-stage renal disease: unraveling etiologies, clinical presentations, management strategies, and complications—a narrative review. *J Pancreatol*.
89. Epidemiology, pathophysiology and management of acute pancreatitis: A literature review. *Res Soc Dev*.
90. A comparative analysis of gradient boosting algorithms. *Artif Intell Rev*. doi:10.1007/s10462-020-09896-5.
91. Measuring the Coverage of the HL7® FHIR® Standard in Supporting Data Acquisition for 3 Public Health Registries. PMC10853080.
92. Evaluation of four scoring systems in prognostication of acute pancreatitis for elderly patients. PMC7268671.
93. Role of Scoring Systems in Prognosticating Outcomes of Patients With Acute Pancreatitis: A Prospective Cohort Study. PMC11953751.

This document is an exploratory secondary-analysis report generated as part of the PenuX-AP-Severity research project. It is not a peer-reviewed publication. Reference entries lacking a full author list reflect the information available from the original search of publicly indexed abstracts at the time of writing; readers wishing to cite the underlying primary literature should verify full bibliographic details against the original source before citation in a formal academic work.
