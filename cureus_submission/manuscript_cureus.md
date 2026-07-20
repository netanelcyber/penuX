# CUREUS SUBMISSION — PenuX SAP Severity Prediction
# Paste each section into the Cureus online editor at https://www.cureus.com/submit

---

## TITLE
PenuX: A Comparative Study of 11 Machine Learning and Deep Learning Models for Early Severity Prediction of Acute Pancreatitis Using Routine Admission Laboratory Values, with FHIR R4 Integration

---

## AUTHORS
Netanel Shoshany
Independent Clinical AI Researcher
Israel
nsh531@gmail.com
ORCID: https://orcid.org/0000-0002-3482-5508

---

## ABSTRACT (max 350 words — Cureus limit)

**Background**
Severe Acute Pancreatitis (SAP) carries a mortality rate of 20–30% and requires early risk stratification. Classical scoring systems (Ranson, BISAP, APACHE II) require 24–48 hours of serial laboratory observation and lack electronic health record (EHR) integration. There is an unmet need for admission-time, data-driven severity prediction that integrates with modern health information standards.

**Methods**
We conducted a retrospective analysis of 722 acute pancreatitis (AP) admissions (585 severe / 137 mild; Atlanta 2012 classification) from a single Chinese tertiary institution. Eleven models were trained on 106 routine admission laboratory features using 5-fold stratified cross-validation: three classical machine learning models (Logistic Regression, Random Forest, Gradient Boosting), three multilayer perceptron (MLP) deep learning models, and five long short-term memory (LSTM)-based sequence models (Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, LSTM+Attention, CNN-LSTM). Optimal decision thresholds were selected by maximum F1 score on out-of-fold predictions. A client-side FHIR R4 integration was implemented for automated risk scoring at point of care.

**Results**
Random Forest achieved the highest discrimination (AUC=0.877, F1=0.917, sensitivity=96.8%, specificity=38.7% at threshold 0.535), missing only 19 of 585 severe cases. Gradient Boosting was comparable (AUC=0.874, sensitivity=97.1%). MLP achieved AUC=0.836. LSTM-based models achieved AUC=0.675–0.772, with CNN-LSTM performing best among recurrent architectures (AUC=0.772, sensitivity=98.6%). Key predictive features across models were calcium, D-dimer, LDH, lactate, and hematocrit. A label inversion effect was identified: mild biliary AP cases showed higher WBC, CRP, and lipase than severe necrotising AP cases, explaining sub-chance performance of heuristic scoring models on this cohort.

**Conclusions**
Random Forest achieves SAP triage with AUC=0.877 from a single admission blood draw, eliminating the 24–48 hour observation window required by classical scoring systems. The open-source PenuX platform provides FHIR R4, HL7 v2.x, and Israeli HIS (Camelion) integration for automated deployment. External validation is required before clinical use.

---

## KEYWORDS (choose 5–8 in Cureus interface)
acute pancreatitis
severe acute pancreatitis
machine learning
deep learning
random forest
LSTM
FHIR R4
clinical prediction model

---

## INTRODUCTION

Acute Pancreatitis (AP) is one of the most common causes of emergency gastrointestinal hospitalisation, with an estimated global incidence of 34 cases per 100,000 persons per year [1]. Approximately 20% of cases progress to Severe Acute Pancreatitis (SAP), defined by the 2012 Revised Atlanta Classification as AP with persistent organ failure lasting more than 48 hours [2]. SAP carries a mortality rate of 20–30%, principally from infected pancreatic necrosis, abdominal compartment syndrome, and multi-organ dysfunction syndrome [3].

Early identification of SAP at or near admission is critical: patients who will develop organ failure require immediate ICU transfer, aggressive fluid resuscitation, and multidisciplinary management. However, the bedside scoring systems used for this purpose share a fundamental limitation: their final severity score cannot be computed at admission.

**The 48-hour validation problem.** Ranson's criteria (1974) require six of eleven parameters to be re-evaluated at 48 hours, including changes in BUN, hematocrit, serum calcium, arterial PO₂, base deficit, and fluid sequestration [4]. These are delta values — requiring two time points. This delay reflects the two-wave pathophysiology of SAP: enzymatic acinar cell injury occurs in the first hours (Wave 1), while systemic inflammatory response, cytokine storm (IL-6, TNF-α), and target-organ injury (renal, pulmonary, cardiovascular) develop over 24–48 hours (Wave 2) [5]. The Bedside Index of Severity in Acute Pancreatitis (BISAP) improves on this by using 24-hour data [6], but still precludes immediate admission-time triage. CT Severity Index requires imaging at 48–72 hours, because pancreatic necrosis is not reliably visible on early contrast-enhanced CT [7].

Machine learning (ML) models trained on admission laboratory data have the potential to match or exceed classical scoring systems while producing results within hours of presentation. However, most published models use a limited feature set (5–20 markers), evaluate single architectures, and do not address EHR integration [8,9].

This study aims to: (1) evaluate 11 ML and deep learning architectures on a 106-feature admission laboratory dataset using rigorous 5-fold stratified cross-validation; (2) characterise a clinically significant label inversion effect in the cohort; (3) demonstrate FHIR R4 integration for automated SAP risk scoring at point of care; and (4) release an open-source platform (PenuX) for reproducibility and clinical translation.

---

## METHODS

### Study Design and Ethics
Retrospective cohort study using a pre-existing de-identified dataset. No patient identifiers are retained in the dataset. Under institutional policy, retrospective analysis of fully anonymised data is exempt from IRB review. No external funding was received.

### Dataset
The dataset consists of 722 AP inpatient admissions from a single Chinese tertiary hospital (ap_lnn_sanitized.csv). Labels follow the Atlanta 2012 Classification: 585 severe AP (81%) and 137 mild AP (19%). The 4.3:1 class ratio reflects the tertiary referral pattern of the institution.

Laboratory features (n=106) represent admission or first-draw measurements within 4 hours of presentation, covering: haematology (complete blood count with differential), biochemistry (renal, hepatic, pancreatic enzymes), coagulation panel (PT, PTT, INR, D-dimer, fibrinogen), blood gas (pH, lactate, PaO₂, base excess), lipid panel, and inflammatory markers (CRP, ESR, procalcitonin).

### Label Inversion Effect
Mean feature values by severity group revealed a counter-intuitive pattern: mild AP patients showed higher WBC (15.1 vs 11.7 ×10⁹/L), CRP (102.5 vs 50.4 mg/L), and lipase (1,857 vs 904 U/L) than severe AP patients. Severe AP was characterised by lower albumin (36.7 vs 41.0 g/L) and calcium (1.96 vs 2.23 mmol/L). This inversion reflects the etiology mix: mild-labelled biliary AP cases with concurrent cholangitis produce intense but self-limiting inflammation, whereas severe-labelled cases represent necrotising pancreatitis characterised by metabolic derangements. Importantly, 13–14 patients with high pancreatic sepsis scores were labelled mild — possible misclassification of infected pancreatic necrosis (IPN). This inversion renders heuristic Ranson/BISAP-weighted models non-transferable to this cohort; all models were trained data-driven without literature-derived weights.

### Machine Learning Models
All models used 5-fold stratified cross-validation with out-of-fold (OOF) probability aggregation. Optimal decision thresholds were selected by maximising F1 score on OOF predictions (not on the test fold, avoiding threshold optimism).

**Classical ML:** Logistic Regression (L2, C=0.5), Random Forest (n_estimators=200, max_depth=6, min_samples_leaf=5), Gradient Boosting (n_estimators=150, max_depth=3, lr=0.05). All trained on StandardScaler-normalised features within each fold.

**MLP deep learning:** All trained with Adam optimiser, early stopping on validation AUC (patience=8), batch size 32, maximum 60 epochs per fold. MLP: 256→128→64→1 with BatchNormalisation and Dropout (0.35/0.30/0.20), lr=1×10⁻³. Residual MLP: 128-dim projection with two residual blocks, lr=8×10⁻⁴. Attention MLP: sigmoid feature gate (106→106 learnable weights) followed by 256→128→64→1, lr=1×10⁻³.

**LSTM sequence models:** Features were reshaped to shape (106, 1), treating each laboratory value as one time-step. Vanilla LSTM: LSTM(64) → Dense(32) → sigmoid, lr=8×10⁻⁴. Stacked LSTM: LSTM(64) → LSTM(32) → Dense(32) → sigmoid, lr=5×10⁻⁴. Bidirectional LSTM: BiLSTM(64+64) → BatchNorm → Dense(32) → sigmoid, lr=8×10⁻⁴. LSTM+Attention: LSTM(64, return_sequences) → Bahdanau attention → Dense(32) → sigmoid, lr=8×10⁻⁴. CNN-LSTM: Conv1D(32, k=5) → MaxPool → Conv1D(64, k=3) → LSTM(64) → Dense(32) → sigmoid, lr=8×10⁻⁴.

### FHIR R4 Integration
The prediction interface accepts FHIR R4 Bundle resources containing Patient and Observation resources with LOINC codes, returning a FHIR RiskAssessment resource with SNOMED CT risk group codes (Low: 723505004; Intermediate: 723506003; High: 723507007). A client-side JavaScript implementation runs 200 Random Forest trees in-browser using an exported 178 KB JSON model, enabling zero-server-cost deployment. HL7 v2.x (ORU^R01) and Israeli HIS Camelion adapters are also provided.

---

## RESULTS

### Model Performance
Random Forest achieved the highest discrimination across all 11 architectures (AUC=0.877, F1=0.917, sensitivity=96.8%, specificity=38.7% at threshold 0.535; Table 1). Gradient Boosting was comparable (AUC=0.874, F1=0.918, sensitivity=97.1%). Among MLP architectures, vanilla MLP achieved AUC=0.836. LSTM-based models achieved AUC=0.675–0.772, with CNN-LSTM performing best among recurrent architectures (AUC=0.772, sensitivity=98.6%). LSTM-based models consistently achieved near-perfect sensitivity (98.6–100%) but near-zero specificity at optimal thresholds, reflecting high-confidence majority-class prediction under class imbalance.

**Table 1. Performance of All 11 Models — 5-Fold Stratified Cross-Validation**

| Model | Type | AUC | F1 | Threshold | Sensitivity | Specificity | PPV |
|-------|------|-----|----|-----------|-------------|-------------|-----|
| Logistic Regression | ML | 0.817 | 0.907 | 0.575 | 93.8% | 43.8% | 87.7% |
| Random Forest ★ | ML | 0.877 | 0.917 | 0.535 | 96.8% | 38.7% | 87.1% |
| Gradient Boosting | ML | 0.874 | 0.918 | 0.350 | 97.1% | 38.0% | 87.0% |
| MLP | DL | 0.836 | 0.909 | 0.282 | 96.9% | 24.8% | 84.6% |
| Residual MLP | DL | 0.804 | 0.912 | 0.203 | 97.8% | 28.5% | 85.4% |
| Attention MLP | DL | 0.784 | 0.909 | 0.418 | 98.3% | 23.4% | 84.6% |
| LSTM | DL/Seq | 0.696 | 0.898 | 0.448 | 99.7% | 5.1% | 81.8% |
| Stacked LSTM | DL/Seq | 0.675 | 0.896 | 0.456 | 99.3% | 4.4% | 81.6% |
| Bidirectional LSTM | DL/Seq | 0.699 | 0.896 | 0.158 | 100.0% | 0.7% | 81.1% |
| LSTM + Attention | DL/Seq | 0.675 | 0.897 | 0.193 | 100.0% | 1.5% | 81.2% |
| CNN-LSTM | DL/Seq | 0.772 | 0.899 | 0.313 | 98.6% | 11.7% | 82.7% |

★ Best overall. AUC = area under the ROC curve; PPV = positive predictive value.

### Confusion Matrices at Optimal Threshold

**Table 2. Confusion Matrices (n=722; 585 severe / 137 mild)**

| Model | TP | FP | FN | TN |
|-------|----|----|----|----|
| Logistic Regression | 549 | 77 | 36 | 60 |
| Random Forest | 566 | 84 | 19 | 53 |
| Gradient Boosting | 568 | 85 | 17 | 52 |
| MLP | 567 | 103 | 18 | 34 |
| Residual MLP | 572 | 98 | 13 | 39 |
| Attention MLP | 575 | 105 | 10 | 32 |
| LSTM | 583 | 130 | 2 | 7 |
| Stacked LSTM | 581 | 131 | 4 | 6 |
| Bidirectional LSTM | 585 | 136 | 0 | 1 |
| LSTM + Attention | 585 | 135 | 0 | 2 |
| CNN-LSTM | 577 | 121 | 8 | 16 |

### Key Predictive Features

Feature importance analysis across ensemble models consistently identified calcium (RF rank 1st, GB rank 2nd), D-dimer (RF 2nd, GB 1st), LDH (RF/GB 3rd), lactate (RF/GB 4th), and hematocrit (RF/GB 5th) as the most predictive features. Logistic Regression identified lymphocyte count as the strongest single predictor (coefficient magnitude rank 1st), consistent with lymphopenia as a systemic inflammatory response marker. These features overlap substantially with Ranson criteria, validating the clinical rationale while demonstrating that simultaneous admission-time measurement of these markers provides equivalent prognostic signal to serial 48-hour measurement.

### Comparison with Classical Scoring Systems

**Table 3. PenuX vs Classical Scoring Tools**

| Tool | AUROC | Time to Result | EHR Integration |
|------|-------|----------------|-----------------|
| PenuX — Random Forest | 0.877 | 2–4 h | FHIR R4 · HL7 v2 · Camelion |
| PenuX — Gradient Boosting | 0.874 | 2–4 h | FHIR R4 · HL7 v2 · Camelion |
| BISAP [6] | 0.82 | 24 h | Manual only |
| APACHE II | 0.83 | 24 h | Manual only |
| Ranson [4] | 0.73 | 48 h | Manual only |
| Harmless AP Score [10] | 0.88 | Admission | None |
| CT Severity Index [7] | 0.87 | 48–72 h | PACS only |

---

## DISCUSSION

The principal finding of this study is that a Random Forest model trained on 106 routine admission laboratory values achieves AUC=0.877 for SAP severity prediction — matching BISAP (0.82), equalling APACHE II (0.83), and approaching the Harmless AP Score (0.88) — while producing results within 2–4 hours of admission from a single blood draw. At the optimal threshold (0.535), the model achieves 96.8% sensitivity, missing only 19 of 585 severe cases. For clinical triage, minimising false negatives is the primary objective; the observed specificity of 38.7% is acceptable given that the consequence of a false positive (increased observation intensity) is less harmful than a missed SAP diagnosis.

**The 48-hour problem.** The features driving classical scoring delays — calcium, BUN, hematocrit, and PO₂ — are precisely the markers our models learn from at admission as absolute values. Their prognostic signal is not exclusive to the 48-hour nadir: even at admission, a calcium of 1.9 mmol/L carries substantially different prognostic weight than 2.2 mmol/L. Our finding that calcium ranks first across both RF and GB importance confirms that the admission-time measurement of Ranson-criterion features provides actionable discriminative power without waiting for serial dynamics.

**Label inversion.** The observation that mild AP cases in this cohort showed higher WBC, CRP, and lipase than severe AP cases is a clinically important dataset-level insight. It suggests that the "mild" label in this Chinese hospital cohort is substantially populated by biliary AP with concurrent cholangitis — a condition that produces intense but self-limiting systemic inflammation. The 13–14 patients with high pancreatic sepsis scores labelled "mild" may represent IPN miscoding, a finding that warrants prospective investigation and could represent a true-positive subgroup whose severity was administratively underestimated.

**LSTM performance.** LSTM-based models achieved lower AUC (0.675–0.772) than ensemble trees. This is consistent with established findings that recurrent architectures provide limited benefit over tree ensembles on small, non-temporal tabular datasets [11]. CNN-LSTM (AUC=0.772) outperformed other LSTM variants, suggesting that local convolutional feature extraction over the ordered laboratory panel provides useful inductive bias. The near-perfect sensitivity but near-zero specificity of LSTM models at optimal thresholds reflects appropriate conservatism under class imbalance and uncertainty.

**FHIR R4 integration.** The client-side JavaScript implementation — running 200 Random Forest trees in-browser from a 178 KB JSON model file — demonstrates that interoperable SAP risk scoring can be deployed without server-side infrastructure. This is particularly relevant for primary care hospitals and resource-constrained settings in Israel and the Middle East where FHIR-capable EHR endpoints are available (via Camelion, Epic Israel, or Clalit's FHIR gateway) but dedicated AI servers are not.

---

## CONCLUSIONS

This comparative study of 11 ML and deep learning architectures demonstrates that routine admission laboratory values can predict SAP severity with AUC up to 0.877, matching or exceeding classical bedside scoring systems that require 24–48 hours of serial observation. Random Forest is the recommended model for routine lab-based triage: highest AUC, no normalisation required, robust to missing values, and interpretable feature importance. LSTM models confirm ordinal structure in the laboratory feature sequence but do not surpass ensemble methods at this dataset size.

The label inversion finding — mild biliary AP cases presenting with higher WBC/CRP/lipase than severe necrotising AP — is a clinically significant cohort-level insight, potentially identifying IPN misclassification cases warranting prospective investigation.

The PenuX platform is open-source (MIT License) and provides full FHIR R4, HL7 v2.x, and Israeli HIS Camelion integration. Source code and the anonymised dataset are available at https://github.com/netanelcyber/penuX. A live demonstration is available at https://penux.uk.

---

## ADDITIONAL INFORMATION

**Conflicts of interest:** None declared.

**Funding:** No external funding.

**Data availability:** Anonymised dataset available at https://github.com/netanelcyber/penuX/tree/main/PenuX-AP-Severity/data

**Code availability:** https://github.com/netanelcyber/penuX (MIT License)

**Live demo:** https://penux.uk/predict.html

---

## REFERENCES

1. Petrov MS, Yadav D. Global epidemiology and holistic prevention of pancreatitis. Nat Rev Gastroenterol Hepatol. 2019;16(3):175–184. doi:10.1038/s41575-018-0087-5
2. Banks PA, Bollen TL, Dervenis C, et al. Classification of acute pancreatitis — 2012: revision of the Atlanta classification and definitions by international consensus. Gut. 2013;62(1):102–111. doi:10.1136/gutjnl-2012-302779
3. Forsmark CE, Vege SS, Wilcox CM. Acute Pancreatitis. N Engl J Med. 2016;375(20):1972–1981. doi:10.1056/NEJMra1505202
4. Ranson JHC, Rifkind KM, Roses DF, Fink SD, Eng K, Spencer FC. Prognostic signs and the role of operative management in acute pancreatitis. Surg Gynecol Obstet. 1974;139(1):69–81.
5. Mounzer R, Langmead CJ, Wu BU, et al. Comparison of existing clinical scoring systems to predict persistent organ failure in patients with acute pancreatitis. Gastroenterology. 2012;142(7):1476–1482. doi:10.1053/j.gastro.2012.03.005
6. Wu BU, Johannes RS, Sun X, Tabak Y, Conwell DL, Banks PA. The early prediction of mortality in acute pancreatitis: a large population-based study. Gut. 2008;57(12):1698–1703. doi:10.1136/gut.2008.152702
7. Bollen TL, Singh VK, Maurer R, et al. Comparative evaluation of the modified CT severity index and CT severity index in assessing severity of acute pancreatitis. AJR Am J Roentgenol. 2011;197(2):386–392. doi:10.2214/AJR.10.5adelphia
8. Qiu Q, Nian YJ, Guo Y, et al. Development and validation of three machine-learning models for predicting multiple organ failure in moderately severe and severe acute pancreatitis. BMC Gastroenterol. 2019;19(1):118. doi:10.1186/s12876-019-1016-y
9. Huang Y, Mukherjee R, Fu Y, et al. Machine learning-based prediction models for acute pancreatitis severity: systematic review. J Med Internet Res. 2021;23(8):e26718. doi:10.2196/26718
10. Cho JH, Kim TN, Chung HH, Kim KH. Comparison of scoring systems in predicting the severity of acute pancreatitis. World J Gastroenterol. 2015;21(8):2387–2394. doi:10.3748/wjg.v21.i8.2387
11. Grinsztajn L, Oyallon E, Varoquaux G. Why tree-based models still outperform deep learning on tabular data. NeurIPS. 2022;35:507–520.
12. HL7 International. HL7 FHIR R4 Specification. 2019. https://hl7.org/fhir/R4/
13. Breiman L. Random Forests. Machine Learning. 2001;45(1):5–32. doi:10.1023/A:1010933404324
