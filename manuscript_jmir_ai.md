# PenuX: Machine Learning and Deep Learning Models for Early Severity Prediction of Acute Pancreatitis Using Routine Admission Laboratory Values — A Comparative Study of 11 Architectures with FHIR R4 Integration

**Netanel Shoshany**  
Independent Clinical AI Researcher  
Email: nsh531@gmail.com  
GitHub: https://github.com/netanelcyber/penuX  
Web: https://penux.uk  

---

## Abstract

**Background:** Severe Acute Pancreatitis (SAP) carries a mortality rate of 20–30% and requires early risk stratification. Classical scoring tools — BISAP, Ranson, APACHE II — require 24–48 hours of serial observation data and are not designed for automated EHR integration.

**Objective:** To develop and compare 11 machine learning and deep learning models for SAP severity prediction using routine admission laboratory values, and to evaluate feasibility of FHIR R4 integration for automated clinical deployment.

**Methods:** Retrospective cohort of 722 AP admissions (585 severe / 137 mild, Atlanta 2012 classification) from a single Chinese institution. Eleven models were trained with 5-fold stratified cross-validation on 106 routine laboratory features: three classical ML models (Logistic Regression, Random Forest, Gradient Boosting), three MLP-based deep learning models (MLP, Residual MLP, Attention MLP), and five LSTM-based sequence models (Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, LSTM+Attention, CNN-LSTM). Optimal decision thresholds were selected by maximum F1 score on out-of-fold predictions.

**Results:** Random Forest achieved the highest AUC (0.877; 95% CI estimated by fold variance), F1=0.917, sensitivity=96.8%, specificity=38.7% at threshold 0.535. Gradient Boosting was comparable (AUC=0.874, F1=0.918, sensitivity=97.1%). Among MLP architectures, vanilla MLP achieved AUC=0.836. LSTM-based models achieved AUC=0.675–0.772, with CNN-LSTM performing best among recurrent architectures (AUC=0.772, sensitivity=98.6%). Key predictive features across models: calcium, D-dimer, LDH, lactate, and hematocrit. A clinically significant label inversion effect was identified: "mild" biliary AP cases presented with higher WBC, CRP, and lipase than "severe" necrotising AP cases, explaining sub-chance performance of heuristic models and highlighting possible IPN misclassification.

**Conclusions:** Random Forest achieves SAP triage with AUC 0.877 from a single admission blood draw, eliminating the 24–48 hour observation window required by classical scoring systems. The system provides full FHIR R4, HL7 v2.x, and Israeli HIS (Camelion) integration for automated clinical deployment. External validation on Western and Israeli cohorts is required before clinical use.

**Keywords:** acute pancreatitis; severe acute pancreatitis; machine learning; deep learning; LSTM; random forest; FHIR R4; clinical prediction model; severity scoring; EHR integration

---

## 1. Introduction

Acute Pancreatitis (AP) is one of the most common causes of emergency gastrointestinal hospitalisation, with an estimated global incidence of 34 cases per 100,000 persons per year [1]. Approximately 20% of cases progress to Severe Acute Pancreatitis (SAP), defined by the 2012 Revised Atlanta Classification as AP with persistent organ failure (>48 hours) [2]. SAP carries a mortality rate of 20–30%, principally from infected pancreatic necrosis, abdominal compartment syndrome, and multi-organ dysfunction syndrome [3].

Early identification of SAP at or near admission is critical: patients who will develop organ failure require immediate ICU transfer, aggressive fluid resuscitation, and multidisciplinary management. However, the bedside scoring tools used for this purpose — Ranson (1974) [4], APACHE II, and BISAP (2008) [5] — share a fundamental limitation: their final severity score cannot be computed at admission.

**The 48-hour validation problem.** Ranson's criteria require 6 of 11 parameters to be re-evaluated at 48 hours (including ΔBUN, ΔHematocrit, Ca²⁺ <8 mg/dL, PaO₂ <60 mmHg, base deficit >4 mEq/L, fluid sequestration >6 L). These are delta values — requiring two time points. This delay reflects the two-wave pathophysiology of SAP: enzymatic acinar cell injury occurs in the first hours (Wave 1), while systemic inflammatory response, cytokine storm (IL-6, TNF-α), and target-organ injury (renal, pulmonary, cardiovascular) develop over 24–48 hours (Wave 2) [6]. Similarly, CT Severity Index requires imaging at 48–72 hours, because pancreatic necrosis is not reliably visible on early-phase contrast-enhanced CT [7].

A substantial body of literature has demonstrated that machine learning models trained on admission laboratory data can match or exceed classical scoring systems while producing results within 2–4 hours of admission [8,9]. However, most published models use a limited feature set (5–20 markers), single architectures, and do not address EHR integration.

**Study objectives.** This study aims to: (1) evaluate 11 ML and DL architectures on a 106-feature admission laboratory dataset; (2) characterise a clinically significant label inversion effect in this cohort; (3) demonstrate FHIR R4 integration for automated SAP risk scoring; and (4) release an open-source platform (PenuX) for reproducibility and further research.

---

## 2. Methods

### 2.1 Dataset and Cohort

The dataset consists of 722 AP inpatient admissions from a single Chinese tertiary hospital, exported as a de-identified CSV (`ap_lnn_sanitized.csv`). Labels follow the Atlanta 2012 Classification (ICD-coded): 585 severe AP (81%) and 137 mild AP (19%).

**Laboratory features (n=106):** Haematology (CBC differential), biochemistry (renal, hepatic, pancreatic enzymes), coagulation (PT, PTT, INR, D-dimer, fibrinogen), blood gas (pH, lactate, PaO₂, base excess), lipid panel, and inflammatory markers (CRP, ESR, procalcitonin). All values represent admission or first-draw measurements within 4 hours of presentation.

**Ethics:** This analysis uses a pre-existing anonymised retrospective dataset. No patient identifiers are retained. Under institutional policy, retrospective analysis of fully anonymised data is exempt from IRB review.

### 2.2 Label Inversion Effect

A systematic comparison of mean feature values by severity group revealed a counter-intuitive pattern: mild AP patients showed higher WBC (15.1 vs 11.7 ×10⁹/L), CRP (102.5 vs 50.4 mg/L), and lipase (1,857 vs 904 U/L) than severe AP patients. Severe AP was characterised by lower albumin (36.7 vs 41.0 g/L) and calcium (1.96 vs 2.23 mmol/L).

This inversion is consistent with the likely etiology mix in a Chinese tertiary hospital: mild-labelled biliary AP cases with concurrent cholangitis produce an intense inflammatory response without progressing to organ failure, whereas severe-labelled cases represent necrotising pancreatitis dominated by hypoalbuminaemia and hypocalcaemia. Importantly, 13–14 patients with high pancreatic sepsis scores were labelled mild — possible ICD misclassification of infected pancreatic necrosis (IPN). This inversion explains why Ranson/BISAP-weighted heuristic models applied to this cohort yield AUC <0.5; data-driven models trained on the cohort itself are necessary.

### 2.3 Model Architectures

All models used 5-fold stratified cross-validation with out-of-fold (OOF) probability aggregation. Optimal thresholds were selected by maximising F1 score on OOF predictions.

**Classical ML models:** All three models used StandardScaler normalisation within each fold. Logistic Regression: L2 regularisation, C=0.5, max_iter=1000. Random Forest: n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42. Gradient Boosting: n_estimators=150, max_depth=3, learning_rate=0.05.

**MLP-based deep learning models:** All trained with Adam optimiser, early stopping on validation AUC (patience=8), batch size 32, maximum 60 epochs per fold. MLP: 256→128→64→1 with BatchNorm and Dropout (0.35/0.30/0.20), lr=1e-3. Residual MLP: 128-dim projection with 2 residual blocks and skip connections, lr=8e-4. Attention MLP: sigmoid feature gate (106→106) followed by 256→128→64→1, lr=1e-3.

**LSTM-based sequence models:** Features are reshaped to sequence shape (106, 1), treating each laboratory value as one time-step. This allows recurrent units to capture ordinal co-occurrence patterns within related panels (e.g., coagulation cascade: D-dimer → fibrinogen → PT → PTT). Vanilla LSTM: LSTM(64) → Dense(32) → sigmoid, lr=8e-4. Stacked LSTM: LSTM(64) → LSTM(32) → Dense(32) → sigmoid, lr=5e-4. Bidirectional LSTM: BiLSTM(64+64) → BatchNorm → Dense(32) → sigmoid, lr=8e-4. LSTM+Attention: LSTM(64, return_sequences) → Bahdanau attention → Dense(32) → sigmoid, lr=8e-4. CNN-LSTM: Conv1D(32, k=5) → MaxPool → Conv1D(64, k=3) → LSTM(64) → Dense(32) → sigmoid, lr=8e-4.

### 2.4 FHIR R4 Integration

The prediction API accepts FHIR R4 Bundle resources containing Patient and Observation resources with LOINC codes. The response is a FHIR RiskAssessment resource with SNOMED CT risk group codes (Low: 723505004, Intermediate: 723506003, High: 723507007) and probabilityDecimal output. A client-side JavaScript implementation (traverseTree / predictRF) runs the 200-tree Random Forest entirely in-browser using the exported model JSON (178 KB), enabling zero-server-cost deployment. HL7 v2.x (ORU^R01) and Israeli HIS Camelion adapters are also provided.

---

## 3. Results

### 3.1 Model Performance — All 11 Architectures

| Model | Type | AUC | F1 | Threshold | Sensitivity | Specificity | PPV |
|-------|------|-----|----|-----------|-------------|-------------|-----|
| Logistic Regression | ML | 0.817 | 0.907 | 0.575 | 93.8% | 43.8% | 87.7% |
| **Random Forest ★** | **ML** | **0.877** | **0.917** | **0.535** | **96.8%** | **38.7%** | **87.1%** |
| Gradient Boosting | ML | 0.874 | 0.918 | 0.350 | 97.1% | 38.0% | 87.0% |
| MLP (3-layer) | DL | 0.836 | 0.909 | 0.282 | 96.9% | 24.8% | 84.6% |
| Residual MLP | DL | 0.804 | 0.912 | 0.203 | 97.8% | 28.5% | 85.4% |
| Attention MLP | DL | 0.784 | 0.909 | 0.418 | 98.3% | 23.4% | 84.6% |
| LSTM | DL/Seq | 0.696 | 0.898 | 0.448 | 99.7% | 5.1% | 81.8% |
| Stacked LSTM | DL/Seq | 0.675 | 0.896 | 0.456 | 99.3% | 4.4% | 81.6% |
| Bidirectional LSTM | DL/Seq | 0.699 | 0.896 | 0.158 | 100.0% | 0.7% | 81.1% |
| LSTM + Attention | DL/Seq | 0.675 | 0.897 | 0.193 | 100.0% | 1.5% | 81.2% |
| CNN-LSTM | DL/Seq | 0.772 | 0.899 | 0.313 | 98.6% | 11.7% | 82.7% |

*★ Best overall model. All metrics from 5-fold stratified cross-validation OOF predictions.*

### 3.2 Confusion Matrices at Optimal Threshold (n=722; 585 severe / 137 mild)

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

### 3.3 Key Predictive Features

Feature importance analysis across ML models consistently identified the same cluster of biomarkers:

| Feature | LR (rank) | RF (rank) | GB (rank) | Clinical Rationale |
|---------|-----------|-----------|-----------|-------------------|
| Calcium | 2nd | 1st | 2nd | Hypocalcaemia from saponification in necrotic fat; Ranson criterion |
| D-dimer | 4th | 2nd | 1st | Coagulopathy / DIC in severe disease |
| LDH | 5th | 3rd | 3rd | Tissue necrosis marker; Ranson criterion (>250 U/L) |
| Lactate | 4th | 4th | 4th | Hypoperfusion / organ dysfunction |
| Hematocrit | — | 5th | 5th | Haemoconcentration — early marker of necrotising pancreatitis |
| Lymphocytes | 1st | — | — | Lymphopenia in systemic inflammatory response |
| Creatinine | 3rd | 9th | 10th | AKI — Ranson criterion |

### 3.4 Comparison with Classical Scoring Systems

| Tool | AUROC | Time to Result | EHR Integration |
|------|-------|----------------|-----------------|
| **PenuX — Random Forest** | **0.877** | **2–4 h (admission)** | **FHIR · HL7 · Camelion** |
| PenuX — Gradient Boosting | 0.874 | 2–4 h | FHIR · HL7 · Camelion |
| BISAP | 0.82 | 24 h | Manual |
| APACHE II | 0.83 | 24 h | Manual |
| Ranson | 0.73 | 48 h | Manual |
| Harmless AP Score | 0.88 | Admission | None |
| CT Severity Index | 0.87 | 48–72 h (post-CT) | PACS only |

---

## 4. Discussion

### 4.1 Clinical Significance

The principal finding is that a Random Forest model trained on 106 routine admission laboratory values achieves AUC=0.877 for SAP severity prediction, matching or exceeding BISAP (0.82) and APACHE II (0.83) while producing results within 2–4 hours of admission. At the optimal threshold (0.535), the model achieves 96.8% sensitivity — missing only 19 of 585 severe cases — with a specificity of 38.7%. This trade-off is appropriate for triage purposes: minimising false negatives (missed SAP) is the primary clinical objective.

The 48-hour diagnostic delay imposed by Ranson and similar systems is clinically dangerous for the highest-risk patients. The features driving the delay (ΔBUN, ΔHematocrit, PaO₂, Ca²⁺ nadir) are precisely the markers our model learns from at admission — not as serial deltas but as admission-time absolute values that already carry prognostic signal.

### 4.2 Label Inversion as a Generalisability Signal

The label inversion effect identified in this cohort — mild cases showing higher WBC, CRP, and lipase than severe cases — is a methodologically important finding. It indicates that severity labels in this dataset do not follow the Western-population biomarker directionality assumed by BISAP, Ranson, and most published prediction models. This likely reflects the dominance of biliary AP with concurrent cholangitis (producing intense but self-limiting inflammation) in the mild group, versus pancreatic necrosis (producing the metabolic derangements of calcium and albumin loss) in the severe group. The 13–14 patients with high pancreatic-sepsis risk scores labelled "mild" may represent IPN misclassification — a clinically actionable finding warranting prospective investigation.

### 4.3 LSTM Performance

LSTM-based models achieved lower AUC (0.675–0.772) than ensemble tree models, consistent with the known limitations of recurrent architectures on small, non-temporal tabular datasets (n=722). The CNN-LSTM variant performed best among sequence models (AUC=0.772), suggesting that local convolutional feature extraction over the lab panel sequence provides useful inductive bias. All LSTM models achieved very high sensitivity (98.6–100.0%) at the cost of near-zero specificity at optimal thresholds, reflecting the class imbalance (4.3:1) and model tendency to predict majority-class membership under high uncertainty.

### 4.4 FHIR R4 Integration

The client-side implementation of Random Forest inference in JavaScript — running 200 trees against 106 features in the browser using an exported 178 KB JSON model — demonstrates that FHIR-integrated SAP risk scoring can be deployed without server-side infrastructure. This is significant for resource-constrained healthcare settings where API endpoints may not be available or maintainable.

---

## 5. Limitations

1. **Single-institution cohort:** All 722 admissions originate from one Chinese hospital. Etiology mix, lab reference ranges, and severity classification may differ in Israeli, Western, or mixed-etiology populations.
2. **No external validation set:** All performance figures are from 5-fold cross-validation on the same cohort. An independent holdout set or prospective validation study is required before clinical translation.
3. **Class imbalance:** 585 severe vs 137 mild (4.3:1 ratio). Models are optimised for sensitivity; specificity is consistently lower (1–44%) across architectures.
4. **DL dataset size:** n=722 is near the lower bound of DL benefit for tabular data. Rapid convergence (7–11 epochs for MLPs) and lower AUC than ensemble trees confirm this. A larger multi-site dataset would likely narrow the gap.
5. **Label quality:** 13–14 patients with high sepsis-risk markers were labelled "mild" — possible ICD coding misclassification of IPN, which may suppress true model performance.
6. **Regulatory status:** Clinical use requires ethics committee approval, Ministry of Health certification, and registration as Software as a Medical Device (SaMD, MDR Class IIa / FDA 510(k)).

---

## 6. Future Directions

- External validation on Israeli multi-site data (Sheba, Hadassah, Rambam) under prospective IRB approval
- Integration of CT findings (Modified CTSI) via FHIR ImagingStudy resources
- Federated Learning for distributed training without cross-site data sharing
- XGBoost and TabNet evaluation on larger datasets
- Prospective investigation of IPN misclassification in the mild-labelled high-risk subgroup
- Point-of-care mobile app for emergency physicians

---

## 7. Conclusions

PenuX demonstrates that routine admission laboratory values, processed by ensemble ML models, can identify Severe Acute Pancreatitis with AUC up to 0.877 — matching or exceeding classical bedside scoring tools that require 24–48 hours of serial observation. Random Forest is the top performer; Gradient Boosting offers equivalent F1 and higher sensitivity. LSTM-based sequence models confirm ordinal structure in the feature space but do not surpass ensemble methods at this dataset size.

The label inversion finding — mild biliary AP cases presenting with higher WBC/CRP/lipase than severe necrotising AP — is a clinically significant dataset-level insight, potentially identifying a subgroup of IPN misclassification warranting prospective investigation.

The platform is open-source (MIT License) and provides full FHIR R4, HL7 v2.x, and Camelion integration, enabling automated SAP risk scoring within hours of admission without 24–48 hour waiting periods.

---

## Conflicts of Interest

None declared.

## Funding

No external funding was received for this study.

## Data Availability

The anonymised dataset (`ap_lnn_sanitized.csv`) is available in the project repository at https://github.com/netanelcyber/penuX under the PenuX-AP-Severity/data/ directory. No patient identifiers are retained.

## Code Availability

All training code, model architectures, evaluation scripts, and the client-side prediction interface are available at: https://github.com/netanelcyber/penuX (MIT License). A live demonstration is available at https://penux.uk/predict.html.

---

## References

1. Petrov MS, Yadav D. Global epidemiology and holistic prevention of pancreatitis. Nat Rev Gastroenterol Hepatol. 2019;16(3):175–184.
2. Banks PA, Bollen TL, Dervenis C, et al. Classification of acute pancreatitis — 2012: revision of the Atlanta classification and definitions by international consensus. Gut. 2013;62(1):102–111.
3. Forsmark CE, Vege SS, Wilcox CM. Acute Pancreatitis. N Engl J Med. 2016;375(20):1972–1981.
4. Ranson JHC, Rifkind KM, Roses DF, et al. Prognostic signs and the role of operative management in acute pancreatitis. Surg Gynecol Obstet. 1974;139(1):69–81.
5. Wu BU, Johannes RS, Sun X, et al. The early prediction of mortality in acute pancreatitis: a large population-based study. Gut. 2008;57(12):1698–1703.
6. Mounzer R, Langmead CJ, Wu BU, et al. Comparison of existing clinical scoring systems to predict persistent organ failure in patients with acute pancreatitis. Gastroenterology. 2012;142(7):1476–1482.
7. Bollen TL, Singh VK, Maurer R, et al. Comparative evaluation of the modified CT severity index and CT severity index in assessing severity of acute pancreatitis. AJR. 2011;197(2):386–392.
8. Qiu Q, Nian YJ, Guo Y, et al. Development and validation of three machine-learning models for predicting multiple organ failure in moderately severe and severe acute pancreatitis. BMC Gastroenterol. 2019;19(1):118.
9. Huang Y, Mukherjee R, Fu Y, et al. Machine learning-based prediction models for acute pancreatitis severity: systematic review. J Med Internet Res. 2021;23(8):e26718.
10. Breiman L. Random Forests. Machine Learning. 2001;45(1):5–32.
11. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. NeurIPS. 2017.
12. HL7 International. HL7 FHIR R4 Specification. 2019. https://hl7.org/fhir/R4/
13. Dellinger EP, Forsmark CE, Layer P, et al. Determinant-based classification of acute pancreatitis severity. Ann Surg. 2012;256(6):875–880.
