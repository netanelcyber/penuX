# PenuX-SAP Stage I Observational Cohort Study Protocol

**Full Title:** Prospective Observational Validation of a Machine-Learning–Based Severity Prediction System (PenuX) in Adults Admitted with Acute Pancreatitis: A Stage I Cohort Study

**Protocol Version:** 1.0  
**Date:** June 2026  
**Principal Investigator:** Netanel Stern, PenuX Research Initiative  
**Contact:** netanel@penux.uk | +972-55-970-8708  
**ClinicalTrials.gov:** [Pending Registration]

---

## 1. Background and Scientific Rationale

Acute pancreatitis (AP) is one of the most common gastrointestinal emergencies, with a global annual incidence of approximately 34 per 100,000 persons and a rising trend. Roughly 10–20% of patients progress to severe acute pancreatitis (SAP), defined by the 2012 Revised Atlanta Classification (RAC) as persistent organ failure lasting more than 48 hours. SAP carries a mortality of 20–40% and places substantial demands on intensive care resources.

Current severity scoring systems — Ranson criteria, APACHE II, BISAP, and the Modified CT Severity Index — require 24–48 hours of observation before reliable stratification is possible, or depend on contrast-enhanced CT imaging that may be unsafe in the acute phase. This delay limits early triage and timely escalation of care.

PenuX is a machine-learning ensemble that predicts SAP severity at the time of hospital admission using routine laboratory values. In a retrospective computational study of 722 AP patients labelled according to the 2012 Revised Atlanta Classification, PenuX (Random Forest core) achieved AUC-ROC 0.877, sensitivity 96.8%, F1 0.917 and PPV 87.1% — outperforming all deep-learning architectures tested. A secondary pancreatic sepsis risk sub-model provides probabilistic sepsis risk stratification at admission.

**This Stage I study is the first prospective external validation** of the PenuX system in a real-world hospital population, conducted in silent-mode (prediction-only, no clinical intervention). The outputs of this stage will inform the design of a Stage II prospective interventional pilot.

---

## 2. Objectives

### 2.1 Primary Objective
Assess the external discriminative validity of the PenuX SAP severity prediction model in a prospective cohort of AP admissions, measured by AUC-ROC against the RAC gold-standard severity label at 48 hours.

### 2.2 Secondary Objectives
1. Evaluate sensitivity, specificity, PPV, NPV, and F1 at the pre-specified operating threshold (optimised for maximal F1 in the development set).
2. Assess calibration of predicted probabilities (Brier score, reliability diagram, Hosmer–Lemeshow test).
3. Determine the lead-time advantage over BISAP score (comparing prediction accuracy at admission vs. 24 h).
4. Evaluate the secondary pancreatic sepsis risk sub-model against culture-confirmed or clinically diagnosed pancreatic sepsis.
5. Characterise any systematic performance differences across age, sex, aetiology (gallstone vs. alcohol vs. other), and BMI strata.
6. Quantify clinical alert burden: proportion of patients flagged as high-risk, false-alert rate.
7. Assess feasibility of data harmonisation between local EHR variables and PenuX input features.

---

## 3. Study Design

| Parameter | Value |
|-----------|-------|
| Design | Prospective, observational, silent-mode (non-interventional) |
| Setting | One or more hospital sites with dedicated gastroenterology / acute medicine service |
| Duration | 12 months enrolment + 3 months follow-up |
| Comparator | Standard clinical severity scores (BISAP, APACHE II at 24 h) |
| Blinding | Clinicians blinded to PenuX outputs throughout Stage I |
| Ethics | Full IRB/Helsinki Committee approval required before initiation |

**Silent-mode rationale:** PenuX predictions are logged but not shown to treating clinicians during Stage I. This eliminates feedback loops, preserves standard-of-care as the sole treatment driver, and produces unbiased outcome labels.

---

## 4. Study Population

### 4.1 Inclusion Criteria
- Age ≥ 18 years
- Admission diagnosis of acute pancreatitis meeting the 2012 Revised Atlanta Classification diagnostic criteria (≥2 of: characteristic abdominal pain; serum lipase or amylase ≥3× upper limit of normal; characteristic imaging findings)
- Admission within 24 hours of symptom onset
- Availability of routine admission blood tests (see Section 6) within 6 hours of presentation
- Informed consent (or waiver of consent approved by IRB for deferred consent in obtunded patients)

### 4.2 Exclusion Criteria
- Chronic pancreatitis without acute-on-chronic exacerbation
- Post-ERCP pancreatitis (distinct physiology)
- Pre-existing pancreatic malignancy
- Patients admitted directly to ICU from another hospital after ≥48 hours of disease course (severity already established)
- Refusal of consent
- Pregnancy (separate physiological reference ranges; deferred to a dedicated sub-study)

### 4.3 Sample Size

Based on the development-set AUC of 0.877 and assuming a conservative expected external-validation AUC of 0.82 (10% degradation), with H₀: AUC = 0.70, α = 0.05 (one-sided), power = 0.80, and SAP prevalence ~25% in a general hospital setting:

- **Required sample: n = 180 patients** (80 SAP, 100 non-SAP expected)
- Inflated to **n = 220** to account for 15% attrition (incomplete blood panels, loss to follow-up, withdrawal)
- Enrolment target: **~18–20 patients/month** over 12 months at a single site

*Sample size calculation performed using the Obuchowski–Mcclish method for AUC comparison (MedCalc v22).*

---

## 5. Enrolment and Consent Procedure

1. Attending physician or research nurse screens all AP admissions against inclusion/exclusion criteria within 6 hours of presentation.
2. Eligible patients (or their next of kin if patient is obtunded) are approached for written informed consent.
3. An IRB-approved deferred-consent pathway may be used for obtunded patients; full consent obtained at the earliest opportunity.
4. Enrolment logged in REDCap with timestamp, enroller ID, and consent status.

---

## 6. Data Collection

### 6.1 Admission Variables (PenuX Model Inputs)
Collected from the EHR within 6 hours of presentation:

| Domain | Variables |
|--------|-----------|
| **Demographics** | Age, sex, BMI |
| **Inflammatory** | CRP, WBC, Neutrophil %, Neutrophil count, Lymphocyte %, Lymphocyte count, Monocyte %, Monocyte count |
| **Pancreatic enzymes** | Amylase (AMY), Lipase (where available) |
| **Liver/biliary** | ALT, AST, GGT, ALP, TBIL, DBIL, IBIL, ALB, TP, GLB, A/G Ratio |
| **Renal/electrolytes** | Creatinine (Cr), Urea, Na⁺, K⁺, Cl⁻, Ca²⁺ |
| **Coagulation** | PT, APTT, INR, FIB |
| **Metabolic** | Glucose (Glu), TG, LDL-C, CO₂-CP |
| **Haematology** | RBC, Hb, HCT, MCV, MCH, MCHC, RDW-CV, RDW-SD, PLT, PDW, PCT, P-LCR, MPV |
| **Cardiac** | LDH, α-HBDH, CK, CK-MB |
| **Other** | Eosinophil %, Basophil %, Eosinophil count, Basophil count |

### 6.2 Clinical Outcome Variables
| Variable | Timing | Source |
|----------|--------|--------|
| SAP label (RAC 2012) | 48–72 h post-admission | Clinical assessment by gastroenterologist |
| BISAP score | Admission, 24 h | Clinical notes |
| APACHE II score | 24 h | ICU chart / clinical notes |
| Ranson criteria | 48 h | Clinical notes |
| Organ failure (SOFA sub-scores: renal, respiratory, cardiovascular) | Daily ×5 | EHR |
| Pancreatic necrosis (CT) | If CT performed (clinical indication) | Radiology report |
| Infected necrosis / pancreatic sepsis | Confirmed by culture or clinical criteria | Microbiology, clinical notes |
| ICU admission | Any point during admission | EHR |
| Length of hospital stay (days) | Discharge | EHR |
| 30-day mortality | 30 days | Medical record or phone follow-up |
| 30-day readmission | 30 days | Medical record or phone follow-up |

### 6.3 AP Aetiology
Classified by attending physician: gallstone, alcohol, hypertriglyceridaemia, post-ERCP (excluded), idiopathic, other.

---

## 7. PenuX System Operation (Silent Mode)

1. Admission lab values are extracted from the EHR (manually or via HL7 interface) and entered into the PenuX web portal or API endpoint.
2. The system outputs:
   - **SAP Severity Probability** (0–1) and binary classification (high-risk / low-risk) at the pre-specified F1-optimal threshold
   - **Pancreatic Sepsis Risk Probability** (0–1)
   - **Risk group label** (low / moderate / high)
   - **Feature attribution** (top-3 contributing variables per patient)
3. All outputs are logged to a secure, access-controlled database. No output is displayed in clinical systems or to treating staff during Stage I.
4. PenuX outputs are linked to patient records via pseudonymised study ID only.

---

## 8. Primary and Secondary Endpoints

### 8.1 Primary Endpoint
- AUC-ROC of PenuX SAP severity probability vs. RAC 2012 severity label at 48 h

### 8.2 Secondary Endpoints
| Endpoint | Metric |
|----------|--------|
| Sensitivity at pre-specified threshold | % (95% CI) |
| Specificity at pre-specified threshold | % (95% CI) |
| PPV | % (95% CI) |
| NPV | % (95% CI) |
| F1 score | (95% CI, bootstrap) |
| Calibration | Brier score, Hosmer–Lemeshow p-value |
| Lead time vs. BISAP | AUC at admission vs. AUC at 24 h (DeLong test) |
| Pancreatic sepsis AUC | AUC-ROC vs. culture-confirmed/clinical sepsis |
| Subgroup AUC: age ≥65 | AUC-ROC (95% CI) |
| Subgroup AUC: gallstone aetiology | AUC-ROC (95% CI) |
| Alert burden | % patients flagged high-risk per day |
| Data completeness | % of 59 features available at admission |

---

## 9. Statistical Analysis Plan

### 9.1 Primary Analysis
AUC-ROC calculated using the DeLong non-parametric estimator with 95% CI. Pre-specified H₀: AUC ≤ 0.70 (worse than existing scores). One-sided test, α = 0.05.

### 9.2 Secondary Analyses
- Sensitivity/specificity/PPV/NPV: Wilson score 95% CI
- Calibration: Brier score; HL χ² statistic (10 groups); reliability plot
- DeLong test for paired AUC comparison (PenuX-admission vs. BISAP-24h)
- Subgroup analyses: Cochran's Q test for heterogeneity across pre-specified strata; no p-value correction (exploratory)
- Missing data: Multiple imputation by chained equations (MICE, m=10) for features missing in <30% of cases; complete-case sensitivity analysis

### 9.3 Multiple Comparisons
Secondary endpoints are exploratory. No Bonferroni adjustment is applied, but findings are interpreted as hypothesis-generating.

### 9.4 Software
Python 3.11 (scikit-learn, scipy, statsmodels), R 4.4 (pROC, rms), REDCap for data capture.

---

## 10. Data Management and Security

- All patient data stored in REDCap (ISO 27001 certified, hosted within hospital firewall or approved cloud region)
- Pseudonymisation: patient linked to 8-character alphanumeric study ID; linkage key held by site PI only
- PenuX receives only pseudonymised structured data; no free-text, no imaging, no direct identifiers
- Data transfer to PenuX servers (if any) via TLS 1.3 encrypted API; server-side encryption at rest (AES-256)
- Right to access / erasure fulfilled by deleting the study ID from the linkage key
- Data retention: 10 years post-study completion per GCP guidelines; anonymised dataset archived for open science (post-publication)

---

## 11. Ethical and Regulatory Considerations

- **IRB/Helsinki approval** required at each site before enrolment commences
- **Informed consent** obtained in writing per Declaration of Helsinki (2013 revision)
- **Non-interventional**: PenuX outputs are never used for clinical decision-making during Stage I; no additional procedures or interventions are imposed on participants
- **CE/FDA:** PenuX is not a CE-marked or FDA-cleared device; Stage I is conducted under the research exemption. Any future clinical use will require appropriate regulatory clearance
- **IEC 62304 lifecycle documentation** is maintained for the PenuX software module
- **Data Protection:** Compliant with GDPR (EU) / Israeli Privacy Protection Law 5741-1981 as applicable

---

## 12. Site Requirements

| Requirement | Minimum |
|-------------|---------|
| Annual AP admissions | ≥ 80 patients/year |
| Gastroenterology or acute medicine service | Required |
| Clinical PI (MD) | Required |
| Research nurse or data coordinator | Highly recommended |
| EHR with structured lab data | Required |
| REDCap access or equivalent | Required |
| Local IRB/ethics committee | Required |
| Data protection officer sign-off | Required |

---

## 13. Timeline

| Milestone | Target Month |
|-----------|-------------|
| IRB submission | M1 |
| IRB approval | M2–M3 |
| Site setup, staff training, REDCap configuration | M3–M4 |
| First patient enrolled | M4 |
| 50% enrolment (n ≈ 110) | M9 |
| Last patient enrolled | M16 |
| Last patient 30-day follow-up complete | M17 |
| Database lock | M17 |
| Statistical analysis complete | M18 |
| Manuscript submission | M19 |

---

## 14. Outputs and Dissemination

- Primary manuscript: AUC/calibration/lead-time results submitted to a peer-reviewed gastroenterology or clinical AI journal (target: *Gut*, *Pancreatology*, *The Lancet Digital Health*, or *PLOS Medicine*)
- Anonymised dataset and analysis code released to GitHub (github.com/netanelcyber/penuX) upon publication
- Stage II pilot protocol: if primary AUC ≥ 0.80, initiate design of a prospective silent-to-active pilot with workflow integration and clinical impact assessment
- Conference abstract: European Pancreatic Club (EPC) or United European Gastroenterology Week (UEGW)

---

## 15. Amendments

Protocol amendments require written approval from the IRB/ethics committee before implementation. Amendments are tracked in the protocol version history and notified to ClinicalTrials.gov within 30 days.

---

## 16. References

1. Banks PA et al. Classification of acute pancreatitis — 2012: revision of the Atlanta classification and definitions by international consensus. *Gut.* 2013;62(1):102–111.
2. Lankisch PG et al. Acute pancreatitis. *Nat Rev Dis Primers.* 2015;1:15054.
3. Petrov MS, Shanbhag S, Chakraborty M, Phillips AR, Windsor JA. Organ failure and infection of pancreatic necrosis as determinants of mortality in patients with acute pancreatitis. *Gastroenterology.* 2010;139(3):813–820.
4. Papachristou GI et al. Comparison of BISAP, Ranson's, APACHE-II, and CTSI scores in predicting organ failure, complications, and mortality in acute pancreatitis. *Am J Gastroenterol.* 2010;105(2):435–441.
5. Qiu Q, Nian YJ, Guo Y et al. Development and validation of three machine-learning models for predicting multiple organ failure in moderately severe and severe acute pancreatitis. *BMC Gastroenterol.* 2019;19(1):118.
6. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics.* 1988;44(3):837–845.
7. Stern N. Comparative evaluation of machine learning and deep learning models for early prediction of severe acute pancreatitis. PenuX Research Initiative. June 2026. [Preprint in preparation]

---

## Appendix A — PenuX Input Feature Mapping

| PenuX Feature Name | Common EHR Label | Units | LOINC Code |
|-------------------|-----------------|-------|-----------|
| CRP | C-reactive protein | mg/L | 1988-5 |
| WBC | White blood cell count | ×10⁹/L | 6690-2 |
| Neutrophil % | Neutrophils/100 leukocytes | % | 770-8 |
| AMY | Amylase | U/L | 1798-8 |
| ALT | Alanine aminotransferase | U/L | 1742-6 |
| Cr | Creatinine | μmol/L | 2160-0 |
| Ca | Calcium, total | mmol/L | 17861-6 |
| ALB | Albumin | g/L | 1751-7 |
| GLU | Glucose | mmol/L | 2345-7 |
| LDH | Lactate dehydrogenase | U/L | 2532-0 |
| PLT | Platelet count | ×10⁹/L | 777-3 |
| INR | International normalised ratio | ratio | 6301-6 |
| TG | Triglycerides | mmol/L | 2571-8 |
| Na⁺ | Sodium | mmol/L | 2951-2 |
| K⁺ | Potassium | mmol/L | 2823-3 |

*Full 59-feature mapping available in the PenuX technical documentation.*

---

## Appendix B — RAC 2012 Severity Classification

| Category | Criteria |
|----------|---------|
| **Mild AP** | No organ failure, no local or systemic complications |
| **Moderately Severe AP** | Transient organ failure (<48 h) and/or local complications |
| **Severe AP (SAP)** | Persistent organ failure (>48 h) — single or multi-organ |

*Organ failure assessed by Modified Marshall Scoring System (respiratory, renal, cardiovascular).*

---

*This document is a research protocol and does not constitute clinical guidance. PenuX is an investigational system not approved for clinical use.*
