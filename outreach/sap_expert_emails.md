# Outreach Emails — Severe Acute Pancreatitis Early Prediction (PenuX-AP-Severity)

A collection of targeted expert outreach emails for the PenuX-AP-Severity project.
Each email is tailored to a specific expert audience.

---

## EMAIL 1 — Gastroenterologist / Pancreatologist (Collaboration Inquiry)

**To:** [Gastroenterology/HPB Pancreatologist]  
**Subject:** Research collaboration inquiry — ML early prediction of Severe Acute Pancreatitis

---

Dear Dr. [Last Name],

I hope this message finds you well. My name is Netanel Shoshany, and I am an independent clinical AI researcher leading **PenuX-AP-Severity**, an open-source machine learning project focused on early prediction of Severe Acute Pancreatitis (SAP).

**The clinical problem:** SAP — defined as persistent organ failure >48 hours per the 2012 Revised Atlanta Classification — affects 10–20% of acute pancreatitis admissions and carries significant morbidity and mortality. Yet most existing scoring systems (BISAP, APACHE II, Ranson, Modified CTSI) require data that accumulates over 24–48 hours, limiting early triage utility.

**What we have built:** A retrospective ML pipeline that uses only **routine admission variables available within the first 24 hours** — vital signs, basic labs (WBC, CRP, BUN, creatinine, calcium, glucose, LDH, hematocrit) — to predict SAP progression. Models include logistic regression, random forest, and gradient-boosted ensembles, calibrated and benchmarked against classical clinical scores.

**Why I am reaching out:** We are seeking collaboration with a clinical pancreatologist or gastroenterologist for:

1. **Clinical review** of feature selection and outcome operationalization (Atlanta 2012 label logic)
2. **External validation** on an institutional AP cohort (retrospective, de-identified, IRB-approved)
3. **Co-authorship** on a planned peer-reviewed manuscript

The repository is publicly available: [https://github.com/netanelcyber/penux](https://github.com/netanelcyber/penux)

I would be grateful for a brief 20-minute call to discuss feasibility. I can send a one-page summary and draft manuscript in advance.

Thank you for your time and consideration.

Best regards,  
Netanel Shoshany  
Independent Clinical AI Researcher  
netanel@penux.uk  
https://github.com/netanelcyber/penux

---

## EMAIL 2 — ICU / Critical Care Physician (Validation Study Proposal)

**To:** [Intensivist / Critical Care MD]  
**Subject:** SAP severity ML model — validation collaboration proposal

---

Dear Dr. [Last Name],

I am writing to propose a research collaboration on machine-learning-based early warning for **Severe Acute Pancreatitis (SAP)** in the ICU setting.

SAP — defined as persistent organ failure >48 hours (Atlanta 2012) — is one of the most resource-intensive presentations in general and surgical ICUs. Early identification of patients at risk for SAP could meaningfully improve triage, bed allocation, and escalation timing.

**The PenuX-AP-Severity project** has developed an ML pipeline trained on MIMIC-IV data that predicts SAP from the first 24 hours of admission using only routine variables: vital signs, WBC, CRP, creatinine, BUN, calcium, glucose, LDH, hematocrit, and basic demographics. The models are benchmarked against BISAP, APACHE II, and Ranson scores and achieve clinically meaningful discrimination on the MIMIC-IV cohort.

**Proposed collaboration (Phase 1 — Retrospective Validation):**

| Step | Description |
|------|-------------|
| 1 | Define local AP cohort (inclusion/exclusion criteria) |
| 2 | Map local variables to model input schema |
| 3 | External validation (AUROC, AUPRC, calibration) |
| 4 | Lead-time and alert-burden analysis |
| 5 | Joint manuscript submission |

This is a **retrospective, non-interventional study** requiring only de-identified EHR data and standard IRB/ethics committee approval. No changes to patient care during the study.

I would welcome a brief call at your convenience. Please find the open-source repository at [https://github.com/netanelcyber/penux](https://github.com/netanelcyber/penux).

Thank you for your consideration.

Sincerely,  
Netanel Shoshany  
netanel@penux.uk

---

## EMAIL 3 — Clinical AI Researcher / Informatics Collaborator (Technical Collaboration)

**To:** [Clinical Informatics / Medical AI Researcher]  
**Subject:** PenuX-AP-Severity — open-source SAP prediction pipeline, seeking co-investigator

---

Dear Dr. [Last Name],

I came across your work on [relevant publication / EHR-based prediction models / clinical NLP] and believe there may be strong complementarity with an open-source project I am leading.

**PenuX-AP-Severity** is a Python-based retrospective ML pipeline for predicting progression to Severe Acute Pancreatitis (SAP, persistent organ failure >48h, Atlanta 2012) from routine 24-hour admission data. The project includes:

- **Full modelling stack:** logistic regression, random forest, XGBoost, LightGBM — with isotonic/Platt calibration and bootstrapped confidence intervals
- **Classical benchmark integration:** BISAP, APACHE II, Ranson, Modified CTSI implemented natively for head-to-head comparison
- **Explainability:** SHAP + permutation importance
- **FHIR/HL7 adapters** for integration-ready inference endpoint (FastAPI)
- **MIMIC-IV SQL extraction** pipeline for reproducibility

The repository is at [https://github.com/netanelcyber/penux](https://github.com/netanelcyber/penux).

**I am seeking a co-investigator** for:
- External validation on a non-MIMIC dataset
- Prospective silent-mode pilot design
- Grant application co-authorship (NIH R01 / ERC / ISF)

If this aligns with your current research agenda, I would be delighted to discuss. I can share the draft manuscript and dataset documentation in advance.

Best regards,  
Netanel Shoshany  
netanel@penux.uk

---

## EMAIL 4 — Hospital Research Committee / IRB Chair (Ethics Approval Preparation)

**To:** [Research Ethics Committee / IRB Chair]  
**Subject:** Pre-submission inquiry — retrospective ML study for SAP prediction (non-interventional)

---

Dear Committee Chair / IRB Administrator,

I am writing to inquire about the process for obtaining ethical approval for a **retrospective, non-interventional** machine-learning model development study at your institution.

**Study summary:**

- **Title:** Early prediction of Severe Acute Pancreatitis using routine 24-hour admission data: a machine learning approach
- **Design:** Retrospective observational cohort study
- **Population:** Adult patients (≥18 years) admitted with acute pancreatitis
- **Outcome:** Severe Acute Pancreatitis (SAP) — persistent organ failure >48 hours per the 2012 Revised Atlanta Classification
- **Intervention:** None — the model is not used to guide patient care
- **Data:** Existing de-identified EHR records; no patient contact; no prospective data collection
- **Data minimization:** Only variables required for prediction and outcome definition are extracted
- **Identifiers:** No direct identifiers stored or published
- **Framework:** TRIPOD reporting guidelines; Declaration of Helsinki (2013); GDPR Article 89 (research exemption if applicable)

I would appreciate guidance on the appropriate application pathway (expedited vs. full review) and whether a data protection officer (DPO) notification is also required.

I am happy to provide the full study protocol, data flow diagram, de-identification methodology, and TRIPOD checklist at your request.

Thank you for your time.

Respectfully,  
Netanel Shoshany  
netanel@penux.uk

---

## EMAIL 5 — Pancreatitis Research Network / Society Contact (Dataset Access)

**To:** [International Pancreatitis Study Group / EUROPAC / IAP contact]  
**Subject:** Inquiry — access to multi-center AP dataset for SAP ML model external validation

---

Dear [Name / Network Coordinator],

I am an independent clinical AI researcher working on **PenuX-AP-Severity**, an open-source machine learning project for early prediction of Severe Acute Pancreatitis (SAP) from 24-hour admission data.

The model has been developed and internally validated on MIMIC-IV (retrospective critical-care dataset). To advance toward publication and potential clinical utility, we urgently need **external validation on a prospectively collected or multi-center cohort** — ideally with:

- ≥200 confirmed acute pancreatitis admissions (any severity)
- Atlanta 2012 severity classification available or reconstructable from records
- Routine labs and vital signs from the first 24 hours of admission
- De-identified, IRB/ethics-approved for secondary AI research

I would be very grateful if you could advise whether any member institutions or network datasets would be available for such a collaboration, or if you could connect me with the appropriate contacts.

All outputs would be shared with the contributing site. Co-authorship is offered to all contributing centers in the manuscript.

Thank you for your support.

Kind regards,  
Netanel Shoshany  
Independent Clinical AI Researcher  
netanel@penux.uk  
https://github.com/netanelcyber/penux

---

## EMAIL 6 — Journal / Conference (Manuscript Pre-submission Inquiry)

**To:** [Editor-in-Chief / Guest Editor]  
**Subject:** Pre-submission inquiry — ML prediction of Severe Acute Pancreatitis from 24-hour admission data

---

Dear Dr. [Editor Name],

I would like to submit a brief pre-submission inquiry regarding a manuscript currently under preparation, tentatively titled:

**"Early Prediction of Severe Acute Pancreatitis from Routine 24-Hour Admission Data: A Machine Learning Approach with Classical Score Benchmarking"**

**Key points:**

- **Design:** Retrospective observational cohort study (MIMIC-IV dataset)
- **Outcome:** SAP per the 2012 Revised Atlanta Classification (persistent organ failure >48h)
- **Methods:** Logistic regression, random forest, and gradient-boosted classifiers trained on routine 24-hour variables; calibrated; SHAP-explained; benchmarked against BISAP, APACHE II, Ranson, and Modified CTSI
- **Novel contribution:** Head-to-head comparison of modern ML against established clinical scores using the same variable availability constraint (first 24 hours only); open-source reproducible pipeline; FHIR-ready inference endpoint
- **Reporting:** TRIPOD guideline-compliant
- **Word count (estimate):** ~3,500 words + tables + figures
- **Code availability:** Open source (https://github.com/netanelcyber/penux)

I believe this work fits well within [Journal Name]'s scope of [clinical informatics / pancreatology / critical care AI]. I would be happy to discuss whether this aligns with your editorial priorities or a special issue before full submission.

Thank you for your consideration.

Sincerely,  
Netanel Shoshany  
netanel@penux.uk

---

*Generated for PenuX-AP-Severity outreach — June 2026*
