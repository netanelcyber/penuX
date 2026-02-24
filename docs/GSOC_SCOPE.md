# GSoC 2026 – Project Scope
This document summarizes the scope discussed in Issue #8.

# 1. Proposed Tasks
The project will focus on two supervised learning tasks using structured ICU clinical data.

 1.1 Pneumonia Prediction (Binary classification)
The objective is to predict whether a patient develops pneumonia during their ICU stay.  
The label will be defined using diagnosis codes.  
Features will be extracted from the first 24 hours of ICU admission.

 1.2 In-Hospital Mortality Prediction (Binary: In-Hospital)
The objective is to predict whether a patient dies before hospital discharge.
The target will be defined using the hospital discharge status (in-hospital mortality indicator).

Both tasks will:
- Share a common preprocessing pipeline  
- Use a shared structured feature set  
- Start with simple baseline models (e.g., Logistic Regression, Random Forest)  

# 2. Dataset Plan
Initially, we will construct a canonical modeling dataset by:
- Filtering adult ICU patients (age ≥ 18)  
- Selecting the first ICU stay per patient  
- Extracting demographic, vital sign, and laboratory features  
- Defining labels using diagnosis codes and discharge status  

To prevent data leakage, data will be split patient-wise.

# 3. Evaluation Metrics
Pneumonia:
    - PR-AUC  
    - F1-score  

Mortality:
    - ROC-AUC  
If time permits, model calibration analysis may also be performed.

# 4. Out of Scope (Initial Phase)
The following items are explicitly outside the scope of the initial implementation:
- Deep learning / imaging models  
- Real-time deployment  
- Clinical decision support claims  

# Disclaimer
This project is intended for research, experimentation and benchmarking purposes only.
