# Cross-Repository Collaboration Issues (Copy-Paste Ready)

## Issue 1: netanelcyber/PenuX

**Title:**
```
Cross-Repository Collaboration: PyHealth & Clinical Prediction Ecosystem
```

**Body:**
```markdown
## Related Projects & Collaboration Opportunities

### PyHealth (sunlabuiuc/PyHealth)
- **URL**: https://github.com/sunlabuiuc/PyHealth
- **Discussion**: https://github.com/sunlabuiuc/PyHealth/discussions/1154
- **Connection**: Comprehensive healthcare deep learning toolkit with calibration & conformal prediction modules
- **Overlap**: Both use MIMIC datasets, clinical EHR data, and implement calibration/fairness checks
- **Synergy**: PenuX demonstrates single-file hybrid architecture; PyHealth provides modular toolkit approach

### Related MIMIC Prediction Projects
1. **yirang-vet-ai/mimic-transformer-clinical-ai**: Transformer-based vital sign + mortality prediction
2. **z-awan-lab/clinical-llm**: LLM fine-tuning for outcome prediction on MIMIC-IV
3. **NithinGowda67/mimic-iv-v3-1-icu-mortality-xai**: Explainable deep learning with SHAP/LIME
4. **sintabh/icu-sepsis-mortality-prediction**: Sepsis mortality prediction on MIMIC-IV

### Suggested Collaboration Areas
- [ ] Cross-validate calibration methods (ECE, Brier, temperature scaling)
- [ ] Benchmark hybrid architectures vs. transformer-based approaches
- [ ] Integrate conformal prediction with PenuX's complete-case filtering
- [ ] Standardize subgroup fairness evaluation methodologies
- [ ] Share MIMIC ETL and feature engineering best practices

### Next Steps
Tag collaborators from PyHealth and related projects for feedback and cross-linking.
```

**Link to create:** https://github.com/netanelcyber/PenuX/issues/new

---

## Issue 2: yirang-vet-ai/mimic-transformer-clinical-ai

**Title:**
```
Cross-Repository Collaboration: PyHealth & MIMIC Clinical Prediction Ecosystem
```

**Body:**
```markdown
## Related Projects & Synergies

### PyHealth (sunlabuiuc/PyHealth)
- **URL**: https://github.com/sunlabuiuc/PyHealth
- **Discussion**: https://github.com/sunlabuiuc/PyHealth/discussions/1154
- **Focus**: Modular deep learning toolkit for healthcare with emphasis on calibration, conformal prediction, and fairness
- **Relevance**: PyHealth's calibration module could strengthen transformer-based vital sign predictions
- **Opportunity**: Benchmark transformer vital sign predictions against PyHealth.calib methods

### PenuX - Focused MIMIC Pathogen Pipeline
- **URL**: https://github.com/netanelcyber/PenuX
- **Focus**: Single-file PyTorch pipeline for pathogen/pneumonia class prediction on MIMIC-III/IV
- **Relevance**: Demonstrates calibration, ECE, and subgroup bias checks on clinical data
- **Comparison**: PenuX uses structured+text hybrid; this project uses transformer sequences

### Other Related Projects
- **z-awan-lab/clinical-llm**: LLM-based outcome prediction
- **NithinGowda67/mimic-iv-v3-1-icu-mortality-xai**: Explainable mortality prediction (SHAP/LIME)
- **sintabh/icu-sepsis-mortality-prediction**: Sepsis mortality ML pipeline

### Collaboration Ideas
- [ ] Cross-validate calibration methods across architectures (Transformers vs. Conv1D+LSTM vs. LSTMs)
- [ ] Benchmark vital sign prediction performance on PyHealth-compatible datasets
- [ ] Integrate conformal prediction for uncertainty-aware vital sign forecasting
- [ ] Standardize fairness evaluation: age bins, sex/gender, admission context

### Questions for Discussion
1. How do transformer calibration curves compare to traditional neural architectures?
2. Could PyHealth's conformal prediction improve transformer-based predictions?
3. Are there shared feature engineering strategies across projects?
```

**Link to create:** https://github.com/yirang-vet-ai/mimic-transformer-clinical-ai/issues/new

---

## Issue 3: z-awan-lab/clinical-llm

**Title:**
```
Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Community
```

**Body:**
```markdown
## Building a Connected Clinical AI Ecosystem

### PyHealth (sunlabuiuc/PyHealth)
- **URL**: https://github.com/sunlabuiuc/PyHealth
- **Discussion**: https://github.com/sunlabuiuc/PyHealth/discussions/1154
- **Vision**: Comprehensive deep learning framework for healthcare with calibration & conformal prediction
- **Our Overlap**: Both implement MIMIC-IV outcome prediction with emphasis on reliability
- **Integration Potential**: Integrate PyHealth's uncertainty quantification (ECE, conformal sets) with LLM predictions

### PenuX - Structured Data Pathogen Pipeline
- **URL**: https://github.com/netanelcyber/PenuX
- **Focus**: MIMIC-III/IV pathogen classification with hybrid PyTorch model
- **Contrast**: PenuX focuses on structured+categorical data; this project focuses on NLP/LLM
- **Synergy**: Could LLM-generated summaries augment PenuX's structured features?

### Other Related Projects
- **yirang-vet-ai/mimic-transformer-clinical-ai**: Transformer vital sign + mortality prediction
- **NithinGowda67/mimic-iv-v3-1-icu-mortality-xai**: Explainable mortality prediction with SHAP/LIME
- **sintabh/icu-sepsis-mortality-prediction**: Sepsis mortality on MIMIC-IV

### Collaboration Opportunities
- [ ] **Calibration Benchmarking**: Compare LLM confidence calibration vs. traditional architectures
- [ ] **Fairness Evaluation**: Standardize subgroup analysis (age, sex, admission type) across NLP and structured approaches
- [ ] **Methodology Sharing**: ETL strategies, feature engineering best practices for MIMIC
- [ ] **Model Fusion**: Can structured PyHealth features augment LLM inputs? (multimodal learning)
- [ ] **Uncertainty Quantification**: Integrate PyHealth's conformal prediction with LLM outputs

### Questions for Community
1. How should we calibrate LLM probability outputs for clinical decision support?
2. What are best practices for fairness auditing in LLM-based clinical prediction?
3. Could hybrid models (LLM + structured data) outperform single-modality approaches?
```

**Link to create:** https://github.com/z-awan-lab/clinical-llm/issues/new

---

## Issue 4: NithinGowda67/mimic-iv-v3-1-icu-mortality-xai

**Title:**
```
Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Ecosystem
```

**Body:**
```markdown
## Connected Clinical ML Initiatives

### PyHealth (sunlabuiuc/PyHealth)
- **URL**: https://github.com/sunlabuiuc/PyHealth
- **Discussion**: https://github.com/sunlabuiuc/PyHealth/discussions/1154
- **Alignment**: Both prioritize explainability (XAI) and uncertainty quantification in clinical predictions
- **Synergy**: PyHealth's calibration module + conformal prediction could strengthen reliability of SHAP/LIME explanations
- **Version Advantage**: We use MIMIC-IV v3.1 (latest); PyHealth supports multiple versions

### PenuX - Pathogen Prediction on MIMIC
- **URL**: https://github.com/netanelcyber/PenuX
- **Focus**: MIMIC-III/IV pathogen classification with calibration & fairness checks
- **Shared Goals**: Calibration (ECE, Brier), fairness checks (subgroup analysis), research-only ethics
- **Key Difference**: We emphasize interpretability (SHAP/LIME); PenuX emphasizes architectural efficiency

### Other Related Projects
- **yirang-vet-ai/mimic-transformer-clinical-ai**: Transformer-based vital sign + mortality prediction
- **z-awan-lab/clinical-llm**: LLM-based outcome prediction with MIMIC-IV benchmarks
- **sintabh/icu-sepsis-mortality-prediction**: Sepsis mortality ML pipeline

### Collaboration Opportunities
- [ ] **XAI + Calibration**: Combine SHAP explanations with calibration confidence intervals for clinical decision support
- [ ] **Version Compatibility**: Share strategies for working with MIMIC-IV v3.1 (latest version)
- [ ] **Fairness Standardization**: Align subgroup evaluation methodologies (age bins, sex/gender, admission type)
- [ ] **Mortality Prediction Benchmarks**: Compare performance/calibration/explainability across projects
- [ ] **Interpretability Standards**: Best practices for communicating uncertainty to clinicians via SHAP + calibration curves

### Research Questions
1. How much do calibration-aware methods improve trust in SHAP-based explanations?
2. Should we standardize demographic subgroups for fairness evaluation across projects?
3. What's the performance/explainability trade-off in MIMIC-IV v3.1 vs. earlier versions?
```

**Link to create:** https://github.com/NithinGowda67/mimic-iv-v3-1-icu-mortality-xai/issues/new

---

## Issue 5: sintabh/icu-sepsis-mortality-prediction

**Title:**
```
Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Community
```

**Body:**
```markdown
## Clinical ML Ecosystem Connections

### PyHealth (sunlabuiuc/PyHealth)
- **URL**: https://github.com/sunlabuiuc/PyHealth
- **Discussion**: https://github.com/sunlabuiuc/PyHealth/discussions/1154
- **Scope**: Modular deep learning toolkit for healthcare with calibration, conformal prediction, and fairness
- **Relevance**: PyHealth's calibration methods (temperature scaling, histogram binning, etc.) directly applicable to sepsis mortality
- **Integration**: Could leverage PyHealth.calib for improved uncertainty quantification in sepsis risk scores

### PenuX - Sepsis/Pathogen Prediction Pipeline
- **URL**: https://github.com/netanelcyber/PenuX
- **Focus**: MIMIC-III/IV pathogen class prediction with hybrid PyTorch model
- **Overlap**: Both tackle sepsis-related tasks using MIMIC data; both implement calibration and fairness checks
- **Comparison**: PenuX focuses on pathogen identification; this project focuses on mortality risk
- **Synergy**: Could PenuX's pathogen predictions improve this project's mortality modeling?

### Other Related Projects
- **yirang-vet-ai/mimic-transformer-clinical-ai**: Transformer-based vital sign + mortality prediction
- **z-awan-lab/clinical-llm**: LLM-based outcome prediction on MIMIC-IV
- **NithinGowda67/mimic-iv-v3-1-icu-mortality-xai**: Explainable mortality prediction (SHAP/LIME)

### Collaboration Possibilities
- [ ] **Sepsis Mortality + Pathogen Joint Prediction**: Could PenuX's pathogen predictions improve mortality modeling?
- [ ] **Calibration Benchmarking**: Compare ECE/Brier scores across different loss functions and architectures
- [ ] **Fairness Framework Alignment**: Standardize demographic subgroup analysis across projects
- [ ] **MIMIC ETL Best Practices**: Share streaming CSV strategies and feature engineering techniques
- [ ] **Mortality Prediction Benchmarks**: Performance comparison across PyHealth + PenuX + this project

### Research Questions
1. Does knowing the identified pathogen improve sepsis mortality prediction (joint modeling)?
2. How do calibration methods perform specifically for sepsis mortality vs. general mortality?
3. Are fairness concerns different for sepsis cohorts vs. general ICU populations?
4. What's the optimal combination of structured data + vital signs + pathogen information?
```

**Link to create:** https://github.com/sintabh/icu-sepsis-mortality-prediction/issues/new

---

## Instructions to Create Issues

1. **Copy the Title** from each section
2. **Copy the Body** (markdown text)
3. Click the **Link to create** for that repository
4. **Paste Title** into the "Title" field
5. **Paste Body** into the "Write" field
6. Click **"Submit new issue"**

Repeat for all 5 repositories! 🚀
