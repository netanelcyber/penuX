#!/bin/bash

# Cross-Repository Issue Creation Script (Using curl + GitHub API)
# This script creates cross-reference issues in 5 clinical ML repositories
# Prerequisites: curl, jq, and GitHub Personal Access Token (PAT)
# 
# Setup:
# 1. Create a Personal Access Token: https://github.com/settings/tokens
#    - Select scopes: repo (full control of private repositories)
# 2. Export token: export GITHUB_TOKEN="your_token_here"
# 3. Run: bash create_issues_curl.sh

set -e

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Error: GITHUB_TOKEN environment variable not set"
    echo "Create a token at: https://github.com/settings/tokens"
    echo "Then run: export GITHUB_TOKEN='your_token_here'"
    exit 1
fi

echo "🚀 Creating cross-repository collaboration issues..."
echo ""

# Helper function to create an issue
create_issue() {
    local repo=$1
    local title=$2
    local body=$3
    
    echo "📝 Creating issue in $repo..."
    
    curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$repo/issues" \
        -d "{\"title\":\"$title\",\"body\":$(echo -n "$body" | jq -R -s .)}" | jq '.html_url'
    
    echo "✅ Issue created in $repo"
    echo ""
}

# Issue 1: netanelcyber/PenuX
create_issue "netanelcyber/PenuX" \
    "Cross-Repository Collaboration: PyHealth & Clinical Prediction Ecosystem" \
    "## Related Projects & Collaboration Opportunities

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
Tag collaborators from PyHealth and related projects for feedback and cross-linking."

# Issue 2: yirang-vet-ai/mimic-transformer-clinical-ai
create_issue "yirang-vet-ai/mimic-transformer-clinical-ai" \
    "Cross-Repository Collaboration: PyHealth & MIMIC Clinical Prediction Ecosystem" \
    "## Related Projects & Synergies

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
3. Are there shared feature engineering strategies across projects?"

# Issue 3: z-awan-lab/clinical-llm
create_issue "z-awan-lab/clinical-llm" \
    "Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Community" \
    "## Building a Connected Clinical AI Ecosystem

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
3. Could hybrid models (LLM + structured data) outperform single-modality approaches?"

# Issue 4: NithinGowda67/mimic-iv-v3-1-icu-mortality-xai
create_issue "NithinGowda67/mimic-iv-v3-1-icu-mortality-xai" \
    "Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Ecosystem" \
    "## Connected Clinical ML Initiatives

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
3. What's the performance/explainability trade-off in MIMIC-IV v3.1 vs. earlier versions?"

# Issue 5: sintabh/icu-sepsis-mortality-prediction
create_issue "sintabh/icu-sepsis-mortality-prediction" \
    "Cross-Repository Collaboration: PyHealth, PenuX & Clinical Prediction Community" \
    "## Clinical ML Ecosystem Connections

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
4. What's the optimal combination of structured data + vital signs + pathogen information?"

echo "=================================="
echo "✨ All 5 cross-reference issues created successfully!"
echo "=================================="
echo ""
echo "📊 Summary:"
echo "  1. ✅ netanelcyber/PenuX"
echo "  2. ✅ yirang-vet-ai/mimic-transformer-clinical-ai"
echo "  3. ✅ z-awan-lab/clinical-llm"
echo "  4. ✅ NithinGowda67/mimic-iv-v3-1-icu-mortality-xai"
echo "  5. ✅ sintabh/icu-sepsis-mortality-prediction"
echo ""
echo "🔗 Main Reference Discussion: https://github.com/sunlabuiuc/PyHealth/discussions/1154"
echo ""
