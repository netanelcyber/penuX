# Project Report: penuX – Pathogen Prediction Pipeline

**Author:** Netanel  
**Date:** May 2026  
**Repository:** https://github.com/netanelcyber/penuX

## Abstract

penuX is a single-file deep learning pipeline developed for multiclass pathogen prediction using Electronic Health Record (EHR) data from the MIMIC-III and MIMIC-IV databases. The project implements a hybrid Conv1D + Bidirectional LSTM architecture combined with structured clinical features.

## Objectives

- Develop an efficient ETL pipeline for large clinical datasets
- Implement and evaluate a hybrid deep learning model
- Focus on model calibration and reliability
- Perform subgroup analysis for fairness assessment

## Methodology

- **Data Processing**: Memory-efficient streaming from CSV without heavy pandas dependency
- **Architecture**: Conv1D for local feature extraction + BiLSTM for temporal dependencies + Dense layers for clinical features
- **Training**: Custom loss functions with attention to calibration
- **Evaluation**: ROC-AUC, PR-AUC, Expected Calibration Error (ECE), Brier Score, Reliability Diagrams

## Key Results

- Strong calibration performance on MIMIC demo datasets
- Identification of important clinical predictors (Fever, WBC, SpO₂, etc.)
- Subgroup analysis across age, gender, and admission type

## Limitations

- Utilizes demonstration subsets only
- Not intended for clinical deployment
- Requires further validation on full datasets

## Future Work

- Multimodal integration (clinical notes + time-series)
- Explainability (SHAP/LIME)
- API deployment

## Conclusion

This project demonstrates practical application of deep learning techniques to critical healthcare challenges while maintaining high standards of model reliability and transparency.