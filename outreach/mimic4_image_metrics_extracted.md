# Extracted Metrics from MIMIC-IV Figures (commit c6f4f99)

Source images (local repo files from commit assets):
- `roc_auc__mimic4_mimic_iv_clinical_database_demo_2_2_pre__gelu.png`
- `roc_auc__mimic4_mimic_iv_clinical_database_demo_2_2_post__gelu.png`
- `pr_auc__mimic4_mimic_iv_clinical_database_demo_2_2_pre__gelu.png`
- `pr_auc__mimic4_mimic_iv_clinical_database_demo_2_2_post__gelu.png`

> Note: These are OvR pathogen-class metrics (not sepsis/AKI endpoint metrics).

## ROC-AUC (OvR) — MIMIC-IV demo 2.2

| Class | PRE AUC | POST AUC |
|---|---:|---:|
| B:PSEUDOMONAS AERUGINOSA | 0.996 | 1.000 |
| B:STAPH AUREUS COAG + | 0.982 | 0.979 |
| B:SERRATIA MARCESCENS | 1.000 | 1.000 |
| B:ESCHERICHIA COLI | 0.958 | 0.992 |
| B:PROTEUS MIRABILIS | 1.000 | 0.989 |
| B:GRAM POSITIVE COCCUS(COCCI) | 0.929 | 1.000 |
| B:OTHER | 1.000 | 1.000 |

## PR-AUC / Average Precision (OvR) — MIMIC-IV demo 2.2

| Class | PRE AP | POST AP |
|---|---:|---:|
| B:PSEUDOMONAS AERUGINOSA | 0.957 | 1.000 |
| B:STAPH AUREUS COAG + | 0.955 | 0.938 |
| B:SERRATIA MARCESCENS | 1.000 | 1.000 |
| B:ESCHERICHIA COLI | 0.864 | 0.977 |
| B:PROTEUS MIRABILIS | 1.000 | 0.667 |
| B:GRAM POSITIVE COCCUS(COCCI) | 0.741 | 1.000 |
| B:OTHER | 1.000 | 1.000 |

## Key deltas (POST - PRE)

- Largest ROC-AUC gains: **GRAM POSITIVE COCCUS(COCCI) +0.071**, **ESCHERICHIA COLI +0.034**.
- Largest ROC-AUC decline: **PROTEUS MIRABILIS -0.011**.
- Largest PR-AP gains: **GRAM POSITIVE COCCUS(COCCI) +0.259**, **ESCHERICHIA COLI +0.113**.
- Largest PR-AP decline: **PROTEUS MIRABILIS -0.333**.

## Ready-to-paste summary sentence

On MIMIC-IV demo 2.2 (OvR pathogen classes), POST-mode achieved AUC=1.000 for Pseudomonas, Serratia, Gram-positive cocci, and Other; E. coli improved from 0.958 to 0.992 AUC and from 0.864 to 0.977 AP, while Proteus PR-AP decreased from 1.000 to 0.667.
