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

## Additional Figure Provided by User — "Before Lab" (MIMIC-III demo 1.4 PRE)

From the attached ROC figure titled:
`ROC curves (OvR) — TEST mimic3_mimic_iii_clinical_database_demo_1_4_pre`

### ROC-AUC (OvR) values read from legend

| Class | AUC |
|---|---:|
| B:PSEUDOMONAS AERUGINOSA | 0.924 |
| B:STAPH AUREUS COAG + | 0.976 |
| B:SERRATIA MARCESCENS | 0.991 |
| B:ESCHERICHIA COLI | 1.000 |
| B:PROTEUS MIRABILIS | 0.974 |
| B:POSITIVE FOR METHICILLIN R | 1.000 |
| B:GRAM POSITIVE COCCUS(COCCI) | 1.000 |
| B:OTHER | 0.966 |

### Quick interpretation note

This "before lab" (PRE) MIMIC-III figure shows strong OvR discrimination overall, with perfect AUC values for E. coli, Methicillin-resistant positive class, and Gram-positive cocci; the lowest class AUC in this panel is Pseudomonas (0.924).
