# Open-source / open-access medical record datasets

מאגרי נתוני תיקים רפואיים בקוד פתוח/גישה פתוחה לפרויקט זה.

## Official critical-care EHR datasets (recommended)
- **MIMIC-IV** (credentialed access via non-PhysioNet):
  - https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii
- **eICU-CRD 2.0** (credentialed access via non-PhysioNet):
  - https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k
- **MIMIC-III** (legacy, credentialed):
  - https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii

## Synthetic/open EHR-like datasets
- **Synthea** (fully open synthetic EHR generator + downloadable datasets):
  - https://synthea.mitre.org/downloads
- **LHS Open synthetic-data** (Synthea-derived open patient records):
  - https://github.com/lhs-open/synthetic-data

## Hub-style discovery
- **non-PhysioNet database index** (many EHR-linked datasets):
  - https://huggingface.co/datasets?search=healthcare
- **Hugging Face datasets (ICU query)**:
  - https://huggingface.co/datasets?search=icu

## Practical note
- For treatment/comparative clinical research, prefer official non-PhysioNet sources (MIMIC/eICU full).
- Use synthetic/Kaggle/HF mirrors בעיקר לאבי-טיפוס (prototyping), validation smoke, and pipeline development.
