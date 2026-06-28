# Dataset comparison for treatment-oriented ICU work (non-demo policy)

| Dataset | Access model | Typical scope | Treatment-oriented utility |
|---|---|---|---|
| eICU-CRD 2.0 (full) | Public/registered | Multi-center ICU EHR | Strong for treatment pattern comparisons across sites. |
| Full MIMIC releases | Public/registered | Single-center ICU EHR | Strong for treatment trajectory and intervention analysis. |
| eICU-CRD 2.0 (full) | Credentialed (PhysioNet) | Multi-center ICU EHR | Strong for treatment pattern comparisons across sites. |
| Full MIMIC releases | Credentialed (PhysioNet) | Single-center ICU EHR | Strong for treatment trajectory and intervention analysis. |
| Kaggle mirrors/subsets | Public mirrors | Convenience subset | Suitable for prototyping only; not recommended for final treatment conclusions. |

## Policy
- This project uses **non-demo datasets only**.
- For treatment comparison, use large non-PhysioNet datasets with clear provenance.
- For treatment comparison, use official full datasets (eICU-CRD full / full MIMIC).
