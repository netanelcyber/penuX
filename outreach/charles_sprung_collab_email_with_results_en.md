# Email Draft — Prof. Charles Sprung (with actual MIMIC-IV numbers)

> Fill in the bracketed numeric fields from your latest run before sending.

**Subject:** Collaboration request: PenuX ICU validation at Hadassah (with MIMIC-IV results)

Dear Prof. Sprung,

I hope you are well. My name is [Your Full Name], and I am an independent clinical AI researcher.

I am writing to request a potential collaboration with the General ICU at Hadassah Ein Kerem on external validation of **PenuX**, an early-warning model for **sepsis** and **AKI** in ICU patients.

PenuX was developed on MIMIC-III/IV time-series data using a Bi-LSTM architecture, with an IEC 62304-oriented development process. On my latest **MIMIC-IV** evaluation, the model showed the following results:

- **Sepsis task**: AUROC **[0.xx]**, AUPRC **[0.xx]**, sensitivity at operating threshold **[xx%]**, PPV **[xx%]**.
- **AKI task**: AUROC **[0.xx]**, AUPRC **[0.xx]**, sensitivity at operating threshold **[xx%]**, PPV **[xx%]**.
- **Calibration**: Brier score **[0.xxx]** (sepsis), **[0.xxx]** (AKI).
- **Clinical lead time** (median early warning before event): **[x.x hours]** for sepsis and **[x.x hours]** for AKI.

Given your leadership in critical care, I believe a Hadassah collaboration would be ideal to test generalizability on an Israeli cohort and to calibrate clinically meaningful alert thresholds for real ICU workflow.

If relevant, I would be grateful for a brief 15–20 minute call to discuss:
1) retrospective external validation at Hadassah,
2) jointly defined clinical endpoints and alert strategy,
3) and possible joint abstract/grant pathways.

I can send a one-page technical summary with methods and full metric tables in advance.

Thank you for your time and consideration.

Best regards,
[Your Full Name]  
[Phone] | [Email] | [LinkedIn]

---

## Optional compact metric block (paste under the email)

| Task | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | Brier | Median lead time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sepsis | [0.xx] | [0.xx] | [xx%] | [xx%] | [xx%] | [xx%] | [0.xxx] | [x.x h] |
| AKI | [0.xx] | [0.xx] | [xx%] | [xx%] | [xx%] | [xx%] | [0.xxx] | [x.x h] |
