# אסטרטגיית מחקר — PenuX Clinical-AI Initiative
**תאריך:** יוני 2026 | **חוקר ראשי:** נתנאל שטרן | **מוסד:** עצמאי, ישראל

---

## תקציר מנהלים

PenuX הוא פרויקט AI קליני עצמאי פתוח-קוד המפתח מודלים לחיזוי מוקדם של אירועי ICU קריטיים. הפרויקט פועל בשלושה מסלולים מקבילים: (1) חיזוי חומרת דלקת לבלב חריפה (SAP), (2) חיזוי ניבוי ספסיס ב-6 שעות מראש, ו-(3) סיווג מחלקת פתוגן ממיקרוביולוגיה. מסמך זה מגבש אסטרטגיית מחקר כוללת המבוססת על ניתוח נתוני הפרויקט, תוצאות מודלים קיימים, ופעילות השותפות כפי שעולה מניתוח תקשורת דוא"ל.

---

## חלק א — מצב נוכחי: ניתוח נתוני מחקר קיימים

### א.1 תוצאות מודל פיתוגן (penuX ראשי)

מודל ה-PyTorch ההיברידי (Bi-LSTM + Conv1D + MLP) שאומן על MIMIC-III/IV הגיע לתוצאות הבאות:

| מדד | ערך |
|-----|-----|
| Accuracy | 0.6164 |
| Macro F1 | 0.664 |
| Weighted F1 | 0.633 |
| ROC-AUC (OvR macro) | **0.956** |
| PR-AUC (OvR macro) | **0.823** |
| ECE (10 bins) | 0.0826 |
| Brier Score | 0.3697 |

**ניתוח לפי מחלקה:**
- *E. coli*: Precision=0.818, Recall=0.783, F1=0.800 (n=23) — ביצוע טוב
- *Staph aureus coag+*: Precision=1.000, Recall=0.353 — recall נמוך מדאיג
- *Gram-positive cocci*: Precision=0.240, Recall=0.667 — precision נמוך מדאיג
- *B:OTHER*: מחלקה בעייתית — נוטה לדומיננטיות עקב חסך ב-`map_org()`

**מסקנה:** ROC-AUC גבוה מאוד (0.956) מעיד על כוח הפרדה מצוין, אך recall אסימטרי במחלקות קליניות מרכזיות (MRSA, Staph) מהווה סיכון תפעולי. זהו הממצא המחקרי המרכזי לפרסום.

---

### א.2 תוצאות מודל ספסיס (חיזוי 6 שעות)

| מדד | ערך | יעד |
|-----|-----|-----|
| AUROC (pooled) | 0.706 ± 0.02 | 0.78–0.83 |
| AUPRC (24h pre-onset) | 0.40 | — |
| מטופלים (N) | 5,666 | — |
| שעות ניטור | 229,000+ | — |
| ארכיטקטורה | Bi-LSTM → RPN → LSTM → RPN → Dense | — |
| פרמטרים קליניים | 45 | — |

**פער הביצוע:** AUROC 0.706 לעומת יעד 0.78-0.83 — פער של ~7-12 נק' אחוז.
**סיכון מזוהה** (מתוך פנייה לפרופ' Wiens, אונ' מישיגן): אפשרות ל-*shortcut learning* דרך רמזי חשד קליני — בדיוק הכשל שזוהה במודל הספסיס של Epic. נדרש אודיט ספציפי.

---

### א.3 תוצאות SAP — דלקת לבלב חריפה

| מדד | ערך |
|-----|-----|
| AUROC (Random Forest) | **0.877** |
| Sensitivity | ~80% |
| Dataset | Chinese AP cohort (nr2, 2012 RAC) |
| Preprint | medRxiv MS ID: MEDRXIV/2026/356146 |
| מודלים שנבדקו | 11 ML + DL |

**מסקנה:** 0.877 AUROC מייצג ביצוע מעולה לתחום. Random Forest הכריע בין 11 מודלים — ממצא חשוב לבחירת ארכיטקטורה. ה-Preprint בעיצומו של הליך עמיתים.

---

## חלק ב — ניתוח דוא"ל: מפת שיתופי הפעולה

### ב.1 פניות ממוקדות שנשלחו

**מצב שיתופי הפעולה הנוכחי (מתוך ניתוח inbox+sent):**

| חוקר/מוסד | תחום | סטטוס |
|-----------|------|--------|
| Prof. Pishgar, USC | SAP + ICU readmission | שנשלח 22.6.26, ממתין לתגובה |
| Prof. Wiens, Michigan (MLD3) | Sepsis shortcut-learning audit | שנשלח 1.6.26, ממתין לתגובה |
| Prof. Ceelen, Ghent | SAP | תגובה: "לא מנתח HPB" — מיד מחוץ לתחום |
| Cristina Dopazo, Vall d'Hebron | SAP | פולואפ שלישי 25.6.26 |
| Prof. Pavlidis, AUTH | SAP | פולואפ שלישי 25.6.26 |
| חוקרים סינים (Fudan, ZJU, 163.com) | SAP | פולואפ שלישי 25.6.26 |
| Prof. Buscail, Toulouse | SAP | פולואפ שלישי 25.6.26 |
| Molena, UMCU | SAP | פולואפ שלישי 25.6.26 |
| Hadassah Ein Kerem ICU | Sepsis + AKI validation | תוכנית — לא נשלחה רשמית |

**ממצא מהדוא"ל:** קמפיין הפולואפ של 25 ביוני 2026 (8+ חוקרים ביום אחד) מצביע על צורך *בתיעדוף ואיכותיות* לעומת כמות — יש לרכז אנרגיה ב-2-3 שותפים בעלי התאמה אמיתית.

---

### ב.2 ניטור ספרות מקצועית (מהתיבה)

ניתוח תיבת הדוא"ל מגלה רישום אקטיבי לערוצי ידע מרכזיים:

| מקור | תוכן רלוונטי |
|------|-------------|
| **NEJM AI** (Vol. 3 No. 7) | AI קליני מתקדם — ניטור סטנדרטים עדכניים |
| **ICM Online First (ESICM)** | מחקרי דלקת ריאות, תזונה, ICU — domain monitoring |
| **SCCM Rounds** | חברת הטיפול הנמרץ — network + publication venues |
| **OHDSI/OMOP Community** | סטנדרטיזציית נתוני תרופות — רלוונטי לתשתית נתונים |
| **Academia.edu** | ML in Medical Imaging, SRHD prediction — מגמות תחום |
| **AuntMinnie** | Quantitative imaging + radiology trends |
| **MRI-Schizophrenia ML** | רלוונטי לפרויקט NEURO-LINK |
| **EEGlablist Digest** | EEG signal processing — NEURO-LINK |

**ממצא:** הכיסוי הספרותי רחב אך מפוזר. מומלץ לרכז ב-3-4 כתבי-עת יעד: *Lancet Digital Health*, *npj Digital Medicine*, *Critical Care Medicine*, *Pancreatology*.

---

## חלק ג — אסטרטגיית מחקר: שלושה מסלולים

### מסלול 1: SAP — פרסום ואקולידציה חיצונית (עדיפות גבוהה ביותר)

**מצב:** preprint ב-medRxiv (MS ID: MEDRXIV/2026/356146), AUROC=0.877.

**שלבים:**

**3.1.1 — תיקון Preprint וכניסה לכתב עת (0–8 שבועות)**
- עיון מחדש בטבלאות ביצוע ברמת TRIPOD checklist
- הוספת Decision-Curve Analysis (DCA) — תוכנן, טרם מומש
- Calibration reliability curves + ECE + Brier report
- יעד כתב עת: *Pancreatology* (IF ~4.5) או *HPB* (IF ~3.8)

**3.1.2 — ולידציה חיצונית (2–6 חודשים)**
- **יעד עדיפות א':** MIMIC-IV מלא (לא demo) — תת-קבוצת AP עם BISAP/APACHE II
- **יעד עדיפות ב':** eICU-CRD 2.0 — מבט multi-center
- **יעד עדיפות ג':** שותף קליני (Hadassah / Vall d'Hebron) — cohort מקומי

**3.1.3 — מחקר יעד (Joint study עם שותף)**
- כותרת מוצעת: *"Combining Early SAP Severity Prediction with ICU Readmission Modeling: A Multi-Institutional Machine Learning Study"*
- שלב הצעת שיתוף עם Prof. Pishgar (USC) — מחקרו על ICU readmission ב-AP משלים ישירות

**מדדי הצלחה:**
- קבלה לכתב-עת עמיתים עד Q4 2026
- ולידציה חיצונית ≥1 cohort עד Q1 2027
- AUC ≥0.85 בולידציה חיצונית

---

### מסלול 2: Sepsis — שיפור ביצועים וביטול shortcut risk (עדיפות גבוהה)

**מצב:** AUROC 0.706, פער מהיעד (0.78-0.83), סיכון shortcut מזוהה.

**שלבים:**

**3.2.1 — אודיט Shortcut-Learning (4–8 שבועות)**
- מימוש בדיקת *"pre-suspicion cutoff"* — הסרת נתונים לאחר חשד קליני ראשון
- ניתוח כל 45 הפרמטרים: Feature Importance + SHAP → זיהוי leakage candidates
- בנצ'מארק: השוואה לתוצאות ה-Epic ESM audit (Wiens et al.)

**3.2.2 — שיפור ביצועים**
- **נתונים:** הרחבה ל-MIMIC-IV מלא (עד N=10,000+ חולי ICU)
- **ארכיטקטורה:** בחינת Transformer (PatchTST / Mamba) לעומת Bi-LSTM+RPN
- **אימון:** Temporal split validation (train 2008-2014, test 2015-2019) לסימולציית dataset shift
- **Rational Polynomial Neuron:** ולידציה — האם מפחית/מגביר shortcut risk?

**3.2.3 — פרסום**
- כותרת מוצעת: *"Shortcut-Learning Audit of an Open-Source ICU Sepsis Prediction Model: Architecture Analysis and Temporal Validation on MIMIC-III/IV"*
- כתב עת יעד: *Critical Care Medicine* או *Intensive Care Medicine*

**מדדי הצלחה:**
- הוכחת shortcut-free AUROC ≥0.74 (pre-suspicion)
- AUROC ≥0.78 לאחר שיפורי ארכיטקטורה
- אודיט מלא מוכן לביקורת חיצונית עד Q3 2026

---

### מסלול 3: Pathogen Classification — כיול ופרסום (עדיפות בינונית)

**מצב:** ROC-AUC 0.956, אך recall אסימטרי ובעיית B:OTHER.

**שלבים:**

**3.3.1 — שיפור Map_org() ו-Class Balance (2–4 שבועות)**
- הרחבת כללי המיפוי ב-`map_org()` — צמצום B:OTHER מתחת ל-20%
- Focal loss / CB-Focal + class-specific thresholds
- Temperature scaling post-hoc calibration

**3.3.2 — מחקר antimicrobial stewardship**
- ניתוח קשר בין חשיפה אנטיביוטית → מחלקת פתוגן בפועל (לפי MIMIC)
- הצגה כ-decision support tool ל-empirical antibiotic selection
- כותרת מוצעת: *"Pathogen-Category Prediction from Microbiology and Antibiotic Exposure: A Calibrated Multiclass Classifier for ICU Stewardship Support"*

**מדדי הצלחה:**
- B:OTHER < 20% בקורפוס המיפוי
- Recall ≥0.70 לכל מחלקה קלינית מרכזית
- ECE ≤0.05 לאחר calibration

---

## חלק ד — תשתית נתונים ודרישות

### ד.1 מפת הנתונים הנוכחית

| Dataset | גישה | שימוש |
|---------|------|--------|
| MIMIC-III demo v1.4 | פתוח | פיתוח ודגמאות |
| MIMIC-IV demo v2.2 | פתוח | פיתוח ודגמאות |
| MIMIC-III/IV מלא | PhysioNet credentialing | ולידציה ואימון |
| Chinese AP cohort (nr2) | SAP preprint | SAP — Mסלול 1 |
| eICU-CRD 2.0 | PhysioNet credentialing | Multi-center validation |
| AmsterdamUMCdb | הרשמה | External ICU validation |

### ד.2 צרכים עתידיים

1. **MIMIC-IV מלא** — דחוף לכל שלושת המסלולים
2. **eICU-CRD** — critical לספסיס multi-center
3. **Cohort מקומי (Hadassah)** — ולידציה ישראלית, נדרש אישור Helsinki

---

## חלק ה — ניהול שותפויות: המלצות מהניתוח

### ה.1 הערכת מצב שיתוף הפעולה

מניתוח תיבת הדוא"ל עולים שלושה דפוסים:

**דפוס א — שיתוף ממוקד עם הצלחה אפשרית:**
- Prof. Pishgar (USC): תחום משלים בדיוק (ICU readmission in AP), ניסיון פרסום 2025, receptive
- **המלצה:** Zoom call → Joint study proposal על MIMIC-IV

**דפוס ב — תגובות שגויות-כוונה:**
- Prof. Ceelen (Ghent): "לא מנתח HPB" — בעיית targeting
- **המלצה:** שיפור מיפוי מוסדי לפני שליחת פנייה

**דפוס ג — רשת בינלאומית רחבה ללא מוקד:**
- 8+ חוקרים מסין, ספרד, הולנד, צרפת — פולואפ מרובה ביום אחד
- **המלצה:** לצמצם ל-2-3 שותפים אמיתיים עם:
  - (1) Cohort מקומי AP מוכן
  - (2) רצון מוכח לשיתוף פעולה
  - (3) קיבולת ל-co-authorship

### ה.2 אסטרטגיית Hadassah ICU

לפי ה-one-pager הקיים (`outreach/hadassah_icu_onepager_en.md`):

**תוכנית שלבים:**
1. **Phase 0 (כעת):** הכנת פרוטוקול מחקר + Helsinki draft
2. **Phase 1 (2-3 חודשים):** ולידציה רטרוספקטיבית על נתוני ICU ישראליים
3. **Phase 2 (6-12 חודשים):** פיילוט "Silent Mode" — המודל רץ ברקע ללא התערבות

**נדרש:**
- Clinical PI מ-Hadassah
- מסלול גישה לנתונים דה-מזוהים (IRB)
- פגישות סקירה קלינית תקופתיות

---

## חלק ו — תוכנית GSoC 2026

**מה מוגדר (docs/GSOC_SCOPE.md):**
- חיזוי דלקת ריאות (binary) — label מקודי אבחנה
- חיזוי תמותה תוך-אשפוזית (binary)
- Features: 24h ראשונות, adult ICU, first stay per patient
- Baseline: LR + Random Forest

**מה עדיין חסר — המלצות לתיוסף:**
1. **Temporal validation** — חלוקה לפי זמן (לא random split)
2. **Subgroup analysis** — age/sex/admission type
3. **Calibration reporting** — Brier + ECE (מוזכר כ"optional if time permits" — יש להפוך לחובה)
4. **Benchmark vs. clinical scores** — APACHE II, SOFA

---

## חלק ז — יומן שגיאות ולקחים

### ז.1 לקחים ממסמכי הפרויקט

| בעיה | לקח |
|------|-----|
| B:OTHER דומיננטי | תמיד לבדוק class distribution לפני אימון |
| No vitals complete rows | להגדיל HOURS_WINDOW + לאפשר imputation |
| Recall אסימטרי בספסיס | Threshold חייב להיות per-class, לא global |
| Dataset shift (ICU) | Temporal split בכל ניסוי |
| Shortcut learning (ספסיס) | Feature audit לפני כל פרסום |

### ז.2 לקחים מניתוח הדוא"ל

| תצפית | לקח אסטרטגי |
|--------|------------|
| תגובת Ceelen: "לא HPB" | בדיקת פרופיל מחקר לפני שליחת פנייה |
| 8 פניות ביום אחד | Spray-and-pray → לא אפקטיבי; מומלץ 1-2 מוצלבים |
| אין תגובות מ-USC/Michigan | לחכות 2-3 שבועות לפולואפ אחד בלבד |
| OHDSI/OMOP participation | הזדמנות: OMOP mapping יפשט ולידציה חיצונית |
| NEJM AI subscription | publish-or-perish: target high-IF venue for each arm |

---

## חלק ח — ציר זמן מסכם

```
יולי 2026
  ├── SAP: תיקון preprint + submission לכתב עת
  ├── Sepsis: shortcut audit + feature SHAP analysis
  └── Pathogen: הרחבת map_org() + calibration

אוגוסט–ספטמבר 2026
  ├── SAP: ולידציה על MIMIC-IV מלא
  ├── Sepsis: אימון מחדש + temporal validation
  └── Hadassah: Helsinki draft + פגישה ראשונה

אוקטובר–נובמבר 2026
  ├── SAP: revision ותגובה לסוקרים
  ├── Sepsis: preprint shortcut audit
  └── GSoC: deliverables finalized

Q1 2027
  ├── SAP: פרסום (עמיתים)
  ├── Sepsis: joint submission (אם שותף מאוגוסט)
  └── Hadassah: Phase 1 retrospective results
```

---

## חלק ט — מדדי הצלחה כלליים

| יעד | מדד | מועד |
|-----|-----|------|
| SAP בכתב עת | קבלת peer-review | Q4 2026 |
| Sepsis audit | shortcut-free AUROC ≥0.74 | Q3 2026 |
| Pathogen calibration | ECE ≤0.05 | Q3 2026 |
| שותף קליני | Zoom ממומש עם ≥1 מוסד | Q3 2026 |
| Hadassah Helsinki | protocol submitted | Q4 2026 |
| MIMIC-IV מלא | גישה מאושרת | Q3 2026 |

---

## נספח — מקורות עיקריים

- TRIPOD reporting: Collins GS et al., BMJ 2015;350:g7594
- Calibration of neural networks: Guo et al., arXiv:1706.04599 (ICML 2017)
- Sepsis-3 definition: Singer M et al., JAMA 2016;315(8):801
- Surviving Sepsis Campaign: Evans L et al., CCM/ICM 2021
- PROBAST: Wolff RF et al., Ann Intern Med 2019;170:51-58
- BISAP: Wu BU et al., Gut 2008;57:1698-1703
- Revised Atlanta Classification: Banks PA et al., Gut 2013;62:102-111
- PhysioNet MIMIC-III/IV: physionet.org
- eICU-CRD 2.0: physionet.org/content/eicu-crd/2.0/
- Epic Sepsis Model audit: Wong A et al., JAMA Intern Med 2021;181:1065

---

*מסמך זה עוגן בנתונים מהפרויקט ומניתוח תקשורת דוא"ל (ינואר–יוני 2026). יש לעדכנו בכל milestone משמעותי.*
