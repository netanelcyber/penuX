# penuX — MIMIC-III / MIMIC-IV Pathogen Class Prediction (`mimic3.py`)

> **מטרה / Objective**
> **עברית:** לספק צינור **מחקר/דמו** בקובץ יחיד (`mimic3.py`) שמאתר אוטומטית נתיבי **MIMIC-III / MIMIC-IV** (כולל `.csv.gz`), מבצע עיבוד וקטלוג פיצ’רים (טקסט/אנטיביוטיקה/ויטלס-מעבדה מוקדמים), מאמן מודל **PyTorch** היברידי לסיווג רב-מחלקתי של **מחלקת פתוגן** (תוויות קבועות), ומפיק הערכה מתקדמת כולל **Calibration**.
> **English:** Provide a **research/demo** single-file (`mimic3.py`) pipeline that auto-discovers **MIMIC-III / MIMIC-IV** roots (including `.csv.gz`), builds mixed text/medication/early-vitals features, trains a hybrid **PyTorch** multiclass classifier for a **fixed pathogen class set**, and produces rich evaluation including **probability calibration**.

**אזהרה / Warning:** זהו קוד **מחקרי/חינוכי בלבד** ואינו מיועד לשימוש קליני או לקבלת החלטות טיפול. MIMIC הוא מאגר דה-מזוהה אך עדיין רגיש ודורש עמידה בהסכמי שימוש. ([PhysioNet][1])

---

## Table of contents

* [תיבת מידע / Infobox](#תיבת-מידע--infobox)
* [Classes (fixed order, CAPITALS)](#classes-fixed-order-capitals)
* [Part A — אנציקלופדיה בעברית](#part-a--אנציקלופדיה-בעברית)

  * [1. רקע והקשר](#1-רקע-והקשר)
  * [2. מאגרי MIMIC והגרסאות הרלוונטיות](#2-מאגרי-mimic-והגרסאות-הרלוונטיות)
  * [3. ארכיטקטורת הפרויקט “ONE FILE” ומה זה אומר בפועל](#3-ארכיטקטורת-הפרויקט-one-file-ומה-זה-אומר-בפועל)
  * [4. דרישות נתונים ופתרון נתיבים (MIMIC-III-like / MIMIC-IV-like)](#4-דרישות-נתונים-ופתרון-נתיבים-mimic-iii-like--mimic-iv-like)
  * [5. הגדרת המשימה והתיוג (Labeling)](#5-הגדרת-המשימה-והתיוג-labeling)
  * [6. בניית פיצ’רים](#6-בניית-פיצרים)
  * [7. מודל היברידי ב-PyTorch](#7-מודל-היברידי-בpyTorch)
  * [8. אימון, Losses ועצירה מוקדמת](#8-אימון-losses-ועצירה-מוקדמת)
  * [9. הערכה: מטריקות, ROC/PR, Calibration](#9-הערכה-מטריקות-rocpr-calibration)
  * [10. בדיקות הטיה (Subgroup/Bias Checks)](#10-בדיקות-הטיה-subgroupbias-checks)
  * [11. תוצרים (Artifacts) וקבצי PNG](#11-תוצרים-artifacts-וקבצי-png)
  * [12. התקנה והרצה](#12-התקנה-והרצה)
  * [13. משתני סביבה מרכזיים](#13-משתני-סביבה-מרכזיים)
  * [14. Troubleshooting](#14-troubleshooting)
  * [15. פרטיות, אתיקה ושימוש אחראי](#15-פרטיות-אתיקה-ושימוש-אחראי)
* [Part B — Encyclopedia in English](#part-b--encyclopedia-in-english)

  * (mirrors the Hebrew sections, expanded)
* [References](#references)

---

## תיבת מידע / Infobox

| Field            | Value                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Project          | **penuX**                                                                                                                 |
| Primary script   | `mimic3.py` (single-file demo pipeline)                                                                                   |
| Data             | MIMIC-III / MIMIC-IV (incl. demo subsets)                                                                                 |
| Task             | Multiclass pathogen-class prediction (fixed label set)                                                                    |
| Core constraints | **ONE FILE**, **NO pandas**, **PyTorch-first**                                                                            |
| Key outputs      | Accuracy/F1, confusion (OvR), ROC-AUC/PR-AUC, **Calibration** (ECE/Brier + reliability table), subgroup checks, PNG plots |
| Intended use     | Research/education only (not clinical)                                                                                    |

**Sources:** repository listing shows `mimic3.py` and calibration images present; MIMIC dataset descriptions come from PhysioNet. ([GitHub][2])

---

## Classes (fixed order, CAPITALS)

```text
B:PSEUDOMONAS AERUGINOSA
B:STAPH AUREUS COAG +
B:SERRATIA MARCESCENS
B:MORGANELLA MORGANII
B:ESCHERICHIA COLI
B:PROTEUS MIRABILIS
B:PROVIDENCIA STUARTII
B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS
B:YEAST
B:GRAM POSITIVE COCCUS(COCCI)
B:OTHER
V:OTHER
```

> Mapping is controlled by `map_org()` — if `B:OTHER` dominates, expand rules there.

**Sources:** MIMIC provides microbiology-related tables/fields (content level); the *specific* label inventory and mapping policy are a project design choice. ([PhysioNet][3])

---

# Part A — אנציקלופדיה בעברית

## 1. רקע והקשר

**penuX** הוא פרויקט דמו/מחקר שמדגים צינור למידת מכונה על נתוני רשומה רפואית אלקטרונית (EHR) מתוך משפחת מאגרי **MIMIC**. מאגרי MIMIC נבנו כדי לאפשר מחקר פתוח יותר בעולם הטיפול הנמרץ והרפואה האשפוזית, והם כוללים שילוב רחב של נתונים מובנים כגון דמוגרפיה, מדדים חיוניים, בדיקות מעבדה, תרופות/הזמנות ועוד. ([PhysioNet][3])

הפרויקט “מקטין” את העולם לכדי משימה מדגימה: ניבוי **מחלקת פתוגן** (רשימת תוויות קבועה) מתוך שלושה סוגי אותות:

1. טקסט קטגורי במיקרוביולוגיה (שדות קצרים יחסית),
2. חשיפה לאנטיביוטיקות (one-hot),
3. ויטלס/מעבדה מוקדמים (חלון שעות מהקבלה).

המטרה העיקרית היא **הדגמה**: איך עושים auto-discovery לדאטה, streaming CSV ללא Pandas, בניית פיצ’רים, אימון מודל PyTorch היברידי, והפקת דוח הערכה “עשיר” כולל **Calibration**.

**מקורות:** תיאור היקף MIMIC-III (כולל סוגי נתונים) ומסמך הדסקריפטור של MIMIC-III. ([PhysioNet][3])

---

## 2. מאגרי MIMIC והגרסאות הרלוונטיות

### 2.1 MIMIC-III (v1.4 + demo v1.4)

**MIMIC-III Clinical Database v1.4** הוא מאגר דה-מזוהה גדול של מטופלי טיפול נמרץ (BIDMC, 2001–2012). PhysioNet מציין שהמאגר כולל בין היתר דמוגרפיה, מדדים חיוניים, בדיקות מעבדה, תרופות, פרוצדורות, דוחות ועוד. ([PhysioNet][3])
גרסת **demo v1.4** מספקת תת-קבוצה לצורכי למידה/בדיקה, אך עדיין מחייבת התייחסות כנתונים רגישים (וגם כאשר דה-מזוהים – נדרשת עמידה בכללי/הסכמי שימוש). ([PhysioNet][1])

### 2.2 MIMIC-IV (demo v2.2; וגרסאות מאוחרות יותר)

PhysioNet מתאר את **MIMIC-IV** כמאגר שמאמץ גישה מודולרית לארגון נתונים ומדגיש provenance. ([PhysioNet][4])
גרסת **MIMIC-IV demo v2.2** היא דמו פתוח המכיל תת-קבוצה (PhysioNet מציין ~100 מטופלים) ובדרך כלל *ללא* הערות קליניות חופשיות (free-text notes). ([PhysioNet][5])

> הערת עדכון (חשוב למחקר): ב-PhysioNet קיימות גם גרסאות מאוחרות יותר של MIMIC-IV (למשל v3.1 שפורסמה ב-2024). הפרויקט שלך יכול להישאר ממוקד בדמו v2.2 לצורך ריצה מהירה, אבל טוב לדעת שהאקוסיסטם התקדם. ([PhysioNet][6])

**מקורות:** דפי PhysioNet לגרסאות/דמו, והמאמר Data Descriptor עבור MIMIC-IV. ([PhysioNet][3])

---

## 3. ארכיטקטורת הפרויקט “ONE FILE” ומה זה אומר בפועל

### 3.1 “ONE FILE”

במונחי הנדסה, “ONE FILE” אומר:

* כל הלוגיקה (Discovery → ETL streaming → Modeling → Training → Evaluation → Plotting) נמצאת בקובץ אחד, לרוב כדי להקל על הדגמות, שיתוף, והרצה מהירה בסביבות ללא תלות בפרויקט רב-קבצים.
* המשמעות היא פשרות מסוימות בניקיון ארכיטקטוני (פחות מודולים/חבילות), אבל יתרון בקלות הפעלה והעתקה.

### 3.2 “NO pandas”

במקום טעינת טבלאות ענק ל-RAM, הפרויקט נשען על:

* קריאה שורתית (streaming) של CSV באמצעות `csv` + `gzip` (כאשר `.csv.gz`)
* צבירה אינקרמנטלית (למשל mapping לפי `hadm_id`, ספירה לפי תווית, ומבני נתונים “קטנים” יחסית שנגזרים מתוך הטבלאות הגדולות)

זו התאמה טבעית לעולם MIMIC שבו טבלאות כמו אירועי ICU/מעבדה יכולות להיות ענקיות.

### 3.3 “PyTorch-only”

המודל וריצת האימון בנויים על אבני-בניין סטנדרטיות של PyTorch: Embedding, Conv1d, LSTM, שכבות fully-connected וכו’. ([PyTorch Docs][7])

**מקורות:** תיעוד PyTorch לשכבות Embedding/Conv1d/LSTM; ורשימת קבצים במאגר המדגימה שהסקריפט וה-PNGs נמצאים יחד. ([PyTorch Docs][7])

---

## 4. דרישות נתונים ופתרון נתיבים (MIMIC-III-like / MIMIC-IV-like)

### 4.1 MIMIC-III-like root (flat CSVs)

הפרויקט מצפה ל-root שמכיל CSV/CSV.GZ (לא תלוי רישיות) עבור “ליבה” של ישויות:

* Microbiology events (לתיוג)
* Prescriptions (חשיפות אנטיביוטיות)
* Admissions / Patients (גיל/זמני קבלה)
* Chart events + dictionaries (מדדים חיוניים)
* Lab events + lab dictionaries (מעבדה)

ב-MIMIC-III קיימת זמינות רחבה של סוגי נתונים כאלה, אם כי שמות קבצים/שדות יכולים להיות תלויי גרסה/הפצה. ([PhysioNet][3])

### 4.2 MIMIC-IV-like demo root (folders)

ב-MIMIC-IV PhysioNet מדגיש ארגון מודולרי; בדמו לרוב תראו תיקיות כמו `hosp/` ו-`icu/`. “Auto-discovery” בפרויקט מכוון לזה: לזהות שורש, ואז לבנות paths לפי חוקים (וגם לתמוך ב-`.csv.gz`). ([PhysioNet][4])

### 4.3 Auto-discovery (הגיון פעולה)

הסקריפט סורק נתיבים אפשריים:

* נתיבי ברירת מחדל בתוך הריפו (למשל `dataset/…` או דומים)
* תתי-תיקיות מתחת לנתיב “dataset/mimic”
* ורשימת override דרך משתנה סביבה `MIMIC_AUTOROOTS`

הגיון זה חשוב במיוחד כי משתמשים שונים מורידים את הדמו/המאגר במבנים שונים (למשל zip, AWS, BigQuery, ועוד). ([Registry of Open Data][8])

**מקורות:** PhysioNet מתאר את מבנה MIMIC-IV כמודולרי ואת הדמו כמועיל לבחינת התאמה למחקר; AWS Open Data Registry מתאר נתיבי שימוש נפוצים; ו-PhysioNet מציין אפשרויות גישה/פלטפורמות. ([PhysioNet][4])

---

## 5. הגדרת המשימה והתיוג (Labeling)

### 5.1 מה מנבאים?

היעד הוא **class prediction**: לבחור תווית פתוגן אחת מתוך רשימה קבועה, ולספק גם הסתברויות (softmax).

### 5.2 מקור התווית

בפרויקט, התווית נגזרת מאירועי מיקרוביולוגיה: יש שדה אורגניזם/תוצאה (משתנה לפי סכימה), והפונקציה `map_org()` מבצעת:

* נרמול טקסט (למשל uppercase/strip)
* כללי התאמה (substring / regex / מילון)
* מיפוי ל-12 המחלקות

זהו “לב מדעי” של הפרויקט: שינוי כללי המיפוי משנה דרמטית את class balance ואת איכות ה-ground truth.

### 5.3 “B:OTHER” כבעיה מחקרית

כאשר מיפוי נוקשה מדי:

* הרבה אורגניזמים “נופלים” ל-`B:OTHER`
* המודל נוטה להתכנס (collapse) למחלקה דומיננטית
* יש “תמריץ” מלאכותי ל-accuracy גבוה אך מידע קליני נמוך

לכן ההמלצה המרכזית היא: אם `B:OTHER` דומיננטי, **להרחיב** את `map_org()`.

**מקורות:** MIMIC-III ו-MIMIC-IV כוללים נתוני בדיקות/מעבדה/מדידות ויכולים לכלול גם רכיבי מיקרוביולוגיה (תלוי סכימה/הפצה); אך פרטי ההגדרה של מחלקות פתוגן הם החלטת פרויקט. ([PhysioNet][3])

---

## 6. בניית פיצ’רים

> כלל זהב בפרויקט: פיצ’רים “בעלי משמעות” אך פשוטים מספיק כדי לרוץ מהר על דמו/מכונה רגילה.

### 6.1 טקסט מיקרוביולוגי (categorical text)

שני שדות:

* `spec_type_desc`
* `interpretation`

עיבוד טיפוסי:

* lowercase
* צמצום רווחים מרובים
* tokenization בסיסית (split whitespace)
* מיפוי טוקנים ל-vocab (עד `MAX_TEXT_TOKENS`)
* חיתוך/ריפוד לאורך `TEXT_SEQ_LEN`

הטקסט כאן אינו “נרטיב קליני” ארוך (כמו notes), אלא שדות קטגוריים קצרים—לכן Conv1D יכול להיות יעיל כדי לקלוט n-grams/דפוסים מקומיים.

### 6.2 חשיפה לאנטיביוטיקות (binary one-hot)

חמש אנטיביוטיקות:

* vancomycin, ciprofloxacin, meropenem, piperacillin, ceftriaxone

סטרטגיה שכיחה:

* לייצר לכל `hadm_id` וקטור 0/1
* “1” אם הייתה מרשם/מתן בחלון (או בכלל האשפוז, תלוי החלטת פרויקט)

היתרון: מודל יכול ללמוד קשרים בין חשיפה אנטיביוטית לבין הסתברויות פתוגנים/קטגוריות.

### 6.3 ויטלס/מעבדה מוקדמים (enforced order)

סדר חובה:

1. `temperature_c`
2. `wbc`
3. `spo2`
4. `age`

וכאשר משלבים עם אנטיביוטיקות, מתקבל וקטור “מספרי” מורחב (כפי שתיארת).

**חלון זמן:** `HOURS_WINDOW` שעות מהקבלה (`ADMITTIME`).
טקטיקה מקובלת: עבור כל מדד, לבחור ערך ראשון/ממוצע/חציון בתוך החלון (לפרויקט דמו—בחירה קונסיסטנטית חשובה יותר מ”האופטימום”).

### 6.4 טיפול בחוסרים (missingness)

בעולם EHR חסרים הם נורמה. פרויקט דמו צריך לבחור:

* או “strict complete rows” (פשוט, אך עלול למחוק הרבה נתונים)
* או אימפוטציה קלה (median/mean + missing indicators)

כיוון שאתה מתאר “No vitals complete rows” כתקלה אפשרית, נראה שהצינור בדמו נוטה ל-strictness, עם התאמות דרך `HOURS_WINDOW`.

**מקורות:** תיאור תכולת MIMIC-III ו-MIMIC-IV (מדדים חיוניים, מעבדה, תרופות) ב-PhysioNet ובמאמרי הדסקריפטור. ([PhysioNet][3])

---

## 7. מודל היברידי ב-PyTorch

### 7.1 למה היברידי?

ב-EHR יש לרוב שילוב של:

* טקסט/קטגוריות (תוויות/תיאורים)
* מספרים (מדדים)
* בינאריים (חשיפות/אבחנות)

מודל “ענפים” מאפשר לכל מודאליות ייצוג מתאים, ואז fusion.

### 7.2 ענף טקסט

אבני הבניין שתיארת הן סטנדרטיות ב-PyTorch:

* `torch.nn.Embedding` (מיפוי טוקנים לווקטורים) ([PyTorch Docs][7])
* `torch.nn.Conv1d` (דפוסים מקומיים) ([PyTorch Docs][9])
* RNN / LSTM, כולל BiLSTM לייצוג דו-כיווני לאורך הרצף ([PyTorch Docs][10])

רעיון אפשרי (דמו):

* Embedding → Conv1D → pooling → RNN → BiLSTM → vector

### 7.3 ענף מספרי

וקטור (temperature, wbc, spo2, age + ABX) נכנס ל-MLP קטן:

* Linear → activation → dropout (אופציונלי) → Linear …

### 7.4 Fusion head

איחוד הייצוגים:

* concat([text_vec, numeric_vec])
* MLP → logits → softmax

### 7.5 אקטיבציות

הגדרה דינמית (למשל `MIMIC_ACTIVATIONS="gelu,swish,elu,relu"`) מאפשרת ניסוי מהיר. (גם אם לא כולן “מובנות” ב-torch.nn כברירת מחדל, אפשר לממש חלקן בקלות.)

**מקורות:** תיעוד PyTorch לשכבות Embedding/Conv1d/LSTM. ([PyTorch Docs][7])

---

## 8. אימון, Losses ועצירה מוקדמת

### 8.1 Early stopping לפי יעד

בניגוד ל-early stopping קלאסי (monitor val loss), בדמו שלך יש “יעד”:

* לעצור כשהמודל מגיע ל-`TARGET_ACC` או `TARGET_F1`
* לבחור kind (overall / subset כמו MRSA/MSSA)
* להגן מפני תנודות באמצעות `EARLY_PATIENCE` ו-`MIN_DELTA`

זה שימושי לדמו: אם המטרה “להראות pipeline עובד”, אפשר לקצר ניסויים.

### 8.2 פונקציות הפסד (Loss)

ציינת כמה אפשרויות:

* CE (Cross-Entropy)
* Weighted CE (להתמודדות עם imbalance)
* Focal / Class-balanced focal (CB-Focal)

הרעיון מאחורי focal: להוריד משקל לדוגמאות “קלות” ולהתמקד בטעויות/דוגמאות קשות. ברמת דמו, זה יכול לעזור אם יש מחלקות נדירות.

### 8.3 Label smoothing

`LOSS_LABEL_SMOOTHING` מוסיף “רכות” לתוויות כדי להפחית overconfidence—שיכול להשפיע גם על calibration.

### 8.4 Restarts ו-retrain-on-full-train

* `MAX_TRAIN_RESTARTS`: אם אימון “נתקע” (למשל collapse), להתחיל מחדש עם seed/אתחול חדש
* `RETRAIN_ON_FULL_TRAIN`: לאחר בחירת epoch מיטבי, לאמן מחדש על train מלא

**מקורות:** כיול יתר (overconfidence) של רשתות מודרניות מוזכר היטב בספרות כיול, והקשר בין training choices לבין calibration מופיע למשל אצל Guo et al. ([arXiv][11])

---

## 9. הערכה: מטריקות, ROC/PR, Calibration

### 9.1 דוח סיווג בסיסי

* Accuracy כולל
* Precision/Recall/F1 per class
* macro/weighted F1 (חשוב ב-imbalance)

### 9.2 Confusion + OvR stats

בנוסף למטריצת בלבול סטנדרטית, הפרויקט מחשב **One-vs-Rest** לכל מחלקה:

* TP/FP/FN/TN
* Sensitivity (Recall), Specificity, PPV (Precision), F1

זה נותן “תמונת רנטגן” למחלקות ספציפיות (למשל MRSA).

### 9.3 ROC-AUC / PR-AUC במולטיקלאס

ב-scikit-learn, `roc_auc_score` תומך ב-multiclass תחת מגבלות מסוימות (למשל OvR/OvO + averaging). ([Scikit-learn][12])
ל-PR, נהוג להשתמש ב-`average_precision_score` (AP), שמסכם precision-recall curve. ([Scikit-learn][13])

> הערה מעשית: ב-multiclass, ROC/PR בדרך כלל מחשבים “לכל מחלקה מול השאר” (OvR) ואז מסכמים macro/weighted.

### 9.4 Calibration: Reliability diagram, ECE, Brier

#### 9.4.1 מה זה calibration?

מודל “מכויל” אומר: אם הוא חוזה הסתברות 0.8, אז בערך ב-80% מהמקרים הוא צודק (ברמת קבוצות/בינים).

scikit-learn מתאר calibration curves / reliability diagrams ככלי ויזואלי להשוואת הסתברות חזויה מול שכיחות אמיתית. ([Scikit-learn][14])

#### 9.4.2 Brier score

`brier_score_loss` מודד ממוצע ריבוע ההפרש בין הסתברות חזויה לתוצאה בפועל (בינארית). ערך נמוך יותר טוב יותר. ([Scikit-learn][15])

#### 9.4.3 ECE (Expected Calibration Error)

ECE הוא מדד נפוץ שמבצע binning על ההסתברויות ומשווה בין **confidence** ממוצע לבין **accuracy** אמפירית בכל bin.
בספרות כיול מודרנית (למשל Guo et al.) מודגש שמודלים עמוקים יכולים להיות לא-מכוילים, ושיטות כמו temperature scaling יכולות לשפר. ([arXiv][11])
קיימות גם עבודות חדשות יותר שמנתחות איך להעריך ECE בצורה עקרונית יותר, מה שמרמז שזהו מדד שימושי אך לא “מושלם”. ([arXiv][16])

**מקורות:** scikit-learn calibration/metrics docs + Guo et al. (2017) + מאמרים על ECE. ([Scikit-learn][14])

---

## 10. בדיקות הטיה (Subgroup/Bias Checks)

הצינור כולל בדיקות “best-effort” עבור תתי-אוכלוסיות אם קיימות עמודות רלוונטיות:

* sex/gender
* age bins
* admission_type / admission_location

בפועל, בדיקות כאלה בדמו אמורות:

* להציג metric per group (למשל accuracy/F1)
* להצביע על פערים בולטים
* לא להכריז “הוגנות” (fairness) — אלא להאיר סיכונים/פערים לבדיקות המשך

**מקורות:** עצם הזמינות של נתוני דמוגרפיה/קבלה ב-MIMIC מוזכרת בתיאור המאגר וב-Data Descriptor; ניתוח פערי ביצועים הוא פרקטיקה מחקרית כללית. ([PhysioNet][3])

---

## 11. תוצרים (Artifacts) וקבצי PNG

### 11.1 שמות תוצרים

כאשר `matplotlib` זמין, נוצרים:

* `calibration__<dataset_tag>__<activation>.png`
* `vitals_feature_signal__<dataset_tag>__<activation>.png`
* ולעיתים גם ROC/PR plots

### 11.2 “Calibration PNG” (מוטמע)

המאגר שלך כבר כולל קבצי calibration לדמו MIMIC-III ו-MIMIC-IV (למשל `...demo_1_4__gelu.png` ו-`...demo_2_2__gelu.png`). ([GitHub][2])

להלן דוגמה כפי שתיארת (נתיב יחסי — מתאים ל-GitHub README):

![Calibration example](calibration__mimic3_mimic_iii_clinical_database_demo_1_4__gelu.png)

**מקורות:** קיומם של ה-PNGs בריפו; והגדרת reliability diagrams בתיעוד scikit-learn. ([GitHub][2])

---

## 12. התקנה והרצה

### 12.1 דרישות

* Python 3.9+ מומלץ
* תלות חובה: `numpy`, `torch`, `scikit-learn`
* תלות אופציונלית: `matplotlib` (ל-PNG), `scipy` (ל-p-values)

### 12.2 התקנה (דוגמה)

```bash
pip install numpy scikit-learn
pip install torch
pip install matplotlib   # optional (PNG plots)
pip install scipy        # optional (better p-values)
```

### 12.3 הרצה

```bash
python mimic3.py
```

במהלך הריצה הסקריפט אמור:

1. לגלות roots אפשריים (ברירות מחדל + `MIMIC_AUTOROOTS`)
2. לפתור נתיבי MIMIC-III/MIMIC-IV
3. לבנות תוויות ממיקרוביולוגיה (`map_org`)
4. לבנות one-hot אנטיביוטיקות
5. לחלץ ויטלס/מעבדה מוקדמים בחלון `HOURS_WINDOW`
6. לאמן מודל היברידי עם עצירה מוקדמת
7. להדפיס מטריקות + לשמור plots (אם plotting זמין)

**מקורות:** קיום הסקריפט וה-artifacts בריפו; תיעוד PyTorch לשכבות; תיעוד scikit-learn למטריקות/כיול. ([GitHub][2])

---

## 13. משתני סביבה מרכזיים

> הרשימה כאן משקפת “פרמטריזציה לדמו”: אפשר לשחק עם חלונות זמן, batch, loss, ארכיטקטורה, device, ומדדי עצירה.

### 13.1 חלון / באצ’ינג

* `HOURS_WINDOW` (default 24)
* `BATCH_SIZE` (default 64)

### 13.2 Stop criteria

* `TARGET_STOP_METRIC` = `acc`/`f1`
* `TARGET_ACC`, `TARGET_ACC_KIND` (overall / subset)
* `TARGET_F1`, `TARGET_F1_KIND` (macro/weighted/subset)
* `MAX_EPOCHS`, `EARLY_PATIENCE`, `MIN_DELTA`
* `RETRAIN_ON_FULL_TRAIN`, `MAX_TRAIN_RESTARTS`

### 13.3 Loss

* `LOSS_NAME` = `ce` / `wce` / `focal` / `cb_focal`
* `LOSS_LABEL_SMOOTHING`, `FOCAL_GAMMA`, `CB_BETA`, `FOCAL_USE_ALPHA`
* `MAX_CLASS_WEIGHT`, `BOTHER_EXTRA_DOWNWEIGHT`

### 13.4 Text / sequence

* `MAX_TEXT_TOKENS`, `TEXT_SEQ_LEN`
* `EMBED_DIM`, `CNN_FILTERS`, `CNN_KERNEL`
* `RNN_UNITS`, `LSTM_UNITS`
* `MIMIC_ACTIVATIONS`

### 13.5 Device

* `DEVICE` = `cpu`/`cuda`/auto

**מקורות:** הנמקה מחקרית כללית (class imbalance, calibration) נתמכת בספרות כיול; ושימוש בשכבות/אימון נתמך בתיעוד PyTorch. ([arXiv][11])

---

## 14. Troubleshooting

### 14.1 “No vitals complete rows”

סיבות נפוצות:

* חלון זמן קטן מדי (`HOURS_WINDOW`)
* אי-התאמה בין מילונים (D_ITEMS / D_LABITEMS) לבין קודי האירועים
* בעיות parsing (פורמט תאריך, ערכים לא מספריים)

פתרונות:

* להגדיל `HOURS_WINDOW`
* להוסיף “רשימות שמות חלופיות” לשדות (label matching)
* לשקול אימפוטציה קלה במקום strict rows

### 14.2 ריצה איטית מאוד

* טבלאות אירועים/מעבדה גדולות → streaming הוא I/O heavy
* דיסק SSD יכול לשפר משמעותית
* בדמו: אפשר לצמצם מראש ל-`hadm_id` רלוונטיים (לניסוי)

### 14.3 מודל מתכנס ל-`B:OTHER`

* לשפר `map_org()` (הפתרון הטוב ביותר)
* לשחק עם `BOTHER_EXTRA_DOWNWEIGHT`, משקלי מחלקה, ו-loss
* לבדוק אם ה-labels באמת מאוזנים/אינפורמטיביים

**מקורות:** תכולת MIMIC-III/MIMIC-IV והיקפי נתונים מתוארים ב-PhysioNet; בעיות imbalance ו-calibration נפוצות בלמידה עמוקה. ([PhysioNet][3])

---

## 15. פרטיות, אתיקה ושימוש אחראי

* MIMIC הוא מאגר **דה-מזוהה** אך כולל מידע קליני רגיש; PhysioNet מציין שהגישה לדאטה המלא דורשת credentialing/הסכמי שימוש (DUA) ועמידה בהכשרות רלוונטיות. ([PhysioNet][1])
* פרויקט ניבוי פתוגנים/עמידות הוא דוגמה קלאסית לתחום “high-stakes”; לכן ההדגשה “not for clinical use” אינה רק פורמלית—אלא הכרחית.
* Calibration ו-subgroup checks הם צעדים לשקיפות, אבל לא תחליף לולידציה קלינית/רגולטורית.

**מקורות:** PhysioNet demo pages (גישה/הגבלות) + ספרות כיול רשתות מודרניות. ([PhysioNet][1])

---

# Part B — Encyclopedia in English

## 1. Background and context

**penuX** is a research/demo project that illustrates an end-to-end machine learning pipeline on the **MIMIC** family of de-identified EHR datasets. MIMIC datasets were created to lower barriers to reproducible critical care and hospital research and include structured information such as demographics, bedside vital signs, lab tests, medications, procedures, reports, and outcomes. ([PhysioNet][3])

The project narrows this broad EHR universe into a compact demonstrator task: predicting a **fixed pathogen class** from (i) short microbiology categorical text fields, (ii) binary antibiotic exposure indicators, and (iii) early vitals/labs within a configurable admission window. The goal is not clinical deployment but a practical demonstration of: data discovery, streaming ETL without pandas, hybrid neural modeling in PyTorch, and “rich” evaluation including probability calibration.

**References:** PhysioNet MIMIC-III overview and the MIMIC-III descriptor PDF; MIMIC-IV descriptor paper. ([PhysioNet][3])

---

## 2. MIMIC datasets and relevant versions

### 2.1 MIMIC-III (v1.4 + demo v1.4)

PhysioNet describes **MIMIC-III v1.4** as a large de-identified critical care database (BIDMC, 2001–2012) with demographics, bedside vitals, labs, medications, notes, imaging reports, and mortality information. ([PhysioNet][3])
The **demo v1.4** subset exists to let researchers inspect structure/content before requesting full access, but PhysioNet still emphasizes credentialing and responsible use. ([PhysioNet][1])

### 2.2 MIMIC-IV (demo v2.2, plus later releases)

PhysioNet highlights **MIMIC-IV** as adopting a modular organization approach that improves provenance and reuse. ([PhysioNet][4])
The **MIMIC-IV demo v2.2** is openly available and described as a 100-patient subset with similar content to MIMIC-IV while excluding free-text clinical notes. ([PhysioNet][5])

> Update note: PhysioNet has also published newer MIMIC-IV versions (e.g., v3.1, 2024). penuX can remain demo-focused, but users should be aware that schema and content may evolve across releases. ([PhysioNet][6])

**References:** PhysioNet pages for MIMIC-III/MIMIC-IV and demo subsets; MIMIC-IV paper PDF. ([PhysioNet][3])

---

## 3. “ONE FILE” design and practical implications

### 3.1 Single-file scope

A “one-file” pipeline puts discovery, streaming ETL, feature building, training, evaluation, and optional plotting into one script for portability and easy demos. The trade-off is reduced modularity, but the benefit is frictionless execution and sharing.

### 3.2 No pandas (streaming CSV)

MIMIC-style event tables can be very large, so streaming row-by-row parsing (with optional gzip support) is a practical approach to avoid loading full tables into RAM.

### 3.3 PyTorch-first modeling

The hybrid network is assembled from standard PyTorch modules such as **Embedding**, **Conv1d**, and **LSTM** blocks. ([PyTorch Docs][7])

**References:** PyTorch module docs; repository listing showing the single script and plot artifacts. ([PyTorch Docs][7])

---

## 4. Data requirements and path resolution (MIMIC-III-like / MIMIC-IV-like)

### 4.1 MIMIC-III-like flat CSV roots

The pipeline expects a dataset root containing core hospital/ICU tables (microbiology, prescriptions, admissions/patients, chart events + dictionaries, lab events + dictionaries). Exact filenames/column names can vary by distribution, so robust resolution logic is helpful.

### 4.2 MIMIC-IV-like folder roots

MIMIC-IV is described as modular; demo distributions often include `hosp/` and `icu/` folders. The “auto-discovery” feature searches for these patterns and resolves equivalent CSV locations accordingly. ([PhysioNet][4])

### 4.3 Discovery strategy

The script searches default candidate roots and accepts overrides via `MIMIC_AUTOROOTS`, which is useful because PhysioNet/AWS/BigQuery workflows place files in different directory layouts. ([Registry of Open Data][8])

**References:** PhysioNet descriptions of MIMIC-IV’s modular structure and demo usage; access workflows across platforms are referenced by PhysioNet/AWS. ([PhysioNet][4])

---

## 5. Task definition and labeling

### 5.1 What is predicted?

A multiclass label from a **fixed pathogen class inventory** (softmax probabilities + argmax class).

### 5.2 Label source and `map_org()`

Ground-truth labels are constructed from microbiology-derived organism/result strings mapped into the fixed classes via `map_org()`. Because this mapping is rule-based, label quality and class balance depend strongly on mapping coverage.

### 5.3 The “B:OTHER dominance” failure mode

If mapping coverage is too narrow:

* many organisms collapse into `B:OTHER`
* the model can overfit to the dominant bucket
* headline accuracy becomes less meaningful

Hence, expanding mapping rules is the primary corrective action.

**References:** MIMIC dataset descriptors (scope and content). The specific class taxonomy is a project choice, but the repository’s file set indicates the pipeline and plots exist. ([PhysioNet][3])

---

## 6. Feature construction

### 6.1 Microbiology categorical text

Two short fields (`spec_type_desc`, `interpretation`) are normalized (lowercase, whitespace collapse) and converted into fixed-length token sequences. Conv1D layers can capture local token patterns, while (Bi)LSTM layers summarize longer dependencies.

### 6.2 Antibiotic exposure one-hot

Binary indicators for five antibiotics (e.g., vancomycin, ciprofloxacin, meropenem, piperacillin, ceftriaxone) aggregated at the admission level.

### 6.3 Early vitals/labs (enforced order)

The required numeric feature order is:

1. temperature (°C)
2. WBC
3. SpO₂
4. age
   (optionally concatenated with antibiotic one-hots)

Values are extracted within the first `HOURS_WINDOW` hours after admission time.

### 6.4 Missingness handling

EHR data is sparse; the demo can choose strict complete-case filtering or lightweight imputation. The troubleshooting hints suggest complete-case filtering may be used by default, configurable through window size and mapping rules.

**References:** MIMIC-III/MIMIC-IV descriptors emphasize availability of vitals, labs, and medication-related data. ([PhysioNet][3])

---

## 7. Hybrid model in PyTorch

### 7.1 Text branch

* `torch.nn.Embedding` for token lookup ([PyTorch Docs][7])
* `torch.nn.Conv1d` for local pattern extraction ([PyTorch Docs][9])
* `torch.nn.LSTM` (bidirectional) for sequence summarization ([PyTorch Docs][10])

### 7.2 Numeric branch

A small MLP transforms early vitals/labs + antibiotic indicators into a compact numeric embedding.

### 7.3 Fusion head

Concatenation of text and numeric embeddings followed by a classification MLP producing multiclass logits.

**References:** PyTorch docs for core modules. ([PyTorch Docs][7])

---

## 8. Training, losses, and target-based early stopping

### 8.1 Target-based early stopping

The pipeline can stop when a target metric is reached (`acc` or `f1`), optionally on a specific slice (e.g., MRSA/MSSA). This is demo-friendly, though it differs from standard research workflows that optimize validation loss or perform thorough cross-validation.

### 8.2 Loss choices and class imbalance

Cross-entropy variants (weighted) and focal/class-balanced focal losses can be used to address skewed label distributions and hard examples.

### 8.3 Calibration-aware options

Label smoothing and post-hoc calibration (e.g., temperature scaling) are commonly discussed when addressing overconfidence. Modern neural networks can be poorly calibrated, and temperature scaling is shown to be effective in many settings. ([arXiv][11])

**References:** Guo et al. (2017) on calibration of modern neural networks. ([arXiv][11])

---

## 9. Evaluation: metrics, ROC/PR, and calibration

### 9.1 Multiclass ROC-AUC / PR-AUC

scikit-learn’s `roc_auc_score` supports multiclass evaluation with specific restrictions and configuration (e.g., OvR/OvO and averaging). ([Scikit-learn][12])
Average precision (AP) summarizes a precision–recall curve and is implemented as `average_precision_score`. ([Scikit-learn][13])

### 9.2 Calibration reporting

scikit-learn documents calibration curves (reliability diagrams) and provides visualization helpers (e.g., `CalibrationDisplay`). ([Scikit-learn][14])
Brier score loss is a standard probability scoring rule and is implemented as `brier_score_loss`. ([Scikit-learn][15])
ECE is widely used but has known estimation subtleties; newer work analyzes principled estimation of calibration errors. ([arXiv][16])

**References:** scikit-learn docs (metrics + calibration) and calibration literature. ([Scikit-learn][12])

---

## 10. Subgroup/bias checks

Where available, the pipeline reports metrics across groups (sex/gender, age bins, admission context). These checks are diagnostic: they can surface disparities but do not establish fairness or clinical validity.

**References:** MIMIC descriptors confirm demographic/admission context exists in the datasets; subgroup evaluation is a general ML auditing practice. ([PhysioNet][3])

---

## 11. Artifacts and PNG outputs (including calibration)

The repository includes multiple calibration plots for MIMIC-III demo and MIMIC-IV demo runs (e.g., `...demo_1_4__gelu.png`, `...demo_2_2__gelu.png`). ([GitHub][2])

Example (relative path, suitable for GitHub README):

![Calibration example](calibration__mimic3_mimic_iii_clinical_database_demo_1_4__gelu.png)

**References:** repository file listing and scikit-learn calibration concept documentation. ([GitHub][2])

---

## 12. Install and run

```bash
pip install numpy scikit-learn
pip install torch
pip install matplotlib   # optional
pip install scipy        # optional
python mimic3.py
```

**References:** PyTorch module docs; scikit-learn metrics/calibration docs. ([PyTorch Docs][10])

---

## 13. Key environment variables

(Expanded list mirrors the Hebrew section; used to configure windows, early stopping, losses, text sequence parameters, and device selection.)

**References:** Calibration literature for why overconfidence and calibration tuning matter; PyTorch docs for implementation primitives. ([arXiv][11])

---

## 14. Troubleshooting

(Expanded list mirrors the Hebrew section: window size, dictionary label matching, I/O bottlenecks, collapse to OTHER, and mitigation via mapping rules and class-balancing.)

**References:** MIMIC dataset scale and breadth described by PhysioNet; calibration literature for overconfidence patterns. ([PhysioNet][3])

---

## 15. Privacy, ethics, and responsible use

PhysioNet emphasizes that even de-identified MIMIC data is sensitive, and access often requires credentialing/training and data use agreements. The pipeline is explicitly research/education only and must not be used for clinical decision-making. ([PhysioNet][1])

**References:** PhysioNet demo pages and dataset access notes. ([PhysioNet][1])

---

# References

* PhysioNet pages for MIMIC-III v1.4 and MIMIC-III demo v1.4. ([PhysioNet][3])
* PhysioNet pages for MIMIC-IV v2.2, MIMIC-IV demo v2.2, and MIMIC-IV v3.1 (newer). ([PhysioNet][4])
* MIMIC-III Data Descriptor (PDF) and MIMIC-IV Data Descriptor (Scientific Data PDF). ([DSpace][17])
* PyTorch docs: `nn.Embedding`, `nn.Conv1d`, `nn.LSTM`. ([PyTorch Docs][7])
* scikit-learn docs: calibration curves / `CalibrationDisplay`, `roc_auc_score`, `average_precision_score`, `brier_score_loss`. ([Scikit-learn][14])
* Guo et al. (2017) “On Calibration of Modern Neural Networks” (arXiv/ICML). ([arXiv][11])
* Estimation of expected calibration errors (ECE) analysis (arXiv). ([arXiv][16])
* Repository file listing demonstrating `mimic3.py` and calibration PNGs exist together. ([GitHub][2])

[1]: https://physionet.org/content/mimiciii-demo/1.4/?utm_source=chatgpt.com "MIMIC-III Clinical Database Demo v1.4 - PhysioNet"
[2]: https://github.com/netanelcyber/penuX "GitHub - netanelcyber/penuX"
[3]: https://physionet.org/content/mimiciii/1.4/?utm_source=chatgpt.com "MIMIC-III Clinical Database v1.4 - PhysioNet"
[4]: https://physionet.org/content/mimiciv/2.2/?utm_source=chatgpt.com "MIMIC-IV v2.2 - PhysioNet"
[5]: https://physionet.org/content/mimic-iv-demo/2.2/?utm_source=chatgpt.com "MIMIC-IV Clinical Database Demo v2.2 - PhysioNet"
[6]: https://physionet.org/content/mimiciv/3.1/?utm_source=chatgpt.com "MIMIC-IV v3.1 - physionet.org"
[7]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html?utm_source=chatgpt.com "Embedding — PyTorch 2.9 documentation"
[8]: https://registry.opendata.aws/mimiciii/?utm_source=chatgpt.com "MIMIC-III (‘Medical Information Mart for Intensive Care’)"
[9]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?utm_source=chatgpt.com "Conv1d — PyTorch 2.9 documentation"
[10]: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html?utm_source=chatgpt.com "LSTM — PyTorch 2.9 documentation"
[11]: https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com "On Calibration of Modern Neural Networks"
[12]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?utm_source=chatgpt.com "roc_auc_score — scikit-learn 1.8.0 documentation"
[13]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html?utm_source=chatgpt.com "average_precision_score — scikit-learn 1.8.0 documentation"
[14]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration — scikit-learn 1.8.0 documentation"
[15]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html?utm_source=chatgpt.com "brier_score_loss — scikit-learn 1.8.0 documentation"
[16]: https://arxiv.org/pdf/2109.03480?utm_source=chatgpt.com "Estimating Expected Calibration Errors - arXiv.org"
[17]: https://dspace.mit.edu/bitstream/handle/1721.1/109192/MIMIC-III.pdf?sequence=1&utm_source=chatgpt.com "Data Descriptor: MIMIC-III, a freely accessible critical care database"
