# MIMIC-III Resistance / Pathogen Prediction (NO-PANDAS)

Train a multi-class model to predict a **fixed pathogen class** (CAPITAL labels) from:

* Microbiology categorical fields (**lowercased values**):

  * `spec_type_desc`
  * `interpretation`
* Antibiotic exposure (**binary one-hot**, **lowercase feature names**)
* Early vitals/labs (**REQUIRED order**, **lowercase feature names**):

  * `temperature_c`, `wbc`, `spo2`, `age`

The pipeline is designed to reduce the common collapse/bias:

`B:OTHER -> 1.0000`

by using:

* stratified split
* train-only OneHotEncoder fitting (no leakage)
* train-only vitals standardization
* class-balanced `sample_weight` (+ optional extra down-weight for `B:OTHER`)
* label smoothing to reduce overconfidence

---

## 1) Project Files

Expected main file:

* `mimic3_resistance_pipeline.py` (NO-PANDAS)

Expected MIMIC-III CSV files in:

`dataset/mimic/mimic-iii-clinical-database-demo-1.4/`

Required CSVs:

* `MICROBIOLOGYEVENTS.csv`
* `PRESCRIPTIONS.csv`
* `ADMISSIONS.csv`
* `PATIENTS.csv`
* `D_ITEMS.csv`
* `CHARTEVENTS.csv`
* `D_LABITEMS.csv`
* `LABEVENTS.csv`

---

## 2) Classes (CAPITALS)

Fixed class order (targets + printing):

* `B:PSEUDOMONAS AERUGINOSA`
* `B:STAPH AUREUS COAG +`
* `B:SERRATIA MARCESCENS`
* `B:MORGANELLA MORGANII`
* `B:ESCHERICHIA COLI`
* `B:PROTEUS MIRABILIS`
* `B:PROVIDENCIA STUARTII`
* `B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS`
* `B:YEAST`
* `B:GRAM POSITIVE COCCUS(COCCI)`
* `B:OTHER`
* `V:OTHER`

> NOTE: Most organisms not matching explicit rules will map to `B:OTHER`.
> If `B:OTHER` dominates your dataset, expand `map_org()`.

---

## 3) Features

### Categorical (values lowercased)

* `spec_type_desc`
* `interpretation`

These values are normalized to lowercase + collapsed whitespace to reduce duplicates and cardinality.

### Numeric (feature names lowercase)

Vitals/labs (**REQUIRED order**):

1. `temperature_c` (mean within window)
2. `wbc` (max within window)
3. `spo2` (min within window)
4. `age`

Antibiotics one-hot (**lowercase feature names**):

* `vancomycin`
* `ciprofloxacin`
* `meropenem`
* `piperacillin`
* `ceftriaxone`

Numeric order is enforced as:

`["temperature_c","wbc","spo2","age", <abx...>]`

---

## 4) Time Window Logic

Vitals/labs are collected within the first **HOURS_WINDOW** hours from `ADMITTIME`.

Default:

* `HOURS_WINDOW = 24`

### Temperature (`temperature_c`)

* source: `CHARTEVENTS.csv`
* item IDs: derived from `D_ITEMS.csv` label heuristics
* supports Celsius and Fahrenheit conversion
* valid range filter: 30–45 °C
* aggregation: **mean** within window

### SpO2 (`spo2`)

* source: `CHARTEVENTS.csv`
* item IDs: derived from `D_ITEMS.csv` label heuristics
* valid range filter: 50–100
* aggregation: **minimum** within window

### WBC (`wbc`)

* source: `LABEVENTS.csv`
* item IDs: derived from `D_LABITEMS.csv` (`wbc` or `white blood` in label)
* valid range filter: 2000–40000
* aggregation: **maximum** within window
* auto scaling heuristic: if median sample suggests units are small, multiply by 1000

### Age (`age`)

* computed from `ADMITTIME - DOB`
* DOB years >= 3000 treated as missing (de-identified dates)
* clipped to 0–110 with safety cap

---

## 5) Install / Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install numpy tensorflow scikit-learn
```

No pandas is used.

---

## 6) Run

From repo root:

```bash
python mimic3.py
```

What it does:

1. Loads microbiology rows → assigns `label` (CAPITALS)
2. Builds antibiotic one-hot per `hadm_id`
3. Computes vitals/labs per `hadm_id` (streaming CSV)
4. Joins everything and trains a DNN
5. Prints:

   * label counts
   * general test accuracy
   * per-HADM predictions sorted by probability

---

## 7) Output

Console output includes:

* dataset size stats
* label distribution
* enforced numeric order
* test set accuracy
* per-HADM probability table ordered descending

Example per-HADM section:

```text
HADM_ID: 12345 (n_rows=3)
B:ESCHERICHIA COLI -> 0.4123
B:OTHER            -> 0.2331
...
```

---

## 8) Fixing “B:OTHER -> 1.0000” (If It Still Happens)

If predictions still collapse to `B:OTHER`:

1. Check printed **label counts**

   * If `B:OTHER` is most rows, the model is learning the dataset imbalance.
2. Expand `map_org()` rules

   * This is the best fix: reduce the “everything goes to OTHER” funnel.
3. Tune balancing knobs (carefully)

   * `MAX_CLASS_WEIGHT`
   * `BOTHER_EXTRA_DOWNWEIGHT`
4. Reduce overconfidence

   * increase `DROPOUT`
   * increase `LABEL_SMOOTHING` (e.g. `0.10`)
5. Train longer only if validation improves

   * increasing epochs helps only when val accuracy rises

---

## 9) Common Troubleshooting

### “No vitals complete rows”

* Increase `HOURS_WINDOW`
* Confirm MIMIC paths are correct
* Confirm `D_ITEMS` contains temperature/spo2 labels and `D_LABITEMS` contains WBC label patterns

### Very slow runtime

* Full `CHARTEVENTS` / `LABEVENTS` are huge
* This script streams row-by-row (I/O heavy)
* Use SSD if possible
* Consider pre-filtering your CSVs to relevant `hadm_id` and time windows

---

## 10) Customization

Edit these in `mimic3_resistance_pipeline.py`:

* `HOURS_WINDOW` : change window length
* `ANTIBIOTICS`  : change which drugs become binary features
* `map_org()`    : expand organism → class mapping (**best way to reduce B:OTHER dominance**)
* `EPOCHS`, `DROPOUT`, `LABEL_SMOOTHING` : model behavior / overconfidence
* `USE_CLASS_WEIGHTS` : toggle class balancing
