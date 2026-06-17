-- Extract first-24h laboratory values for AP cohort
-- Requires: ap_cohort CTE or table from 01_ap_cohort.sql
--
-- Lab item IDs are MIMIC-IV specific and may change between versions.
-- Verify itemids against the d_labitems table.
-- Clinical validation required before use.

WITH ap_cohort AS (
    SELECT a.hadm_id, a.admittime
    FROM mimiciv_hosp.admissions a
    INNER JOIN mimiciv_hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE (d.icd_version = 9 AND d.icd_code = '5770')
       OR (d.icd_version = 10 AND d.icd_code LIKE 'K85%')
),
ranked_labs AS (
    SELECT
        le.hadm_id,
        le.itemid,
        dl.label,
        le.valuenum,
        le.valueuom,
        le.charttime,
        ROW_NUMBER() OVER (
            PARTITION BY le.hadm_id, le.itemid
            ORDER BY le.charttime ASC
        ) AS rn
    FROM mimiciv_hosp.labevents le
    INNER JOIN ap_cohort ap ON le.hadm_id = ap.hadm_id
    INNER JOIN mimiciv_hosp.d_labitems dl ON le.itemid = dl.itemid
    WHERE le.charttime BETWEEN ap.admittime AND ap.admittime + INTERVAL '24 hours'
      AND le.valuenum IS NOT NULL
      AND le.itemid IN (
          51300, 51301,   -- WBC
          50889,          -- CRP (if available)
          51006,          -- BUN / urea nitrogen
          50912,          -- Creatinine
          50893,          -- Calcium total
          50809, 50931,   -- Glucose
          51221,          -- Hematocrit
          50954,          -- LDH
          50878,          -- AST
          50861,          -- ALT
          50862,          -- Albumin
          50883           -- Triglycerides (may not be available in MIMIC)
      )
)
SELECT hadm_id, itemid, label, valuenum, valueuom, charttime
FROM ranked_labs
WHERE rn = 1
ORDER BY hadm_id, label;
