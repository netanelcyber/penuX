-- Extract first-24h vital signs for AP cohort
-- Chartevents itemids are MIMIC-IV specific.
-- Verify itemids against d_items. Clinical validation required.

WITH ap_cohort AS (
    SELECT a.hadm_id, a.admittime
    FROM mimiciv_hosp.admissions a
    INNER JOIN mimiciv_hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE (d.icd_version = 9 AND d.icd_code = '5770')
       OR (d.icd_version = 10 AND d.icd_code LIKE 'K85%')
),
ranked_vitals AS (
    SELECT
        ce.hadm_id,
        ce.itemid,
        di.label,
        ce.valuenum,
        ce.valueuom,
        ce.charttime,
        ROW_NUMBER() OVER (
            PARTITION BY ce.hadm_id, ce.itemid
            ORDER BY ce.charttime ASC
        ) AS rn
    FROM mimiciv_icu.chartevents ce
    INNER JOIN ap_cohort ap ON ce.hadm_id = ap.hadm_id
    INNER JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
    WHERE ce.charttime BETWEEN ap.admittime AND ap.admittime + INTERVAL '24 hours'
      AND ce.valuenum IS NOT NULL
      AND ce.itemid IN (
          220045,  -- Heart rate
          220179,  -- Systolic BP non-invasive
          220180,  -- Diastolic BP non-invasive
          224690,  -- Respiratory rate
          223762,  -- Temperature Celsius
          220277   -- SpO2
      )
)
SELECT hadm_id, itemid, label, valuenum, valueuom, charttime
FROM ranked_vitals
WHERE rn = 1
ORDER BY hadm_id, label;
