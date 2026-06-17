-- Extract candidate outcomes for AP severity labeling
--
-- IMPORTANT: MIMIC-IV does not contain direct Atlanta 2012 SAP labels.
-- Outcome definitions (e.g. persistent organ failure >48h) require
-- careful operationalization and clinical expert review.
-- This script provides proxy variables only.

WITH ap_cohort AS (
    SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.hospital_expire_flag
    FROM mimiciv_hosp.admissions a
    INNER JOIN mimiciv_hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE (d.icd_version = 9 AND d.icd_code = '5770')
       OR (d.icd_version = 10 AND d.icd_code LIKE 'K85%')
),
icu_stays AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays,
           MIN(intime) AS first_icu_intime
    FROM mimiciv_icu.icustays
    GROUP BY hadm_id
),
ventilation AS (
    SELECT hadm_id, 1 AS had_ventilation
    FROM mimiciv_derived.ventilation
    WHERE ventilation_status = 'InvasiveVent'
    GROUP BY hadm_id
),
vasopressors AS (
    SELECT hadm_id, 1 AS had_vasopressors
    FROM mimiciv_derived.norepinephrine_equivalent_dose
    GROUP BY hadm_id
)
SELECT
    c.subject_id,
    c.hadm_id,
    c.admittime,
    c.dischtime,
    EXTRACT(EPOCH FROM (c.dischtime - c.admittime)) / 3600 AS los_hours,
    c.hospital_expire_flag AS in_hospital_mortality,
    CASE WHEN icu.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS icu_admission,
    COALESCE(vent.had_ventilation, 0) AS mechanical_ventilation,
    COALESCE(vaso.had_vasopressors, 0) AS vasopressors
FROM ap_cohort c
LEFT JOIN icu_stays icu ON c.hadm_id = icu.hadm_id
LEFT JOIN ventilation vent ON c.hadm_id = vent.hadm_id
LEFT JOIN vasopressors vaso ON c.hadm_id = vaso.hadm_id
ORDER BY c.subject_id, c.hadm_id;
