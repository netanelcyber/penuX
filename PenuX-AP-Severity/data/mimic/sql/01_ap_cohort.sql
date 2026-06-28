-- Identify Acute Pancreatitis admissions in MIMIC-IV
-- ICD-9: 577.0 | ICD-10: K85.*
--
-- IMPORTANT: This SQL is a starting point only.
-- Cohort definitions must be validated clinically before use.
-- Exact table names depend on MIMIC-IV version and schema.
-- Requires PhysioNet credentialing and a signed MIMIC-IV DUA.

SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.admission_type,
    a.admission_location,
    d.icd_code,
    d.icd_version
FROM
    mimiciv_hosp.admissions a
    INNER JOIN mimiciv_hosp.diagnoses_icd d
        ON a.hadm_id = d.hadm_id
WHERE
    (d.icd_version = 9 AND d.icd_code = '5770')
    OR
    (d.icd_version = 10 AND d.icd_code LIKE 'K85%')
ORDER BY
    a.subject_id, a.hadm_id;
