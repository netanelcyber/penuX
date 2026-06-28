"""AP-relevant feature definitions and column alias normalization."""
import logging

import pandas as pd

log = logging.getLogger(__name__)

AP_CANDIDATE_FEATURES: list[str] = [
    "age", "sex", "body_mass_index", "bmi",
    "heart_rate", "systolic_bp", "diastolic_bp",
    "respiratory_rate", "temperature", "spo2",
    "wbc", "crp", "bun", "urea", "creatinine",
    "calcium", "glucose", "hematocrit", "ldh",
    "ast", "alt", "albumin", "triglycerides", "sirs_count",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "heart_rate":    ["hr", "pulse", "heart rate"],
    "systolic_bp":   ["sbp", "sys_bp", "systolic", "systolic blood pressure"],
    "diastolic_bp":  ["dbp", "dia_bp", "diastolic", "diastolic blood pressure"],
    "bun":           ["urea", "blood_urea_nitrogen", "bun_mg_dl"],
    "wbc":           ["white_blood_cell_count", "leukocytes", "wbc_count"],
    "hematocrit":    ["hct", "hematocrit_pct"],
    "temperature":   ["temp", "body_temperature", "temp_c"],
    "creatinine":    ["creat", "serum_creatinine"],
    "spo2":          ["oxygen_saturation", "o2sat", "spO2"],
    "bmi":           ["body_mass_index", "bmi_kg_m2"],
    "crp":           ["c_reactive_protein", "crp_mg_l"],
    "ldh":           ["lactate_dehydrogenase", "ldh_u_l"],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to canonical names where unambiguous."""
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                log.info("Renaming column '%s' -> '%s'", alias, canonical)
                break
    return df.rename(columns=rename_map)


def available_ap_features(df: pd.DataFrame) -> list[str]:
    """Return AP candidate features that are present in the DataFrame."""
    present = [f for f in AP_CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in AP_CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        log.debug("AP features not found in dataset: %s", missing)
    return present
