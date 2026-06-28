"""Clinical scoring systems for AP severity benchmarking.

IMPORTANT: These implementations are research placeholders.
All score implementations must be reviewed by a clinical expert
before use in any study or clinical workflow.
Scores return NaN when required fields are missing.
"""
import logging
import math

import pandas as pd

log = logging.getLogger(__name__)

_SCORE_FIELDS = {
    "sirs": ["temperature", "heart_rate", "respiratory_rate", "wbc"],
    "bisap": ["bun", "age", "sirs_score", "pleural_effusion", "impaired_mental_status"],
    "ranson": [
        "age", "wbc", "glucose", "ldh", "ast",
        "hematocrit_drop", "bun_rise", "calcium", "pao2", "base_deficit", "fluid_sequestration",
    ],
    "apache_ii": [
        "temperature", "mean_arterial_pressure", "heart_rate", "respiratory_rate",
        "pao2", "arterial_ph", "sodium", "potassium", "creatinine",
        "hematocrit", "wbc", "glasgow_coma_scale", "age", "chronic_health",
    ],
    "modified_ctsi": ["pancreatic_necrosis_pct", "ctsi_grade"],
}


def required_fields_for_score(score_name: str) -> list[str]:
    """Return required fields for a clinical score."""
    score_name = score_name.lower()
    if score_name not in _SCORE_FIELDS:
        raise ValueError(f"Unknown score '{score_name}'. Available: {list(_SCORE_FIELDS)}")
    return _SCORE_FIELDS[score_name]


def _check_fields(row: pd.Series, fields: list[str]) -> list[str]:
    return [f for f in fields if f not in row.index or pd.isna(row.get(f))]


def compute_sirs(row: pd.Series) -> int | float:
    """SIRS score (0-4). Requires: temperature, heart_rate, respiratory_rate, wbc."""
    missing = _check_fields(row, ["temperature", "heart_rate", "respiratory_rate", "wbc"])
    if missing:
        log.debug("SIRS: missing fields %s", missing)
        return float("nan")
    score = 0
    temp = row["temperature"]
    score += 1 if temp < 36.0 or temp > 38.0 else 0
    score += 1 if row["heart_rate"] > 90 else 0
    score += 1 if row["respiratory_rate"] > 20 else 0
    wbc = row["wbc"]
    score += 1 if wbc < 4.0 or wbc > 12.0 else 0
    return score


def compute_bisap(row: pd.Series) -> int | float:
    """BISAP score (0-5). Clinical review required before use.

    Requires: bun (mg/dL), age (years), sirs_score, pleural_effusion (bool), impaired_mental_status (bool).
    """
    missing = _check_fields(row, ["bun", "age", "sirs_score", "pleural_effusion", "impaired_mental_status"])
    if missing:
        log.debug("BISAP: missing fields %s", missing)
        return float("nan")
    score = 0
    score += 1 if row["bun"] > 25 else 0
    score += 1 if row["age"] > 60 else 0
    score += 1 if row["sirs_score"] >= 2 else 0
    score += 1 if bool(row["pleural_effusion"]) else 0
    score += 1 if bool(row["impaired_mental_status"]) else 0
    return score


def compute_ranson(row: pd.Series) -> int | float:
    """Ranson criteria (0-11). Clinical review required before use.

    Admission criteria: age, wbc, glucose, ldh, ast.
    48h criteria: hematocrit_drop, bun_rise, calcium, pao2, base_deficit, fluid_sequestration.
    """
    admission = ["age", "wbc", "glucose", "ldh", "ast"]
    h48 = ["hematocrit_drop", "bun_rise", "calcium", "pao2", "base_deficit", "fluid_sequestration"]
    missing = _check_fields(row, admission + h48)
    if missing:
        log.debug("Ranson: missing fields %s", missing)
        return float("nan")
    score = 0
    score += 1 if row["age"] > 55 else 0
    score += 1 if row["wbc"] > 16 else 0
    score += 1 if row["glucose"] > 200 else 0
    score += 1 if row["ldh"] > 350 else 0
    score += 1 if row["ast"] > 250 else 0
    score += 1 if row["hematocrit_drop"] > 10 else 0
    score += 1 if row["bun_rise"] > 5 else 0
    score += 1 if row["calcium"] < 8 else 0
    score += 1 if row["pao2"] < 60 else 0
    score += 1 if row["base_deficit"] > 4 else 0
    score += 1 if row["fluid_sequestration"] > 6 else 0
    return score


def compute_apache_ii(row: pd.Series) -> int | float:
    """APACHE II score. Clinical review required before use.

    Full implementation requires 15+ physiological variables.
    Returns NaN if any required field is missing.
    """
    required = [
        "temperature", "mean_arterial_pressure", "heart_rate", "respiratory_rate",
        "hematocrit", "wbc", "creatinine", "sodium", "potassium",
        "glasgow_coma_scale", "age", "arterial_ph",
    ]
    missing = _check_fields(row, required)
    if missing:
        log.debug("APACHE II: missing fields %s", missing)
        return float("nan")
    # Simplified version — for illustration only
    # Full APACHE II requires complex lookup tables; use validated implementation in production
    log.warning("APACHE II score is a simplified placeholder. Clinical review required.")
    age_points = (
        0 if row["age"] < 45
        else 2 if row["age"] < 55
        else 3 if row["age"] < 65
        else 5 if row["age"] < 75
        else 6
    )
    gcs_points = 15 - int(row["glasgow_coma_scale"])
    return age_points + gcs_points  # partial — not a complete APACHE II


def compute_modified_ctsi(row: pd.Series) -> int | float:
    """Modified CTSI (0-10). Requires: ctsi_grade (A-E or 0-4), pancreatic_necrosis_pct."""
    missing = _check_fields(row, ["ctsi_grade", "pancreatic_necrosis_pct"])
    if missing:
        log.debug("Modified CTSI: missing fields %s", missing)
        return float("nan")
    grade_map = {"A": 0, "B": 2, "C": 4, "D": 6, "E": 8}
    grade = row["ctsi_grade"]
    if isinstance(grade, str):
        grade_score = grade_map.get(grade.upper(), int(grade) if grade.isdigit() else float("nan"))
    else:
        grade_score = int(grade) * 2
    if math.isnan(grade_score):
        return float("nan")
    necrosis = row["pancreatic_necrosis_pct"]
    necrosis_score = 0 if necrosis == 0 else (2 if necrosis < 30 else (4 if necrosis < 50 else 6))
    return min(grade_score + necrosis_score, 10)
