"""Camelion HIS ↔ PenuX-AP-Severity adapter.

Camelion (קמיליון) is an Israeli hospital information system. This module
provides two integration paths:

1. **FHIR R4** — `bundle_to_admission_input(bundle)`:
   Accepts a FHIR R4 Bundle (Patient + Observation resources) from the
   Camelion FHIR REST gateway and maps it to AdmissionInput.

2. **Camelion native JSON** — `camelion_json_to_admission_input(payload)`:
   Accepts the Camelion HL7-derived flat JSON structure used by Israeli
   hospitals for point-of-care integrations.

IMPORTANT PRIVACY NOTE:
   Patient identifiers (Teudat Zehut, MRN) received in requests are used only
   to correlate observations within a single request. They are NEVER logged,
   stored, or forwarded downstream. The prediction pipeline receives only
   anonymised numerical/categorical values.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from api.fhir_schemas import (
    BundleEntry,
    FHIRBundle,
    ObservationResource,
    PatientResource,
)
from api.schemas import AdmissionInput

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LOINC → AdmissionInput field mapping
# ---------------------------------------------------------------------------
# Maps LOINC observation codes to the flat AdmissionInput field names.
# Only codes that appear in the Camelion order-set for acute pancreatitis
# admission are listed here.

LOINC_TO_FIELD: dict[str, str] = {
    # Vitals
    "8867-4": "heart_rate",         # Heart rate
    "8480-6": "systolic_bp",        # Systolic blood pressure
    "8462-4": "diastolic_bp",       # Diastolic blood pressure
    "9279-1": "respiratory_rate",   # Respiratory rate
    "8310-5": "temperature",        # Body temperature
    "59408-5": "spo2",              # Oxygen saturation (SpO2)
    # Labs — haematology
    "6690-2": "wbc",                # WBC
    "20570-8": "hematocrit",        # Hematocrit
    # Labs — chemistry
    "1988-5": "crp",                # CRP
    "3094-0": "bun",                # BUN (urea nitrogen)
    "2160-0": "creatinine",         # Creatinine
    "17861-6": "calcium",           # Calcium
    "2345-7": "glucose",            # Glucose
    "14804-9": "ldh",               # LDH
    "1920-8": "ast",                # AST
    "1742-6": "alt",                # ALT
    "1751-7": "albumin",            # Albumin
    "2571-8": "triglycerides",      # Triglycerides
    # Anthropometrics
    "39156-5": "bmi",               # BMI
}

# Camelion native JSON key → AdmissionInput field
# Based on the Camelion HL7 v2.x → JSON conversion layer used in Israeli hospitals.
CAMELION_KEY_TO_FIELD: dict[str, str] = {
    # Demographics
    "גיל": "age",
    "מין": "sex",
    "bmi": "bmi",
    # Vitals (Hebrew labels from Camelion nursing flowsheet)
    "דופק": "heart_rate",
    "לחץ דם סיסטולי": "systolic_bp",
    "לחץ דם דיאסטולי": "diastolic_bp",
    "קצב נשימה": "respiratory_rate",
    "חום": "temperature",
    "סטורציה": "spo2",
    # Labs (Hebrew labels as exported from Camelion lab module)
    "כדוריות דם לבנות": "wbc",
    "wbc": "wbc",
    "crp": "crp",
    "CRP": "crp",
    "קראטינין": "creatinine",
    "creatinine": "creatinine",
    "אוריאה": "bun",
    "bun": "bun",
    "סידן": "calcium",
    "calcium": "calcium",
    "גלוקוז": "glucose",
    "glucose": "glucose",
    "המטוקריט": "hematocrit",
    "hematocrit": "hematocrit",
    "ldh": "ldh",
    "LDH": "ldh",
    "ast": "ast",
    "AST": "ast",
    "alt": "alt",
    "ALT": "alt",
    "אלבומין": "albumin",
    "albumin": "albumin",
    "טריגליצרידים": "triglycerides",
    "triglycerides": "triglycerides",
}

SEX_MAP = {
    "male": "M", "female": "F",
    "זכר": "M", "נקבה": "F",
    "m": "M", "f": "F",
    "1": "M", "2": "F",
}


def _age_from_birthdate(birth_date_str: str) -> float | None:
    """Convert FHIR birthDate (YYYY-MM-DD) to age in years."""
    try:
        bd = date.fromisoformat(birth_date_str)
        today = datetime.utcnow().date()
        return float((today - bd).days / 365.25)
    except Exception:
        return None


def _extract_observation_value(obs: ObservationResource) -> float | None:
    if obs.valueQuantity and obs.valueQuantity.value is not None:
        return float(obs.valueQuantity.value)
    # Blood-pressure panels store systolic/diastolic as components
    return None


def _loinc_code(obs: ObservationResource) -> str | None:
    if obs.code and obs.code.coding:
        for coding in obs.code.coding:
            if coding.system in (
                "http://loinc.org",
                "https://loinc.org",
                "urn:oid:2.16.840.1.113883.6.1",
            ):
                return coding.code
    return None


def bundle_to_admission_input(bundle: FHIRBundle) -> AdmissionInput:
    """Map a FHIR R4 Bundle (Patient + Observations) → AdmissionInput.

    Patient identifiers (TZ / MRN) are read only for log correlation and
    are discarded before this function returns.
    """
    fields: dict[str, Any] = {}

    for entry in bundle.entry:
        res = entry.resource
        rtype = res.get("resourceType")

        if rtype == "Patient":
            patient = PatientResource(**res)
            if patient.birthDate:
                age = _age_from_birthdate(patient.birthDate)
                if age is not None:
                    fields["age"] = age
            if patient.gender:
                fields["sex"] = SEX_MAP.get(patient.gender.lower(), patient.gender)

        elif rtype == "Observation":
            obs = ObservationResource(**res)
            loinc = _loinc_code(obs)

            # Blood pressure panel: components carry systolic + diastolic
            if obs.component:
                for comp in obs.component:
                    if comp.code and comp.code.coding:
                        for coding in comp.code.coding:
                            comp_loinc = coding.code
                            if comp_loinc in LOINC_TO_FIELD and comp.valueQuantity:
                                fields[LOINC_TO_FIELD[comp_loinc]] = float(comp.valueQuantity.value)

            if loinc and loinc in LOINC_TO_FIELD:
                val = _extract_observation_value(obs)
                if val is not None:
                    fields[LOINC_TO_FIELD[loinc]] = val

    log.debug("FHIR bundle mapped to fields: %s", list(fields.keys()))
    return AdmissionInput(**fields)


def camelion_json_to_admission_input(payload: dict[str, Any]) -> AdmissionInput:
    """Map a Camelion native JSON payload → AdmissionInput.

    Camelion exports a flat key-value structure from its HL7 v2.x interface.
    Both Hebrew and English keys are supported (see CAMELION_KEY_TO_FIELD).
    """
    fields: dict[str, Any] = {}

    # Top-level flat mapping
    for key, value in payload.items():
        field = CAMELION_KEY_TO_FIELD.get(key)
        if field:
            if field == "sex":
                fields[field] = SEX_MAP.get(str(value).strip().lower(), str(value))
            else:
                try:
                    fields[field] = float(value)
                except (TypeError, ValueError):
                    log.debug("Non-numeric value for Camelion key %r: %r", key, value)

    # Nested "labs" / "vitals" sub-dicts (Camelion REST export format)
    for section in ("labs", "vitals", "בדיקות_מעבדה", "סימנים_חיוניים"):
        sub = payload.get(section, {})
        if isinstance(sub, dict):
            for key, value in sub.items():
                field = CAMELION_KEY_TO_FIELD.get(key)
                if field and field not in fields:
                    try:
                        fields[field] = float(value)
                    except (TypeError, ValueError):
                        pass

    log.debug("Camelion JSON mapped to fields: %s", list(fields.keys()))
    return AdmissionInput(**fields)
