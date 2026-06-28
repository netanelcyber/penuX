"""HL7 v2.x message parser for EHR system integration.

Parses HL7 v2.x segments (OBR, OBX, PID, PV1) commonly exported by Epic,
Cerner, OpenEMR, and other EHR systems. Maps lab results and vitals to
AdmissionInput for the prediction pipeline.

HL7 v2 segment reference:
  - PID: Patient identification (age, sex)
  - PV1: Patient visit / admission
  - OBR: Observation request (panel headers)
  - OBX: Observation/result (individual lab/vital values)
  - NTE: Notes/comments

IMPORTANT PRIVACY NOTE:
  Patient identifiers (MRN, name, account#, SSN) are extracted for encounter
  routing only and are never stored, logged, or forwarded downstream. Only
  numerical lab/vital values are passed to the prediction pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from api.schemas import AdmissionInput

log = logging.getLogger(__name__)

# HL7 v2 field separator (usually | in standard HL7)
FIELD_SEP = "|"
COMPONENT_SEP = "^"

# Observation codes (LOINC or LIS) mapped to AdmissionInput fields
LOINC_CODE_MAP: dict[str, str] = {
    "8867-4": "heart_rate",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "9279-1": "respiratory_rate",
    "8310-5": "temperature",
    "59408-5": "spo2",
    "39156-5": "bmi",
    "6690-2": "wbc",
    "20570-8": "hematocrit",
    "1988-5": "crp",
    "3094-0": "bun",
    "2160-0": "creatinine",
    "17861-6": "calcium",
    "2345-7": "glucose",
    "14804-9": "ldh",
    "1920-8": "ast",
    "1742-6": "alt",
    "1751-7": "albumin",
    "2571-8": "triglycerides",
}

# LIS (Laboratory Information System) codes — vendor-specific
LIS_CODE_FALLBACK: dict[str, str] = {
    "WBC": "6690-2",
    "HCT": "20570-8",
    "CRP": "1988-5",
    "BUN": "3094-0",
    "CREAT": "2160-0",
    "CA": "17861-6",
    "GLU": "2345-7",
    "LDH": "14804-9",
    "AST": "1920-8",
    "ALT": "1742-6",
    "ALB": "1751-7",
    "TRIG": "2571-8",
}


def _parse_hl7_segment(segment: str, field_sep: str = FIELD_SEP) -> list[str]:
    """Parse a single HL7 segment into fields."""
    return segment.rstrip("\r\n").split(field_sep)


def _extract_component(field: str, index: int, sep: str = COMPONENT_SEP) -> Optional[str]:
    """Extract a component from a ^ -separated field."""
    parts = field.split(sep)
    if index < len(parts):
        return parts[index] or None
    return None


def _loinc_from_obx(fields: list[str]) -> Optional[str]:
    """Extract LOINC code from OBX observation identifier field (field 3)."""
    if len(fields) < 4:
        return None
    obs_id_field = fields[3]
    loinc = _extract_component(obs_id_field, 0)
    return loinc


def _obs_value_from_obx(fields: list[str]) -> Optional[float]:
    """Extract numeric observation value from OBX (field 5)."""
    if len(fields) < 6:
        return None
    val_str = fields[5].strip()
    if not val_str:
        return None
    try:
        return float(val_str)
    except ValueError:
        return None


def _age_from_pid(fields: list[str]) -> Optional[float]:
    """Extract age from PID segment (field 7: YYYYMMDD)."""
    if len(fields) < 8:
        return None
    dob_str = fields[7].strip()
    if not dob_str or len(dob_str) < 8:
        return None
    try:
        from datetime import date, datetime
        year = int(dob_str[0:4])
        month = int(dob_str[4:6])
        day = int(dob_str[6:8])
        dob = date(year, month, day)
        today = datetime.utcnow().date()
        return float((today - dob).days / 365.25)
    except (ValueError, IndexError):
        return None


def _sex_from_pid(fields: list[str]) -> Optional[str]:
    """Extract sex from PID (field 8: M/F/O/U)."""
    if len(fields) < 9:
        return None
    sex = fields[8].strip().upper()
    if sex in ("M", "F", "O", "U"):
        return sex
    return None


def hl7_message_to_admission_input(message: str) -> AdmissionInput:
    """Parse an HL7 v2.x message and extract clinical values.

    Message format (line-separated segments):
        MSH|^~\\&|Epic|Hospital|...
        PID|1||MRN||Doe^John||20640101|M
        OBX|1|NM|2160-0^Creatinine|||1.5|mg/dL|||F
        OBX|2|NM|6690-2^WBC|||18.2|10*9/L|||F

    Patient identifiers are discarded before this function returns.
    """
    fields: dict[str, Any] = {}
    lines = message.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        seg_fields = _parse_hl7_segment(line)
        if not seg_fields:
            continue

        seg_type = seg_fields[0]

        if seg_type == "PID":
            age = _age_from_pid(seg_fields)
            if age is not None:
                fields["age"] = age
            sex = _sex_from_pid(seg_fields)
            if sex is not None:
                fields["sex"] = sex

        elif seg_type == "OBX":
            loinc = _loinc_from_obx(seg_fields)
            if not loinc:
                continue

            field_name = LOINC_CODE_MAP.get(loinc)
            if not field_name:
                field_name = LOINC_CODE_MAP.get(
                    LIS_CODE_FALLBACK.get(loinc, loinc)
                )
            if not field_name:
                log.debug("Unknown observation code: %s", loinc)
                continue

            value = _obs_value_from_obx(seg_fields)
            if value is not None:
                fields[field_name] = value

    log.debug("HL7 message mapped to fields: %s", list(fields.keys()))
    return AdmissionInput(**fields)


def hl7_batch_results_to_admission_input(
    pid_segment: str,
    obx_segments: list[str],
) -> AdmissionInput:
    """Parse HL7 segments separately (PID once, multiple OBX)."""
    fields: dict[str, Any] = {}

    pid_fields = _parse_hl7_segment(pid_segment)
    if pid_fields and pid_fields[0] == "PID":
        age = _age_from_pid(pid_fields)
        if age is not None:
            fields["age"] = age
        sex = _sex_from_pid(pid_fields)
        if sex is not None:
            fields["sex"] = sex

    for obx_line in obx_segments:
        seg_fields = _parse_hl7_segment(obx_line)
        if not seg_fields or seg_fields[0] != "OBX":
            continue

        loinc = _loinc_from_obx(seg_fields)
        if not loinc:
            continue

        field_name = LOINC_CODE_MAP.get(loinc) or LOINC_CODE_MAP.get(
            LIS_CODE_FALLBACK.get(loinc, loinc)
        )
        if not field_name:
            continue

        value = _obs_value_from_obx(seg_fields)
        if value is not None:
            fields[field_name] = value

    log.debug("HL7 batch mapped to fields: %s", list(fields.keys()))
    return AdmissionInput(**fields)
