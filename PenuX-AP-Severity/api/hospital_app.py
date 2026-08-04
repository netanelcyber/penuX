"""Hospital-oriented PenuX-AP research API.

Run locally with:
    uvicorn api.hospital_app:app --reload

A trained model bundle must be supplied through PENUX_AP_HOSPITAL_MODEL_PATH.
The service deliberately has no heuristic fallback: when no trained bundle is
available it returns 503 rather than presenting an unvalidated formula as a
trained hospital model.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.hospital_model import (
    HospitalModelBundle,
    RESEARCH_WARNING,
    prepare_hospital_frame,
)
from api.schemas import AdmissionInput

log = logging.getLogger(__name__)

_DESCRIPTION = """
## PenuX-AP-Severity Hospital Research API

> **Research use only. Not for patient-care decisions.**

This API implements the hospital data policy for acute-pancreatitis severity
model development and research inference:

- prediction window: first 24 hours from T0;
- no imaging field is required;
- missing values remain missing and are not converted to zero;
- a case is not rejected merely because five optional variables are absent;
- primary model development requires at least 8 of 16 core predictor groups;
- outcome fields are excluded from the predictor feature frame;
- preprocessing and imputation are fitted inside cross-validation folds by the
  offline training pipeline;
- no heuristic fallback is used when a trained model is unavailable.
"""

app = FastAPI(
    title="PenuX-AP-Severity Hospital Research API",
    description=_DESCRIPTION,
    version="1.1.0",
    contact={"name": "PenuX Research", "url": "https://penux.uk", "email": "nsh531@gmail.com"},
    docs_url="/docs",
    redoc_url="/redoc",
)

_bundle: HospitalModelBundle | None = None


class ValidationOutput(BaseModel):
    eligible_for_primary_model_development: bool
    missing_essential: list[str]
    sufficient_predictor_coverage: bool
    core_present: int
    core_total: int
    completeness_fraction: float
    completeness_level: str
    present_groups: list[str]
    missing_groups: list[str]
    vital_groups_present: int
    laboratory_groups_present: int
    imaging_required: bool = False
    rule_note: str
    warning: str = RESEARCH_WARNING


class HospitalPredictionOutput(BaseModel):
    prediction_available: bool
    severe_ap_probability: Optional[float] = None
    threshold_used: Optional[float] = None
    risk_group: Optional[str] = None
    model_version: Optional[str] = None
    core_present: int
    core_total: int
    completeness_level: str
    missing_core_groups: list[str] = Field(default_factory=list)
    fields_used: list[str] = Field(default_factory=list)
    missing_model_fields: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    warning: str = RESEARCH_WARNING


def _load_bundle() -> HospitalModelBundle | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    configured = os.environ.get("PENUX_AP_HOSPITAL_MODEL_PATH")
    if not configured:
        return None
    path = Path(configured)
    if not path.exists():
        log.error("Configured hospital model path does not exist")
        return None
    loaded: Any = joblib.load(path)
    if not isinstance(loaded, HospitalModelBundle):
        raise RuntimeError("PENUX_AP_HOSPITAL_MODEL_PATH is not a HospitalModelBundle")
    _bundle = loaded
    return _bundle


def _prediction_gate(data: AdmissionInput) -> tuple[bool, list[str], dict[str, Any]]:
    summary = data.core_data_summary()
    reasons: list[str] = []
    if data.age is None or data.age < 18:
        reasons.append("adult_age_required")
    if data.acute_pancreatitis_diagnosis is not True:
        reasons.append("documented_acute_pancreatitis_diagnosis_required")
    if not data.enzyme_criterion_met():
        reasons.append("lipase_or_amylase_at_least_3x_uln_required_without_imaging")
    if summary["core_present"] < 8:
        reasons.append("fewer_than_8_of_16_core_predictor_groups")

    full = data.hospital_data_dict()
    if not any(full.get(name) is not None for name in ("heart_rate", "systolic_bp", "respiratory_rate", "temperature", "spo2")):
        reasons.append("no_vital_signs_available")
    if not any(full.get(name) is not None for name in ("wbc", "hematocrit", "hemoglobin", "urea_mmol_l", "bun", "creatinine_umol_l", "creatinine", "glucose_mmol_l", "glucose")):
        reasons.append("no_laboratory_predictors_available")
    return not reasons, reasons, summary


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    bundle = _load_bundle()
    return {
        "status": "ok",
        "hospital_model_loaded": bundle is not None,
        "model_version": bundle.model_version if bundle else None,
        "warning": RESEARCH_WARNING,
    }


@app.post("/validate", response_model=ValidationOutput, tags=["data-quality"])
def validate_research_case(data: AdmissionInput) -> ValidationOutput:
    """Validate a retrospective model-development case without scoring it."""
    return ValidationOutput(**data.research_eligibility_summary())


@app.post("/predict", response_model=HospitalPredictionOutput, tags=["prediction"])
def predict_hospital_case(data: AdmissionInput) -> HospitalPredictionOutput:
    """Score a sufficiently complete first-24-hour AP record.

    This endpoint does not require outcome follow-up because the outcome is not
    known at prediction time. It does require the operational AP case
    definition used when no imaging data are available.
    """
    allowed, reasons, summary = _prediction_gate(data)
    if not allowed:
        return HospitalPredictionOutput(
            prediction_available=False,
            core_present=summary["core_present"],
            core_total=summary["core_total"],
            completeness_level=summary["completeness_level"],
            missing_core_groups=summary["missing_groups"],
            data_quality_warnings=reasons,
        )

    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained hospital model is configured. Train with "
                "scripts/train_hospital_model.py and set "
                "PENUX_AP_HOSPITAL_MODEL_PATH."
            ),
        )

    prepared = prepare_hospital_frame(data.hospital_data_dict())
    probability = float(bundle.predict_proba(prepared)[0, 1])
    risk_group = "high" if probability >= bundle.threshold else "lower"
    row = prepared.iloc[0].to_dict()
    fields_used = [name for name in bundle.feature_names if row.get(name) is not None and not _is_nan(row.get(name))]
    missing_model_fields = [name for name in bundle.feature_names if name not in fields_used]

    warnings: list[str] = []
    if summary["completeness_level"] == "intermediate":
        warnings.append("intermediate_input_completeness")
    if missing_model_fields:
        warnings.append("model_pipeline_imputed_missing_predictors")

    return HospitalPredictionOutput(
        prediction_available=True,
        severe_ap_probability=round(probability, 6),
        threshold_used=bundle.threshold,
        risk_group=risk_group,
        model_version=bundle.model_version,
        core_present=summary["core_present"],
        core_total=summary["core_total"],
        completeness_level=summary["completeness_level"],
        missing_core_groups=summary["missing_groups"],
        fields_used=fields_used,
        missing_model_fields=missing_model_fields,
        data_quality_warnings=warnings,
    )


def _is_nan(value: Any) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False
