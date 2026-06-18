"""FastAPI research-only prediction endpoint for SAP severity.

RESEARCH USE ONLY. Not validated for clinical use.
Do not use for patient-care decisions.

Integration endpoints:
  POST /predict          — plain JSON (AdmissionInput)
  POST /fhir/predict     — FHIR R4 Bundle (Patient + Observations, LOINC coded)
  POST /camelion/predict — Camelion (קמיליון) HIS native JSON
"""
import os
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    AdmissionInput,
    CamelionPredictionResponse,
    CamelionRequest,
    HealthResponse,
    PredictionOutput,
    RESEARCH_WARNING,
)
from api.fhir_schemas import FHIRBundle, RiskAssessmentResource, RiskAssessmentPrediction, CodeableConcept, Coding
from api.camelion_adapter import bundle_to_admission_input, camelion_json_to_admission_input
from penux_ap.config import RISK_THRESHOLDS

log = logging.getLogger(__name__)

app = FastAPI(
    title="PenuX-AP-Severity Research API",
    description=(
        "Research prototype for early prediction of Severe Acute Pancreatitis. "
        "NOT validated for clinical use. NOT for patient-care decisions.\n\n"
        "**Camelion (קמיליון) HIS integration**: POST /camelion/predict or POST /fhir/predict"
    ),
    version="0.1.0",
)

_model = None


def _load_model():
    global _model
    model_path = os.environ.get("PENUX_AP_MODEL_PATH")
    if not model_path:
        return None
    p = Path(model_path)
    if not p.exists():
        log.warning("Model path configured but file not found: %s", p)
        return None
    _model = joblib.load(p)
    log.info("Model loaded from %s", p)
    return _model


def _run_prediction(admission: AdmissionInput) -> tuple[float, str]:
    """Return (probability, risk_group). Raises HTTPException on failure."""
    model = _model or _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model loaded. Set the PENUX_AP_MODEL_PATH environment variable "
                "to the path of a trained .joblib model file."
            ),
        )
    row = pd.DataFrame([admission.model_dump()])
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if proba < RISK_THRESHOLDS["low"]:
        risk_group = "low"
    elif proba < RISK_THRESHOLDS["intermediate"]:
        risk_group = "intermediate"
    else:
        risk_group = "high"
    return proba, risk_group


@app.on_event("startup")
def startup():
    _load_model()


# ---------------------------------------------------------------------------
# Standard health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


# ---------------------------------------------------------------------------
# Plain JSON endpoint (original)
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionOutput)
def predict(data: AdmissionInput):
    model = _model or _load_model()
    if model is None:
        return PredictionOutput(
            error=(
                "No model loaded. Set the PENUX_AP_MODEL_PATH environment variable "
                "to the path of a trained .joblib model file."
            )
        )
    row = pd.DataFrame([data.model_dump()])
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if proba < RISK_THRESHOLDS["low"]:
        risk_group = "low"
    elif proba < RISK_THRESHOLDS["intermediate"]:
        risk_group = "intermediate"
    else:
        risk_group = "high"

    return PredictionOutput(
        severe_ap_probability=round(proba, 4),
        threshold_used=0.5,
        risk_group=risk_group,
    )


# ---------------------------------------------------------------------------
# FHIR R4 endpoint — Camelion FHIR gateway
# ---------------------------------------------------------------------------

@app.post("/fhir/predict", response_model=RiskAssessmentResource)
def fhir_predict(bundle: FHIRBundle):
    """Accept a FHIR R4 Bundle (Patient + Observation resources) and return
    a FHIR RiskAssessment.  Intended for the Camelion FHIR REST gateway.

    Patient identifiers received in the Bundle are discarded after mapping
    observations within this request; they are never stored or logged.
    """
    admission = bundle_to_admission_input(bundle)
    proba, risk_group = _run_prediction(admission)

    # Map risk_group to SNOMED qualitative risk codes
    qualitative_map = {
        "low": ("723505004", "Low risk"),
        "intermediate": ("723506003", "Moderate risk"),
        "high": ("723507007", "High risk"),
    }
    snomed_code, snomed_display = qualitative_map.get(risk_group, ("261665006", "Unknown"))

    return RiskAssessmentResource(
        prediction=[
            RiskAssessmentPrediction(
                outcome=CodeableConcept(
                    coding=[Coding(
                        system="http://snomed.info/sct",
                        code="67630002",
                        display="Severe acute pancreatitis",
                    )],
                    text="Severe Acute Pancreatitis",
                ),
                probabilityDecimal=round(proba, 4),
                qualitativeRisk=CodeableConcept(
                    coding=[Coding(
                        system="http://snomed.info/sct",
                        code=snomed_code,
                        display=snomed_display,
                    )]
                ),
                rationale=RESEARCH_WARNING,
            )
        ],
        note=[{"text": RESEARCH_WARNING}],
    )


# ---------------------------------------------------------------------------
# Camelion native JSON endpoint
# ---------------------------------------------------------------------------

@app.post("/camelion/predict", response_model=CamelionPredictionResponse)
async def camelion_predict(request: Request):
    """Accept a Camelion HIS native JSON payload and return a structured
    prediction response for triage alerting.

    Supports both Hebrew and English field names as exported by the Camelion
    REST / HL7 v2 gateway.  Patient identifiers (MRN, Teudat Zehut) are
    accepted in the payload for encounter correlation but are discarded
    immediately and never stored.

    Example payload::

        {
          "encounter_id": "ENC-2024-00123",
          "patient_id": "...",
          "גיל": 62,
          "מין": "זכר",
          "דופק": 105,
          "חום": 38.6,
          "כדוריות דם לבנות": 18.2,
          "crp": 210,
          "קראטינין": 1.4,
          "גלוקוז": 230,
          "סידן": 7.8,
          "ldh": 480,
          "ast": 310,
          "אוריאה": 30
        }
    """
    payload: dict[str, Any] = await request.json()

    encounter_id = payload.get("encounter_id") or payload.get("מזהה_מפגש")

    # Strip identifiers before adapter — not passed downstream
    clean_payload = {
        k: v for k, v in payload.items()
        if k not in ("patient_id", "encounter_id", "מזהה_מטופל", "מזהה_מפגש", "תעודת_זהות")
    }

    admission = camelion_json_to_admission_input(clean_payload)
    admission_dict = admission.model_dump()

    fields_used = [k for k, v in admission_dict.items() if v is not None]
    missing_fields = [k for k, v in admission_dict.items() if v is None]

    model = _model or _load_model()
    if model is None:
        return CamelionPredictionResponse(
            encounter_id=encounter_id,
            fields_used=fields_used,
            missing_fields=missing_fields,
            error=(
                "No model loaded. Set the PENUX_AP_MODEL_PATH environment variable "
                "to the path of a trained .joblib model file."
            ),
        )

    try:
        proba, risk_group = _run_prediction(admission)
    except HTTPException as e:
        return CamelionPredictionResponse(
            encounter_id=encounter_id,
            fields_used=fields_used,
            missing_fields=missing_fields,
            error=str(e.detail),
        )

    return CamelionPredictionResponse(
        encounter_id=encounter_id,
        severe_ap_probability=round(proba, 4),
        risk_group=risk_group,
        fields_used=fields_used,
        missing_fields=missing_fields,
    )
