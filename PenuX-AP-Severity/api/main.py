"""FastAPI research-only prediction endpoint for SAP severity.

RESEARCH USE ONLY. Not validated for clinical use.
Do not use for patient-care decisions.
"""
import os
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import AdmissionInput, HealthResponse, PredictionOutput, RESEARCH_WARNING
from penux_ap.config import RISK_THRESHOLDS

log = logging.getLogger(__name__)

app = FastAPI(
    title="PenuX-AP-Severity Research API",
    description=(
        "Research prototype for early prediction of Severe Acute Pancreatitis. "
        "NOT validated for clinical use. NOT for patient-care decisions."
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


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


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
