"""FastAPI research-only prediction endpoint for SAP severity.

RESEARCH USE ONLY. Not validated for clinical use.
Do not use for patient-care decisions.

Integration endpoints:
  POST /predict          — plain JSON (AdmissionInput)
  POST /fhir/predict     — FHIR R4 Bundle (Patient + Observations, LOINC coded)
  POST /camelion/predict — Camelion (קמיליון) HIS native JSON
"""
import math
import os
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

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
from api.hl7_adapter import hl7_message_to_admission_input
from penux_ap.config import RISK_THRESHOLDS

log = logging.getLogger(__name__)

_DESCRIPTION = """
## PenuX-AP-Severity — Early Prediction of Severe Acute Pancreatitis

> ⚠️ **RESEARCH PROTOTYPE ONLY** — not validated for clinical use, not for patient-care decisions.

Uses routine admission laboratory values to estimate SAP risk within **4 hours of admission**,
implementing the **2012 Revised Atlanta Classification**.

---

### Endpoints

| Endpoint | Protocol | EHR |
|---|---|---|
| `POST /predict` | Plain JSON | Any |
| `POST /fhir/predict` | **FHIR R4** Bundle | Camelion, Epic, Cerner |
| `POST /camelion/predict` | Camelion native JSON | קמיליון HIS |
| `POST /hl7/predict` | HL7 v2.x raw message | Epic, Cerner, OpenEMR |

### FHIR Compliance

- Resource type: **RiskAssessment** (FHIR R4)
- Input: **Bundle** (Patient + Observation with LOINC codes)
- Output SNOMED codes: `723505004` Low · `723506003` Moderate · `723507007` High
- Condition SNOMED: `67630002` Severe acute pancreatitis

### Privacy

Patient identifiers (MRN, Teudat Zehut) are accepted for encounter correlation
and **immediately discarded** — never stored, logged, or forwarded.

### Resources

- GitHub: [netanelcyber/penuX](https://github.com/netanelcyber/penuX)
- Website: [penux.uk](https://penux.uk)
- LOINC codes: see `/openapi.json`
"""

app = FastAPI(
    title="PenuX-AP-Severity API",
    description=_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "PenuX Research Team",
        "url": "https://penux.uk",
        "email": "nsh531@gmail.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/netanelcyber/penuX/blob/main/PenuX-AP-Severity/LICENSE",
    },
    openapi_tags=[
        {"name": "health",    "description": "Service health check"},
        {"name": "predict",   "description": "Plain JSON prediction endpoint"},
        {"name": "fhir",      "description": "FHIR R4 — RiskAssessment (Camelion / Epic / Cerner)"},
        {"name": "camelion",  "description": "Camelion (קמיליון) HIS native JSON adapter"},
        {"name": "hl7",       "description": "HL7 v2.x — any EHR (Epic, Cerner, OpenEMR)"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
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


_HEURISTIC_WEIGHTS = {
    "wbc": 0.08, "crp": 0.004, "creatinine": 0.30, "bun": 0.015,
    "glucose": 0.002, "ldh": 0.001, "hematocrit": 0.03, "ast": 0.002,
    "albumin": -0.30, "calcium": -0.40, "bilirubin_total": 0.05,
}
_HEURISTIC_THRESHOLDS = {
    "wbc": 12.0, "crp": 150.0, "creatinine": 1.5, "bun": 25.0,
    "glucose": 200.0, "ldh": 250.0, "hematocrit": 44.0, "ast": 250.0,
    "albumin": 3.5, "calcium": 8.0, "bilirubin_total": 3.0,
}

def _heuristic_score(admission: AdmissionInput) -> float:
    """Logistic heuristic (BISAP/Ranson-weighted) used when no ML model is loaded."""
    logit = -1.8 + 0.015 * max(0, (admission.age or 55) - 55)
    if str(admission.sex or "").upper() in ("M", "MALE", "זכר"):
        logit += 0.15
    data = admission.model_dump()
    for feat, w in _HEURISTIC_WEIGHTS.items():
        v = data.get(feat)
        if v is None:
            continue
        t = _HEURISTIC_THRESHOLDS[feat]
        logit += w * (max(0, t - v) if feat in ("albumin", "calcium") else max(0, v - t))
    return round(1.0 / (1.0 + math.exp(-logit)), 4)


def _run_prediction(admission: AdmissionInput) -> tuple[float, str]:
    """Return (probability, risk_group). Falls back to heuristic when no ML model."""
    model = _model or _load_model()
    if model is not None:
        row = pd.DataFrame([admission.model_dump()])
        try:
            proba = float(model.predict_proba(row)[0, 1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    else:
        proba = _heuristic_score(admission)

    if proba < RISK_THRESHOLDS["low"]:
        risk_group = "low"
    elif proba < RISK_THRESHOLDS["intermediate"]:
        risk_group = "intermediate"
    else:
        risk_group = "high"
    return proba, risk_group


# ---------------------------------------------------------------------------
# Keras pathogen model (latest from github.com/netanelcyber/penuX main branch)
# Features: temperature_c (°C), wbc (cells/µL), spo2 (%), age (years)
# Output: 12-class pathogen classification
# ---------------------------------------------------------------------------
_PATHOGEN_CLASSES = [
    "B:PSEUDOMONAS AERUGINOSA",
    "B:STAPH AUREUS COAG +",
    "B:SERRATIA MARCESCENS",
    "B:MORGANELLA MORGANII",
    "B:ESCHERICHIA COLI",
    "B:PROTEUS MIRABILIS",
    "B:PROVIDENCIA STUARTII",
    "B:MRSA",
    "B:YEAST",
    "B:GRAM POSITIVE COCCUS",
    "B:OTHER",
    "V:OTHER",
]

_KERAS_ENCODER = None
_KERAS_HEAD    = None
_KERAS_SCALER  = None   # dict with 'mu' and 'sd'

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_keras_model():
    global _KERAS_ENCODER, _KERAS_HEAD, _KERAS_SCALER
    enc_path = _MODEL_DIR / "clin_encoder.keras"
    hd_path  = _MODEL_DIR / "clin_head.keras"
    sc_path  = _MODEL_DIR / "clin_scaler.npz"
    if not (enc_path.exists() and hd_path.exists() and sc_path.exists()):
        log.warning("Keras model files not found in %s", _MODEL_DIR)
        return False
    try:
        import tensorflow as tf
        _KERAS_ENCODER = tf.keras.models.load_model(str(enc_path))
        _KERAS_HEAD    = tf.keras.models.load_model(str(hd_path))
        sc = np.load(str(sc_path))
        _KERAS_SCALER  = {"mu": sc["mu"], "sd": sc["sd"]}
        log.info("Keras pathogen model loaded from %s", _MODEL_DIR)
        return True
    except Exception as e:
        log.warning("Failed to load Keras model: %s", e)
        return False


class PathogenInput(BaseModel):
    temperature_c: float
    wbc: float
    spo2: float
    age: float


class PathogenOutput(BaseModel):
    predicted_pathogen: str
    confidence: float
    top3: list
    model_source: str = "github.com/netanelcyber/penuX main branch"
    warning: str = RESEARCH_WARNING


# ---------------------------------------------------------------------------
# Sepsis risk — SIRS + qSOFA heuristic from routine tests
# ---------------------------------------------------------------------------

class SepsisInput(BaseModel):
    temperature_c: Optional[float] = Field(None, description="Body temperature °C")
    heart_rate: Optional[float] = Field(None, description="Heart rate bpm")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate /min")
    systolic_bp: Optional[float] = Field(None, description="Systolic BP mmHg")
    wbc: Optional[float] = Field(None, description="WBC ×10⁹/L (e.g. 12.5 = 12,500 cells/µL)")
    lactate: Optional[float] = Field(None, description="Lactate mmol/L")
    creatinine: Optional[float] = Field(None, description="Creatinine mg/dL")
    bilirubin: Optional[float] = Field(None, description="Bilirubin total mg/dL")
    platelets: Optional[float] = Field(None, description="Platelets ×10³/µL")
    map_mmhg: Optional[float] = Field(None, description="Mean arterial pressure mmHg")
    spo2: Optional[float] = Field(None, description="SpO2 %")
    age: Optional[float] = Field(None, description="Age in years")


class SepsisOutput(BaseModel):
    sepsis_risk_probability: float
    risk_group: str = Field(description="low | moderate | high | critical")
    sirs_score: int = Field(description="SIRS criteria met (0-4)")
    qsofa_score: int = Field(description="qSOFA score (0-3)")
    criteria_met: list[str] = Field(description="Specific criteria that were triggered")
    warning: str = RESEARCH_WARNING


def _sepsis_score(inp: SepsisInput) -> tuple[float, str, int, int, list[str]]:
    """SIRS + qSOFA logistic model from routine tests only.

    SIRS: temp, HR, RR, WBC  — 2+ criteria = SIRS
    qSOFA: SBP≤100, RR≥22, altered mentation (not tested here) — 2+ = high risk
    Augmented with lactate, creatinine, bilirubin, platelets for organ dysfunction.
    """
    criteria: list[str] = []
    sirs = 0
    qsofa = 0

    if inp.temperature_c is not None:
        if inp.temperature_c > 38.3:
            sirs += 1
            criteria.append(f"Fever ({inp.temperature_c}°C > 38.3)")
        elif inp.temperature_c < 36.0:
            sirs += 1
            criteria.append(f"Hypothermia ({inp.temperature_c}°C < 36.0)")

    if inp.heart_rate is not None and inp.heart_rate > 90:
        sirs += 1
        criteria.append(f"Tachycardia (HR {inp.heart_rate} > 90)")

    if inp.respiratory_rate is not None:
        if inp.respiratory_rate > 20:
            sirs += 1
            criteria.append(f"Tachypnea (RR {inp.respiratory_rate} > 20)")
        if inp.respiratory_rate >= 22:
            qsofa += 1

    if inp.wbc is not None:
        if inp.wbc > 12.0:
            sirs += 1
            criteria.append(f"Leukocytosis (WBC {inp.wbc} > 12.0)")
        elif inp.wbc < 4.0:
            sirs += 1
            criteria.append(f"Leukopenia (WBC {inp.wbc} < 4.0)")

    if inp.systolic_bp is not None and inp.systolic_bp <= 100:
        qsofa += 1
        criteria.append(f"Hypotension (SBP {inp.systolic_bp} ≤ 100)")

    # Organ dysfunction markers (Sepsis-3)
    organ_score = 0.0
    if inp.creatinine is not None and inp.creatinine > 2.0:
        organ_score += 0.15
        criteria.append(f"Renal dysfunction (Creatinine {inp.creatinine} > 2.0)")
    if inp.bilirubin is not None and inp.bilirubin > 2.0:
        organ_score += 0.10
        criteria.append(f"Hepatic dysfunction (Bilirubin {inp.bilirubin} > 2.0)")
    if inp.platelets is not None and inp.platelets < 100:
        organ_score += 0.15
        criteria.append(f"Thrombocytopenia (Platelets {inp.platelets} < 100)")
    if inp.lactate is not None and inp.lactate > 2.0:
        organ_score += 0.20
        criteria.append(f"Elevated lactate ({inp.lactate} > 2.0 mmol/L)")
        if inp.lactate > 4.0:
            organ_score += 0.10
            criteria.append(f"Critical lactate ({inp.lactate} > 4.0 — septic shock)")
    if inp.map_mmhg is not None and inp.map_mmhg < 65:
        organ_score += 0.20
        criteria.append(f"Low MAP ({inp.map_mmhg} < 65 mmHg — vasopressor territory)")
    if inp.spo2 is not None and inp.spo2 < 94:
        organ_score += 0.08
        criteria.append(f"Hypoxia (SpO2 {inp.spo2}% < 94%)")

    # Age adjustment
    age_adj = 0.0
    if inp.age is not None and inp.age > 65:
        age_adj = 0.05

    # Logistic: intercept anchored so 0 criteria ≈ 5% risk
    logit = (
        -2.9
        + 0.55 * sirs
        + 0.65 * qsofa
        + organ_score * 4.0
        + age_adj
    )
    proba = round(1.0 / (1.0 + math.exp(-logit)), 4)

    if proba < 0.15:
        risk = "low"
    elif proba < 0.40:
        risk = "moderate"
    elif proba < 0.70:
        risk = "high"
    else:
        risk = "critical"

    return proba, risk, min(sirs, 4), qsofa, criteria


@app.on_event("startup")
def startup():
    _load_model()
    _load_keras_model()


# ---------------------------------------------------------------------------
# Standard health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Service health check",
    responses={200: {"description": "Service is running"}},
)
def health():
    return HealthResponse()


# ---------------------------------------------------------------------------
# Plain JSON endpoint (original)
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["predict"],
    summary="Predict SAP risk — plain JSON",
    description=(
        "Accepts routine admission lab values and returns a SAP risk probability "
        "with risk tier (low / intermediate / high).\n\n"
        "All fields are optional — supply whatever is available at admission. "
        "Prediction reliability increases with more fields provided."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "high_risk": {
                            "summary": "High-risk patient",
                            "value": {
                                "age": 65, "sex": "M",
                                "wbc": 18.5, "crp": 220, "creatinine": 1.8,
                                "glucose": 180, "ldh": 450, "ast": 90,
                                "hematocrit": 47, "bun": 32, "calcium": 7.8,
                                "albumin": 2.9,
                            },
                        },
                        "low_risk": {
                            "summary": "Low-risk patient",
                            "value": {
                                "age": 42, "sex": "F",
                                "wbc": 9.1, "crp": 45, "creatinine": 0.8,
                                "glucose": 110, "ldh": 180, "ast": 38,
                                "hematocrit": 38,
                            },
                        },
                        "minimal": {
                            "summary": "Minimal labs only",
                            "value": {"age": 55, "sex": "M", "wbc": 14.0, "creatinine": 1.5},
                        },
                    }
                }
            }
        }
    },
)
def predict(data: AdmissionInput):
    proba, risk_group = _run_prediction(data)
    return PredictionOutput(
        severe_ap_probability=proba,
        threshold_used=0.5,
        risk_group=risk_group,
    )


# ---------------------------------------------------------------------------
# FHIR R4 endpoint — Camelion FHIR gateway
# ---------------------------------------------------------------------------

@app.post(
    "/fhir/predict",
    response_model=RiskAssessmentResource,
    tags=["fhir"],
    summary="FHIR R4 — RiskAssessment prediction",
    description=(
        "Accepts a **FHIR R4 Bundle** containing a Patient resource and Observation "
        "resources coded with **LOINC codes**, and returns a **FHIR RiskAssessment** "
        "resource.\n\n"
        "This endpoint is the primary integration point for **Camelion (קמיליון) HIS**, "
        "Epic SMART on FHIR, Cerner Millennium, and any FHIR R4-compliant system.\n\n"
        "**Supported LOINC codes:**\n"
        "- `6690-2` WBC · `1988-5` CRP · `2160-0` Creatinine\n"
        "- `3094-0` BUN · `2345-7` Glucose · `2532-0` LDH\n"
        "- `1920-8` AST · `1742-6` ALT · `20570-8` Hematocrit\n"
        "- `17861-6` Calcium · `1751-7` Albumin · `1975-2` Bilirubin\n\n"
        "Patient identifiers are accepted but **immediately discarded** and never stored."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/fhir+json": {
                    "examples": {
                        "fhir_bundle": {
                            "summary": "FHIR R4 Bundle with Patient + CRP + WBC",
                            "value": {
                                "resourceType": "Bundle",
                                "type": "collection",
                                "entry": [
                                    {
                                        "resource": {
                                            "resourceType": "Patient",
                                            "birthDate": "1962-03-15",
                                            "gender": "male",
                                        }
                                    },
                                    {
                                        "resource": {
                                            "resourceType": "Observation",
                                            "status": "final",
                                            "code": {"coding": [{"system": "http://loinc.org", "code": "1988-5", "display": "CRP"}]},
                                            "valueQuantity": {"value": 220, "unit": "mg/L"},
                                        }
                                    },
                                    {
                                        "resource": {
                                            "resourceType": "Observation",
                                            "status": "final",
                                            "code": {"coding": [{"system": "http://loinc.org", "code": "6690-2", "display": "WBC"}]},
                                            "valueQuantity": {"value": 18.5, "unit": "10*3/uL"},
                                        }
                                    },
                                    {
                                        "resource": {
                                            "resourceType": "Observation",
                                            "status": "final",
                                            "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]},
                                            "valueQuantity": {"value": 1.8, "unit": "mg/dL"},
                                        }
                                    },
                                ],
                            },
                        }
                    }
                },
                "application/json": {"schema": {"$ref": "#/components/schemas/FHIRBundle"}},
            }
        },
        "responses": {
            "200": {
                "description": "FHIR RiskAssessment resource",
                "content": {
                    "application/fhir+json": {
                        "example": {
                            "resourceType": "RiskAssessment",
                            "status": "final",
                            "subject": {"reference": "Patient/anonymous"},
                            "prediction": [{
                                "outcome": {
                                    "coding": [{"system": "http://snomed.info/sct", "code": "67630002", "display": "Severe acute pancreatitis"}],
                                    "text": "Severe Acute Pancreatitis",
                                },
                                "probabilityDecimal": 0.782,
                                "qualitativeRisk": {
                                    "coding": [{"system": "http://snomed.info/sct", "code": "723507007", "display": "High risk"}]
                                },
                            }],
                        }
                    }
                },
            }
        },
    },
)
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

@app.post(
    "/camelion/predict",
    response_model=CamelionPredictionResponse,
    tags=["camelion"],
    summary="Camelion (קמיליון) HIS native JSON",
    description=(
        "Accepts a Camelion HIS flat JSON payload with Hebrew **or** English field names "
        "and returns a structured SAP risk response for triage alerting.\n\n"
        "**Hebrew keys supported:** `גיל` (age), `מין` (sex), `כדוריות_דם_לבנות` (WBC), "
        "`CRP`, `קריאטינין` (creatinine), `גלוקוז` (glucose), `סידן` (calcium), "
        "`ldh`, `ast`, `אוריאה` (BUN), `המטוקריט` (hematocrit)\n\n"
        "Patient identifiers (`patient_id`, `תעודת_זהות`) are **never stored or logged**."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "hebrew": {
                            "summary": "Hebrew field names (Camelion native)",
                            "value": {
                                "encounter_id": "ENC-2024-00123",
                                "גיל": 65,
                                "מין": "זכר",
                                "כדוריות_דם_לבנות": 18.2,
                                "CRP": 210,
                                "קריאטינין": 1.4,
                                "גלוקוז": 230,
                                "סידן": 7.8,
                                "ldh": 480,
                                "ast": 310,
                                "אוריאה": 30,
                            },
                        },
                        "english": {
                            "summary": "English field names",
                            "value": {
                                "encounter_id": "ENC-2024-00124",
                                "age": 58, "sex": "F",
                                "wbc": 14.5, "crp": 155,
                                "creatinine": 1.2, "glucose": 190,
                                "ldh": 310, "ast": 140, "bun": 28,
                            },
                        },
                    }
                }
            }
        }
    },
)
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


# ---------------------------------------------------------------------------
# HL7 v2.x endpoint — Epic, Cerner, OpenEMR, and other EHRs
# ---------------------------------------------------------------------------

@app.post(
    "/hl7/predict",
    response_model=PredictionOutput,
    tags=["hl7"],
    summary="HL7 v2.x — Epic / Cerner / OpenEMR",
    description=(
        "Accepts a raw **HL7 v2.x** message (`text/plain`) and returns a SAP risk prediction.\n\n"
        "Supported message types: `ORU^R01` (lab results).\n\n"
        "Supported code systems:\n"
        "- **LOINC** (standard): `2160-0`, `6690-2`, `1988-5`, etc.\n"
        "- **Epic LIS codes**: `CREAT`, `WBC`, `CRP`, `LDH`, `AST`, `ALT`\n"
        "- **Cerner codes**: `CREAT`, `GLUC`, `CA`, `BUN`\n\n"
        "Patient identifiers (PID segment) are parsed for encounter routing and "
        "**immediately discarded** — never stored or logged."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "text/plain": {
                    "schema": {"type": "string"},
                    "examples": {
                        "hl7_orur01": {
                            "summary": "HL7 ORU^R01 with LOINC codes",
                            "value": (
                                "MSH|^~\\&|LIS|HOSPITAL|PENUX|API|20240601||ORU^R01|001|P|2.5\r"
                                "PID|1||MRN123||DOE^JOHN||19620315|M\r"
                                "OBR|1|||AP_PANEL\r"
                                "OBX|1|NM|1988-5^CRP^LN||220|mg/L|0-5||||F\r"
                                "OBX|2|NM|6690-2^WBC^LN||18.5|10*3/uL|4-11||||F\r"
                                "OBX|3|NM|2160-0^Creatinine^LN||1.8|mg/dL|0.7-1.2||||F\r"
                                "OBX|4|NM|14804-9^LDH^LN||450|U/L|120-250||||F"
                            ),
                        },
                        "epic_lis": {
                            "summary": "Epic LIS vendor codes",
                            "value": (
                                "MSH|^~\\&|Epic|Hospital|PENUX|API|20240601||ORU^R01|002|P|2.5\r"
                                "PID|1||MRN456||SMITH^JANE||19780520|F\r"
                                "OBX|1|NM|CREAT^Creatinine||1.4|mg/dL|||F\r"
                                "OBX|2|NM|WBC^White Blood Cells||16.2|K/uL|||F\r"
                                "OBX|3|NM|LDH^LDH||380|IU/L|||F"
                            ),
                        },
                    },
                }
            }
        }
    },
)
async def hl7_predict(request: Request):
    """Accept an HL7 v2.x message from any EHR system and return prediction.

    Supported EHR systems: Epic, Cerner, OpenEMR, Allscripts, VistA, etc.
    Any system that exports OBX (observation) segments with LOINC codes.

    Example HL7 v2.x message::

        MSH|^~\\&|Epic|Hospital|Receiver||||20240618
        PID|1||MRN123456||Doe^John||19640101|M
        OBX|1|NM|2160-0^Creatinine||1.5|mg/dL|||F
        OBX|2|NM|6690-2^WBC||18.2|10*9/L|||F
        OBX|3|NM|1988-5^CRP||210|mg/L|||F
        OBX|4|NM|14804-9^LDH||480|U/L|||F

    Patient identifiers (MRN, name, account#) are extracted for encounter
    routing but discarded immediately and never stored or logged.

    Supports both LOINC (standard) and vendor-specific LIS codes (Epic WBC,
    Cerner CREAT, etc). Unknown codes are logged and skipped.
    """
    # Accept raw HL7 message as plain text
    try:
        message = await request.body()
        if isinstance(message, bytes):
            message = message.decode("utf-8")
    except Exception:
        return PredictionOutput(
            error="Could not parse request body as HL7 message (expected UTF-8 text)"
        )

    if not message.strip():
        return PredictionOutput(error="Empty HL7 message")

    admission = hl7_message_to_admission_input(message)
    admission_dict = admission.model_dump()

    fields_used = [k for k, v in admission_dict.items() if v is not None]

    try:
        proba, risk_group = _run_prediction(admission)
    except HTTPException as e:
        return PredictionOutput(error=str(e.detail))

    return PredictionOutput(
        severe_ap_probability=round(proba, 4),
        threshold_used=0.5,
        risk_group=risk_group,
    )


# ---------------------------------------------------------------------------
# Pathogen prediction — latest Keras model from github.com/netanelcyber/penuX
# ---------------------------------------------------------------------------

@app.post(
    "/predict/pathogen",
    response_model=PathogenOutput,
    tags=["predict"],
    summary="Pathogen classification — latest Keras model (MIMIC-based)",
    description=(
        "Classifies likely pathogen from 4 routine admission vitals/labs "
        "using the **latest trained Keras model** from "
        "[github.com/netanelcyber/penuX](https://github.com/netanelcyber/penuX) `main` branch.\n\n"
        "**Input features** (all required):\n"
        "- `temperature_c` — body temperature in °C\n"
        "- `wbc` — white blood cell count (cells/µL, e.g. 12000)\n"
        "- `spo2` — oxygen saturation (%)\n"
        "- `age` — patient age in years\n\n"
        "**Output:** 12-class pathogen probability distribution "
        "(Bacterial: Pseudomonas, Staph, E.coli, MRSA, Yeast, etc. · Viral)\n\n"
        "Model: `clin_encoder.keras` + `clin_head.keras` + `clin_scaler.npz` "
        "trained on MIMIC-III clinical data."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "bacterial_likely": {
                            "summary": "Bacterial — fever + high WBC + low SpO2",
                            "value": {"temperature_c": 39.2, "wbc": 18000, "spo2": 88.0, "age": 70},
                        },
                        "viral_likely": {
                            "summary": "Viral — moderate fever + normal WBC",
                            "value": {"temperature_c": 38.1, "wbc": 8500, "spo2": 93.0, "age": 45},
                        },
                        "normal": {
                            "summary": "Normal / low risk",
                            "value": {"temperature_c": 36.8, "wbc": 7000, "spo2": 98.0, "age": 35},
                        },
                    }
                }
            }
        }
    },
)
def predict_pathogen(data: PathogenInput):
    if _KERAS_ENCODER is None or _KERAS_HEAD is None:
        if not _load_keras_model():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Keras model not loaded. Place clin_encoder.keras, clin_head.keras, "
                    "clin_scaler.npz in the models/ directory."
                ),
            )

    mu = _KERAS_SCALER["mu"]
    sd = _KERAS_SCALER["sd"]
    x = np.array([[data.temperature_c, data.wbc, data.spo2, data.age]], dtype=np.float32)
    x_norm = (x - mu) / sd

    try:
        enc   = _KERAS_ENCODER.predict(x_norm, verbose=0)
        probs = _KERAS_HEAD.predict(enc, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    top_idx = probs.argsort()[::-1][:3]
    top3 = [
        {"rank": i + 1, "pathogen": _PATHOGEN_CLASSES[idx], "probability": round(float(probs[idx]), 4)}
        for i, idx in enumerate(top_idx)
    ]

    return PathogenOutput(
        predicted_pathogen=_PATHOGEN_CLASSES[int(probs.argmax())],
        confidence=round(float(probs.max()), 4),
        top3=top3,
    )


# ---------------------------------------------------------------------------
# Sepsis risk detection — SIRS + qSOFA + organ dysfunction (routine tests)
# ---------------------------------------------------------------------------

@app.post(
    "/predict/sepsis",
    response_model=SepsisOutput,
    tags=["predict"],
    summary="Sepsis risk — routine tests only (SIRS + qSOFA + organ dysfunction)",
    description=(
        "Estimates sepsis risk from **routine admission tests only** using a "
        "logistic combination of:\n\n"
        "- **SIRS** (Systemic Inflammatory Response Syndrome): temperature, HR, RR, WBC\n"
        "- **qSOFA** (quick Sequential Organ Failure Assessment): RR ≥ 22, SBP ≤ 100\n"
        "- **Organ dysfunction markers** (Sepsis-3): creatinine, bilirubin, platelets, lactate, MAP, SpO2\n\n"
        "All fields are optional — supply whatever routine tests are available at admission.\n\n"
        "**Risk groups:** low (<15%) · moderate (15–40%) · high (40–70%) · critical (>70%)\n\n"
        "⚠️ RESEARCH USE ONLY — not validated for clinical decisions. "
        "Sepsis requires clinical judgment, culture results, and physician assessment."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "septic_shock": {
                            "summary": "Critical — septic shock pattern",
                            "value": {
                                "temperature_c": 39.5, "heart_rate": 118,
                                "respiratory_rate": 26, "systolic_bp": 88,
                                "wbc": 18.5, "lactate": 4.2, "creatinine": 2.4,
                                "platelets": 80, "map_mmhg": 58, "spo2": 90, "age": 72,
                            },
                        },
                        "sirs_without_organ_failure": {
                            "summary": "High — SIRS criteria but no organ failure",
                            "value": {
                                "temperature_c": 38.8, "heart_rate": 102,
                                "respiratory_rate": 22, "systolic_bp": 105,
                                "wbc": 14.2, "age": 58,
                            },
                        },
                        "low_risk": {
                            "summary": "Low — near-normal vitals and labs",
                            "value": {
                                "temperature_c": 37.1, "heart_rate": 78,
                                "respiratory_rate": 16, "systolic_bp": 122,
                                "wbc": 8.5, "spo2": 98, "age": 40,
                            },
                        },
                    }
                }
            }
        }
    },
)
def predict_sepsis(data: SepsisInput):
    proba, risk, sirs_score, qsofa_score, criteria = _sepsis_score(data)
    return SepsisOutput(
        sepsis_risk_probability=proba,
        risk_group=risk,
        sirs_score=sirs_score,
        qsofa_score=qsofa_score,
        criteria_met=criteria,
    )
