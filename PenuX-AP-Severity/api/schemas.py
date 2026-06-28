"""Pydantic schemas for the FastAPI research endpoint and Camelion HIS integration."""
from typing import Any, Optional
from pydantic import BaseModel, Field

RESEARCH_WARNING = (
    "This is a research prototype only. It is not validated for clinical use "
    "and must not be used for patient-care decisions."
)


class AdmissionInput(BaseModel):
    age: Optional[float] = Field(None, description="Age in years")
    sex: Optional[str] = Field(None, description="Sex (M/F)")
    bmi: Optional[float] = Field(None, description="Body mass index kg/m²")
    heart_rate: Optional[float] = Field(None, description="Heart rate bpm")
    systolic_bp: Optional[float] = Field(None, description="Systolic BP mmHg")
    diastolic_bp: Optional[float] = Field(None, description="Diastolic BP mmHg")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate /min")
    temperature: Optional[float] = Field(None, description="Temperature °C")
    spo2: Optional[float] = Field(None, description="SpO2 %")
    wbc: Optional[float] = Field(None, description="WBC ×10⁹/L")
    crp: Optional[float] = Field(None, description="CRP mg/L")
    bun: Optional[float] = Field(None, description="BUN mg/dL")
    creatinine: Optional[float] = Field(None, description="Creatinine mg/dL")
    calcium: Optional[float] = Field(None, description="Calcium mg/dL")
    glucose: Optional[float] = Field(None, description="Glucose mg/dL")
    hematocrit: Optional[float] = Field(None, description="Hematocrit %")
    ldh: Optional[float] = Field(None, description="LDH U/L")
    ast: Optional[float] = Field(None, description="AST U/L")
    alt: Optional[float] = Field(None, description="ALT U/L")
    albumin: Optional[float] = Field(None, description="Albumin g/dL")
    triglycerides: Optional[float] = Field(None, description="Triglycerides mg/dL")


class PredictionOutput(BaseModel):
    severe_ap_probability: Optional[float] = None
    threshold_used: float = 0.5
    risk_group: Optional[str] = None
    model_version: str = "0.1.0"
    warning: str = RESEARCH_WARNING
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    warning: str = RESEARCH_WARNING


# ---------------------------------------------------------------------------
# Camelion HIS native JSON schema
# ---------------------------------------------------------------------------

class CamelionRequest(BaseModel):
    """Camelion (קמיליון) HIS flat JSON request.

    Accepts both Hebrew and English field names as exported by the Camelion
    REST / HL7 v2 gateway. Patient identifiers are accepted for routing but
    are discarded before prediction and are never stored or logged.
    """
    # Patient identifiers — routing only, never stored
    patient_id: Optional[str] = Field(None, description="Camelion patient MRN (not stored)")
    encounter_id: Optional[str] = Field(None, description="Camelion encounter ID (not stored)")

    # Flat clinical values — Hebrew or English keys accepted via the adapter
    # (all fields are optional; pass whatever is available at admission)
    model_config = {"extra": "allow"}


class CamelionPredictionResponse(BaseModel):
    """Response returned to the Camelion system."""
    encounter_id: Optional[str] = None
    severe_ap_probability: Optional[float] = None
    risk_group: Optional[str] = Field(
        None,
        description="low | intermediate | high — for triage alerting only",
    )
    fields_used: list[str] = Field(
        default_factory=list,
        description="Clinical variables that contributed to this prediction",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Variables not supplied; prediction may be less reliable",
    )
    model_version: str = "0.1.0"
    warning: str = RESEARCH_WARNING
    error: Optional[str] = None
