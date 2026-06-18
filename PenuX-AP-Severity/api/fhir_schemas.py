"""FHIR R4 resource schemas for Camelion HIS integration.

Implements a subset of FHIR R4 sufficient for the PenuX-AP-Severity prediction
endpoint. Only the fields consumed by the prediction pipeline are modelled.

Reference: https://hl7.org/fhir/R4/
Camelion (קמיליון) uses FHIR R4 REST for external system integration.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# FHIR primitive / backbone types
# ---------------------------------------------------------------------------

class Coding(BaseModel):
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None


class CodeableConcept(BaseModel):
    coding: list[Coding] = Field(default_factory=list)
    text: Optional[str] = None


class Quantity(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None


class Reference(BaseModel):
    reference: Optional[str] = None
    display: Optional[str] = None


class Identifier(BaseModel):
    system: Optional[str] = None
    value: Optional[str] = None


# ---------------------------------------------------------------------------
# FHIR Patient (subset)
# ---------------------------------------------------------------------------

class PatientResource(BaseModel):
    resourceType: str = "Patient"
    id: Optional[str] = None
    # Israeli Teudat Zehut — received but never stored downstream
    identifier: list[Identifier] = Field(default_factory=list)
    birthDate: Optional[str] = Field(None, description="YYYY-MM-DD")
    gender: Optional[str] = Field(None, description="male | female | other | unknown")


# ---------------------------------------------------------------------------
# FHIR Observation (subset)
# ---------------------------------------------------------------------------

class ObservationComponent(BaseModel):
    code: Optional[CodeableConcept] = None
    valueQuantity: Optional[Quantity] = None
    valueString: Optional[str] = None


class ObservationResource(BaseModel):
    resourceType: str = "Observation"
    id: Optional[str] = None
    status: Optional[str] = None
    code: Optional[CodeableConcept] = None
    subject: Optional[Reference] = None
    valueQuantity: Optional[Quantity] = None
    valueString: Optional[str] = None
    valueCodeableConcept: Optional[CodeableConcept] = None
    component: list[ObservationComponent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FHIR Bundle entry / Bundle
# ---------------------------------------------------------------------------

class BundleEntry(BaseModel):
    resource: dict[str, Any] = Field(default_factory=dict)


class FHIRBundle(BaseModel):
    """FHIR R4 Bundle containing Patient + Observation resources.

    Expected use: POST /fhir/predict with a Bundle of type "collection"
    containing one Patient resource and any number of Observation resources
    with LOINC-coded lab/vital measurements.
    """
    resourceType: str = "Bundle"
    type: str = "collection"
    entry: list[BundleEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FHIR RiskAssessment (output)
# ---------------------------------------------------------------------------

class RiskAssessmentPrediction(BaseModel):
    outcome: CodeableConcept
    probabilityDecimal: Optional[float] = None
    qualitativeRisk: Optional[CodeableConcept] = None
    rationale: Optional[str] = None


class RiskAssessmentResource(BaseModel):
    """FHIR R4 RiskAssessment returned by /fhir/predict."""
    resourceType: str = "RiskAssessment"
    status: str = "final"
    method: CodeableConcept = Field(
        default_factory=lambda: CodeableConcept(
            coding=[Coding(
                system="http://snomed.info/sct",
                code="74964007",
                display="Machine learning prediction model",
            )]
        )
    )
    subject: Optional[Reference] = None
    prediction: list[RiskAssessmentPrediction] = Field(default_factory=list)
    note: list[dict] = Field(default_factory=list)
