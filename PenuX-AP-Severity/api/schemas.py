"""Pydantic schemas for the PenuX-AP research API.

The public prediction contract remains backwards compatible with the original
flat admission schema. The expanded fields support hospital data mapping,
unit normalisation and cohort-quality assessment. Outcome fields are never
included in the predictor feature dictionary.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESEARCH_WARNING = (
    "This is a research prototype only. It is not validated for clinical use "
    "and must not be used for patient-care decisions."
)

LEGACY_MODEL_FIELDS: tuple[str, ...] = (
    "age", "sex", "bmi", "heart_rate", "systolic_bp", "diastolic_bp",
    "respiratory_rate", "temperature", "spo2", "wbc", "crp", "bun",
    "creatinine", "calcium", "glucose", "hematocrit", "ldh", "ast",
    "alt", "albumin", "triglycerides",
)

CORE_GROUPS: dict[str, tuple[str, ...]] = {
    "age": ("age",),
    "sex": ("sex",),
    "heart_rate": ("heart_rate",),
    "systolic_bp": ("systolic_bp",),
    "respiratory_rate": ("respiratory_rate",),
    "temperature": ("temperature",),
    "spo2": ("spo2",),
    "consciousness": ("gcs", "avpu"),
    "wbc": ("wbc",),
    "hemoglobin_or_hematocrit": ("hemoglobin", "hematocrit"),
    "urea_or_bun": ("urea_mmol_l", "bun"),
    "creatinine": ("creatinine", "creatinine_umol_l"),
    "glucose": ("glucose", "glucose_mmol_l"),
    "calcium": ("calcium", "calcium_mmol_l"),
    "albumin": ("albumin", "albumin_g_l"),
    "bilirubin": ("bilirubin_total", "bilirubin_total_umol_l"),
}

VITAL_GROUPS: tuple[str, ...] = (
    "heart_rate", "systolic_bp", "respiratory_rate", "temperature", "spo2",
)
LAB_GROUPS: tuple[str, ...] = (
    "wbc", "hemoglobin_or_hematocrit", "urea_or_bun", "creatinine",
    "glucose", "calcium", "albumin", "bilirubin",
)


class AdmissionInput(BaseModel):
    """Expanded first-24-hour AP predictor input.

    Values omitted from the request remain ``None``. Absence is never treated
    as zero or normal. SI aliases are converted to the legacy units used by
    the current endpoint, while the original SI values remain available in the
    full hospital record.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    patient_id: Optional[str] = Field(None, description="Pseudonymised patient identifier; never log a national ID")
    encounter_id: Optional[str] = Field(None, description="Pseudonymised hospital encounter identifier")
    admission_time: Optional[datetime] = Field(None, description="T0: first hospital arrival/admission timestamp")
    measurement_window_hours: float = Field(24.0, ge=0, le=24)
    data_context: Literal["prediction", "model_development"] = "prediction"
    acute_pancreatitis_diagnosis: Optional[bool] = None
    abdominal_pain_compatible: Optional[bool] = None
    imaging_available: bool = False
    outcome_followup_available: Optional[bool] = None
    persistent_organ_failure_gt_48h: Optional[bool] = Field(None, description="Outcome only; excluded from predictor features")

    age: Optional[float] = Field(None, ge=0, le=130, description="Age at T0, years")
    sex: Optional[str] = Field(None, description="Recorded sex; retain local coding")
    height_cm: Optional[float] = Field(None, gt=0, le=260)
    weight_kg: Optional[float] = Field(None, gt=0, le=500)
    bmi: Optional[float] = Field(None, gt=0, le=100, description="kg/m²")

    heart_rate: Optional[float] = Field(None, ge=0, le=350, description="bpm")
    systolic_bp: Optional[float] = Field(None, ge=0, le=350, description="mmHg")
    diastolic_bp: Optional[float] = Field(None, ge=0, le=250, description="mmHg")
    mean_arterial_pressure: Optional[float] = Field(None, ge=0, le=250, description="mmHg")
    respiratory_rate: Optional[float] = Field(None, ge=0, le=100, description="breaths/min")
    temperature: Optional[float] = Field(None, ge=25, le=45, description="°C")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="%")
    oxygen_support: Optional[str] = None
    oxygen_flow_l_min: Optional[float] = Field(None, ge=0, le=100)
    fio2: Optional[float] = Field(None, ge=0.21, le=1.0)
    gcs: Optional[int] = Field(None, ge=3, le=15)
    avpu: Optional[str] = Field(None, description="A/V/P/U or local equivalent")
    urine_output_ml_24h: Optional[float] = Field(None, ge=0)
    vasopressor_use: Optional[bool] = None
    invasive_ventilation: Optional[bool] = None
    noninvasive_ventilation: Optional[bool] = None

    wbc: Optional[float] = Field(None, description="×10⁹/L")
    anc: Optional[float] = Field(None, description="×10⁹/L")
    alc: Optional[float] = Field(None, description="×10⁹/L")
    monocytes_absolute: Optional[float] = Field(None, description="×10⁹/L")
    rbc: Optional[float] = Field(None, description="×10¹²/L")
    hemoglobin: Optional[float] = Field(None, description="g/dL")
    hematocrit: Optional[float] = Field(None, description="%")
    platelets: Optional[float] = Field(None, description="×10⁹/L")
    mcv: Optional[float] = Field(None, description="fL")
    rdw_cv: Optional[float] = Field(None, description="%")
    mpv: Optional[float] = Field(None, description="fL")

    urea_mmol_l: Optional[float] = None
    bun: Optional[float] = Field(None, description="mg/dL")
    creatinine: Optional[float] = Field(None, description="mg/dL")
    creatinine_umol_l: Optional[float] = None
    egfr: Optional[float] = Field(None, description="mL/min/1.73m²")
    sodium: Optional[float] = Field(None, description="mmol/L")
    potassium: Optional[float] = Field(None, description="mmol/L")
    chloride: Optional[float] = Field(None, description="mmol/L")
    bicarbonate_total: Optional[float] = Field(None, description="mmol/L; chemistry TCO₂")
    glucose: Optional[float] = Field(None, description="mg/dL")
    glucose_mmol_l: Optional[float] = None
    calcium: Optional[float] = Field(None, description="mg/dL; total calcium")
    calcium_mmol_l: Optional[float] = None
    ionized_calcium_mmol_l: Optional[float] = None
    magnesium_mmol_l: Optional[float] = None
    phosphate_mmol_l: Optional[float] = None
    bicarbonate_blood_gas: Optional[float] = Field(None, description="mmol/L; keep separate from chemistry TCO₂")

    albumin: Optional[float] = Field(None, description="g/dL")
    albumin_g_l: Optional[float] = None
    total_protein_g_l: Optional[float] = None
    bilirubin_total: Optional[float] = Field(None, description="mg/dL")
    bilirubin_total_umol_l: Optional[float] = None
    bilirubin_direct_umol_l: Optional[float] = None
    ast: Optional[float] = Field(None, description="U/L")
    alt: Optional[float] = Field(None, description="U/L")
    alp: Optional[float] = Field(None, description="U/L")
    ggt: Optional[float] = Field(None, description="U/L")
    ldh: Optional[float] = Field(None, description="U/L")
    lipase: Optional[float] = Field(None, description="U/L")
    lipase_uln: Optional[float] = Field(None, gt=0, description="Local upper limit of normal, U/L")
    amylase: Optional[float] = Field(None, description="U/L")
    amylase_uln: Optional[float] = Field(None, gt=0, description="Local upper limit of normal, U/L")
    crp: Optional[float] = Field(None, description="mg/L")
    procalcitonin_ng_ml: Optional[float] = None
    triglycerides: Optional[float] = Field(None, description="mg/dL")
    triglycerides_mmol_l: Optional[float] = None

    pt_seconds: Optional[float] = None
    inr: Optional[float] = None
    aptt_seconds: Optional[float] = None
    fibrinogen_g_l: Optional[float] = None
    d_dimer_mg_l_feu: Optional[float] = None
    lactate: Optional[float] = Field(None, description="mmol/L")
    ph: Optional[float] = None
    pao2: Optional[float] = Field(None, description="mmHg")
    paco2: Optional[float] = Field(None, description="mmHg")
    base_excess: Optional[float] = Field(None, description="mmol/L")

    diabetes: Optional[bool] = None
    heart_failure: Optional[bool] = None
    ischemic_heart_disease: Optional[bool] = None
    chronic_kidney_disease: Optional[bool] = None
    chronic_dialysis: Optional[bool] = None
    chronic_liver_disease: Optional[bool] = None
    cirrhosis: Optional[bool] = None
    copd: Optional[bool] = None
    active_malignancy: Optional[bool] = None
    immunosuppression: Optional[bool] = None
    obesity: Optional[bool] = None
    hypertriglyceridemia: Optional[bool] = None
    gallstones: Optional[bool] = None
    chronic_pancreatitis: Optional[bool] = None
    smoking_status: Optional[str] = None
    alcohol_status: Optional[str] = None
    home_medications: list[str] = Field(default_factory=list)
    medication_reconciliation_completed: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def normalise_units_and_aliases(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        if data.get("temperature") is None and data.get("temperature_c") is not None:
            data["temperature"] = data["temperature_c"]
        if data.get("bilirubin_total") is None and data.get("bilirubin") is not None:
            data["bilirubin_total"] = data["bilirubin"]
        if data.get("bun") is None and data.get("urea_mmol_l") is not None:
            data["bun"] = float(data["urea_mmol_l"]) * 2.801
        if data.get("creatinine") is None and data.get("creatinine_umol_l") is not None:
            data["creatinine"] = float(data["creatinine_umol_l"]) / 88.4
        if data.get("glucose") is None and data.get("glucose_mmol_l") is not None:
            data["glucose"] = float(data["glucose_mmol_l"]) * 18.018
        if data.get("calcium") is None and data.get("calcium_mmol_l") is not None:
            data["calcium"] = float(data["calcium_mmol_l"]) * 4.008
        if data.get("albumin") is None and data.get("albumin_g_l") is not None:
            data["albumin"] = float(data["albumin_g_l"]) / 10.0
        if data.get("bilirubin_total") is None and data.get("bilirubin_total_umol_l") is not None:
            data["bilirubin_total"] = float(data["bilirubin_total_umol_l"]) / 17.1
        if data.get("triglycerides") is None and data.get("triglycerides_mmol_l") is not None:
            data["triglycerides"] = float(data["triglycerides_mmol_l"]) * 88.57
        if data.get("bmi") is None and data.get("height_cm") and data.get("weight_kg"):
            height_m = float(data["height_cm"]) / 100.0
            data["bmi"] = float(data["weight_kg"]) / (height_m * height_m)
        if data.get("mean_arterial_pressure") is None and data.get("systolic_bp") is not None and data.get("diastolic_bp") is not None:
            data["mean_arterial_pressure"] = (
                float(data["systolic_bp"]) + 2.0 * float(data["diastolic_bp"])
            ) / 3.0
        return data

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        full = super().model_dump(*args, **kwargs)
        return {name: full.get(name) for name in LEGACY_MODEL_FIELDS}

    def hospital_data_dict(self) -> dict[str, Any]:
        return super().model_dump(exclude_none=False)

    def core_data_summary(self) -> dict[str, Any]:
        full = self.hospital_data_dict()
        present: list[str] = []
        missing: list[str] = []
        for group, alternatives in CORE_GROUPS.items():
            if any(full.get(field) is not None for field in alternatives):
                present.append(group)
            else:
                missing.append(group)
        count = len(present)
        level = "high" if count >= 12 else "intermediate" if count >= 8 else "data_sparse"
        return {
            "core_present": count,
            "core_total": len(CORE_GROUPS),
            "completeness_fraction": round(count / len(CORE_GROUPS), 4),
            "completeness_level": level,
            "present_groups": present,
            "missing_groups": missing,
        }

    def enzyme_criterion_met(self) -> bool:
        lipase_ok = self.lipase is not None and self.lipase_uln is not None and self.lipase >= 3 * self.lipase_uln
        amylase_ok = self.amylase is not None and self.amylase_uln is not None and self.amylase >= 3 * self.amylase_uln
        return bool(lipase_ok or amylase_ok)

    def research_eligibility_summary(self) -> dict[str, Any]:
        missing_essential: list[str] = []
        if not self.encounter_id:
            missing_essential.append("encounter_id")
        if self.admission_time is None:
            missing_essential.append("admission_time")
        if self.age is None:
            missing_essential.append("age")
        elif self.age < 18:
            missing_essential.append("adult_age_required")
        if self.acute_pancreatitis_diagnosis is not True:
            missing_essential.append("documented_acute_pancreatitis_diagnosis")
        if not self.enzyme_criterion_met():
            missing_essential.append("lipase_or_amylase_at_least_3x_uln")
        if self.outcome_followup_available is not True:
            missing_essential.append("outcome_followup_beyond_48h")

        core = self.core_data_summary()
        full = self.hospital_data_dict()
        vitals_present = sum(full.get(name) is not None for name in VITAL_GROUPS)
        labs_present = sum(
            any(full.get(field) is not None for field in CORE_GROUPS[group])
            for group in LAB_GROUPS
        )
        sufficient_predictors = core["core_present"] >= 8 and vitals_present > 0 and labs_present > 0
        eligible = not missing_essential and sufficient_predictors
        return {
            "eligible_for_primary_model_development": eligible,
            "missing_essential": missing_essential,
            "sufficient_predictor_coverage": sufficient_predictors,
            "vital_groups_present": vitals_present,
            "laboratory_groups_present": labs_present,
            **core,
            "imaging_required": False,
            "rule_note": "Do not exclude solely because five optional variables are missing.",
        }


class PredictionOutput(BaseModel):
    severe_ap_probability: Optional[float] = None
    threshold_used: float = 0.5
    risk_group: Optional[str] = None
    model_version: str = "1.1.0"
    warning: str = RESEARCH_WARNING
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.1.0"
    warning: str = RESEARCH_WARNING


class CamelionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    patient_id: Optional[str] = Field(None, description="Camelion patient MRN; routing only")
    encounter_id: Optional[str] = Field(None, description="Camelion encounter ID")


class CamelionPredictionResponse(BaseModel):
    encounter_id: Optional[str] = None
    severe_ap_probability: Optional[float] = None
    risk_group: Optional[str] = Field(None, description="low | intermediate | high")
    fields_used: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    model_version: str = "1.1.0"
    warning: str = RESEARCH_WARNING
    error: Optional[str] = None
