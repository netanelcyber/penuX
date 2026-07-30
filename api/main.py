"""FastAPI research-only prediction endpoint for SAP severity.

RESEARCH USE ONLY. Not validated for clinical use.
Do not use for patient-care decisions.

Integration endpoints:
  POST /predict          — plain JSON (AdmissionInput)
  POST /fhir/predict     — FHIR R4 Bundle (Patient + Observations, LOINC coded)
  POST /camelion/predict — Camelion (קמיליון) HIS native JSON
"""
import json
import math
import os
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

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
        {"name": "models",    "description": "Model evaluation / sweep results"},
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
# Model sweep — serves the pre-computed 294-model hyperparameter sweep
# results (scripts/model_sweep_271.py), real 5-fold-CV numbers, no
# fabricated data. Read-only; the sweep itself is not re-run per request.
# ---------------------------------------------------------------------------

class ModelSweepResult(BaseModel):
    name: str
    auroc_mean: float
    auroc_std: Optional[float] = None
    auprc_mean: float
    f1_mean: float
    accuracy_mean: float
    fit_seconds: Optional[float] = None
    status: str


class ModelSweepResponse(BaseModel):
    dataset: str
    n_samples: int
    n_features: int
    positive_rate: float
    cv_folds: int
    n_configs_attempted: int
    n_configs_succeeded: int
    n_configs_failed: int
    total_runtime_seconds: float
    results: list[ModelSweepResult] = Field(description="Top-N configurations ranked by AUROC")
    caveats: list[str] = Field(description="Important context — different dataset than the primary manuscript cohort, and a known label-direction discrepancy in the source data")


_SWEEP_FILE = Path(__file__).resolve().parent.parent / "models" / "model_sweep_271_results.json"
_SWEEP_CAVEATS = [
    "This sweep was run on data/public_sanitized/ap_multiml_sanitized.csv "
    "(Guilin Medical University, 2016-2024, n=1289) — NOT the primary n=722 "
    "Atlanta-2012-labeled cohort used by /predict and the manuscript. "
    "Results here are exploratory/supplementary and not directly comparable "
    "to the primary model's reported performance.",
    "This dataset's own SOURCES.md documents label=1 as the minority SAP "
    "class (204/15.8%), but the actual CSV has label=1 on the majority "
    "(1085 rows) — a real discrepancy between the source's documentation "
    "and its data. AUROC/AUPRC are mathematically symmetric to which class "
    "is called positive, so the discrimination scores are valid, but which "
    "class means 'severe' is unverified.",
]


@app.get(
    "/models/sweep",
    response_model=ModelSweepResponse,
    tags=["models"],
    summary="294-model hyperparameter sweep results (exploratory, supplementary)",
    description=(
        "Serves the pre-computed results of a 294-configuration hyperparameter "
        "sweep (16 classical ML families — LogisticRegression, RandomForest, "
        "ExtraTrees, GradientBoosting, AdaBoost, XGBoost, LightGBM, CatBoost, "
        "SVC, MLP, KNN, DecisionTree, Ridge, SGD, GaussianNB, BernoulliNB), "
        "each genuinely 5-fold cross-validated — real numbers, not "
        "fabricated. The sweep itself runs offline via "
        "`scripts/model_sweep_271.py`; this endpoint just serves the "
        "resulting JSON, ranked by AUROC descending.\n\n"
        "⚠️ See the `caveats` field in the response — this used a different "
        "dataset than the primary SAP severity model, and there's a known "
        "label-direction discrepancy in that dataset's documentation."
    ),
    responses={
        404: {"description": "Sweep results file not found — run scripts/model_sweep_271.py first"},
    },
)
def get_model_sweep(top_n: int = 30):
    if not _SWEEP_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Sweep results not found at {_SWEEP_FILE}. Run scripts/model_sweep_271.py first.")

    with open(_SWEEP_FILE) as f:
        data = json.load(f)

    top_n = max(1, min(top_n, len(data["results_ranked_by_auroc"])))

    return ModelSweepResponse(
        dataset=data["dataset"],
        n_samples=data["n_samples"],
        n_features=data["n_features"],
        positive_rate=data["positive_rate"],
        cv_folds=data["cv_folds"],
        n_configs_attempted=data["n_configs_attempted"],
        n_configs_succeeded=data["n_configs_succeeded"],
        n_configs_failed=data["n_configs_failed"],
        total_runtime_seconds=data["total_runtime_seconds"],
        results=data["results_ranked_by_auroc"][:top_n],
        caveats=_SWEEP_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Score distribution — lognormal fit over the 3 primary models' predicted
# probabilities on the real n=722 cohort (data/public_sanitized/ap_model_results.csv)
# ---------------------------------------------------------------------------

_SCORE_RESULTS_FILE = Path(__file__).resolve().parent.parent / "data" / "public_sanitized" / "ap_model_results.csv"
_SCORE_MODELS = {
    "sap_prob": "Severe Acute Pancreatitis (SAP) risk",
    "sep_prob": "Sepsis risk",
    "panc_prob": "Pancreatic complication risk",
}


class LognormalFit(BaseModel):
    shape: float = Field(description="Lognormal shape parameter (sigma of the underlying normal)")
    loc: float = Field(description="Location parameter (fixed at 0 — scipy convention for a 2-param lognormal fit)")
    scale: float = Field(description="Scale parameter — exp(mu) of the underlying normal")


class ModelScoreDistribution(BaseModel):
    model_key: str
    model_label: str
    n: int
    mean: float
    median: float
    std: float
    lognormal_fit: LognormalFit
    histogram_bin_edges: list[float]
    histogram_counts: list[int]
    pdf_x: list[float] = Field(description="X values (probability, 0-1) for the fitted lognormal PDF curve")
    pdf_y: list[float] = Field(description="Fitted lognormal PDF density at each pdf_x")


class ScoreDistributionResponse(BaseModel):
    dataset: str
    n_patients: int
    models: list[ModelScoreDistribution]
    caveats: list[str]


_SCORE_DIST_CAVEATS = [
    "Probabilities are bounded in (0, 1); a lognormal is fit as a descriptive "
    "approximation of the right-skewed shape (common for clinical risk scores), "
    "not a claim that the true generating process is lognormal — treat the fit "
    "as illustrative, not a statistical test result.",
    "These are pre-computed predictions on the primary n=722 Atlanta-2012-labeled "
    "cohort (data/public_sanitized/ap_model_results.csv), one score per patient "
    "per model — not a live re-run of /predict.",
]


def _compute_score_distributions() -> tuple[pd.DataFrame, dict[str, ModelScoreDistribution]]:
    if not _SCORE_RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Score results not found at {_SCORE_RESULTS_FILE}")

    from scipy import stats as _stats

    df = pd.read_csv(_SCORE_RESULTS_FILE)
    fits: dict[str, ModelScoreDistribution] = {}

    for key, label in _SCORE_MODELS.items():
        if key not in df.columns:
            continue
        scores = df[key].dropna().clip(lower=1e-6).to_numpy()

        shape, loc, scale = _stats.lognorm.fit(scores, floc=0)

        counts, edges = np.histogram(scores, bins=20, range=(0.0, 1.0))
        pdf_x = np.linspace(0.001, 1.0, 200)
        pdf_y = _stats.lognorm.pdf(pdf_x, shape, loc=loc, scale=scale)

        fits[key] = ModelScoreDistribution(
            model_key=key,
            model_label=label,
            n=len(scores),
            mean=round(float(np.mean(scores)), 4),
            median=round(float(np.median(scores)), 4),
            std=round(float(np.std(scores)), 4),
            lognormal_fit=LognormalFit(shape=round(float(shape), 4), loc=round(float(loc), 4), scale=round(float(scale), 4)),
            histogram_bin_edges=[round(float(e), 4) for e in edges],
            histogram_counts=[int(c) for c in counts],
            pdf_x=[round(float(x), 4) for x in pdf_x],
            pdf_y=[round(float(y), 4) for y in pdf_y],
        )

    return df, fits


@app.get(
    "/models/score-distribution",
    response_model=ScoreDistributionResponse,
    tags=["models"],
    summary="Predicted-score distribution across all 3 models, with lognormal fit",
    description=(
        "Returns the distribution of predicted risk probabilities for all three "
        "PenuX prediction targets — SAP severity, sepsis risk, and pancreatic "
        "complication risk — computed on the real n=722 primary cohort, each "
        "fit to a lognormal distribution for visualization. See `/models/"
        "score-distribution.png` for a rendered chart of the same data.\n\n"
        "⚠️ See `caveats` — a lognormal is a descriptive fit to bounded (0,1) "
        "probabilities, not a formal distributional claim."
    ),
)
def get_score_distribution():
    _, fits = _compute_score_distributions()
    return ScoreDistributionResponse(
        dataset="data/public_sanitized/ap_model_results.csv",
        n_patients=len(pd.read_csv(_SCORE_RESULTS_FILE)),
        models=list(fits.values()),
        caveats=_SCORE_DIST_CAVEATS,
    )


@app.get(
    "/models/score-distribution.png",
    tags=["models"],
    summary="Predicted-score distribution chart (PNG) — histogram + lognormal fit",
    description=(
        "Renders the same data as `/models/score-distribution` as a PNG image: "
        "one panel per model (SAP severity, sepsis risk, pancreatic risk), each "
        "showing a histogram of predicted probabilities with the fitted "
        "lognormal PDF overlaid."
    ),
)
def get_score_distribution_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fastapi.responses import Response

    df, fits = _compute_score_distributions()

    fig, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 4))
    if len(fits) == 1:
        axes = [axes]

    for ax, (key, fit) in zip(axes, fits.items()):
        scores = df[key].dropna().clip(lower=1e-6)
        ax.hist(scores, bins=20, range=(0.0, 1.0), density=True, alpha=0.5,
                color="#4A6C8C", edgecolor="white", label="Observed")
        ax.plot(fit.pdf_x, fit.pdf_y, color="#B4432A", linewidth=2, label="Lognormal fit")
        ax.set_title(fit.model_label, fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle(f"PenuX predicted-score distributions (n={len(df)}, lognormal fit)", fontsize=12)
    fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# lognormal(p) - lognormal(1-p) — asymmetry analysis
#
# For each model, fit a lognormal to p (the predicted probability) and a
# separate lognormal to 1-p (its complement), then take the pointwise
# difference of the two fitted PDFs over x in (0,1). Because lognormal is
# not symmetric under x -> 1-x, this difference curve is a diagnostic for
# how asymmetric the predicted-score distribution really is: a curve at
# ~0 everywhere would mean p and 1-p happen to be fit almost as mirror
# images of each other; the actual shape here (large positive lobe at low
# x, negative lobe at high x) reflects that most predicted probabilities
# sit well below 0.5, so lognormal(p) concentrates near 0 while
# lognormal(1-p) concentrates near 1.
# ---------------------------------------------------------------------------

class LognormalDiffCurve(BaseModel):
    model_key: str
    model_label: str
    fit_p: LognormalFit
    fit_one_minus_p: LognormalFit
    x: list[float] = Field(description="X values (0-1) shared by both PDFs and their difference")
    pdf_p: list[float] = Field(description="Fitted lognormal PDF of p, evaluated at x")
    pdf_one_minus_p: list[float] = Field(description="Fitted lognormal PDF of (1-p), evaluated at x")
    diff: list[float] = Field(description="pdf_p(x) - pdf_one_minus_p(x) — the asymmetry curve")
    max_abs_diff: float = Field(description="max(|diff|) over x — a single-number asymmetry magnitude")


class LognormalDiffResponse(BaseModel):
    dataset: str
    n_patients: int
    curves: list[LognormalDiffCurve]
    caveats: list[str]


def _compute_lognormal_diffs() -> tuple[pd.DataFrame, dict[str, LognormalDiffCurve]]:
    if not _SCORE_RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Score results not found at {_SCORE_RESULTS_FILE}")

    from scipy import stats as _stats

    df = pd.read_csv(_SCORE_RESULTS_FILE)
    curves: dict[str, LognormalDiffCurve] = {}
    x = np.linspace(0.001, 0.999, 200)

    for key, label in _SCORE_MODELS.items():
        if key not in df.columns:
            continue
        p = df[key].dropna().clip(lower=1e-6, upper=1 - 1e-6).to_numpy()
        one_minus_p = 1.0 - p

        shape_p, loc_p, scale_p = _stats.lognorm.fit(p, floc=0)
        shape_q, loc_q, scale_q = _stats.lognorm.fit(one_minus_p, floc=0)

        pdf_p = _stats.lognorm.pdf(x, shape_p, loc=loc_p, scale=scale_p)
        pdf_q = _stats.lognorm.pdf(x, shape_q, loc=loc_q, scale=scale_q)
        diff = pdf_p - pdf_q

        curves[key] = LognormalDiffCurve(
            model_key=key,
            model_label=label,
            fit_p=LognormalFit(shape=round(float(shape_p), 4), loc=round(float(loc_p), 4), scale=round(float(scale_p), 4)),
            fit_one_minus_p=LognormalFit(shape=round(float(shape_q), 4), loc=round(float(loc_q), 4), scale=round(float(scale_q), 4)),
            x=[round(float(v), 4) for v in x],
            pdf_p=[round(float(v), 4) for v in pdf_p],
            pdf_one_minus_p=[round(float(v), 4) for v in pdf_q],
            diff=[round(float(v), 4) for v in diff],
            max_abs_diff=round(float(np.max(np.abs(diff))), 4),
        )

    return df, curves


@app.get(
    "/models/score-distribution/lognormal-diff",
    response_model=LognormalDiffResponse,
    tags=["models"],
    summary="lognormal(p) - lognormal(1-p) asymmetry curve, per model",
    description=(
        "For each of the 3 prediction targets, fits a lognormal to the "
        "predicted probability p and a separate lognormal to its complement "
        "1-p, then returns the pointwise difference of the two fitted PDFs "
        "over x in (0,1) — a diagnostic for how asymmetric the predicted-"
        "score distribution is around 0.5. `max_abs_diff` gives a single-"
        "number summary per model. See `/models/score-distribution/"
        "lognormal-diff.png` for a rendered chart.\n\n"
        "⚠️ See `caveats` — same lognormal-as-descriptive-fit caveat as "
        "`/models/score-distribution`, plus: this compares two *independent* "
        "fits (fit to p, fit to 1-p), not a transform of one fit — the "
        "asymmetry mostly reflects that lognormal is not symmetric under "
        "x -> 1-x combined with predicted probabilities skewing low."
    ),
)
def get_lognormal_diff():
    _, curves = _compute_lognormal_diffs()
    return LognormalDiffResponse(
        dataset="data/public_sanitized/ap_model_results.csv",
        n_patients=len(pd.read_csv(_SCORE_RESULTS_FILE)),
        curves=list(curves.values()),
        caveats=_SCORE_DIST_CAVEATS + [
            "This compares two independently-fit lognormals (one to p, one to "
            "1-p) — the difference curve is not a closed-form transform, it's "
            "computed pointwise from both fitted PDFs."
        ],
    )


@app.get(
    "/models/score-distribution/lognormal-diff.png",
    tags=["models"],
    summary="lognormal(p) - lognormal(1-p) asymmetry chart (PNG)",
    description=(
        "Renders the same data as `/models/score-distribution/lognormal-diff` "
        "as a PNG: one panel per model, each showing both fitted PDFs "
        "(lognormal(p) and lognormal(1-p)) plus their difference curve."
    ),
)
def get_lognormal_diff_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fastapi.responses import Response

    _, curves = _compute_lognormal_diffs()

    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.2))
    if len(curves) == 1:
        axes = [axes]

    for ax, (key, c) in zip(axes, curves.items()):
        ax.plot(c.x, c.pdf_p, color="#4A6C8C", linewidth=1.8, label="lognormal(p)")
        ax.plot(c.x, c.pdf_one_minus_p, color="#9A7A1F", linewidth=1.8, label="lognormal(1-p)")
        ax.plot(c.x, c.diff, color="#B4432A", linewidth=2, linestyle="--", label="diff")
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.set_title(f"{c.model_label}\nmax|diff|={c.max_abs_diff}", fontsize=9.5)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7.5)

    fig.suptitle("lognormal(p) - lognormal(1-p) per model", fontsize=12)
    fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Brier score + calibration curve — proper, established alternatives to the
# lognormal-diff asymmetry curve above for actually assessing model quality.
#
# IMPORTANT LABEL CAVEAT: the only ground-truth outcome in this cohort is
# the SAP-severity label (true_label in ap_model_results.csv). The sepsis-
# risk and pancreatic-complication-risk models are evaluated against this
# SAME label, because data/public_sanitized/ap_sepsis_risk.csv and
# ap_pancreatic_sepsis_risk.csv carry an identically-named/valued
# "sap_label" column (row-for-row identical to true_label — verified by
# direct comparison), not an independently adjudicated sepsis or
# pancreatic-complication diagnosis. So "sep_prob calibration" here really
# means "how well does the sepsis-risk score predict SAP severity" — not
# a validated sepsis-outcome calibration.
# ---------------------------------------------------------------------------

class CalibrationBin(BaseModel):
    bin_mean_predicted: float
    bin_fraction_positive: float
    bin_count: int


class ModelCalibration(BaseModel):
    model_key: str
    model_label: str
    n: int
    brier_score: float = Field(description="Mean squared error between predicted probability and true outcome (0=perfect, 0.25=uninformative at prevalence 0.5)")
    prevalence: float = Field(description="Fraction of patients with the positive outcome in this cohort")
    bins: list[CalibrationBin]


class CalibrationResponse(BaseModel):
    dataset: str
    n_patients: int
    label_caveat: str
    models: list[ModelCalibration]
    caveats: list[str]


_CALIBRATION_LABEL_CAVEAT = (
    "All three probabilities are scored against the SAME ground-truth label "
    "(SAP severity, true_label in ap_model_results.csv) — sep_prob and "
    "panc_prob do NOT have independent sepsis/pancreatic-complication "
    "outcome labels in this cohort (their source CSVs carry an identically-"
    "valued 'sap_label' column, confirmed row-for-row identical to "
    "true_label). Read 'sep_prob calibration' as 'how well does the "
    "sepsis-risk score predict SAP severity', not a validated sepsis-"
    "outcome calibration."
)


def _compute_calibration(n_bins: int = 10) -> tuple[pd.DataFrame, dict[str, ModelCalibration]]:
    if not _SCORE_RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Score results not found at {_SCORE_RESULTS_FILE}")

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    df = pd.read_csv(_SCORE_RESULTS_FILE)
    if "true_label" not in df.columns:
        raise HTTPException(status_code=500, detail="true_label column missing from ap_model_results.csv")

    y = df["true_label"].to_numpy()
    results: dict[str, ModelCalibration] = {}

    for key, label in _SCORE_MODELS.items():
        if key not in df.columns:
            continue
        p = df[key].dropna().clip(0.0, 1.0).to_numpy()
        y_aligned = df.loc[df[key].notna(), "true_label"].to_numpy()

        brier = brier_score_loss(y_aligned, p)
        prob_true, prob_pred = calibration_curve(y_aligned, p, n_bins=n_bins, strategy="quantile")

        # calibration_curve doesn't return per-bin counts directly — recompute via digitize on the same quantile edges.
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        bin_idx = np.digitize(p, edges[1:-1], right=True)
        counts = [int(np.sum(bin_idx == i)) for i in range(len(prob_true))]

        results[key] = ModelCalibration(
            model_key=key,
            model_label=label,
            n=len(p),
            brier_score=round(float(brier), 4),
            prevalence=round(float(np.mean(y_aligned)), 4),
            bins=[
                CalibrationBin(bin_mean_predicted=round(float(pp), 4), bin_fraction_positive=round(float(pt), 4), bin_count=c)
                for pp, pt, c in zip(prob_pred, prob_true, counts)
            ],
        )

    return df, results


@app.get(
    "/models/calibration",
    response_model=CalibrationResponse,
    tags=["models"],
    summary="Brier score + calibration curve for all 3 models",
    description=(
        "Proper model-quality diagnostics, as an alternative to the "
        "lognormal-diff asymmetry curve: Brier score (mean squared error "
        "between predicted probability and true outcome) and a quantile-"
        "binned calibration curve (reliability diagram: mean predicted "
        "probability vs. observed fraction positive, per bin) for each of "
        "the 3 prediction targets.\n\n"
        "⚠️ **Read `label_caveat` first** — sep_prob and panc_prob are "
        "evaluated against the SAP-severity label, not independent outcome "
        "labels (none exist in this cohort). See `/models/calibration.png` "
        "for a rendered reliability diagram."
    ),
)
def get_calibration(n_bins: int = 10):
    _, results = _compute_calibration(n_bins=n_bins)
    return CalibrationResponse(
        dataset="data/public_sanitized/ap_model_results.csv",
        n_patients=len(pd.read_csv(_SCORE_RESULTS_FILE)),
        label_caveat=_CALIBRATION_LABEL_CAVEAT,
        models=list(results.values()),
        caveats=[
            "Brier score and calibration curve are computed on the full n=722 "
            "cohort, not a held-out test split — this reflects in-sample "
            "calibration, not generalization to new patients.",
        ],
    )


@app.get(
    "/models/calibration.png",
    tags=["models"],
    summary="Calibration curve (reliability diagram) chart, PNG",
    description=(
        "Renders the same data as `/models/calibration` as a PNG reliability "
        "diagram: one panel per model, mean predicted probability (x) vs. "
        "observed fraction positive (y), with the perfect-calibration "
        "diagonal for reference, and Brier score annotated per panel."
    ),
)
def get_calibration_png(n_bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fastapi.responses import Response

    _, results = _compute_calibration(n_bins=n_bins)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    for ax, (key, r) in zip(axes, results.items()):
        xs = [b.bin_mean_predicted for b in r.bins]
        ys = [b.bin_fraction_positive for b in r.bins]
        sizes = [max(15, b.bin_count) for b in r.bins]
        ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1, label="Perfect calibration")
        ax.scatter(xs, ys, s=sizes, color="#4A6C8C", alpha=0.8, label="Observed (per bin)")
        ax.plot(xs, ys, color="#4A6C8C", linewidth=1, alpha=0.6)
        ax.set_title(f"{r.model_label}\nBrier={r.brier_score}", fontsize=9.5)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7.5)

    fig.suptitle("Calibration (reliability diagram) — see label_caveat in /models/calibration", fontsize=11)
    fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


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


# ---------------------------------------------------------------------------
# Clinical deterioration risk — NEWS2 (National Early Warning Score 2)
# ---------------------------------------------------------------------------

class DeteriorationInput(BaseModel):
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate /min")
    spo2: Optional[float] = Field(None, description="SpO2 %")
    on_supplemental_oxygen: Optional[bool] = Field(None, description="Patient is on any supplemental oxygen")
    systolic_bp: Optional[float] = Field(None, description="Systolic BP mmHg")
    heart_rate: Optional[float] = Field(None, description="Heart rate bpm")
    consciousness_altered: Optional[bool] = Field(
        None, description="Altered consciousness — anything other than fully Alert on AVPU (Confusion/Voice/Pain/Unresponsive)"
    )
    temperature_c: Optional[float] = Field(None, description="Body temperature °C")


class DeteriorationOutput(BaseModel):
    news2_score: int = Field(description="Total NEWS2 score (0-20)")
    risk_group: str = Field(description="low | low-medium | medium | high")
    component_scores: dict[str, int] = Field(description="Per-parameter NEWS2 sub-scores")
    escalation: str = Field(description="Suggested monitoring/escalation per RCP NEWS2 guidance")
    warning: str = RESEARCH_WARNING


def _news2_score(inp: "DeteriorationInput") -> tuple[int, str, dict, str]:
    """NEWS2 (Royal College of Physicians) — standard, validated deterioration
    early-warning score computed from routine vital signs. Any single
    parameter scoring 3 escalates risk to at least 'medium' regardless of
    total, per RCP guidance (a single very abnormal vital sign matters even
    if the sum is otherwise low).
    """
    scores: dict[str, int] = {}

    if inp.respiratory_rate is not None:
        rr = inp.respiratory_rate
        if rr <= 8: scores["respiratory_rate"] = 3
        elif rr <= 11: scores["respiratory_rate"] = 1
        elif rr <= 20: scores["respiratory_rate"] = 0
        elif rr <= 24: scores["respiratory_rate"] = 2
        else: scores["respiratory_rate"] = 3

    if inp.spo2 is not None:
        s = inp.spo2
        if s <= 91: scores["spo2"] = 3
        elif s <= 93: scores["spo2"] = 2
        elif s <= 95: scores["spo2"] = 1
        else: scores["spo2"] = 0

    if inp.on_supplemental_oxygen is not None:
        scores["supplemental_oxygen"] = 2 if inp.on_supplemental_oxygen else 0

    if inp.systolic_bp is not None:
        bp = inp.systolic_bp
        if bp <= 90: scores["systolic_bp"] = 3
        elif bp <= 100: scores["systolic_bp"] = 2
        elif bp <= 110: scores["systolic_bp"] = 1
        elif bp <= 219: scores["systolic_bp"] = 0
        else: scores["systolic_bp"] = 3

    if inp.heart_rate is not None:
        hr = inp.heart_rate
        if hr <= 40: scores["heart_rate"] = 3
        elif hr <= 50: scores["heart_rate"] = 1
        elif hr <= 90: scores["heart_rate"] = 0
        elif hr <= 110: scores["heart_rate"] = 1
        elif hr <= 130: scores["heart_rate"] = 2
        else: scores["heart_rate"] = 3

    if inp.consciousness_altered is not None:
        scores["consciousness"] = 3 if inp.consciousness_altered else 0

    if inp.temperature_c is not None:
        t = inp.temperature_c
        if t <= 35.0: scores["temperature"] = 3
        elif t <= 36.0: scores["temperature"] = 1
        elif t <= 38.0: scores["temperature"] = 0
        elif t <= 39.0: scores["temperature"] = 1
        else: scores["temperature"] = 2

    total = sum(scores.values())
    has_single_3 = any(v == 3 for v in scores.values())

    if total >= 7:
        risk = "high"
    elif has_single_3 or total >= 5:
        risk = "medium"
    elif total >= 1:
        risk = "low-medium"
    else:
        risk = "low"

    escalation_map = {
        "low": "Routine monitoring — continue per ward protocol.",
        "low-medium": "Increase monitoring frequency; nurse-led review.",
        "medium": "Urgent review by ward-based clinician; consider critical care outreach.",
        "high": "Emergency assessment — critical care outreach team review, consider transfer to higher-acuity setting.",
    }

    return total, risk, scores, escalation_map[risk]


@app.post(
    "/predict/deterioration",
    response_model=DeteriorationOutput,
    tags=["predict"],
    summary="Clinical deterioration risk — NEWS2 early warning score",
    description=(
        "Computes **NEWS2** (National Early Warning Score 2, Royal College of "
        "Physicians), a standard, clinically validated score for detecting "
        "patient deterioration from routine vital signs.\n\n"
        "All fields are optional — supply whatever vitals are available; the "
        "score is a sum over whichever parameters are provided.\n\n"
        "**Risk groups:** low (0) · low-medium (1-4) · medium (5-6, or any "
        "single parameter scoring 3) · high (≥7)\n\n"
        "⚠️ RESEARCH USE ONLY — not a substitute for clinical judgment or "
        "your institution's early-warning-score protocol."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "high_risk": {
                            "summary": "High — multiple abnormal vitals",
                            "value": {
                                "respiratory_rate": 26, "spo2": 90, "on_supplemental_oxygen": True,
                                "systolic_bp": 85, "heart_rate": 135, "consciousness_altered": True,
                                "temperature_c": 39.4,
                            },
                        },
                        "medium_single_red_flag": {
                            "summary": "Medium — one severely abnormal vital (single-3 rule)",
                            "value": {"respiratory_rate": 16, "spo2": 97, "heart_rate": 75, "systolic_bp": 88},
                        },
                        "low_risk": {
                            "summary": "Low — normal vitals",
                            "value": {"respiratory_rate": 16, "spo2": 98, "systolic_bp": 120, "heart_rate": 75, "temperature_c": 36.8},
                        },
                    }
                }
            }
        }
    },
)
def predict_deterioration(data: DeteriorationInput):
    total, risk, component_scores, escalation = _news2_score(data)
    return DeteriorationOutput(
        news2_score=total,
        risk_group=risk,
        component_scores=component_scores,
        escalation=escalation,
    )


# ---------------------------------------------------------------------------
# 30-day in-hospital mortality risk — routine admission variables
# ---------------------------------------------------------------------------

class MortalityInput(BaseModel):
    age: Optional[float] = Field(None, description="Age in years")
    comorbidity_count: Optional[int] = Field(
        None, description="Number of major comorbidities (e.g. CHF, COPD, cirrhosis, CKD, active malignancy, diabetes with end-organ damage)"
    )
    systolic_bp: Optional[float] = Field(None, description="Systolic BP mmHg")
    heart_rate: Optional[float] = Field(None, description="Heart rate bpm")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate /min")
    temperature_c: Optional[float] = Field(None, description="Body temperature °C")
    spo2: Optional[float] = Field(None, description="SpO2 %")
    consciousness_altered: Optional[bool] = Field(None, description="Altered consciousness (anything other than fully Alert)")
    creatinine: Optional[float] = Field(None, description="Creatinine mg/dL")
    bun: Optional[float] = Field(None, description="BUN mg/dL")
    bilirubin_total: Optional[float] = Field(None, description="Total bilirubin mg/dL")
    albumin: Optional[float] = Field(None, description="Albumin g/dL")
    platelets: Optional[float] = Field(None, description="Platelets ×10³/µL")
    lactate: Optional[float] = Field(None, description="Lactate mmol/L")
    wbc: Optional[float] = Field(None, description="WBC ×10⁹/L")


class MortalityOutput(BaseModel):
    mortality_risk_probability: float = Field(description="Estimated 30-day in-hospital mortality risk, 0-1")
    risk_group: str = Field(description="low | moderate | high | critical")
    contributing_factors: list[str] = Field(description="Variables that drove the risk estimate upward")
    warning: str = RESEARCH_WARNING


def _mortality_score(inp: "MortalityInput") -> tuple[float, str, list[str]]:
    """Logistic combination of age, comorbidity burden, hemodynamic
    instability, and organ-dysfunction labs — a lightweight, explainable
    stand-in for a full APACHE II / SAPS-style score, intended for research
    use on routine admission-panel variables (not curve-fit to any specific
    cohort).
    """
    factors: list[str] = []
    logit = -4.2  # anchors ~1.5% baseline risk when nothing else is abnormal

    if inp.age is not None:
        if inp.age > 65:
            age_term = 0.03 * (inp.age - 65)
            logit += age_term
            if inp.age > 75:
                factors.append(f"Advanced age ({inp.age:.0f})")

    if inp.comorbidity_count is not None and inp.comorbidity_count > 0:
        logit += 0.35 * inp.comorbidity_count
        factors.append(f"Comorbidity burden ({inp.comorbidity_count} major comorbidities)")

    if inp.systolic_bp is not None and inp.systolic_bp < 90:
        logit += 0.9
        factors.append(f"Hypotension (SBP {inp.systolic_bp} < 90)")

    if inp.heart_rate is not None and inp.heart_rate > 120:
        logit += 0.4
        factors.append(f"Severe tachycardia (HR {inp.heart_rate} > 120)")

    if inp.respiratory_rate is not None and inp.respiratory_rate > 24:
        logit += 0.4
        factors.append(f"Tachypnea (RR {inp.respiratory_rate} > 24)")

    if inp.temperature_c is not None and (inp.temperature_c < 35.0 or inp.temperature_c > 39.5):
        logit += 0.5
        factors.append(f"Temperature dysregulation ({inp.temperature_c}°C)")

    if inp.spo2 is not None and inp.spo2 < 90:
        logit += 0.6
        factors.append(f"Hypoxia (SpO2 {inp.spo2}% < 90%)")

    if inp.consciousness_altered:
        logit += 0.8
        factors.append("Altered consciousness")

    if inp.creatinine is not None and inp.creatinine > 2.0:
        logit += 0.5
        factors.append(f"Renal dysfunction (Creatinine {inp.creatinine} > 2.0)")

    if inp.bun is not None and inp.bun > 40:
        logit += 0.3
        factors.append(f"Elevated BUN ({inp.bun} > 40)")

    if inp.bilirubin_total is not None and inp.bilirubin_total > 3.0:
        logit += 0.4
        factors.append(f"Hepatic dysfunction (Bilirubin {inp.bilirubin_total} > 3.0)")

    if inp.albumin is not None and inp.albumin < 2.5:
        logit += 0.5
        factors.append(f"Hypoalbuminemia (Albumin {inp.albumin} < 2.5)")

    if inp.platelets is not None and inp.platelets < 100:
        logit += 0.4
        factors.append(f"Thrombocytopenia (Platelets {inp.platelets} < 100)")

    if inp.lactate is not None and inp.lactate > 2.0:
        logit += 0.5 * min(inp.lactate / 2.0, 3.0)
        factors.append(f"Elevated lactate ({inp.lactate} mmol/L)")

    if inp.wbc is not None and (inp.wbc > 15.0 or inp.wbc < 3.0):
        logit += 0.3
        factors.append(f"Leukocyte abnormality (WBC {inp.wbc})")

    proba = round(1.0 / (1.0 + math.exp(-logit)), 4)

    if proba < 0.05:
        risk = "low"
    elif proba < 0.20:
        risk = "moderate"
    elif proba < 0.50:
        risk = "high"
    else:
        risk = "critical"

    return proba, risk, factors


@app.post(
    "/predict/mortality",
    response_model=MortalityOutput,
    tags=["predict"],
    summary="30-day in-hospital mortality risk — routine admission variables",
    description=(
        "Estimates 30-day in-hospital mortality risk from routine admission "
        "variables: age, comorbidity burden, hemodynamic/respiratory "
        "instability, consciousness, and organ-dysfunction labs (renal, "
        "hepatic, coagulation, lactate).\n\n"
        "All fields are optional — supply whatever is available.\n\n"
        "**Risk groups:** low (<5%) · moderate (5–20%) · high (20–50%) · "
        "critical (>50%)\n\n"
        "This is a lightweight, explainable logistic combination of known "
        "mortality risk factors — **not** a validated score like APACHE II "
        "or SAPS II, and not specific to any single diagnosis.\n\n"
        "⚠️ RESEARCH USE ONLY — not for prognostication in individual "
        "patient care."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "critical": {
                            "summary": "Critical — elderly, multi-organ dysfunction",
                            "value": {
                                "age": 82, "comorbidity_count": 3, "systolic_bp": 82, "heart_rate": 128,
                                "respiratory_rate": 28, "temperature_c": 35.2, "spo2": 87,
                                "consciousness_altered": True, "creatinine": 3.1, "bilirubin_total": 4.2,
                                "albumin": 2.1, "platelets": 78, "lactate": 5.5,
                            },
                        },
                        "moderate": {
                            "summary": "Moderate — older patient, mild organ stress",
                            "value": {"age": 75, "comorbidity_count": 2, "bun": 45, "albumin": 2.4},
                        },
                        "low_risk": {
                            "summary": "Low — young, no comorbidities, normal vitals/labs",
                            "value": {"age": 38, "comorbidity_count": 0, "systolic_bp": 118, "heart_rate": 76, "spo2": 98},
                        },
                    }
                }
            }
        }
    },
)
def predict_mortality(data: MortalityInput):
    proba, risk, factors = _mortality_score(data)
    return MortalityOutput(
        mortality_risk_probability=proba,
        risk_group=risk,
        contributing_factors=factors,
    )


# ---------------------------------------------------------------------------
# SAPS II — Simplified Acute Physiology Score II (Le Gall et al., 1993)
#
# A real, published, externally-validated ICU mortality score, implemented
# faithfully from the original point tables and logistic regression
# equation — unlike /predict/mortality above (a lightweight, unvalidated
# heuristic), this reproduces a citable clinical instrument so the two can
# be compared directly on the same patient.
# ---------------------------------------------------------------------------

class SAPS2Input(BaseModel):
    age: Optional[float] = Field(None, description="Age in years")
    heart_rate: Optional[float] = Field(None, description="Worst (highest deviation from normal) heart rate in 24h, bpm")
    systolic_bp: Optional[float] = Field(None, description="Worst systolic BP in 24h, mmHg")
    temperature_c: Optional[float] = Field(None, description="Worst (highest) temperature in 24h, °C")
    pao2_fio2: Optional[float] = Field(None, description="PaO2/FiO2 ratio — only if mechanically ventilated or on CPAP")
    ventilated_or_cpap: bool = Field(False, description="True if mechanically ventilated or on CPAP (required for PaO2/FiO2 to score)")
    urine_output_l_24h: Optional[float] = Field(None, description="Urine output, liters/24h")
    bun_mg_dl: Optional[float] = Field(None, description="Blood urea nitrogen, mg/dL")
    wbc: Optional[float] = Field(None, description="WBC ×10⁹/L")
    potassium: Optional[float] = Field(None, description="Serum potassium mmol/L")
    sodium: Optional[float] = Field(None, description="Serum sodium mmol/L")
    bicarbonate: Optional[float] = Field(None, description="Serum bicarbonate (HCO3-) mmol/L")
    bilirubin_total: Optional[float] = Field(None, description="Total bilirubin mg/dL")
    gcs: Optional[int] = Field(None, description="Glasgow Coma Scale, 3-15")
    admission_type: Optional[str] = Field(
        None, description="scheduled_surgical | unscheduled_surgical | medical — omit for 0 points"
    )
    chronic_disease: Optional[str] = Field(
        None, description="metastatic_cancer | hematologic_malignancy | aids — omit if none"
    )


class SAPS2Output(BaseModel):
    saps2_score: int = Field(description="Total SAPS II points")
    predicted_mortality_probability: float = Field(description="SAPS II logistic-regression predicted hospital mortality, 0-1")
    point_breakdown: dict[str, int] = Field(description="Points contributed by each scored variable")
    missing_variables: list[str] = Field(description="Variables not supplied — score is a lower bound until they're provided")
    warning: str = RESEARCH_WARNING


def _saps2_score(inp: "SAPS2Input") -> tuple[int, float, dict, list]:
    """Original Le Gall et al. 1993 point tables + logistic regression
    equation: logit = -7.7631 + 0.0737*score + 0.9971*ln(score+1).
    """
    points: dict[str, int] = {}
    missing: list[str] = []

    def band(value, table, label):
        for lo, hi, pts in table:
            if (lo is None or value >= lo) and (hi is None or value < hi):
                points[label] = pts
                return

    if inp.age is not None:
        band(inp.age, [(None, 40, 0), (40, 60, 7), (60, 70, 12), (70, 75, 15), (75, 80, 16), (80, None, 18)], "age")
    else:
        missing.append("age")

    if inp.heart_rate is not None:
        band(inp.heart_rate, [(None, 40, 11), (40, 70, 2), (70, 120, 0), (120, 160, 4), (160, None, 7)], "heart_rate")
    else:
        missing.append("heart_rate")

    if inp.systolic_bp is not None:
        band(inp.systolic_bp, [(None, 70, 13), (70, 100, 5), (100, 200, 0), (200, None, 2)], "systolic_bp")
    else:
        missing.append("systolic_bp")

    if inp.temperature_c is not None:
        points["temperature"] = 3 if inp.temperature_c >= 39.0 else 0
    else:
        missing.append("temperature_c")

    if inp.ventilated_or_cpap and inp.pao2_fio2 is not None:
        band(inp.pao2_fio2, [(None, 100, 11), (100, 200, 9), (200, None, 6)], "pao2_fio2")
    elif inp.ventilated_or_cpap:
        missing.append("pao2_fio2 (ventilated but ratio not supplied)")
    # else: not ventilated -> 0 points, correctly omitted

    if inp.urine_output_l_24h is not None:
        band(inp.urine_output_l_24h, [(None, 0.5, 11), (0.5, 1.0, 4), (1.0, None, 0)], "urine_output")
    else:
        missing.append("urine_output_l_24h")

    if inp.bun_mg_dl is not None:
        band(inp.bun_mg_dl, [(None, 28, 0), (28, 84, 6), (84, None, 10)], "bun")
    else:
        missing.append("bun_mg_dl")

    if inp.wbc is not None:
        band(inp.wbc, [(None, 1, 12), (1, 20, 0), (20, None, 3)], "wbc")
    else:
        missing.append("wbc")

    if inp.potassium is not None:
        band(inp.potassium, [(None, 3, 3), (3, 5, 0), (5, None, 3)], "potassium")
    else:
        missing.append("potassium")

    if inp.sodium is not None:
        band(inp.sodium, [(None, 125, 5), (125, 145, 0), (145, None, 1)], "sodium")
    else:
        missing.append("sodium")

    if inp.bicarbonate is not None:
        band(inp.bicarbonate, [(None, 15, 6), (15, 20, 3), (20, None, 0)], "bicarbonate")
    else:
        missing.append("bicarbonate")

    if inp.bilirubin_total is not None:
        band(inp.bilirubin_total, [(None, 4.0, 0), (4.0, 6.0, 4), (6.0, None, 9)], "bilirubin")
    else:
        missing.append("bilirubin_total")

    if inp.gcs is not None:
        band(inp.gcs, [(None, 6, 26), (6, 9, 13), (9, 11, 7), (11, 14, 5), (14, None, 0)], "gcs")
    else:
        missing.append("gcs")

    admission_points = {"scheduled_surgical": 0, "medical": 6, "unscheduled_surgical": 8}
    if inp.admission_type in admission_points:
        points["admission_type"] = admission_points[inp.admission_type]
    elif inp.admission_type is not None:
        missing.append("admission_type (unrecognized value)")

    chronic_points = {"metastatic_cancer": 9, "hematologic_malignancy": 10, "aids": 17}
    if inp.chronic_disease in chronic_points:
        points["chronic_disease"] = chronic_points[inp.chronic_disease]

    total = sum(points.values())
    logit = -7.7631 + 0.0737 * total + 0.9971 * math.log(total + 1)
    proba = round(1.0 / (1.0 + math.exp(-logit)), 4)

    return total, proba, points, missing


@app.post(
    "/predict/saps2",
    response_model=SAPS2Output,
    tags=["predict"],
    summary="SAPS II — Simplified Acute Physiology Score II (validated ICU mortality score)",
    description=(
        "Computes **SAPS II** (Le Gall et al., 1993) — a real, published, "
        "externally-validated ICU mortality score, using the original point "
        "tables (age, heart rate, systolic BP, temperature, PaO2/FiO2 if "
        "ventilated, urine output, BUN, WBC, potassium, sodium, "
        "bicarbonate, bilirubin, GCS, admission type, chronic disease) and "
        "the original logistic regression equation:\n\n"
        "`logit = -7.7631 + 0.0737 × score + 0.9971 × ln(score + 1)`\n\n"
        "Unlike **`/predict/mortality`** (a lightweight, unvalidated "
        "heuristic on a different, smaller variable set), this reproduces "
        "a citable, externally-validated clinical instrument — run both "
        "endpoints on the same patient to compare a real validated score "
        "against the simpler heuristic.\n\n"
        "All fields are optional; missing variables are listed in the "
        "response and the score is a lower bound until they're supplied — "
        "SAPS II is designed to use the *worst* value of each variable "
        "in the first 24 ICU hours.\n\n"
        "⚠️ RESEARCH USE ONLY — SAPS II was validated on general ICU "
        "populations, not specifically on acute pancreatitis."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "severe": {
                            "summary": "Severe — score ≈ 90, ~97% predicted mortality",
                            "value": {
                                "age": 78, "heart_rate": 128, "systolic_bp": 82, "temperature_c": 39.4,
                                "ventilated_or_cpap": True, "pao2_fio2": 150, "urine_output_l_24h": 0.4,
                                "bun_mg_dl": 90, "wbc": 22, "potassium": 5.5, "sodium": 148,
                                "bicarbonate": 14, "bilirubin_total": 5.0, "gcs": 9,
                                "admission_type": "unscheduled_surgical",
                            },
                        },
                        "mild": {
                            "summary": "Mild — low score, low predicted mortality",
                            "value": {
                                "age": 45, "heart_rate": 90, "systolic_bp": 120, "temperature_c": 37.5,
                                "ventilated_or_cpap": False, "urine_output_l_24h": 1.8, "bun_mg_dl": 18,
                                "wbc": 9, "potassium": 4.0, "sodium": 138, "bicarbonate": 24,
                                "bilirubin_total": 0.8, "gcs": 15, "admission_type": "medical",
                            },
                        },
                    }
                }
            }
        }
    },
)
def predict_saps2(data: SAPS2Input):
    total, proba, points, missing = _saps2_score(data)
    return SAPS2Output(
        saps2_score=total,
        predicted_mortality_probability=proba,
        point_breakdown=points,
        missing_variables=missing,
    )


# ---------------------------------------------------------------------------
# Polynomial-logit risk index — f(x) polynomial -> logistic -> natural log
#
# risk_probability = sigmoid(f(x)),  where f(x) is a degree-2 polynomial
# (linear + quadratic + one interaction term) over standardized lab
# deviations. log_risk_index = ln(risk_probability).
#
# Terminology note: strictly, "logit" is the *inverse* of the logistic/
# sigmoid function (logit(p) = ln(p/(1-p))), while what's applied to f(x)
# here is the logistic/sigmoid transform itself, sigmoid(f(x)) =
# 1/(1+e^-f(x)), which is the conventional way to turn an unbounded
# polynomial score into a (0,1) probability. That probability is what gets
# logged. This is an original, exploratory scoring construct — not a
# validated clinical instrument like SAPS II above.
# ---------------------------------------------------------------------------

class PolynomialLogitInput(BaseModel):
    age: Optional[float] = Field(None, description="Age in years")
    wbc: Optional[float] = Field(None, description="WBC ×10⁹/L")
    crp: Optional[float] = Field(None, description="CRP mg/L")
    creatinine: Optional[float] = Field(None, description="Creatinine mg/dL")
    glucose: Optional[float] = Field(None, description="Glucose mg/dL")
    ldh: Optional[float] = Field(None, description="LDH U/L")
    ast: Optional[float] = Field(None, description="AST U/L")
    hematocrit: Optional[float] = Field(None, description="Hematocrit %")
    calcium: Optional[float] = Field(None, description="Calcium mg/dL")
    albumin: Optional[float] = Field(None, description="Albumin g/dL")


class PolynomialLogitOutput(BaseModel):
    polynomial_score: float = Field(description="f(x) — the raw degree-2 polynomial score, unbounded")
    risk_probability: float = Field(description="sigmoid(f(x)) ∈ (0,1) — the logistic transform of the polynomial score")
    log_risk_index: float = Field(description="ln(risk_probability) — always ≤ 0; closer to 0 means HIGHER risk, more negative means LOWER risk")
    risk_group: str = Field(description="low | intermediate | high, derived from risk_probability")
    terms_used: list[str] = Field(description="Variables that contributed to f(x)")
    warning: str = RESEARCH_WARNING


# (center, scale, "higher_is_worse") per variable — z = (x - center)/scale,
# sign-flipped for variables where a LOW value is the dangerous direction
# (calcium, albumin), so that in every case a larger positive z means more
# abnormal/risk-associated, keeping the polynomial's linear-term signs
# uniformly positive.
_POLY_VARS = {
    "age":        (55.0, 15.0, True),
    "wbc":        (10.0, 4.0,  True),
    "crp":        (80.0, 60.0, True),
    "creatinine": (1.0,  0.4,  True),
    "glucose":    (110.0, 40.0, True),
    "ldh":        (200.0, 80.0, True),
    "ast":        (35.0, 30.0, True),
    "hematocrit": (42.0, 6.0,  True),
    "calcium":    (9.2,  0.8,  False),
    "albumin":    (4.0,  0.6,  False),
}
_POLY_LINEAR_WEIGHT = 0.10
_POLY_QUADRATIC_WEIGHT = 0.02
_POLY_INTERACTION_WEIGHT = 0.02  # wbc x crp — combined inflammatory signal
_POLY_INTERCEPT = -2.5


def _polynomial_logit_score(inp: "PolynomialLogitInput") -> tuple[float, float, float, str, list]:
    data = inp.model_dump()
    z = {}
    terms_used = []

    f_x = _POLY_INTERCEPT
    for name, (center, scale, higher_is_worse) in _POLY_VARS.items():
        value = data.get(name)
        if value is None:
            continue
        raw_z = (value - center) / scale
        z[name] = raw_z if higher_is_worse else -raw_z
        f_x += _POLY_LINEAR_WEIGHT * z[name] + _POLY_QUADRATIC_WEIGHT * (z[name] ** 2)
        terms_used.append(name)

    if "wbc" in z and "crp" in z:
        f_x += _POLY_INTERACTION_WEIGHT * z["wbc"] * z["crp"]
        terms_used.append("wbc×crp interaction")

    risk_probability = 1.0 / (1.0 + math.exp(-f_x))
    log_risk_index = math.log(max(risk_probability, 1e-12))

    if risk_probability < 0.2:
        risk_group = "low"
    elif risk_probability < 0.5:
        risk_group = "intermediate"
    else:
        risk_group = "high"

    return round(f_x, 4), round(risk_probability, 6), round(log_risk_index, 6), risk_group, terms_used


@app.post(
    "/predict/polynomial-logit",
    response_model=PolynomialLogitOutput,
    tags=["predict"],
    summary="Polynomial-logit risk index — ln(sigmoid(f(x))), f(x) a degree-2 polynomial",
    description=(
        "Exploratory scoring construct: builds a degree-2 polynomial `f(x)` "
        "over standardized deviations of routine labs (linear + quadratic "
        "terms per variable, plus a WBC×CRP interaction term), passes it "
        "through the logistic/sigmoid function to get a probability in "
        "(0,1), then returns the natural log of that probability as the "
        "`log_risk_index`.\n\n"
        "`f(x) = b0 + Σ(w·zᵢ + w₂·zᵢ²) + w₃·(z_wbc·z_crp)`, "
        "`risk_probability = sigmoid(f(x))`, "
        "`log_risk_index = ln(risk_probability)`\n\n"
        "**Reading `log_risk_index`:** it is always ≤ 0 (natural log of a "
        "probability ≤ 1). Values closer to 0 mean *higher* risk "
        "(probability near 1); very negative values mean *lower* risk "
        "(probability near 0) — this is the opposite of most clinical "
        "scores where higher-is-worse, so read `risk_probability` or "
        "`risk_group` if that's more intuitive.\n\n"
        "This is an original, exploratory formula — not a validated "
        "clinical instrument. All fields optional.\n\n"
        "⚠️ RESEARCH USE ONLY."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "high_risk": {
                            "summary": "High — multiple abnormal labs",
                            "value": {"age": 72, "wbc": 19.5, "crp": 260, "creatinine": 2.1, "glucose": 210, "ldh": 480, "ast": 140, "hematocrit": 48, "calcium": 7.4, "albumin": 2.6},
                        },
                        "low_risk": {
                            "summary": "Low — near-normal labs",
                            "value": {"age": 40, "wbc": 8.0, "crp": 20, "creatinine": 0.9, "glucose": 95, "hematocrit": 41, "calcium": 9.4, "albumin": 4.2},
                        },
                    }
                }
            }
        }
    },
)
def predict_polynomial_logit(data: PolynomialLogitInput):
    f_x, proba, log_risk, risk_group, terms_used = _polynomial_logit_score(data)
    return PolynomialLogitOutput(
        polynomial_score=f_x,
        risk_probability=proba,
        log_risk_index=log_risk,
        risk_group=risk_group,
        terms_used=terms_used,
    )
