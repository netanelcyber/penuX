"""Hospital-oriented model training and inference utilities.

This module does not claim clinical validity. It implements the agreed data
policy: no imaging dependency, no complete-case deletion, no outcome leakage,
minimum patient-level predictor coverage, and preprocessing fitted inside each
cross-validation fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RESEARCH_WARNING = (
    "Research use only. Not externally or prospectively validated. "
    "Not for patient-care decisions."
)

OUTCOME_OR_LEAKAGE_COLUMNS: frozenset[str] = frozenset({
    "persistent_organ_failure_gt_48h",
    "severe_acute_pancreatitis",
    "sap",
    "target",
    "label",
    "icu_admission",
    "invasive_ventilation_after_24h",
    "renal_replacement_therapy",
    "length_of_stay",
    "mortality_30d",
    "discharge_disposition",
})

NUMERIC_CANDIDATES: tuple[str, ...] = (
    # Demographics and first-24-hour vital signs.
    "age", "bmi", "heart_rate", "systolic_bp", "diastolic_bp",
    "mean_arterial_pressure", "respiratory_rate", "temperature", "spo2",
    "oxygen_flow_l_min", "fio2", "gcs", "urine_output_ml_24h",
    # Haematology.
    "wbc", "anc", "alc", "monocytes_absolute", "rbc", "hemoglobin",
    "hematocrit", "platelets", "mcv", "rdw_cv", "mpv",
    # Renal, electrolyte and metabolic tests.
    "urea_mmol_l", "creatinine_umol_l", "egfr", "sodium", "potassium",
    "chloride", "bicarbonate_total", "glucose_mmol_l", "calcium_mmol_l",
    "ionized_calcium_mmol_l", "magnesium_mmol_l", "phosphate_mmol_l",
    "bicarbonate_blood_gas",
    # Liver, pancreatic and inflammatory tests.
    "albumin_g_l", "total_protein_g_l", "bilirubin_total_umol_l",
    "bilirubin_direct_umol_l", "ast", "alt", "alp", "ggt", "ldh",
    "lipase", "amylase", "crp", "procalcitonin_ng_ml",
    "triglycerides_mmol_l",
    # Coagulation and blood gas.
    "pt_seconds", "inr", "aptt_seconds", "fibrinogen_g_l",
    "d_dimer_mg_l_feu", "lactate", "ph", "pao2", "paco2", "base_excess",
)

CATEGORICAL_CANDIDATES: tuple[str, ...] = (
    "sex", "avpu", "oxygen_support", "smoking_status", "alcohol_status",
    "diabetes", "heart_failure", "ischemic_heart_disease",
    "chronic_kidney_disease", "chronic_liver_disease", "cirrhosis", "copd",
    "active_malignancy", "immunosuppression", "obesity",
    "hypertriglyceridemia", "gallstones", "chronic_pancreatitis",
)

CORE_ALTERNATIVES: dict[str, tuple[str, ...]] = {
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
    "creatinine": ("creatinine_umol_l", "creatinine"),
    "glucose": ("glucose_mmol_l", "glucose"),
    "calcium": ("calcium_mmol_l", "calcium"),
    "albumin": ("albumin_g_l", "albumin"),
    "bilirubin": ("bilirubin_total_umol_l", "bilirubin_total"),
}

REQUIRED_DEVELOPMENT_COLUMNS: tuple[str, ...] = (
    "encounter_id", "admission_time", "age", "acute_pancreatitis_diagnosis",
    "lipase", "lipase_uln", "amylase", "amylase_uln",
    "outcome_followup_available",
)


@dataclass
class HospitalModelBundle:
    """Serializable model wrapper that enforces the fitted feature contract."""

    estimator: Any
    feature_names: list[str]
    threshold: float
    model_version: str = "hospital-1.0.0"
    metrics: dict[str, float] = field(default_factory=dict)
    feature_availability: dict[str, float] = field(default_factory=dict)
    warning: str = RESEARCH_WARNING

    def _frame(self, records: Any) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, Mapping):
            frame = pd.DataFrame([dict(records)])
        else:
            frame = pd.DataFrame(records)
        for name in self.feature_names:
            if name not in frame.columns:
                frame[name] = np.nan
        return frame.loc[:, self.feature_names]

    def predict_proba(self, records: Any) -> np.ndarray:
        frame = self._frame(records)
        return self.estimator.predict_proba(frame)

    def predict(self, records: Any) -> np.ndarray:
        probabilities = self.predict_proba(records)[:, 1]
        return (probabilities >= self.threshold).astype(int)


def _core_presence(frame: pd.DataFrame) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for group, alternatives in CORE_ALTERNATIVES.items():
        available = [name for name in alternatives if name in frame.columns]
        if not available:
            result[group] = pd.Series(False, index=frame.index)
        else:
            result[group] = frame[available].notna().any(axis=1)
    return pd.DataFrame(result, index=frame.index)


def development_eligibility_mask(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return the prespecified primary-development eligibility mask.

    Imaging is intentionally absent from the rule. A documented AP diagnosis
    plus lipase or amylase at least three times the local ULN is required for
    the operational case definition. The outcome must be observable beyond
    48 hours, and at least 8 of 16 core predictor groups must be present.
    """
    missing_columns = [name for name in REQUIRED_DEVELOPMENT_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Missing columns required to define the development cohort: "
            + ", ".join(missing_columns)
        )

    lipase_ok = (
        frame["lipase"].notna()
        & frame["lipase_uln"].notna()
        & (frame["lipase"] >= 3.0 * frame["lipase_uln"])
    )
    amylase_ok = (
        frame["amylase"].notna()
        & frame["amylase_uln"].notna()
        & (frame["amylase"] >= 3.0 * frame["amylase_uln"])
    )
    core = _core_presence(frame)
    core_count = core.sum(axis=1)

    vital_groups = ["heart_rate", "systolic_bp", "respiratory_rate", "temperature", "spo2"]
    lab_groups = [
        "wbc", "hemoglobin_or_hematocrit", "urea_or_bun", "creatinine",
        "glucose", "calcium", "albumin", "bilirubin",
    ]
    has_any_vital = core[vital_groups].any(axis=1)
    has_any_lab = core[lab_groups].any(axis=1)

    eligibility = (
        frame["encounter_id"].notna()
        & frame["admission_time"].notna()
        & frame["age"].notna()
        & (frame["age"] >= 18)
        & frame["acute_pancreatitis_diagnosis"].fillna(False).astype(bool)
        & (lipase_ok | amylase_ok)
        & frame["outcome_followup_available"].fillna(False).astype(bool)
        & (core_count >= 8)
        & has_any_vital
        & has_any_lab
    )

    audit = core.copy()
    audit["core_present"] = core_count
    audit["enzyme_criterion_met"] = lipase_ok | amylase_ok
    audit["has_any_vital"] = has_any_vital
    audit["has_any_lab"] = has_any_lab
    audit["eligible"] = eligibility
    return eligibility, audit


def select_features_by_availability(
    frame: pd.DataFrame,
    *,
    minimum_availability: float = 0.80,
    numeric_candidates: Sequence[str] = NUMERIC_CANDIDATES,
    categorical_candidates: Sequence[str] = CATEGORICAL_CANDIDATES,
) -> tuple[list[str], list[str], dict[str, float]]:
    if not 0 < minimum_availability <= 1:
        raise ValueError("minimum_availability must be in (0, 1]")

    availability = frame.notna().mean().to_dict()
    numeric = [
        name for name in numeric_candidates
        if name in frame.columns
        and name not in OUTCOME_OR_LEAKAGE_COLUMNS
        and availability.get(name, 0.0) >= minimum_availability
    ]
    categorical = [
        name for name in categorical_candidates
        if name in frame.columns
        and name not in OUTCOME_OR_LEAKAGE_COLUMNS
        and availability.get(name, 0.0) >= minimum_availability
    ]
    if not numeric and not categorical:
        raise ValueError("No predictor met the requested availability threshold")
    selected_availability = {name: float(availability[name]) for name in numeric + categorical}
    return numeric, categorical, selected_availability


def build_pipeline(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, list(numeric_features)))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, list(categorical_features)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def threshold_for_target_sensitivity(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    target_sensitivity: float = 0.98,
) -> tuple[float, dict[str, float]]:
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))

    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= target_sensitivity:
            candidate = (specificity, threshold, sensitivity)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        threshold = 0.0
    else:
        _, threshold, _ = best

    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return float(threshold), {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "true_negative": float(tn),
        "false_negative": float(fn),
    }


def train_hospital_model(
    frame: pd.DataFrame,
    *,
    target_column: str,
    minimum_availability: float = 0.80,
    target_sensitivity: float = 0.98,
    cv_folds: int = 5,
) -> tuple[HospitalModelBundle, pd.DataFrame]:
    """Fit a leakage-resistant hospital model and return its audit table."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if target_column in OUTCOME_OR_LEAKAGE_COLUMNS - {"target", "label", "sap", "severe_acute_pancreatitis", "persistent_organ_failure_gt_48h"}:
        raise ValueError(f"Unsupported target column: {target_column}")

    eligibility, audit = development_eligibility_mask(frame)
    development = frame.loc[eligibility].copy()
    if development.empty:
        raise ValueError("No case met the prespecified development eligibility rules")

    y = pd.to_numeric(development[target_column], errors="raise").astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("Target must be binary and encoded as 0/1")
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Both target classes are required")
    if class_counts.min() < cv_folds:
        raise ValueError("The minority class has fewer cases than cv_folds")

    numeric, categorical, availability = select_features_by_availability(
        development,
        minimum_availability=minimum_availability,
    )
    feature_names = numeric + categorical
    pipeline = build_pipeline(numeric, categorical)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_probability = cross_val_predict(
        pipeline,
        development[feature_names],
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    threshold, operating = threshold_for_target_sensitivity(
        y,
        oof_probability,
        target_sensitivity=target_sensitivity,
    )
    oof_prediction = (oof_probability >= threshold).astype(int)
    metrics = {
        "auroc_oof": float(roc_auc_score(y, oof_probability)),
        "average_precision_oof": float(average_precision_score(y, oof_probability)),
        "f1_oof": float(f1_score(y, oof_prediction, zero_division=0)),
        "f2_oof": float(fbeta_score(y, oof_prediction, beta=2, zero_division=0)),
        "threshold": threshold,
        "target_sensitivity": float(target_sensitivity),
        **operating,
        "n_development_cases": float(len(development)),
        "n_predictors": float(len(feature_names)),
    }

    pipeline.fit(development[feature_names], y)
    bundle = HospitalModelBundle(
        estimator=pipeline,
        feature_names=feature_names,
        threshold=threshold,
        metrics=metrics,
        feature_availability=availability,
    )

    audit = audit.copy()
    audit["selected_for_development"] = eligibility
    return bundle, audit
