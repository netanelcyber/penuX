"""Tests for the hospital-oriented AP data contract and model pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.hospital_model import train_hospital_model
from api.schemas import AdmissionInput


def test_si_units_are_normalised_for_legacy_estimator():
    record = AdmissionInput(
        age=60,
        urea_mmol_l=10.0,
        creatinine_umol_l=176.8,
        glucose_mmol_l=10.0,
        calcium_mmol_l=2.0,
        albumin_g_l=30.0,
        bilirubin_total_umol_l=34.2,
        triglycerides_mmol_l=2.0,
        height_cm=180,
        weight_kg=81,
        systolic_bp=120,
        diastolic_bp=60,
    )
    legacy = record.model_dump()
    assert legacy["bun"] == pytest.approx(28.01)
    assert legacy["creatinine"] == pytest.approx(2.0)
    assert legacy["glucose"] == pytest.approx(180.18)
    assert legacy["calcium"] == pytest.approx(8.016)
    assert legacy["albumin"] == pytest.approx(3.0)
    assert record.bilirubin_total == pytest.approx(2.0)
    assert legacy["triglycerides"] == pytest.approx(177.14)
    assert record.bmi == pytest.approx(25.0)
    assert record.mean_arterial_pressure == pytest.approx(80.0)


def test_research_case_can_be_eligible_without_imaging():
    record = AdmissionInput(
        encounter_id="ENC-001",
        admission_time="2026-01-01T08:00:00",
        age=64,
        sex="M",
        acute_pancreatitis_diagnosis=True,
        lipase=360,
        lipase_uln=60,
        outcome_followup_available=True,
        imaging_available=False,
        heart_rate=105,
        systolic_bp=102,
        respiratory_rate=22,
        temperature=38.1,
        spo2=94,
        wbc=16.0,
        hematocrit=47,
        urea_mmol_l=11.0,
    )
    result = record.research_eligibility_summary()
    assert result["eligible_for_primary_model_development"] is True
    assert result["imaging_required"] is False
    assert result["core_present"] >= 8


def _synthetic_frame(n: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    target = np.array([0, 0, 1] * (n // 3), dtype=int)
    frame = pd.DataFrame({
        "encounter_id": [f"E{i:03d}" for i in range(n)],
        "admission_time": pd.date_range("2025-01-01", periods=n, freq="D"),
        "age": rng.integers(20, 90, n),
        "sex": np.where(np.arange(n) % 2, "F", "M"),
        "acute_pancreatitis_diagnosis": True,
        "lipase": rng.normal(500, 80, n),
        "lipase_uln": 60.0,
        "amylase": np.nan,
        "amylase_uln": np.nan,
        "outcome_followup_available": True,
        "heart_rate": rng.normal(95 + target * 10, 12, n),
        "systolic_bp": rng.normal(118 - target * 12, 15, n),
        "respiratory_rate": rng.normal(18 + target * 4, 3, n),
        "temperature": rng.normal(37.2 + target * 0.4, 0.6, n),
        "spo2": rng.normal(97 - target * 2, 2, n),
        "wbc": rng.normal(10 + target * 5, 3, n),
        "hematocrit": rng.normal(41 + target * 3, 4, n),
        "urea_mmol_l": rng.normal(6 + target * 4, 2, n),
        "creatinine_umol_l": rng.normal(80 + target * 35, 20, n),
        "glucose_mmol_l": rng.normal(6 + target * 2, 1.5, n),
        "calcium_mmol_l": rng.normal(2.25 - target * 0.18, 0.12, n),
        "albumin_g_l": rng.normal(39 - target * 5, 4, n),
        "bilirubin_total_umol_l": rng.normal(16 + target * 10, 8, n),
        "crp": rng.normal(50 + target * 100, 35, n),
        "persistent_organ_failure_gt_48h": target,
    })
    # Deliberately introduce predictor missingness. Cases remain eligible
    # because enough core groups are still present.
    frame.loc[frame.index[::5], "crp"] = np.nan
    frame.loc[frame.index[::7], "albumin_g_l"] = np.nan
    return frame


def test_training_pipeline_handles_missing_predictors_inside_cv():
    frame = _synthetic_frame()
    bundle, audit = train_hospital_model(
        frame,
        target_column="persistent_organ_failure_gt_48h",
        minimum_availability=0.75,
        target_sensitivity=0.80,
        cv_folds=3,
    )
    assert audit["selected_for_development"].all()
    assert 0.0 <= bundle.threshold <= 1.0
    assert "crp" in bundle.feature_names
    assert "persistent_organ_failure_gt_48h" not in bundle.feature_names
    probabilities = bundle.predict_proba(frame.iloc[:3])
    assert probabilities.shape == (3, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
