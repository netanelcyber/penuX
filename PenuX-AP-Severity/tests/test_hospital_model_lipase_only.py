"""Regression test: amylase columns are not mandatory when lipase is available."""
import pandas as pd

from api.hospital_model import development_eligibility_mask, prepare_hospital_frame


def test_lipase_only_file_is_accepted_without_imaging_or_amylase_columns():
    frame = pd.DataFrame([{
        "encounter_id": "E1",
        "admission_time": "2026-01-01T08:00:00",
        "age": 55,
        "sex": "M",
        "acute_pancreatitis_diagnosis": True,
        "lipase": 360.0,
        "lipase_uln": 60.0,
        "outcome_followup_available": True,
        "heart_rate": 105.0,
        "systolic_bp": 100.0,
        "respiratory_rate": 22.0,
        "temperature": 38.0,
        "spo2": 94.0,
        "wbc": 16.0,
        "hematocrit": 47.0,
        "urea_mmol_l": 10.0,
    }])
    prepared = prepare_hospital_frame(frame)
    assert prepared.loc[0, "lipase_uln_ratio"] == 6.0
    eligibility, audit = development_eligibility_mask(frame)
    assert bool(eligibility.iloc[0]) is True
    assert bool(audit.loc[0, "enzyme_criterion_met"]) is True
