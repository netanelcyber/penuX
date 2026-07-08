"""Tests for penux_ap.clinical_scores, in-memory fixtures only."""
import math

import pandas as pd

from penux_ap.clinical_scores import compute_quasi_sofa_labs, required_fields_for_score


def test_required_fields_for_quasi_sofa():
    assert required_fields_for_score("quasi_sofa_labs") == [
        "creatinine_umol_l", "bilirubin_umol_l", "platelets_e9_l",
    ]


def test_quasi_sofa_normal_labs_score_zero():
    row = pd.Series({"creatinine_umol_l": 70, "bilirubin_umol_l": 10, "platelets_e9_l": 250})
    assert compute_quasi_sofa_labs(row) == 0


def test_quasi_sofa_missing_required_field_returns_nan():
    row = pd.Series({"creatinine_umol_l": 70, "bilirubin_umol_l": 10})
    assert math.isnan(compute_quasi_sofa_labs(row))


def test_quasi_sofa_severe_organ_dysfunction():
    row = pd.Series({"creatinine_umol_l": 500, "bilirubin_umol_l": 250, "platelets_e9_l": 15})
    # renal=4, hepatic=4, coag=4 -> 12
    assert compute_quasi_sofa_labs(row) == 12


def test_quasi_sofa_optional_wbc_and_lactate_add_points():
    base = {"creatinine_umol_l": 70, "bilirubin_umol_l": 10, "platelets_e9_l": 250}
    assert compute_quasi_sofa_labs(pd.Series(base)) == 0

    with_wbc = dict(base, wbc_e9_l=20.0)
    assert compute_quasi_sofa_labs(pd.Series(with_wbc)) == 1

    with_lactate = dict(base, lactate_mmol_l=3.5)
    assert compute_quasi_sofa_labs(pd.Series(with_lactate)) == 1

    with_both = dict(base, wbc_e9_l=20.0, lactate_mmol_l=3.5)
    assert compute_quasi_sofa_labs(pd.Series(with_both)) == 2
