"""Tests for leakage-resistant research validation utilities."""
import numpy as np
import pytest

from penux_ap.research_validation import (
    bootstrap_metric_intervals,
    decision_curve_analysis,
    fbeta_at_threshold,
    select_threshold_for_sensitivity,
)


def test_select_threshold_for_sensitivity_hits_target_and_is_specific():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    p = np.array([0.95, 0.80, 0.65, 0.40, 0.70, 0.30, 0.20, 0.10])

    result = select_threshold_for_sensitivity(
        y,
        p,
        target_sensitivity=0.75,
        beta=2.5,
    )

    assert result["threshold"] == pytest.approx(0.65)
    assert result["achieved_sensitivity"] == pytest.approx(0.75)
    assert result["specificity"] == pytest.approx(0.75)
    assert result["fn"] == 1


def test_target_one_can_fall_back_to_low_threshold():
    y = np.array([1, 1, 0, 0], dtype=int)
    p = np.array([0.80, 0.20, 0.70, 0.10])

    result = select_threshold_for_sensitivity(y, p, target_sensitivity=1.0)

    assert result["achieved_sensitivity"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.20)


def test_fbeta_recall_weighting():
    y = np.array([1, 1, 1, 0, 0], dtype=int)
    p = np.array([0.9, 0.8, 0.4, 0.2, 0.1])

    f1 = fbeta_at_threshold(y, p, threshold=0.5, beta=1.0)
    f25 = fbeta_at_threshold(y, p, threshold=0.5, beta=2.5)

    assert 0.0 < f25 < f1 < 1.0


def test_bootstrap_metric_intervals_are_reproducible():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    p = np.array([0.95, 0.82, 0.71, 0.52, 0.61, 0.33, 0.18, 0.05])

    first = bootstrap_metric_intervals(
        y,
        p,
        threshold=0.5,
        beta=2.5,
        n_bootstraps=100,
        random_state=7,
    )
    second = bootstrap_metric_intervals(
        y,
        p,
        threshold=0.5,
        beta=2.5,
        n_bootstraps=100,
        random_state=7,
    )

    assert first == second
    assert first["metrics"]["sensitivity"]["n_valid"] == 100
    assert 0.0 <= first["metrics"]["auroc"]["ci_lower"] <= 1.0
    assert 0.0 <= first["metrics"]["auroc"]["ci_upper"] <= 1.0


def test_decision_curve_matches_manual_net_benefit():
    y = np.array([1, 1, 0, 0], dtype=int)
    p = np.array([0.9, 0.8, 0.7, 0.1])

    dca = decision_curve_analysis(y, p, threshold_probabilities=[0.5])
    row = dca.iloc[0]

    assert row["model_net_benefit"] == pytest.approx(0.25)
    assert row["treat_all_net_benefit"] == pytest.approx(0.0)
    assert row["treat_none_net_benefit"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad_target", [0.0, -0.1, 1.01])
def test_invalid_target_sensitivity_rejected(bad_target):
    y = np.array([1, 0])
    p = np.array([0.8, 0.2])
    with pytest.raises(ValueError):
        select_threshold_for_sensitivity(y, p, target_sensitivity=bad_target)
