import numpy as np

from ml.explain import _source_feature
from ml.model_utils import (
    PlattCalibrator,
    expected_calibration_error,
    priority_from_risk,
    priority_thresholds,
    top_k_recall,
)


def test_top_k_recall_counts_positives_in_top_fraction():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05])
    assert top_k_recall(y, p, fraction=0.3) == 2 / 3


def test_priority_thresholds_and_tiers():
    risk = np.linspace(0, 1, 101)
    thresholds = priority_thresholds(risk, high_fraction=0.2, medium_fraction=0.3)
    assert priority_from_risk(0.95, thresholds) == "HIGH"
    assert priority_from_risk(0.5, thresholds) == "MEDIUM"
    assert priority_from_risk(0.1, thresholds) == "LOW"


def test_platt_calibration_is_monotonic():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p ** 2).astype(int)
    calibrated = PlattCalibrator().fit(p, y).transform(np.sort(p))
    assert np.all(np.diff(calibrated) >= -1e-12)
    assert expected_calibration_error(y, PlattCalibrator().fit(p, y).transform(p)) < 0.1


def test_shap_names_map_back_to_inputs():
    features = ["mmse", "adas", "sex"]
    assert _source_feature("num__mmse", features) == "mmse"
    assert _source_feature("num__missingindicator_adas", features) == "adas"
    assert _source_feature("cat__sex_2.0", features) == "sex"
