"""Visit-window suggestion from the TEMPO decline rate."""

import pytest

from backend.services.visit_window_service import MAX_DAYS, MIN_DAYS, suggest_visit_window


def test_faster_decline_gives_tighter_window():
    stable = suggest_visit_window(0.4)["half_width_days"]
    moderate = suggest_visit_window(1.0)["half_width_days"]
    fast = suggest_visit_window(2.5)["half_width_days"]
    assert stable > moderate > fast


def test_window_is_clamped():
    assert suggest_visit_window(0.0)["half_width_days"] == MAX_DAYS
    assert suggest_visit_window(-1.0)["half_width_days"] == MAX_DAYS
    assert suggest_visit_window(50.0)["half_width_days"] == MIN_DAYS


def test_drift_at_edge_stays_within_budget():
    for rate in (0.3, 0.7, 1.2, 2.0):
        window = suggest_visit_window(rate)
        assert window["expected_drift_at_edge"] <= window["drift_budget"] + 1e-9


@pytest.mark.parametrize("rate, days", [(0.5, 28), (1.0, 18), (2.0, 9)])
def test_reference_values(rate, days):
    assert suggest_visit_window(rate)["half_width_days"] == days


def test_assessment_returns_visit_window(client, admin_headers):
    payload = {"patient_ref": "window-1", "age": 74, "education": 14, "sex": 2, "mmse": 23, "adas": 20, "faq": 8}
    decline = client.post("/assessment", headers=admin_headers, json=payload).json()["decline"]
    window = decline["visit_window"]
    assert MIN_DAYS <= window["half_width_days"] <= MAX_DAYS
    assert window["rate_used"] == pytest.approx(max(decline["interval_high"], 0.05))
    assert "safety" in window["scope"].lower()
