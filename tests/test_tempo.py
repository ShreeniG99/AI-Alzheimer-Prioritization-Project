"""TEMPO: progression-rate target, models and API integration."""

import importlib.util
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, STAGES
from ml.model_utils import speed_tier

spec = importlib.util.spec_from_file_location("tempo_target", ROOT / "scripts" / "10_build_tempo_target.py")
tempo_target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tempo_target)

STAGE1 = {"age": 74, "education": 14, "sex": 2, "mmse": 23, "adas": 20, "faq": 8}


def test_ols_slope_is_points_per_year():
    assert tempo_target.ols_slope([0, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert tempo_target.ols_slope([0, 0.5, 1.5], [2, 2, 2]) == pytest.approx(0.0)


def test_speed_tiers():
    thresholds = {"moderate": 0.3, "fast": 1.0}
    assert speed_tier(1.2, thresholds) == "FAST"
    assert speed_tier(0.5, thresholds) == "MODERATE"
    assert speed_tier(-0.1, thresholds) == "STABLE"


def test_tempo_target_table(workspace):
    target = pd.read_csv(workspace["ALZ_PROCESSED_DIR"] / "tempo_target.csv")
    assert len(target) > 200
    assert target["RID"].is_unique
    assert (target["n_visits"] >= 3).all()
    assert (target["span_years"] >= 1.0).all()
    # Converters decline faster than non-converters.
    cohort = pd.read_csv(workspace["ALZ_PROCESSED_DIR"] / "stage1.csv")[["RID", "target"]]
    merged = target.merge(cohort, on="RID")
    means = merged.groupby("target")["cdrsb_slope"].mean()
    assert means[1] > means[0]


@pytest.mark.parametrize("stage", STAGES)
def test_tempo_bundles(workspace, stage):
    bundle = joblib.load(workspace["ALZ_ARTIFACTS_DIR"] / f"tempo_{stage}_model.joblib")
    assert bundle["kind"] == "tempo"
    assert bundle["interval_halfwidth"] > 0
    assert "cdr" not in bundle["features"] and "cdrsb_baseline" not in bundle["features"]
    meta = json.loads((workspace["ALZ_ARTIFACTS_DIR"] / f"tempo_{stage}_metadata.json").read_text())
    metrics = meta["untouched_test_metrics"]
    # Better than always predicting the cohort mean.
    assert metrics["mae"] < meta["baselines"]["mean_prediction"]["mae"]
    assert metrics["spearman"] > 0.2


def test_assessment_includes_decline(client, admin_headers):
    body = client.post("/assessment", headers=admin_headers, json={"patient_ref": "tempo-1", **STAGE1}).json()
    decline = body["decline"]
    assert decline["speed_tier"] in {"FAST", "MODERATE", "STABLE"}
    assert decline["interval_low"] < decline["predicted_cdrsb_per_year"] < decline["interval_high"]
    assert decline["outlook"]["label"]
    assert decline["explanation"]["output_space"] == "CDR-SB points per year"

    queue = client.get("/patients", headers=admin_headers).json()["patients"]
    saved = next(p for p in queue if p["id"] == body["id"])
    assert saved["decline"]["speed_tier"] == decline["speed_tier"]
    assert "explanation" not in saved["decline"]


def test_worse_cognition_predicts_faster_decline(client, admin_headers):
    def slope(**overrides):
        payload = {**STAGE1, **overrides}
        return client.post("/assessment", headers=admin_headers, json=payload).json()["decline"]["predicted_cdrsb_per_year"]

    assert slope(mmse=19, adas=32, faq=18) > slope(mmse=30, adas=4, faq=0)


def test_model_info_and_health_list_tempo(client, admin_headers):
    assert client.get("/health").json()["tempo_stages"] == STAGES
    info = client.get("/model-info", headers=admin_headers).json()
    assert all(s["tempo"] for s in info["stages"])
