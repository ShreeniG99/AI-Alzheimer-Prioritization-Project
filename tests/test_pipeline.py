import json

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import STAGES


@pytest.mark.parametrize("stage", STAGES)
def test_processed_dataset_is_binary_and_leakage_free(workspace, stage):
    data = pd.read_csv(workspace["ALZ_PROCESSED_DIR"] / f"{stage}.csv")
    assert len(data) > 100
    assert data["RID"].is_unique
    assert set(data["target"].unique()) == {0, 1}
    # CDR and diagnosis must never be model inputs (label leakage).
    assert not {"cdr", "CDR", "dx", "DIAGNOSIS"} & set(data.columns)


@pytest.mark.parametrize("stage", STAGES)
def test_trained_bundle_contents(workspace, stage):
    bundle = joblib.load(workspace["ALZ_ARTIFACTS_DIR"] / f"{stage}_model.joblib")
    assert bundle["stage"] == stage
    assert bundle["selected_model"] in {"logistic_regression", "random_forest", "xgboost"}
    assert bundle["model_version"].startswith(stage)
    assert len(bundle["background"]) > 0
    thresholds = bundle["priority_thresholds"]
    assert thresholds["medium_threshold"] <= thresholds["high_threshold"]

    metadata = json.loads((workspace["ALZ_ARTIFACTS_DIR"] / f"{stage}_metadata.json").read_text())
    assert metadata["cohort"]["participants"] > 0
    assert 0.5 < metadata["untouched_test_metrics"]["roc_auc"] <= 1.0


def test_later_stages_include_their_modality(workspace):
    def features(stage):
        return set(joblib.load(workspace["ALZ_ARTIFACTS_DIR"] / f"{stage}_model.joblib")["features"])

    assert {"ptau217", "ab42_ab40"} <= features("stage2")
    assert {"eTIV", "nWBV", "ASF"} <= features("stage3")
    assert {"amyloid_centiloid", "fdg"} <= features("stage4")
    assert {"apoe4", "ptau181"} <= features("stage2")
    assert {"bmi", "gds", "hx_cardiovascular"} <= features("stage1")
    assert not {"eTIV", "ptau217", "amyloid_centiloid"} & features("stage1")


def test_bootstrap_artifact_matches_dataset(workspace):
    npz = np.load(workspace["ALZ_ARTIFACTS_DIR"] / "stage1_bootstrap_predictions.npz")
    data = pd.read_csv(workspace["ALZ_PROCESSED_DIR"] / "stage1.csv")
    assert npz["predictions"].shape == (20, len(data))
    assert ((npz["predictions"] >= 0) & (npz["predictions"] <= 1)).all()


def test_prioritized_cohort_report(workspace):
    cohort = pd.read_csv(workspace["ALZ_REPORTS_DIR"] / "prioritized_cohort.csv")
    assert cohort["risk_score"].between(0, 1).all()


def test_age_derived_and_missing_codes_cleaned(workspace):
    data = pd.read_csv(workspace["ALZ_PROCESSED_DIR"] / "stage1.csv")
    assert data["age"].notna().mean() > 0.99
    assert data["age"].between(40, 110).all()
    for column in ["education", "faq"]:
        assert (data[column].dropna() >= 0).all()


def test_inference_summary_is_aggregate_and_consistent(workspace):
    summary = json.loads((workspace["ALZ_REPORTS_DIR"] / "inference_summary.json").read_text())

    overlap = summary["cohort_overlap"]
    assert set(overlap["participants"]) == set(STAGES)
    assert overlap["in_every_stage"] <= min(overlap["participants"].values())
    assert overlap["participants_with_conflicting_labels"] == 0

    test = summary["stage1_test_breakdown"]
    assert test["top20_converters"] + test["converters_outside_top20"] == test["converters"]
    assert test["top20_converters"] + test["top20_non_converters"] == test["top20_patients"]
    assert sum(t["patients"] for t in test["tiers"].values()) == test["patients"]
    assert sum(t["converters"] for t in test["tiers"].values()) == test["converters"]
    assert test["pairs_ranked_wrong"] <= test["converter_vs_non_converter_pairs"]

    for scores in summary["stage1_input_reduction"].values():
        assert 0.5 < scores["roc_auc"] <= 1.0
    for stage in STAGES[1:]:
        assert "screening_plus_this_stage" in summary["stage_value_same_participants"][stage]

    # Aggregates only: no participant identifiers in the report.
    assert "RID" not in json.dumps(summary)
