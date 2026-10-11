"""
09_sensitivity_analysis.py

Stage-1 model sensitivity analysis for the
AI-Driven Prioritization for Early Alzheimer's
Diagnostic Pathways research prototype.

Purpose
-------
Evaluate how the trained Stage-1 model's calibrated risk estimate
changes when individual model inputs are varied while the remaining
inputs are held constant.

This is a MODEL-BEHAVIOR analysis.

It must NOT be interpreted as:
    - causal evidence,
    - clinical effect estimation,
    - treatment guidance,
    - diagnostic validation.

The Stage-1 production model uses:

    education
    mmse
    adas
    faq
    sex

The analysis uses the production model artifact and its Platt
calibrator so that reported risk values correspond to the same
calibrated risk definition used by the API /predict endpoint.

Outputs
-------
reports/sensitivity_analysis.csv
reports/sensitivity_analysis.png
reports/sensitivity_summary.csv
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

from common import artifacts_dir, processed_dir, reports_dir

MODEL_PATH = artifacts_dir() / "stage1_model.joblib"

DATA_PATH = processed_dir() / "stage1.csv"

REPORTS_DIR = reports_dir()

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    REPORTS_DIR
    / "sensitivity_analysis.csv"
)

SUMMARY_PATH = (
    REPORTS_DIR
    / "sensitivity_summary.csv"
)

PLOT_PATH = (
    REPORTS_DIR
    / "sensitivity_analysis.png"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FEATURES = [
    "age",
    "education",
    "mmse",
    "adas",
    "faq",
    "sex",
]

NUMERIC_FEATURES = [
    "age",
    "education",
    "mmse",
    "adas",
    "faq",
]

CATEGORICAL_FEATURES = [
    "sex",
]


# Representative reference profile.

# These values are intentionally within the observed Stage-1
# feature ranges and are used only to demonstrate model behavior.

REFERENCE_PROFILE = {
    "age": 72.0,
    "education": 16.0,
    "mmse": 28.0,
    "adas": 10.67,
    "faq": 0.0,
    "sex": 1.0,
}


# Feature-specific sensitivity values.

# MMSE: clinically plausible scale range.
MMSE_VALUES = [
    18.0,
    20.0,
    22.0,
    24.0,
    26.0,
    28.0,
    30.0,
]

# ADAS-11 total score: 0-70.
ADAS_VALUES = [
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
]

# FAQ total: 0-30.
FAQ_VALUES = [
    0.0,
    2.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
]

# Age at baseline (years); ADNI cohort range is roughly 55-91.
AGE_VALUES = [
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
]

# Education: ADNI PTEDUCAT is coded 0-20.
EDUCATION_VALUES = [
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0,
]

# ADNI coding:
# 1 = Male
# 2 = Female
SEX_VALUES = [
    1.0,
    2.0,
]


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model() -> Dict[str, Any]:
    """Load the trained Stage-1 model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage-1 model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    required_keys = [
        "pipeline",
        "calibrator",
        "selected_model",
        "features",
        "priority_thresholds",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:
        raise RuntimeError(
            "Stage-1 model artifact is missing required keys: "
            f"{missing_keys}"
        )

    return artifact


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_reference_profile(
    artifact: Dict[str, Any],
) -> None:
    """Verify that the reference profile matches the production model."""

    artifact_features = artifact["features"]

    missing = [f for f in FEATURES if f not in artifact_features]
    if missing:
        raise RuntimeError(
            "Production Stage-1 model does not use features varied by this "
            f"analysis: {missing}\nModel features: {artifact_features}"
        )

    # Other model inputs (e.g. comorbidities) are held at "not recorded",
    # i.e. the model's training-median imputation, a typical value.
    for feature in artifact_features:
        if feature not in FEATURES:
            REFERENCE_PROFILE.setdefault(feature, None)

    for feature in FEATURES:
        if feature not in REFERENCE_PROFILE:
            raise RuntimeError(
                f"Reference profile is missing feature: {feature}"
            )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def calibrated_probability(
    artifact: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    """
    Calculate the production-model calibrated probability.

    This follows the same conceptual sequence as backend prediction:

        preprocessing pipeline
                    ↓
        raw logistic probability
                    ↓
        Platt calibration
                    ↓
        calibrated risk score
    """

    pipeline = artifact["pipeline"]
    calibrator = artifact["calibrator"]

    row = {
        feature: profile.get(feature)
        for feature in artifact["features"]
    }

    dataframe = pd.DataFrame(
        [row]
    ).apply(pd.to_numeric, errors="coerce").astype(float)

    raw_probability = float(
        pipeline.predict_proba(
            dataframe
        )[:, 1][0]
    )

    calibrated = float(
        calibrator.transform(
            np.asarray(
                [raw_probability]
            )
        )[0]
    )

    return calibrated


# ---------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------

def priority_from_probability(
    probability: float,
    thresholds: Dict[str, float],
) -> str:
    """Convert calibrated model score into the project's priority label."""

    high_threshold = float(
        thresholds["high_threshold"]
    )

    medium_threshold = float(
        thresholds["medium_threshold"]
    )

    if probability >= high_threshold:
        return "HIGH"

    if probability >= medium_threshold:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_stage1_dataset() -> pd.DataFrame:
    """Load Stage-1 dataset for reference-range reporting."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Stage-1 dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH
    )

    missing_columns = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Stage-1 dataset is missing expected features: "
            f"{missing_columns}"
        )

    return dataframe


# ---------------------------------------------------------------------
# Sensitivity profiles
# ---------------------------------------------------------------------

def build_sensitivity_profiles() -> List[Dict[str, Any]]:
    """
    Build one-at-a-time sensitivity profiles.

    Every row starts from REFERENCE_PROFILE.

    Only one feature is changed at a time.

    This allows the analysis to describe model response to individual
    input variation without claiming that the variation represents a
    causal intervention.
    """

    profiles = []

    feature_values = {
        "age": AGE_VALUES,
        "education": EDUCATION_VALUES,
        "mmse": MMSE_VALUES,
        "adas": ADAS_VALUES,
        "faq": FAQ_VALUES,
        "sex": SEX_VALUES,
    }

    for feature in FEATURES:

        for value in feature_values[feature]:

            profile = dict(
                REFERENCE_PROFILE
            )

            profile[feature] = value

            profiles.append(
                {
                    "analysis": "one_at_a_time",
                    "feature_changed": feature,
                    "input_value": value,
                    **profile,
                }
            )

    return profiles


# ---------------------------------------------------------------------
# Missing-data sensitivity
# ---------------------------------------------------------------------

def build_missingness_profiles() -> List[Dict[str, Any]]:
    """
    Create profiles where one Stage-1 input is missing.

    The production pipeline's existing preprocessing strategy is used.

    This evaluates model behavior under missing inputs; it does not
    establish that missingness itself is clinically meaningful.
    """

    profiles = []

    for feature in FEATURES:

        profile = dict(
            REFERENCE_PROFILE
        )

        profile[feature] = np.nan

        profiles.append(
            {
                "analysis": "single_feature_missing",
                "feature_changed": feature,
                "input_value": np.nan,
                **profile,
            }
        )

    return profiles


# ---------------------------------------------------------------------
# Prediction table
# ---------------------------------------------------------------------

def evaluate_profiles(
    artifact: Dict[str, Any],
    profiles: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Generate calibrated model results for all sensitivity profiles."""

    rows = []

    thresholds = artifact[
        "priority_thresholds"
    ]

    reference_risk = calibrated_probability(
        artifact,
        REFERENCE_PROFILE,
    )

    for profile_record in profiles:

        profile = {
            feature: profile_record[feature]
            for feature in FEATURES
        }

        risk = calibrated_probability(
            artifact,
            profile,
        )

        priority = priority_from_probability(
            risk,
            thresholds,
        )

        rows.append(
            {
                "analysis": profile_record[
                    "analysis"
                ],
                "feature_changed": profile_record[
                    "feature_changed"
                ],
                "input_value": profile_record[
                    "input_value"
                ],
                "age": profile[
                    "age"
                ],
                "education": profile[
                    "education"
                ],
                "mmse": profile[
                    "mmse"
                ],
                "adas": profile[
                    "adas"
                ],
                "faq": profile[
                    "faq"
                ],
                "sex": profile[
                    "sex"
                ],
                "calibrated_risk": risk,
                "risk_change_from_reference": (
                    risk - reference_risk
                ),
                "priority": priority,
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the model's sensitivity to each feature.

    The range is descriptive of model output over the tested values.
    """

    rows = []

    one_at_a_time = results[
        results["analysis"]
        == "one_at_a_time"
    ]

    for feature in FEATURES:

        subset = one_at_a_time[
            one_at_a_time[
                "feature_changed"
            ]
            == feature
        ]

        if subset.empty:
            continue

        minimum_row = subset.loc[
            subset[
                "calibrated_risk"
            ].idxmin()
        ]

        maximum_row = subset.loc[
            subset[
                "calibrated_risk"
            ].idxmax()
        ]

        rows.append(
            {
                "feature": feature,
                "tested_values": len(subset),
                "minimum_risk": float(
                    subset[
                        "calibrated_risk"
                    ].min()
                ),
                "maximum_risk": float(
                    subset[
                        "calibrated_risk"
                    ].max()
                ),
                "risk_range": float(
                    subset[
                        "calibrated_risk"
                    ].max()
                    -
                    subset[
                        "calibrated_risk"
                    ].min()
                ),
                "minimum_risk_input": float(
                    minimum_row[
                        "input_value"
                    ]
                ),
                "maximum_risk_input": float(
                    maximum_row[
                        "input_value"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def create_plot(
    results: pd.DataFrame,
) -> None:
    """
    Create one figure containing separate feature sensitivity curves.

    A single figure is used for reporting convenience.

    No causal interpretation is implied.
    """

    one_at_a_time = results[
        results["analysis"]
        == "one_at_a_time"
    ]

    plt.figure(
        figsize=(11, 7)
    )

    for feature in FEATURES:

        subset = one_at_a_time[
            one_at_a_time[
                "feature_changed"
            ]
            == feature
        ].copy()

        subset = subset.sort_values(
            "input_value"
        )

        plt.plot(
            subset[
                "input_value"
            ],
            subset[
                "calibrated_risk"
            ],
            marker="o",
            label=feature,
        )

    plt.axhline(
        y=0.0785421039309634,
        linestyle="--",
        label="High-priority threshold",
    )

    plt.axhline(
        y=0.013819882151742317,
        linestyle=":",
        label="Medium-priority threshold",
    )

    plt.xlabel(
        "Changed feature value"
    )

    plt.ylabel(
        "Calibrated model risk score"
    )

    plt.title(
        "Stage-1 Model Sensitivity Analysis"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "STAGE-1 MODEL SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 70
    )

    # Load model.
    artifact = load_model()

    print(
        f"Model: {artifact['selected_model']}"
    )

    print(
        f"Features: {artifact['features']}"
    )

    # Validate model/data configuration.
    validate_reference_profile(
        artifact
    )

    dataset = load_stage1_dataset()

    print(
        f"Stage-1 cohort rows: {len(dataset)}"
    )

    # Reference profile.
    reference_risk = calibrated_probability(
        artifact,
        REFERENCE_PROFILE,
    )

    reference_priority = priority_from_probability(
        reference_risk,
        artifact[
            "priority_thresholds"
        ],
    )

    print()
    print(
        "Reference profile:"
    )

    for feature in FEATURES:
        print(
            f"  {feature}: "
            f"{REFERENCE_PROFILE[feature]}"
        )

    print(
        f"Reference calibrated risk: "
        f"{reference_risk:.6f}"
    )

    print(
        f"Reference priority: "
        f"{reference_priority}"
    )

    # Build sensitivity profiles.
    sensitivity_profiles = (
        build_sensitivity_profiles()
    )

    missingness_profiles = (
        build_missingness_profiles()
    )

    all_profiles = (
        sensitivity_profiles
        +
        missingness_profiles
    )

    print()
    print(
        f"Profiles evaluated: "
        f"{len(all_profiles)}"
    )

    # Evaluate.
    results = evaluate_profiles(
        artifact,
        all_profiles,
    )

    # Add reference profile metadata.
    results[
        "reference_calibrated_risk"
    ] = reference_risk

    results[
        "reference_priority"
    ] = reference_priority

    # Save detailed results.
    results.to_csv(
        CSV_PATH,
        index=False,
    )

    # Summary.
    summary = build_summary(
        results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # Plot.
    create_plot(
        results
    )

    # Print summary.
    print()
    print(
        "Sensitivity summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # Missing-data results.
    print()
    print(
        "Single-feature missingness:"
    )

    missing_results = results[
        results["analysis"]
        == "single_feature_missing"
    ][
        [
            "feature_changed",
            "calibrated_risk",
            "risk_change_from_reference",
            "priority",
        ]
    ]

    print(
        missing_results.to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {CSV_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        f"  {PLOT_PATH}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  This analysis describes model behavior only."
    )

    print(
        "  Changes in model score are not causal effects."
    )

    print(
        "  Priority thresholds are research/resource-"
        "oriented and are not clinical cutoffs."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()

