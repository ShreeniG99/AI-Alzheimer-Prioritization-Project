from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from common import artifacts_dir, processed_dir, reports_dir

DATA_PATH = processed_dir() / "stage1.csv"
MODEL_PATH = artifacts_dir() / "stage1_model.joblib"
REPORTS_DIR = reports_dir()

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CAPACITIES = [
    5,
    10,
    20,
    50,
    100,
    200,
    335,
]

RANDOM_SEED = 42


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def recall_at_k(
    ranked_target: np.ndarray,
    k: int,
) -> float:

    k = min(k, len(ranked_target))

    if ranked_target.sum() == 0:
        return 0.0

    return float(
        ranked_target[:k].sum()
        / ranked_target.sum()
    )


def precision_at_k(
    ranked_target: np.ndarray,
    k: int,
) -> float:

    k = min(k, len(ranked_target))

    if k == 0:
        return 0.0

    return float(
        ranked_target[:k].sum()
        / k
    )


def random_metrics(
    target: np.ndarray,
    k: int,
    rng: np.random.Generator,
    repeats: int = 1000,
) -> tuple[float, float]:

    recalls = []
    precisions = []

    n = len(target)

    for _ in range(repeats):

        indices = rng.choice(
            n,
            size=min(k, n),
            replace=False,
        )

        selected = target[indices]

        total_positive = target.sum()

        if total_positive > 0:
            recalls.append(
                selected.sum()
                / total_positive
            )

        if len(selected) > 0:
            precisions.append(
                selected.sum()
                / len(selected)
            )

    return (
        float(np.mean(recalls)),
        float(np.mean(precisions)),
    )


def clinical_baseline_score(df: pd.DataFrame) -> pd.Series:
    """
    Simple research baseline using cognitive/functional
    measures.

    Higher score = greater impairment.

    This is a baseline for comparison only and is not
    a clinical scoring system.
    """

    mmse_component = (
        30 - df["mmse"]
    )

    adas_component = df["adas"]

    faq_component = df["faq"]

    score = (
        mmse_component
        + adas_component
        + faq_component
    )

    return score


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print()
print("=" * 70)
print("PHASE 3B - RETROSPECTIVE PRIORITIZATION EVALUATION")
print("=" * 70)
print()

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print(
    f"Loaded dataset: {DATA_PATH}"
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {list(df.columns)}"
)

print()


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
calibrator = artifact["calibrator"]

features = artifact["features"]

print(
    f"Model: {artifact['selected_model']}"
)

print(
    f"Stage: {artifact['stage']}"
)

print(
    f"Features: {features}"
)

print()


# ---------------------------------------------------------
# Validate target
# ---------------------------------------------------------

if "target" not in df.columns:
    raise ValueError(
        "The processed dataset does not contain a "
        "'target' column."
    )


target = df["target"].astype(int).to_numpy()

print(
    "Target distribution:"
)

print(
    pd.Series(target)
    .value_counts()
    .sort_index()
)

print()


# ---------------------------------------------------------
# Generate calibrated risk scores
# ---------------------------------------------------------

X = df[features]

raw_scores = pipeline.predict_proba(X)[:, 1]

risk_scores = calibrator.transform(
    raw_scores
)

risk_scores = np.asarray(
    risk_scores,
    dtype=float,
)


df["risk_score"] = risk_scores


# ---------------------------------------------------------
# Model ranking
# ---------------------------------------------------------

model_ranked = (
    df.sort_values(
        "risk_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

model_target = (
    model_ranked["target"]
    .astype(int)
    .to_numpy()
)


# ---------------------------------------------------------
# Clinical baseline ranking
# ---------------------------------------------------------

baseline_df = df.copy()

baseline_df["clinical_baseline_score"] = (
    clinical_baseline_score(
        baseline_df
    )
)

baseline_ranked = (
    baseline_df.sort_values(
        "clinical_baseline_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

baseline_target = (
    baseline_ranked["target"]
    .astype(int)
    .to_numpy()
)


# ---------------------------------------------------------
# Overall cohort statistics
# ---------------------------------------------------------

total_patients = len(df)

total_positive = int(
    target.sum()
)

overall_rate = (
    total_positive / total_patients
    if total_patients > 0
    else 0.0
)

print("=" * 70)
print("COHORT")
print("=" * 70)

print(
    f"Total participants: {total_patients}"
)

print(
    f"Target-positive participants: {total_positive}"
)

print(
    f"Target rate: {overall_rate:.4%}"
)

print()


# ---------------------------------------------------------
# Evaluate capacities
# ---------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)

results = []

for capacity in CAPACITIES:

    k = min(
        capacity,
        total_patients,
    )

    # -------------------------
    # AI/model prioritization
    # -------------------------

    model_recall = recall_at_k(
        model_target,
        k,
    )

    model_precision = precision_at_k(
        model_target,
        k,
    )

    model_lift = (
        model_precision / overall_rate
        if overall_rate > 0
        else 0.0
    )

    # -------------------------
    # Clinical baseline
    # -------------------------

    baseline_recall = recall_at_k(
        baseline_target,
        k,
    )

    baseline_precision = precision_at_k(
        baseline_target,
        k,
    )

    baseline_lift = (
        baseline_precision / overall_rate
        if overall_rate > 0
        else 0.0
    )

    # -------------------------
    # Random baseline
    # -------------------------

    random_recall, random_precision = (
        random_metrics(
            target,
            k,
            rng,
        )
    )

    random_lift = (
        random_precision / overall_rate
        if overall_rate > 0
        else 0.0
    )

    # -------------------------
    # Store
    # -------------------------

    results.append(
        {
            "capacity": capacity,
            "selection_fraction": (
                k / total_patients
            ),
            "model_recall": model_recall,
            "model_precision": model_precision,
            "model_lift": model_lift,
            "baseline_recall": baseline_recall,
            "baseline_precision": baseline_precision,
            "baseline_lift": baseline_lift,
            "random_recall": random_recall,
            "random_precision": random_precision,
            "random_lift": random_lift,
        }
    )


results_df = pd.DataFrame(
    results
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_path = (
    REPORTS_DIR
    / "prioritization_evaluation.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 70)
print("PRIORITIZATION RESULTS")
print("=" * 70)
print()

display_df = results_df.copy()

for column in [
    "selection_fraction",
    "model_recall",
    "model_precision",
    "model_lift",
    "baseline_recall",
    "baseline_precision",
    "baseline_lift",
    "random_recall",
    "random_precision",
    "random_lift",
]:

    display_df[column] = (
        display_df[column]
        .astype(float)
        .round(4)
    )

print(
    display_df.to_string(
        index=False
    )
)

print()

print(
    f"Saved results to: {results_path}"
)

print()


# ---------------------------------------------------------
# Cumulative recall curve
# ---------------------------------------------------------

selection_percent = (
    np.arange(
        1,
        total_patients + 1,
    )
    / total_patients
    * 100
)

cumulative_model = (
    np.cumsum(model_target)
    / total_positive
    * 100
)

cumulative_baseline = (
    np.cumsum(baseline_target)
    / total_positive
    * 100
)

random_curve = selection_percent


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    selection_percent,
    cumulative_model,
    label="AI model",
)

plt.plot(
    selection_percent,
    cumulative_baseline,
    label="Clinical baseline",
)

plt.plot(
    selection_percent,
    random_curve,
    label="Random selection",
    linestyle="--",
)

plt.xlabel(
    "Percentage of cohort selected"
)

plt.ylabel(
    "Recall of 24-month progressors (%)"
)

plt.title(
    "Cumulative Gains for Retrospective Prioritization"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

curve_path = (
    REPORTS_DIR
    / "prioritization_recall_curve.png"
)

plt.savefig(
    curve_path,
    dpi=200,
)

plt.close()


print(
    f"Saved curve to: {curve_path}"
)

print()


# ---------------------------------------------------------
# Top-K ranked cohort export
# ---------------------------------------------------------

ranked_path = (
    REPORTS_DIR
    / "prioritized_cohort.csv"
)

model_ranked[
    [
        column
        for column in [
            "RID",
            "risk_score",
            "target",
            "education",
            "mmse",
            "adas",
            "faq",
            "sex",
        ]
        if column in model_ranked.columns
    ]
].to_csv(
    ranked_path,
    index=False,
)


print(
    f"Saved ranked cohort to: {ranked_path}"
)

print()

print("=" * 70)
print("PHASE 3B COMPLETE")
print("=" * 70)