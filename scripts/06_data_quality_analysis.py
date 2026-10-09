from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


from common import artifacts_dir, processed_dir, reports_dir

DATA_PATH = processed_dir() / "stage1.csv"

REPORTS_DIR = reports_dir()

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print()
print("=" * 70)
print("PHASE 4 - DATA QUALITY ANALYSIS")
print("=" * 70)
print()

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
# Feature groups
# ---------------------------------------------------------

stage1_features = [
    "education",
    "mmse",
    "adas",
    "faq",
    "sex",
]

contextual_features = [
    "moca",
]

all_features = (
    stage1_features
    + contextual_features
)


# ---------------------------------------------------------
# Overall missingness
# ---------------------------------------------------------

summary_rows = []

for feature in all_features:

    missing_count = int(
        df[feature].isna().sum()
    )

    available_count = (
        len(df) - missing_count
    )

    missing_percent = (
        missing_count
        / len(df)
        * 100
    )

    available_percent = (
        available_count
        / len(df)
        * 100
    )

    summary_rows.append(
        {
            "feature": feature,
            "missing_count": missing_count,
            "available_count": available_count,
            "missing_percent": missing_percent,
            "available_percent": available_percent,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------
# Missingness by target
# ---------------------------------------------------------

target_rows = []

for target_value in sorted(
    df["target"].dropna().unique()
):

    target_df = df[
        df["target"] == target_value
    ]

    target_count = len(target_df)

    for feature in all_features:

        missing_count = int(
            target_df[feature]
            .isna()
            .sum()
        )

        missing_percent = (
            missing_count
            / target_count
            * 100
            if target_count > 0
            else 0
        )

        target_rows.append(
            {
                "target": int(target_value),
                "target_count": target_count,
                "feature": feature,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )


missingness_target_df = pd.DataFrame(
    target_rows
)


# ---------------------------------------------------------
# Patient-level completeness
# ---------------------------------------------------------

df["stage1_missing_count"] = (
    df[stage1_features]
    .isna()
    .sum(axis=1)
)

df["stage1_complete"] = (
    df["stage1_missing_count"] == 0
)

complete_count = int(
    df["stage1_complete"].sum()
)

incomplete_count = (
    len(df) - complete_count
)

print("=" * 70)
print("STAGE-1 COMPLETENESS")
print("=" * 70)

print(
    f"Complete Stage-1 records: "
    f"{complete_count}"
)

print(
    f"Incomplete Stage-1 records: "
    f"{incomplete_count}"
)

print(
    f"Complete percentage: "
    f"{complete_count / len(df):.2%}"
)

print()


# ---------------------------------------------------------
# Print overall missingness
# ---------------------------------------------------------

print("=" * 70)
print("OVERALL MISSINGNESS")
print("=" * 70)

display_summary = summary_df.copy()

display_summary[
    "missing_percent"
] = display_summary[
    "missing_percent"
].round(2)

display_summary[
    "available_percent"
] = display_summary[
    "available_percent"
].round(2)

print(
    display_summary.to_string(
        index=False
    )
)

print()


# ---------------------------------------------------------
# Print missingness by target
# ---------------------------------------------------------

print("=" * 70)
print("MISSINGNESS BY TARGET")
print("=" * 70)

display_target = (
    missingness_target_df.copy()
)

display_target[
    "missing_percent"
] = display_target[
    "missing_percent"
].round(2)

print(
    display_target.to_string(
        index=False
    )
)

print()


# ---------------------------------------------------------
# Save CSV reports
# ---------------------------------------------------------

summary_path = (
    REPORTS_DIR
    / "data_quality_summary.csv"
)

target_path = (
    REPORTS_DIR
    / "missingness_by_target.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

missingness_target_df.to_csv(
    target_path,
    index=False,
)

print(
    f"Saved: {summary_path}"
)

print(
    f"Saved: {target_path}"
)

print()


# ---------------------------------------------------------
# Missingness bar chart
# ---------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    summary_df["feature"],
    summary_df["missing_percent"],
)

plt.ylabel(
    "Missing values (%)"
)

plt.xlabel(
    "Feature"
)

plt.title(
    "Feature Missingness - Stage 1 and Contextual Data"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

missingness_plot = (
    REPORTS_DIR
    / "data_quality.png"
)

plt.savefig(
    missingness_plot,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Missingness by target plot
# ---------------------------------------------------------

pivot = (
    missingness_target_df
    .pivot(
        index="feature",
        columns="target",
        values="missing_percent",
    )
    .fillna(0)
)

plt.figure(
    figsize=(10, 6)
)

pivot.plot(
    kind="bar",
)

plt.ylabel(
    "Missing values (%)"
)

plt.xlabel(
    "Feature"
)

plt.title(
    "Feature Missingness by 24-Month Target"
)

plt.xticks(
    rotation=30
)

plt.legend(
    title="Target"
)

plt.tight_layout()

target_plot = (
    REPORTS_DIR
    / "missingness_by_target.png"
)

plt.savefig(
    target_plot,
    dpi=200,
)

plt.close()


print(
    f"Saved: {missingness_plot}"
)

print(
    f"Saved: {target_plot}"
)

print()

print("=" * 70)
print("PHASE 4 DATA QUALITY ANALYSIS COMPLETE")
print("=" * 70)