from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


from common import artifacts_dir, processed_dir, reports_dir

DATA_PATH = processed_dir() / "stage1.csv"
REPORTS_DIR = reports_dir()

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "education",
    "mmse",
    "adas",
    "faq",
    "moca",
]

CATEGORICAL_FEATURES = [
    "sex",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Stage-1 dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("PHASE 4B - FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ---------------------------------------------------------
    # 1. Overall numeric feature summary
    # ---------------------------------------------------------
    summary_rows = []

    for feature in NUMERIC_FEATURES:
        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce")
        valid = series.dropna()

        if len(valid) == 0:
            continue

        summary_rows.append(
            {
                "feature": feature,
                "count": int(valid.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "q1": float(valid.quantile(0.25)),
                "median": float(valid.median()),
                "q3": float(valid.quantile(0.75)),
                "max": float(valid.max()),
                "iqr": float(
                    valid.quantile(0.75)
                    - valid.quantile(0.25)
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = REPORTS_DIR / "feature_distribution_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---------------------------------------------------------
    # 2. Numeric distributions by target
    # ---------------------------------------------------------
    by_target_rows = []

    if "target" in df.columns:
        for feature in NUMERIC_FEATURES:
            if feature not in df.columns:
                continue

            for target_value in sorted(df["target"].dropna().unique()):
                subset = pd.to_numeric(
                    df.loc[
                        df["target"] == target_value,
                        feature,
                    ],
                    errors="coerce",
                ).dropna()

                if len(subset) == 0:
                    continue

                by_target_rows.append(
                    {
                        "target": int(target_value),
                        "feature": feature,
                        "count": int(len(subset)),
                        "mean": float(subset.mean()),
                        "std": float(subset.std()),
                        "min": float(subset.min()),
                        "median": float(subset.median()),
                        "max": float(subset.max()),
                    }
                )

    by_target_df = pd.DataFrame(by_target_rows)

    by_target_path = (
        REPORTS_DIR / "feature_distribution_by_target.csv"
    )
    by_target_df.to_csv(by_target_path, index=False)

    # ---------------------------------------------------------
    # 3. Categorical feature distribution
    # ---------------------------------------------------------
    categorical_rows = []

    for feature in CATEGORICAL_FEATURES:
        if feature not in df.columns:
            continue

        counts = df[feature].value_counts(
            dropna=False
        )

        total = len(df)

        for category, count in counts.items():
            category_label = (
                "MISSING"
                if pd.isna(category)
                else str(category)
            )

            categorical_rows.append(
                {
                    "feature": feature,
                    "category": category_label,
                    "count": int(count),
                    "percent": float(
                        count / total * 100
                    ),
                }
            )

    categorical_df = pd.DataFrame(categorical_rows)

    categorical_path = (
        REPORTS_DIR / "categorical_distribution_summary.csv"
    )
    categorical_df.to_csv(
        categorical_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Outlier analysis using IQR
    # ---------------------------------------------------------
    outlier_rows = []

    for feature in NUMERIC_FEATURES:
        if feature not in df.columns:
            continue

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[
            (series < lower) | (series > upper)
        ]

        outlier_rows.append(
            {
                "feature": feature,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(len(outliers)),
                "outlier_percent": float(
                    len(outliers) / len(series) * 100
                ),
            }
        )

    outlier_df = pd.DataFrame(outlier_rows)

    outlier_path = (
        REPORTS_DIR / "feature_outlier_analysis.csv"
    )
    outlier_df.to_csv(
        outlier_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Distribution plots
    # ---------------------------------------------------------
    numeric_plot_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature in df.columns
    ]

    for feature in numeric_plot_features:
        plt.figure(figsize=(8, 5))

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        plt.hist(series, bins=20)

        plt.title(
            f"{feature.upper()} Distribution"
        )
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()

        output_path = (
            REPORTS_DIR
            / f"distribution_{feature}.png"
        )

        plt.savefig(output_path, dpi=150)
        plt.close()

    # ---------------------------------------------------------
    # 6. Target-separated distributions
    # ---------------------------------------------------------
    if "target" in df.columns:

        for feature in numeric_plot_features:
            plt.figure(figsize=(8, 5))

            for target_value in [0, 1]:
                subset = pd.to_numeric(
                    df.loc[
                        df["target"] == target_value,
                        feature,
                    ],
                    errors="coerce",
                ).dropna()

                if len(subset) > 0:
                    plt.hist(
                        subset,
                        bins=20,
                        alpha=0.5,
                        label=f"Target {target_value}",
                    )

            plt.title(
                f"{feature.upper()} Distribution by Target"
            )
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            output_path = (
                REPORTS_DIR
                / f"distribution_{feature}_by_target.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
            )
            plt.close()

    # ---------------------------------------------------------
    # 7. Print important findings
    # ---------------------------------------------------------
    print()
    print("NUMERIC FEATURE SUMMARY")
    print(summary_df.to_string(index=False))

    print()
    print("OUTLIER ANALYSIS")
    print(outlier_df.to_string(index=False))

    print()
    print("CATEGORICAL DISTRIBUTIONS")
    print(categorical_df.to_string(index=False))

    print()
    print("FILES CREATED")
    print(summary_path)
    print(by_target_path)
    print(categorical_path)
    print(outlier_path)

    print()
    print("PHASE 4B COMPLETE")


if __name__ == "__main__":
    main()
