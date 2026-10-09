import pandas as pd

from backend.settings import REPORTS_DIR

COHORT_PATH = REPORTS_DIR / "prioritized_cohort.csv"


def get_population_risk_distribution() -> dict:
    if not COHORT_PATH.exists():
        return {
            "available": False,
            "total_patients": 0,
            "risk_scores": [],
            "reason": "Run scripts/05_evaluate_prioritization.py to create the cohort risk report.",
        }

    df = pd.read_csv(COHORT_PATH)
    risk = pd.to_numeric(df["risk_score"], errors="coerce").dropna()

    return {
        "available": True,
        "total_patients": int(len(df)),
        "risk_scores": [round(float(value), 6) for value in risk.tolist()],
        "scope": (
            "Retrospective ADNI research cohort; "
            "risk distribution is for research analysis only."
        ),
    }
