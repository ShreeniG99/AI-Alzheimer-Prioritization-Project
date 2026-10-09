from typing import Any, Dict, List, Optional

from backend.settings import SCOPE


def rank_scored(
    entries: List[Dict[str, Any]],
    capacity: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rank already-scored patients from highest to lowest calibrated risk and
    mark the top ``capacity`` as selected for a simulated limited resource
    (e.g. available blood-test, MRI or PET slots).

    This is a research prioritization function. It does not make a clinical
    diagnosis or treatment decision.
    """

    if capacity is not None and capacity < 1:
        raise ValueError("Capacity must be greater than or equal to 1.")

    results = sorted(entries, key=lambda x: x["risk_score"], reverse=True)
    total = len(results)
    selected_count = min(capacity, total) if capacity is not None else 0

    for index, patient in enumerate(results, start=1):
        patient["rank"] = index
        patient["percentile"] = round(100 * (total - index) / (total - 1), 2) if total > 1 else 100.0
        patient["selected_for_capacity"] = index <= selected_count

    risk_scores = [p["risk_score"] for p in results]

    def count(priority):
        return sum(p["priority"] == priority for p in results)

    return {
        "total_patients": total,
        "capacity": capacity,
        "selected_count": selected_count,
        "cohort_summary": {
            "mean_risk": round(sum(risk_scores) / total, 6) if total else 0.0,
            "minimum_risk": round(min(risk_scores), 6) if total else 0.0,
            "maximum_risk": round(max(risk_scores), 6) if total else 0.0,
            "high_priority_count": count("HIGH"),
            "medium_priority_count": count("MEDIUM"),
            "low_priority_count": count("LOW"),
        },
        "patients": results,
        "scope": SCOPE + " Ranking supports prioritization research only.",
    }
