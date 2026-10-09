from typing import Any, Dict, List, Optional

from backend.services.model_registry import STAGE_INFO

STAGE1_FEATURES = STAGE_INFO["stage1"]["modality_features"]

CONTEXTUAL_FEATURES = ["moca", "cdr"]


def calculate_quality(features: Dict[str, Any], model_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate completeness and basic quality warnings for the inputs of the
    model actually used (Stage 1 by default).

    This is a data-quality indicator only.
    It is not a measure of diagnostic certainty.
    """

    expected = list(model_features or STAGE1_FEATURES)

    available = [f for f in expected if features.get(f) is not None]
    missing = [f for f in expected if features.get(f) is None]

    total = len(expected)
    completeness = len(available) / total if total else 0.0

    if completeness == 1.0:
        level = "GOOD"
    elif completeness >= 0.6:
        level = "LIMITED"
    else:
        level = "POOR"

    warnings = [
        f"{feature} is missing; the model will use its imputation strategy and missingness indicator."
        for feature in missing
    ]

    if features.get("moca") is None:
        warnings.append(
            "MoCA is unavailable. This does not prevent prediction "
            "because MoCA is contextual only (not a model input)."
        )

    return {
        "available": len(available),
        "total": total,
        "completeness": round(completeness, 4),
        "level": level,
        "available_features": available,
        "missing_features": missing,
        "warnings": warnings,
        "contextual_data": {
            "moca_available": features.get("moca") is not None,
            "cdr_available": features.get("cdr") is not None,
        },
        "interpretation": (
            "Data completeness indicator only; "
            "not a measure of clinical certainty."
        ),
    }
