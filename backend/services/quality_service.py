from typing import Any, Dict, List, Optional

from backend.services.model_registry import STAGE_INFO

STAGE1_FEATURES = ["age", "education", "mmse", "adas", "faq", "sex"]

CONTEXTUAL_FEATURES = ["moca", "cdr"]


def calculate_quality(
    features: Dict[str, Any],
    model_features: Optional[List[str]] = None,
    stage_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate completeness and basic quality warnings for the inputs of the
    model actually used (Stage 1 by default).

    This is a data-quality indicator only.
    It is not a measure of diagnostic certainty.
    """

    model_features = list(model_features or STAGE1_FEATURES)

    # Quality is graded on the core screening inputs plus the measurements of
    # the stage actually used; comorbidities and other extras are optional.
    expected = [f for f in model_features if f in STAGE1_FEATURES or f in (stage_features or [])]
    optional = [f for f in model_features if f not in expected]

    available = [f for f in expected if features.get(f) is not None]
    missing = [f for f in expected if features.get(f) is None]
    optional_missing = [f for f in optional if features.get(f) is None]

    total = len(expected)
    completeness = len(available) / total if total else 0.0

    if completeness == 1.0:
        level = "GOOD"
    elif completeness >= 0.6:
        level = "LIMITED"
    else:
        level = "POOR"

    warnings = [
        f"{feature} is missing; the model will use its imputation strategy."
        for feature in missing
    ]

    if optional_missing:
        warnings.append(
            f"{len(optional_missing)} optional input(s) not recorded; "
            "typical values are imputed for them."
        )

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
        "optional_missing_features": optional_missing,
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
