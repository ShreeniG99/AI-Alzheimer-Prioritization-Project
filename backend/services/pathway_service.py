"""Progressive diagnostic pathway (PDF slide 8).

Stage 1 cognitive screening -> Stage 2 blood biomarkers -> Stage 3 MRI ->
Stage 4 PET. A patient "reaches" a stage once any of that stage's modality
inputs are supplied. Each reached stage with a trained model refines the
risk score; the deepest one determines the current priority tier.

The recommended next step is a workflow routing suggestion for clinician
review. It is never a diagnosis or treatment recommendation.
"""

from typing import Any, Dict, List

from backend.services.model_registry import STAGE_INFO, STAGES, registry
from backend.services.visit_window_service import suggest_visit_window


class ModelUnavailableError(RuntimeError):
    pass


def reached_stages(features: Dict[str, Any]) -> List[str]:
    reached = ["stage1"]
    for stage in STAGES[1:]:
        triggers = STAGE_INFO[stage].get("trigger_features", STAGE_INFO[stage]["modality_features"])
        if any(features.get(f) is not None for f in triggers):
            reached.append(stage)
    return reached


def next_step(stage: str, priority: str) -> Dict[str, Any]:
    index = STAGES.index(stage)
    following = STAGES[index + 1] if index + 1 < len(STAGES) else None
    following_name = STAGE_INFO[following]["name"] if following else None
    following_deployed = bool(following and registry.get(following))

    if priority == "HIGH":
        if following:
            action = f"Prioritise for Stage {index + 2}: {following_name}"
            detail = (
                f"Highest priority tier at {STAGE_INFO[stage]['name'].lower()}. Subject to clinician "
                f"review, local protocol and capacity, schedule Stage {index + 2} ({following_name}) and enter "
                "the results to refine the score."
            )
        else:
            action = "Prioritise for specialist neurology review"
            detail = (
                "Highest priority after PET. Refer for specialist review; any assessment of "
                "eligibility for further care or therapy is a clinician decision."
            )
    elif priority == "MEDIUM":
        action = "Clinician review"
        detail = (
            "Intermediate priority. Review with the clinical team"
            + (f" and consider Stage {index + 2} ({following_name}) if capacity allows." if following else ".")
        )
    else:
        action = "Routine pathway"
        detail = "Lower priority. Continue routine clinician-led follow-up and re-screen per local protocol."

    return {
        "action": action,
        "detail": detail,
        "next_stage": following if priority in ("HIGH", "MEDIUM") else None,
        "next_stage_model_available": following_deployed,
    }


OUTLOOK = {
    # (priority is high?, decline is fast?) -> combined workflow outlook
    (True, True): ("Test first, refer urgently",
                   "High progression risk and fast expected decline: prioritise diagnostic work-up "
                   "and specialist review."),
    (True, False): ("Test soon, routine follow-up",
                    "High progression risk but slower expected decline: prioritise testing; "
                    "follow up on the routine schedule."),
    (False, True): ("Re-check sooner",
                    "Lower progression risk but faster expected decline: shorten the interval "
                    "to the next cognitive re-assessment."),
    (False, False): ("Routine pathway",
                     "Lower progression risk and slow expected decline: continue routine "
                     "clinician-led follow-up."),
}


def decline_outlook(priority: str, tier: str) -> Dict[str, str]:
    """Combine who-to-test-first (priority) with how-fast (TEMPO speed tier).

    MEDIUM priority counts as lower risk, MODERATE decline as slower; the
    detailed tiers are still shown alongside.
    """
    label, detail = OUTLOOK[(priority == "HIGH", tier == "FAST")]
    return {"label": label, "detail": detail}


def estimate_decline(features: Dict[str, Any], stages: List[str], priority: str, explain: bool) -> Dict[str, Any] | None:
    """TEMPO estimate from the deepest reached stage that has a TEMPO model."""
    stage = next((s for s in reversed(stages) if registry.get_tempo(s)), None)
    if stage is None:
        return None
    model = registry.get_tempo(stage)
    result = model.predict(features)
    result["outlook"] = decline_outlook(priority, result["speed_tier"])
    result["visit_window"] = suggest_visit_window(result["interval_high"])
    if explain:
        result["explanation"] = model.explain(features)
    return result


def evaluate_pathway(features: Dict[str, Any], explain: bool = True) -> Dict[str, Any]:
    if registry.get("stage1") is None:
        raise ModelUnavailableError("No Stage-1 model is deployed. Train it with scripts/03_train.py.")

    reached = reached_stages(features)
    trace = []
    for stage in reached:
        model = registry.get(stage)
        if model is None:
            trace.append({
                "stage": stage,
                "name": STAGE_INFO[stage]["name"],
                "available": False,
                "note": "Inputs supplied but no trained model is deployed for this stage.",
            })
            continue
        trace.append({"name": STAGE_INFO[stage]["name"], "available": True, **model.predict(features)})

    scored = [t for t in trace if t["available"]]
    current = scored[-1]
    model = registry.get(current["stage"])

    return {
        "current_stage": current["stage"],
        "current_stage_name": current["name"],
        "risk_score": current["risk_score"],
        "priority": current["priority"],
        "model": current["model"],
        "model_version": current["model_version"],
        "stages_reached": reached,
        "trace": trace,
        "next_step": next_step(current["stage"], current["priority"]),
        "decline": estimate_decline(features, [t["stage"] for t in scored], current["priority"], explain),
        "explanation": model.explain(features) if explain else None,
    }
