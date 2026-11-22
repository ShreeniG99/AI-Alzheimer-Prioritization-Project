"""Visit-window suggestion from the TEMPO decline rate.

A protocol visit such as "Day 14 +/- 1" fixes the same window for every
participant. For a cognitive or functional assessment, the window only needs
to be tight enough that the patient cannot change meaningfully between the
earliest and latest allowed day. That depends on how fast the patient is
declining, which TEMPO estimates.

Rule (a design heuristic, not a validated standard):

    half-width (days) = drift_budget / upper_rate * 365

* ``upper_rate`` is the upper end of TEMPO's prediction interval (CDR-SB
  points per year), so the window is sized for a patient declining faster
  than the point estimate.
* ``drift_budget`` is how much expected CDR-SB change we accept across half
  the window. The default, 0.05 points, is one tenth of the smallest step a
  CDR-SB score can take (0.5).
* The result is clamped to [MIN_DAYS, MAX_DAYS].

The suggestion applies only to visits whose purpose is measuring cognition
or function. Windows set for safety, dosing, pharmacokinetics or imaging
logistics are outside its scope and must follow the protocol.
"""

from typing import Any, Dict

DRIFT_BUDGET = 0.05        # CDR-SB points accepted across half the window
MIN_DAYS = 3               # never suggest tighter than +/- 3 days
MAX_DAYS = 28              # never suggest wider than +/- 4 weeks
RATE_FLOOR = 0.05          # points/year; avoids division by ~0 for stable patients

SCOPE_NOTE = (
    "Applies only to cognitive or functional assessment visits. Safety, dosing, "
    "pharmacokinetic and imaging windows follow the protocol. A research heuristic "
    "for protocol design and clinician review, not a validated standard."
)


def suggest_visit_window(
    upper_rate: float,
    drift_budget: float = DRIFT_BUDGET,
    min_days: int = MIN_DAYS,
    max_days: int = MAX_DAYS,
) -> Dict[str, Any]:
    """Suggested +/- days for an assessment visit, given an upper decline rate."""
    rate = max(float(upper_rate), RATE_FLOOR)
    raw = drift_budget / rate * 365.0
    days = int(max(min_days, min(max_days, raw)))
    return {
        "half_width_days": days,
        "label": f"±{days} days",
        "rate_used": rate,
        "drift_budget": drift_budget,
        "expected_drift_at_edge": round(rate * days / 365.0, 3),
        "basis": (
            f"At up to {rate:.2f} CDR-SB points/year, moving the visit {days} days "
            f"changes the expected score by at most {rate * days / 365.0:.3f} points."
        ),
        "scope": SCOPE_NOTE,
    }
