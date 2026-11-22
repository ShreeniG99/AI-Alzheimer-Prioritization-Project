# Suggested visit windows from the decline rate

## The problem

Protocols often give every visit the same tight window, for example *Day 14 ± 1*.
When a participant can't come on that day, the coordinator has to choose between a
protocol deviation and a missed visit. For safety, dosing or pharmacokinetic visits
a tight window is necessary. For a visit whose purpose is to **measure cognition or
function**, the window only has to be narrow enough that the person cannot change
meaningfully between the earliest and the latest allowed day. How narrow that is
depends on how fast the person is declining, and that is what TEMPO estimates.

## The rule

```
half-width (days) = drift budget / upper decline rate × 365      clamped to ±3 … ±28 days
```

- **Upper decline rate**: the upper end of TEMPO's 80% prediction interval, in CDR-SB
  points per year. Using the upper end sizes the window for someone declining faster
  than the point estimate.
- **Drift budget**: 0.05 CDR-SB points across half the window, one tenth of the
  smallest step a CDR-SB score can take (0.5).
- **Clamp**: never tighter than ±3 days, never wider than ±28 days. Rates under 0.05
  points/year are treated as 0.05.

| Upper decline rate (points/year) | Suggested window | Expected drift at the window edge |
|---|---|---|
| 0.5 (typical STABLE patient after adding the ≈ 0.5 interval) | ±28 days (cap) | 0.038 |
| 1.0 | ±18 days | 0.049 |
| 2.0 (typical FAST patient) | ±9 days | 0.049 |
| 6.0 or more | ±3 days (floor) | ≤ 0.05 |

On ADNI, 70% of the Stage-1 cohort is STABLE (< 0.3 points/year), so most people
would get a wider window than a one-size-fits-all *± 1 day*, and the tight windows
go to the people whose scores can actually move.

## Where it appears

- API: `POST /assessment`, `GET /patients/{id}` → `decline.visit_window`
  (`half_width_days`, `label`, `basis`, `scope`).
- Dashboard: *Suggested assessment visit window* under the combined outlook.
- Code: `backend/services/visit_window_service.py`; tests in
  `tests/test_visit_window.py`.

## What it is not

- **Not validated.** The drift budget is a design choice. Validating it needs
  test–retest data: how much CDR-SB (or MMSE/ADAS-Cog) varies between two visits a
  few weeks apart, by decline tier. ADNI visit-date deviations from the schedule are
  a possible retrospective source.
- **Not for safety, dosing, PK or imaging windows.** Those follow the protocol.
- **Not a scheduling system.** It is one input for protocol designers and for the
  clinician or coordinator deciding whether a requested change is reasonable.
