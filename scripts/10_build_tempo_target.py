"""
TEMPO target: annual rate of change in CDR Sum of Boxes (CDR-SB).

For every participant with a baseline diagnosis (same baseline as the
pathway models), fits an ordinary least-squares line through their CDR-SB
visits from baseline to 36 months and stores the slope in points per year.

Output: data/processed/tempo_target.csv
    RID, cdrsb_slope, cdrsb_baseline, n_visits, span_years

CDR-SB is only the outcome; it is never used as a model input.

Usage:
    python scripts/10_build_tempo_target.py
"""

import numpy as np
import pandas as pd

from common import ensure_rid, find_column, load_config, processed_dir, read_prefix


def baseline_dates():
    dx = ensure_rid(read_prefix("DXSUM_"))
    datecol = find_column(dx, ["EXAMDATE", "EXAMDATE2", "USERDATE"])
    dx["_date"] = pd.to_datetime(dx[datecol], errors="coerce")
    return dx.dropna(subset=["_date"]).sort_values("_date").groupby("RID")["_date"].first().rename("baseline_date")


def ols_slope(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    centred = years - years.mean()
    return float((centred * (values - values.mean())).sum() / (centred ** 2).sum())


def build(cfg):
    t = cfg["tempo"]
    cdr = ensure_rid(read_prefix("CDR_"))
    datecol = find_column(cdr, ["VISDATE", "EXAMDATE", "USERDATE"])
    score = find_column(cdr, ["CDRSB", "CDR_SB", "CDRSUM"])
    visits = pd.DataFrame({
        "RID": cdr["RID"],
        "date": pd.to_datetime(cdr[datecol], errors="coerce"),
        "cdrsb": pd.to_numeric(cdr[score], errors="coerce"),
    })
    # CDR-SB ranges 0-18; negative values are ADNI missing codes.
    visits = visits[visits["cdrsb"].between(0, 18)].dropna(subset=["date"])
    visits = visits.merge(baseline_dates(), left_on="RID", right_index=True)
    visits["months"] = (visits["date"] - visits["baseline_date"]).dt.days / 30.4375

    lo = -t["baseline_tolerance_months"]
    hi = t["window_months"] + t["window_tolerance_months"]
    visits = visits[visits["months"].between(lo, hi)]
    visits = visits.sort_values(["RID", "months"]).drop_duplicates(["RID", "date"])

    rows = []
    for rid, group in visits.groupby("RID"):
        if len(group) < t["min_visits"]:
            continue
        if group["months"].iloc[0] > t["baseline_tolerance_months"]:
            continue  # no CDR-SB at baseline
        span = group["months"].iloc[-1] - group["months"].iloc[0]
        if span < t["min_span_months"]:
            continue
        rows.append({
            "RID": rid,
            "cdrsb_slope": round(ols_slope(group["months"] / 12.0, group["cdrsb"]), 4),
            "cdrsb_baseline": float(group["cdrsb"].iloc[0]),
            "n_visits": int(len(group)),
            "span_years": round(span / 12.0, 3),
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    target = build(cfg)
    processed_dir().mkdir(parents=True, exist_ok=True)
    out = processed_dir() / "tempo_target.csv"
    target.to_csv(out, index=False)

    thresholds = cfg["tempo"]["speed_thresholds"]
    slope = target["cdrsb_slope"]
    print(f"Saved {out}")
    print(f"Participants: {len(target):,}")
    print(slope.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_string())
    print(f"Stable   (< {thresholds['moderate']}/yr): {(slope < thresholds['moderate']).mean():.1%}")
    print(f"Moderate          : {slope.between(thresholds['moderate'], thresholds['fast'], inclusive='left').mean():.1%}")
    print(f"Fast     (>= {thresholds['fast']}/yr): {(slope >= thresholds['fast']).mean():.1%}")


if __name__ == "__main__":
    main()
