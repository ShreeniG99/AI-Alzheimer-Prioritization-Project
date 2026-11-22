"""
TEMPO: train the progression-rate model for one pathway stage.

Predicts each participant's annual change in CDR-SB (points per year)
from the same baseline inputs as that stage's prioritization model.

- Candidates: ridge regression, random forest, XGBoost.
- Participant-level 60/20/20 split, stratified by observed speed tier.
- Selection on validation: Spearman rank correlation, fast-decliner capture
  (share of true FAST decliners in the predicted top 20%) and
  interpretability.
- The winner is refit on train+validation. An 80% prediction interval is
  set from 5-fold cross-validated absolute residuals (split conformal).
- Evaluated once on the untouched test split, against two baselines:
  predicting the mean, and ranking by a simple impairment sum.

Outputs:
    artifacts/tempo_<stage>_model.joblib, artifacts/tempo_<stage>_metadata.json
    reports/tempo_<stage>_model_comparison.csv, reports/tempo_<stage>_observed_vs_predicted.png

Usage:
    python scripts/11_train_tempo.py --stage stage1
"""

import argparse
import json
import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from common import artifacts_dir, load_config, processed_dir, reports_dir
from ml.model_utils import build_preprocessor, make_regressors, save_bundle, speed_tier

INTERPRETABILITY = {"ridge_regression": 1.0, "random_forest": 0.8, "xgboost": 0.7}


def fast_capture(y_true, score, fast_threshold, fraction):
    """Share of truly fast decliners found in the top `fraction` by score."""
    y_true = np.asarray(y_true, dtype=float)
    fast = y_true >= fast_threshold
    if fast.sum() == 0:
        return float("nan")
    k = max(1, int(math.ceil(len(y_true) * fraction)))
    top = np.argsort(-np.asarray(score, dtype=float))[:k]
    return float(fast[top].sum() / fast.sum())


def evaluate(y, pred, t):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    thresholds = t["speed_thresholds"]
    true_tier = np.array([speed_tier(v, thresholds) for v in y])
    pred_tier = np.array([speed_tier(v, thresholds) for v in pred])
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        # Undefined for constant predictions (e.g. the mean baseline).
        "spearman": None if np.ptp(pred) == 0 else float(spearmanr(y, pred).statistic),
        "fast_capture": fast_capture(y, pred, thresholds["fast"], t["top_fraction"]),
        "tier_accuracy": float((true_tier == pred_tier).mean()),
        "n": int(len(y)),
        "n_fast": int((y >= thresholds["fast"]).sum()),
    }


def impairment_sum(X):
    adas = X["adas"].fillna(X["adas"].median())
    faq = X["faq"].fillna(X["faq"].median())
    return (30 - X["mmse"].fillna(X["mmse"].median())) + adas + faq


def main(stage):
    cfg = load_config()
    t = cfg["tempo"]
    rs = cfg["training"]["random_state"]

    data_path = processed_dir() / f"{stage}.csv"
    target_path = processed_dir() / "tempo_target.csv"
    for path, script in [(data_path, "02_build_dataset.py"), (target_path, "10_build_tempo_target.py")]:
        if not path.exists():
            raise SystemExit(f"{path} does not exist. Run {script} first.")

    df = pd.read_csv(data_path).merge(pd.read_csv(target_path), on="RID", how="inner")
    feature_cfg = cfg["features"][stage]
    numeric = [c for c in feature_cfg["numeric"] if c in df.columns and df[c].notna().any()]
    categorical = [c for c in feature_cfg["categorical"] if c in df.columns and df[c].notna().any()]
    features = numeric + categorical
    X = df[features]
    y = df["cdrsb_slope"].astype(float)
    tiers = y.map(lambda v: speed_tier(v, t["speed_thresholds"]))

    test_size = cfg["training"]["test_size"]
    val_size = cfg["training"]["validation_size"]
    X_train, X_tmp, y_train, y_tmp, tier_train, tier_tmp = train_test_split(
        X, y, tiers, test_size=test_size + val_size, stratify=tiers, random_state=rs
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (test_size + val_size), stratify=tier_tmp, random_state=rs
    )

    def pipeline(estimator):
        return Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", estimator)])

    w = t["selection_weights"]
    results = []
    for name, estimator in make_regressors(rs).items():
        model = pipeline(estimator).fit(X_train, y_train)
        m = evaluate(y_val, model.predict(X_val), t)
        m["interpretability_score"] = INTERPRETABILITY.get(name, 0.7)
        m["selection_score"] = (
            w["spearman"] * m["spearman"]
            + w["fast_capture"] * m["fast_capture"]
            + w["interpretability"] * m["interpretability_score"]
        )
        m["model"] = name
        results.append(m)

    comparison = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    selected = str(comparison.iloc[0]["model"])

    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    final = pipeline(make_regressors(rs)[selected])

    # Split-conformal interval from cross-validated residuals on dev data.
    oof = cross_val_predict(final, X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=rs))
    halfwidth = float(np.quantile(np.abs(y_dev - oof), t["interval_coverage"]))
    final.fit(X_dev, y_dev)

    pred_test = final.predict(X_test)
    test_metrics = evaluate(y_test, pred_test, t)
    test_metrics["interval_coverage_observed"] = float((np.abs(y_test - pred_test) <= halfwidth).mean())

    baselines = {
        "mean_prediction": evaluate(y_test, np.full(len(y_test), y_dev.mean()), t),
        "impairment_sum_ranking": {
            "spearman": float(spearmanr(y_test, impairment_sum(X_test)).statistic),
            "fast_capture": fast_capture(y_test, impairment_sum(X_test), t["speed_thresholds"]["fast"], t["top_fraction"]),
        },
    }

    model_version = f"tempo-{stage}-{selected}-{datetime.now(timezone.utc):%Y%m%d}"
    cohort = {
        "participants": int(len(df)),
        "fast": int((y >= t["speed_thresholds"]["fast"]).sum()),
        "moderate": int(tiers.eq("MODERATE").sum()),
        "stable": int(tiers.eq("STABLE").sum()),
        "mean_slope": float(y.mean()),
        "median_slope": float(y.median()),
    }
    background = X_dev.sample(n=min(200, len(X_dev)), random_state=rs).reset_index(drop=True)

    bundle = {
        "kind": "tempo",
        "stage": stage,
        "pipeline": final,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "selected_model": selected,
        "model_version": model_version,
        "interval_halfwidth": halfwidth,
        "interval_coverage": t["interval_coverage"],
        "speed_thresholds": t["speed_thresholds"],
        "cohort": cohort,
        "background": background,
    }
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    save_bundle(artifacts_dir() / f"tempo_{stage}_model.joblib", bundle)

    metadata = {
        "stage": stage,
        "selected_model": selected,
        "model_version": model_version,
        "features": features,
        "cohort": cohort,
        "interval_halfwidth": halfwidth,
        "validation_comparison": comparison.to_dict(orient="records"),
        "untouched_test_metrics": test_metrics,
        "baselines": baselines,
        "target": "Annual change in CDR-SB (points/year), OLS slope over 0-36 months",
        "note": "Research prototype; not a validated clinical prognostic tool.",
    }
    with open(artifacts_dir() / f"tempo_{stage}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=float)

    reports_dir().mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir() / f"tempo_{stage}_model_comparison.csv", index=False)

    plt.figure(figsize=(6.5, 6))
    plt.scatter(pred_test, y_test, s=12, alpha=0.6)
    lim = [min(y_test.min(), pred_test.min()) - 0.2, max(y_test.max(), pred_test.max()) + 0.2]
    plt.plot(lim, lim, "--", color="grey")
    for v in t["speed_thresholds"].values():
        plt.axvline(v, color="lightgrey", lw=0.8)
    plt.xlabel("Predicted CDR-SB change (points/year)")
    plt.ylabel("Observed CDR-SB change (points/year)")
    plt.title(f"TEMPO {stage}: untouched test set (Spearman {test_metrics['spearman']:.2f})")
    plt.tight_layout()
    plt.savefig(reports_dir() / f"tempo_{stage}_observed_vs_predicted.png", dpi=160)
    plt.close()

    print("\nTEMPO MODEL COMPARISON (validation)")
    print(comparison[["model", "mae", "spearman", "fast_capture", "r2", "selection_score"]].round(3).to_string(index=False))
    print("\nSELECTED:", selected)
    print("\nUNTOUCHED TEST METRICS")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in test_metrics.items()}, indent=2))
    print("\nBASELINES (test)")
    print(json.dumps(baselines, indent=2, default=lambda v: round(v, 3)))
    print(f"\n{int(t['interval_coverage'] * 100)}% interval half-width: ±{halfwidth:.2f} points/year")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(load_config()["features"]), required=True)
    main(ap.parse_args().stage)
