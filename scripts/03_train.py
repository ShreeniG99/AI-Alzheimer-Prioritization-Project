import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)

from common import ROOT, load_config, processed_dir, artifacts_dir, reports_dir
import sys
sys.path.insert(0, str(ROOT))
from ml.model_utils import (
    build_preprocessor, make_models, PlattCalibrator, expected_calibration_error,
    top_k_recall, priority_thresholds, save_bundle, feature_names_from_pipeline
)

CORE_FEATURES = ["age", "education", "mmse", "adas", "faq", "sex"]


def evaluate(y, p, top_k_fraction):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "specificity": float(tn/(tn+fp)) if (tn+fp) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "top_k_recall": top_k_recall(y, p, top_k_fraction),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def main(stage):
    cfg = load_config()
    path = processed_dir() / f"{stage}.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Run 02_build_dataset.py first.")

    df = pd.read_csv(path)
    feature_cfg = cfg["features"][stage]
    numeric = [c for c in feature_cfg["numeric"] if c in df.columns]
    categorical = [c for c in feature_cfg["categorical"] if c in df.columns]
    features = numeric + categorical

    if not features:
        raise SystemExit("No model features were found in the processed dataset.")

    X = df[features].copy()
    y = df["target"].astype(int).copy()

    # Remove completely missing features.
    numeric = [c for c in numeric if X[c].notna().any()]
    categorical = [c for c in categorical if X[c].notna().any()]
    features = numeric + categorical

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=cfg["training"]["test_size"] + cfg["training"]["validation_size"],
        stratify=y, random_state=cfg["training"]["random_state"]
    )
    relative_test = cfg["training"]["test_size"] / (
        cfg["training"]["test_size"] + cfg["training"]["validation_size"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test,
        stratify=y_tmp, random_state=cfg["training"]["random_state"]
    )

    results = []
    fitted = {}
    missing_indicators = bool(cfg["training"].get("missing_indicators", False))

    for name, estimator in make_models(cfg["training"]["random_state"]).items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical, missing_indicators)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        p_val_raw = pipe.predict_proba(X_val)[:,1]

        cal = PlattCalibrator().fit(p_val_raw, y_val)
        p_val = cal.transform(p_val_raw)

        m = evaluate(y_val, p_val, cfg["training"]["top_k_fraction"])
        interpretability = {
            "logistic_regression": 1.0,
            "random_forest": 0.8,
            "xgboost": 0.7,
        }.get(name, 0.7)
        calibration_score = max(0.0, 1.0 - m["ece"])
        w = cfg["selection_weights"]
        selection_score = (
            w["pr_auc"] * m["pr_auc"] +
            w["top_k_recall"] * m["top_k_recall"] +
            w["calibration"] * calibration_score +
            w["interpretability"] * interpretability
        )
        m.update({
            "model": name,
            "selection_score": float(selection_score),
            "interpretability_score": interpretability,
        })
        results.append(m)
        fitted[name] = (pipe, cal)

    comparison = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    reports_dir().mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir() / f"{stage}_model_comparison.csv", index=False)

    selected_name = str(comparison.iloc[0]["model"])

    # Refit selected model on the full train+validation set.
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)

    estimator = make_models(cfg["training"]["random_state"])[selected_name]
    final_pipe = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical, missing_indicators)),
        ("model", estimator),
    ])

    # Out-of-fold probabilities for calibration, with model refit after.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["training"]["random_state"])
    p_oof = cross_val_predict(
        final_pipe, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=1
    )[:,1]
    final_cal = PlattCalibrator().fit(p_oof, y_dev)
    final_pipe.fit(X_dev, y_dev)

    p_test_raw = final_pipe.predict_proba(X_test)[:,1]
    p_test = final_cal.transform(p_test_raw)
    test_metrics = evaluate(y_test, p_test, cfg["training"]["top_k_fraction"])

    thresholds = priority_thresholds(
        final_cal.transform(p_oof),
        cfg["priority"]["high_capacity_fraction"],
        cfg["priority"]["medium_capacity_fraction"],
    )

    model_version = f"{stage}-{selected_name}-{datetime.now(timezone.utc):%Y%m%d}"

    cohort = {
        "participants": int(len(df)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        # Completeness of the core screening inputs (comorbidities and other
        # extras are optional and often unrecorded).
        "complete_input_fraction": float(
            X[[f for f in features if f in CORE_FEATURES]].notna().all(axis=1).mean()
        ),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }

    # Small background sample of development data for SHAP explanations.
    background = X_dev[features].sample(
        n=min(200, len(X_dev)), random_state=cfg["training"]["random_state"]
    ).reset_index(drop=True)

    bundle = {
        "stage": stage,
        "model_version": model_version,
        "cohort": cohort,
        "background": background,
        "pipeline": final_pipe,
        "calibrator": final_cal,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "priority_thresholds": thresholds,
        "selected_model": selected_name,
        "training_config": cfg["training"],
        "selection_weights": cfg["selection_weights"],
    }
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    save_bundle(artifacts_dir() / f"{stage}_model.joblib", bundle)

    metadata = {
        "stage": stage,
        "selected_model": selected_name,
        "model_version": model_version,
        "features": features,
        "cohort": cohort,
        "priority_thresholds": thresholds,
        "validation_comparison": comparison.to_dict(orient="records"),
        "untouched_test_metrics": test_metrics,
        "target_definition": cfg["target"],
        "note": "Research prototype; not a validated clinical diagnostic system.",
    }
    with open(artifacts_dir() / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=float)

    # Test ROC/PR curves.
    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC={test_metrics['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{stage}: ROC curve on untouched test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir() / f"{stage}_roc.png", dpi=160)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    plt.figure(figsize=(7,5))
    plt.plot(rec, prec, label=f"AP={test_metrics['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{stage}: Precision-recall curve on untouched test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir() / f"{stage}_pr.png", dpi=160)
    plt.close()

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nSELECTED:", selected_name)
    print("\nUNTOUCHED TEST METRICS")
    print(json.dumps(test_metrics, indent=2))
    print("\nPriority thresholds:", thresholds)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(load_config()["features"]), required=True)
    args = ap.parse_args()
    main(args.stage)
