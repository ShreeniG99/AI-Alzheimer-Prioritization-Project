"""
Inference analysis: the numbers behind "what should we infer from this project?"

Answers the review questions with aggregate figures only (counts, rates,
medians, AUCs). No participant-level row is printed or saved.

    cohort_overlap      Are the same participants in Stages 1-4?
    held_out_breakdown      What "top 20% captures 81%" means in patients; tiers;
                        the pair-wise meaning of ROC-AUC
    missed_vs_caught    Which converters does the Stage-1 ranking miss?
    subgroups           Stage-1 performance by sex, age, education, smoking
    education           Conversion rate and MMSE by years of education
    input_reduction     Can staff enter fewer Stage-1 inputs?
    stage_value         Same participants, with and without each stage's tests
    held_out_auc_ci         Bootstrap 95% interval of each stage's test ROC-AUC

Cross-validated figures refit logistic regression (the Stage-1 model family)
with the project's own preprocessing; test-split figures use the saved models.

Output: reports/inference_summary.json

Usage:
    python scripts/12_inference_analysis.py
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline

from common import artifacts_dir, load_config, processed_dir, reports_dir
from ml.model_utils import build_preprocessor, make_models, priority_from_risk, top_k_recall

CORE_TESTS = ["mmse", "adas", "faq"]
CORE_SCREENING = ["age", "education", "mmse", "adas", "faq"]


def load_stages(cfg):
    """Processed dataset and saved model bundle for every stage that exists."""
    stages = {}
    for stage in cfg["features"]:
        data_path = processed_dir() / f"{stage}.csv"
        model_path = artifacts_dir() / f"{stage}_model.joblib"
        if data_path.exists() and model_path.exists():
            stages[stage] = (pd.read_csv(data_path), joblib.load(model_path))
    if "stage1" not in stages:
        raise SystemExit("Stage 1 data and model are required. Run 02_build_dataset.py and 03_train.py first.")
    return stages


def held_out_split(df, bundle, cfg):
    """The untouched test split, reproduced exactly as in 03_train.py."""
    tr = cfg["training"]
    X, y = df[bundle["features"]], df["target"].astype(int)
    held_out = tr["test_size"] + tr["validation_size"]
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=held_out, stratify=y, random_state=tr["random_state"])
    _, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp, test_size=tr["test_size"] / held_out, stratify=y_tmp, random_state=tr["random_state"]
    )
    risk = bundle["calibrator"].transform(bundle["pipeline"].predict_proba(X_test)[:, 1])
    return X_test, y_test.to_numpy(), risk


def logistic_pipeline(numeric, categorical, cfg):
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", make_models(cfg["training"]["random_state"])["logistic_regression"]),
    ])


def cv_scores(df, numeric, categorical, cfg, repeats):
    """Repeated stratified 5-fold CV: mean ROC-AUC, PR-AUC and top-20% capture."""
    numeric = [c for c in numeric if c in df.columns and df[c].notna().any()]
    categorical = [c for c in categorical if c in df.columns and df[c].notna().any()]
    X, y = df[numeric + categorical], df["target"].astype(int).to_numpy()
    fraction = cfg["training"]["top_k_fraction"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=cfg["training"]["random_state"])
    template = logistic_pipeline(numeric, categorical, cfg)
    auc, pr, cap = [], [], []
    for train, test in cv.split(X, y):
        p = clone(template).fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])[:, 1]
        auc.append(roc_auc_score(y[test], p))
        pr.append(average_precision_score(y[test], p))
        cap.append(top_k_recall(y[test], p, fraction))
    return {
        "inputs": len(numeric) + len(categorical),
        "roc_auc": round(float(np.mean(auc)), 3), "roc_auc_sd": round(float(np.std(auc)), 3),
        "pr_auc": round(float(np.mean(pr)), 3), "pr_auc_sd": round(float(np.std(pr)), 3),
        "top20_capture": round(float(np.mean(cap)), 3),
    }


def cohort_overlap(stages):
    ids = {s: set(df["RID"]) for s, (df, _) in stages.items()}
    converters = {s: set(df.loc[df["target"] == 1, "RID"]) for s, (df, _) in stages.items()}
    labels = pd.concat([df[["RID", "target"]] for df, _ in stages.values()])
    return {
        "participants": {s: len(v) for s, v in ids.items()},
        "subset_of_stage1": {s: v <= ids["stage1"] for s, v in ids.items()},
        "in_every_stage": len(set.intersection(*ids.values())),
        "converters_in_every_stage": len(set.intersection(*converters.values())),
        "participants_with_conflicting_labels": int((labels.groupby("RID")["target"].nunique() > 1).sum()),
    }


def held_out_breakdown(df, bundle, cfg):
    _, y, risk = held_out_split(df, bundle, cfg)
    k = max(1, int(np.ceil(len(y) * cfg["training"]["top_k_fraction"])))
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-risk)[:k]] = True
    pos, neg = risk[y == 1], risk[y == 0]
    pairs = len(pos) * len(neg)
    tiers = np.array([priority_from_risk(r, bundle["priority_thresholds"]) for r in risk])
    return {
        "patients": int(len(y)), "converters": int(y.sum()),
        "top20_patients": int(k),
        "top20_converters": int(y[top].sum()),
        "top20_non_converters": int((y[top] == 0).sum()),
        "converters_outside_top20": int(y[~top].sum()),
        "expected_converters_if_random": round(float(y.sum() * k / len(y)), 1),
        "converter_vs_non_converter_pairs": int(pairs),
        "pairs_ranked_wrong": int((pos[:, None] < neg[None, :]).sum()),
        "tiers": {t: {"patients": int((tiers == t).sum()), "converters": int(y[tiers == t].sum())}
                  for t in ("HIGH", "MEDIUM", "LOW")},
    }


def out_of_fold_risk(df, bundle, cfg):
    """Stage-1 out-of-fold risk for the whole cohort (no one scored by a model that saw them)."""
    pipe = logistic_pipeline(bundle["numeric_features"], bundle["categorical_features"], cfg)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg["training"]["random_state"])
    return cross_val_predict(pipe, df[bundle["features"]], df["target"], cv=cv, method="predict_proba")[:, 1]


def missed_vs_caught(df, oof, cfg):
    cut = np.quantile(oof, 1 - cfg["training"]["top_k_fraction"])
    converters = df[df["target"] == 1]
    groups = {"missed": converters[oof[converters.index] < cut], "caught": converters[oof[converters.index] >= cut]}
    return {
        name: {
            "converters": int(len(g)),
            "share_mmse_28_or_more": round(float((g["mmse"] >= 28).mean()), 2),
            "share_faq_zero": round(float((g["faq"] == 0).mean()), 2),
            "median_adas": round(float(g["adas"].median()), 2),
            "median_mmse": float(g["mmse"].median()),
            "median_faq": float(g["faq"].median()),
            "share_female": round(float((g["sex"] == 2).mean()), 2),
        }
        for name, g in groups.items() if len(g)
    }


def subgroups(df, oof, cfg):
    groups = {
        "male": df["sex"] == 1, "female": df["sex"] == 2,
        "age_under_70": df["age"] < 70, "age_70_to_79": df["age"].between(70, 80, inclusive="left"),
        "age_80_plus": df["age"] >= 80,
        "education_12_or_less": df["education"] <= 12,
        "education_13_to_16": df["education"].between(12, 16, inclusive="right"),
        "education_over_16": df["education"] > 16,
    }
    if "hx_smoking" in df.columns:
        groups.update({"smoking_history": df["hx_smoking"] == 1, "no_smoking_history": df["hx_smoking"] == 0})
    rows = {}
    for name, mask in groups.items():
        y, p = df.loc[mask, "target"].to_numpy(), oof[mask.to_numpy()]
        if len(np.unique(y)) < 2:
            continue
        rows[name] = {
            "patients": int(len(y)), "converters": int(y.sum()),
            "conversion_rate": round(float(y.mean()), 3),
            "roc_auc": round(float(roc_auc_score(y, p)), 3),
            "top20_capture_within_group": round(top_k_recall(y, p, cfg["training"]["top_k_fraction"]), 2),
        }
    return rows


def education(df, bundle):
    bands = pd.cut(df["education"], [0, 12, 16, 30], labels=["12_or_less", "13_to_16", "over_16"])
    stable = df["target"] == 0
    out = {
        "conversion_rate": df.groupby(bands, observed=True)["target"].mean().round(3).to_dict(),
        "mean_mmse_of_stable_participants": df[stable].groupby(bands[stable], observed=True)["mmse"].mean().round(2).to_dict(),
    }
    model = bundle["pipeline"].named_steps["model"]
    if hasattr(model, "coef_"):  # logistic regression: standardised log-odds coefficients
        names = bundle["pipeline"].named_steps["preprocess"].get_feature_names_out()
        coef = dict(zip(names, model.coef_[0]))
        out["standardised_coefficients"] = {k.split("__")[1]: round(float(v), 3) for k, v in coef.items()}
    return out


def input_reduction(df, bundle, cfg, repeats):
    numeric, categorical = bundle["numeric_features"], bundle["categorical_features"]
    return {
        "all_stage1_inputs": cv_scores(df, numeric, categorical, cfg, repeats),
        "age_education_sex_mmse_adas_faq": cv_scores(df, CORE_SCREENING, ["sex"], cfg, repeats),
        "mmse_adas_faq": cv_scores(df, CORE_TESTS, [], cfg, repeats),
        "adas_faq": cv_scores(df, ["adas", "faq"], [], cfg, repeats),
        "faq_only": cv_scores(df, ["faq"], [], cfg, repeats),
    }


def stage_value(stages, cfg, repeats):
    """Each later-stage cohort scored with screening inputs only, then with that stage's tests added."""
    screening = cfg["features"]["stage1"]
    previous = set(screening["numeric"])
    out = {}
    for stage, (df, _) in stages.items():
        own = cfg["features"][stage]["numeric"]
        added = [c for c in own if c not in previous]
        previous = set(own)
        entry = {
            "participants": int(len(df)), "converters": int(df["target"].sum()),
            "screening_only": cv_scores(df, screening["numeric"], screening["categorical"], cfg, repeats),
        }
        if added:
            entry["added_inputs"] = added
            entry["screening_plus_this_stage"] = cv_scores(
                df, screening["numeric"] + added, screening["categorical"], cfg, repeats
            )
        out[stage] = entry
    return out


def held_out_auc_ci(stages, cfg, draws):
    rng = np.random.default_rng(cfg["training"]["random_state"])
    out = {}
    for stage, (df, bundle) in stages.items():
        _, y, risk = held_out_split(df, bundle, cfg)
        aucs = []
        while len(aucs) < draws:
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                aucs.append(roc_auc_score(y[idx], risk[idx]))
        low, high = np.percentile(aucs, [2.5, 97.5])
        out[stage] = {
            "test_patients": int(len(y)), "test_converters": int(y.sum()),
            "roc_auc": round(float(roc_auc_score(y, risk)), 3),
            "ci95": [round(float(low), 3), round(float(high), 3)],
        }
    return out


def main(repeats, draws):
    cfg = load_config()
    stages = load_stages(cfg)
    df1, bundle1 = stages["stage1"]
    oof = out_of_fold_risk(df1, bundle1, cfg)

    summary = {
        "note": "Aggregate figures only. Research prototype; not a validated clinical tool.",
        "cohort_overlap": cohort_overlap(stages),
        "stage1_test_breakdown": held_out_breakdown(df1, bundle1, cfg),
        "stage1_missed_vs_caught_converters": missed_vs_caught(df1, oof, cfg),
        "stage1_subgroups": subgroups(df1, oof, cfg),
        "education": education(df1, bundle1),
        "stage1_input_reduction": input_reduction(df1, bundle1, cfg, repeats),
        "stage_value_same_participants": stage_value(stages, cfg, repeats),
        "test_auc_ci": held_out_auc_ci(stages, cfg, draws),
    }

    reports_dir().mkdir(parents=True, exist_ok=True)
    out = reports_dir() / "inference_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5, help="Repeats of the 5-fold cross-validation (default 5).")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws for the test ROC-AUC interval.")
    args = ap.parse_args()
    main(args.repeats, args.bootstrap)
