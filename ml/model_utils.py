import os
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

def build_preprocessor(numeric, categorical, missing_indicators=False):
    """Median/mode imputation, scaling and one-hot encoding.

    Missingness indicators are off by default: in the ADNI cohort the few
    records with a missing ADAS/FAQ contained no converters, so an indicator
    learned "missing -> low risk" and pushed incomplete records to LOW
    priority. Imputing without an indicator keeps them at a neutral value.
    """
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=missing_indicators)),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)

def _n_jobs(default):
    """Worker count for tree models. IDEA_N_JOBS lets an HPC job match its CPU allocation."""
    value = os.environ.get("IDEA_N_JOBS")
    return int(value) if value else default


def make_models(random_state=42):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=_n_jobs(-1), random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=_n_jobs(4), random_state=random_state
        )
    return models

def make_regressors(random_state=42):
    """Candidate models for TEMPO (CDR-SB points per year)."""
    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, n_jobs=_n_jobs(-1), random_state=random_state
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5, reg_lambda=2.0,
            objective="reg:squarederror", tree_method="hist", n_jobs=_n_jobs(4), random_state=random_state,
        )
    return models

def speed_tier(slope, thresholds):
    """Decline-speed tier for a predicted CDR-SB slope (points per year)."""
    if slope >= thresholds["fast"]:
        return "FAST"
    if slope >= thresholds["moderate"]:
        return "MODERATE"
    return "STABLE"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs")
        self.fitted = False

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
        z = np.log(p/(1-p)).reshape(-1,1)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        self.model.fit(z, y)
        self.fitted = True
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
        if not self.fitted:
            return p
        z = np.log(p/(1-p)).reshape(-1,1)
        return self.model.predict_proba(z)[:,1]

def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)

def top_k_recall(y, p, fraction=0.2):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    k = max(1, int(math.ceil(len(y)*fraction)))
    idx = np.argsort(-p)[:k]
    positives = y.sum()
    if positives == 0:
        return float("nan")
    return float(y[idx].sum()/positives)

def priority_thresholds(validation_risk, high_fraction=0.20, medium_fraction=0.30):
    r = np.asarray(validation_risk, dtype=float)
    high_thr = float(np.quantile(r, 1-high_fraction))
    medium_thr = float(np.quantile(r, medium_fraction))
    return {"high_threshold": high_thr, "medium_threshold": medium_thr}

def priority_from_risk(risk, thresholds):
    if risk >= thresholds["high_threshold"]:
        return "HIGH"
    if risk >= thresholds["medium_threshold"]:
        return "MEDIUM"
    return "LOW"

def pathway_for_priority(priority):
    return {
        "HIGH": {
            "stage": "Priority review queue",
            "workflow": "Prioritize clinician review and, subject to local protocol/capacity, move the patient to the next diagnostic evaluation queue.",
        },
        "MEDIUM": {
            "stage": "Clinician review queue",
            "workflow": "Route for clinician review and consider additional structured assessment according to the local diagnostic pathway.",
        },
        "LOW": {
            "stage": "Routine review queue",
            "workflow": "Keep in the routine clinician-led assessment pathway; no automatic diagnostic or treatment recommendation is generated.",
        },
    }[priority]

def feature_names_from_pipeline(pipeline):
    pre = pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())

def save_bundle(path, bundle):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

def load_bundle(path):
    return joblib.load(path)
