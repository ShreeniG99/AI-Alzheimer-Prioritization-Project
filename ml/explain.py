"""SHAP explanations for saved stage model bundles.

Explanations are computed on the preprocessed feature space (imputed,
scaled, one-hot encoded, plus missingness indicators) and then aggregated
back to the original clinical inputs, so that e.g. ``num__mmse`` and
``num__missingindicator_mmse`` are reported together as "MMSE".

Values are in the model's raw output space (log-odds for logistic regression
and XGBoost, probability for random forest). Platt calibration is monotonic,
so the direction of each contribution also holds for the calibrated risk.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "age": "Age",
    "education": "Education (years)",
    "mmse": "MMSE",
    "adas": "ADAS-Cog 11",
    "faq": "FAQ",
    "bmi": "BMI",
    "sbp": "Systolic blood pressure",
    "gds": "Depression score (GDS)",
    "hx_cardiovascular": "Cardiovascular history",
    "hx_endocrine": "Endocrine/metabolic history",
    "hx_psychiatric": "Psychiatric history",
    "hx_neurologic": "Neurologic history",
    "hx_smoking": "Smoking history",
    "apoe4": "APOE ε4 alleles",
    "ptau181": "Plasma p-tau181",
    "nfl_legacy": "Plasma NfL (older assay)",
    "fdg": "FDG-PET metabolism",
    "sex": "Sex",
    "ptau217": "Plasma p-tau217",
    "ab42_ab40": "Plasma Aβ42/40",
    "nfl": "Plasma NfL",
    "gfap": "Plasma GFAP",
    "eTIV": "eTIV",
    "nWBV": "nWBV",
    "ASF": "ASF",
    "hippocampus_icv": "Hippocampal volume / ICV",
    "amyloid_centiloid": "Amyloid PET (Centiloid)",
}


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _source_feature(transformed_name: str, features: List[str]) -> str:
    """Map a ColumnTransformer output name back to its input feature."""
    name = transformed_name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        name = name[len("missingindicator_"):]
    if name in features:
        return name
    # One-hot columns look like "sex_2.0"; pick the longest matching prefix.
    matches = [f for f in features if name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


class ShapExplainer:
    def __init__(self, bundle: Dict[str, Any]):
        import shap  # imported lazily: heavy, and only needed for explanations

        self.features: List[str] = bundle["features"]
        pipeline = bundle["pipeline"]
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.transformed_names = list(self.preprocessor.get_feature_names_out())
        self.sources = [_source_feature(n, self.features) for n in self.transformed_names]

        background = bundle.get("background")
        if background is None or len(background) == 0:
            raise ValueError("Model bundle has no background sample; retrain with 03_train.py.")
        background_t = self._dense(self.preprocessor.transform(background[self.features]))

        # Regressors (TEMPO) explain the predicted CDR-SB slope directly.
        regression = not hasattr(self.model, "predict_proba")

        if hasattr(self.model, "coef_"):
            self.explainer = shap.LinearExplainer(self.model, background_t)
            self.method = "SHAP LinearExplainer"
            self.output_space = "log-odds"
        elif hasattr(self.model, "estimators_") or hasattr(self.model, "get_booster"):
            self.explainer = shap.TreeExplainer(self.model)
            self.method = "SHAP TreeExplainer"
            self.output_space = "log-odds" if hasattr(self.model, "get_booster") else "probability"
        else:
            sample = shap.sample(background_t, min(50, len(background_t)), random_state=0)
            predict = self.model.predict if regression else (lambda z: self.model.predict_proba(z)[:, 1])
            self.explainer = shap.KernelExplainer(predict, sample)
            self.method = "SHAP KernelExplainer"
            self.output_space = "probability"
        self.regression = regression
        if regression:
            self.output_space = "CDR-SB points per year"

    @staticmethod
    def _dense(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)

    def explain(self, row: Dict[str, Any], top_n: int | None = None) -> Dict[str, Any]:
        frame = pd.DataFrame([{f: row.get(f) for f in self.features}])
        transformed = self._dense(self.preprocessor.transform(frame))
        result = self.explainer(transformed)
        values = np.asarray(result.values)
        base = np.asarray(result.base_values)
        # Binary classifiers may return one column per class; keep the positive class.
        if values.ndim == 3:
            values, base = values[..., 1], base[..., 1]
        values, base = values[0], float(np.ravel(base)[0])

        totals: Dict[str, float] = {}
        for source, value in zip(self.sources, values):
            totals[source] = totals.get(source, 0.0) + float(value)

        contributions = []
        for feature in self.features:
            value = totals.get(feature, 0.0)
            supplied = row.get(feature)
            contributions.append({
                "feature": feature,
                "label": feature_label(feature),
                "value": supplied,
                "missing": supplied is None,
                "shap_value": value,
                "direction": (
                    ("predicts faster decline" if self.regression else "increases risk score") if value > 0
                    else ("predicts slower decline" if self.regression else "decreases risk score") if value < 0
                    else "neutral"
                ),
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        if top_n:
            contributions = contributions[:top_n]

        return {
            "method": self.method,
            "output_space": self.output_space,
            "base_value": base,
            "contributions": contributions,
            "interpretation": (
                "SHAP values describe how each input moved this model's score relative to "
                "the training-cohort average. They are not causal or independent clinical "
                "effects. Missing inputs are imputed with typical (median) values."
            ),
        }
