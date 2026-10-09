"""
Research uncertainty service.

Bootstrap uncertainty (scripts/08_uncertainty_analysis.py) is available for
participants represented in the retrospective cohort of a stage. The bootstrap
artifact contains raw (uncalibrated) probabilities from model refits, so those
quantities are labelled as raw model probabilities; the production calibrated
risk is reported separately.

This service does not generate clinical confidence intervals and does not
extrapolate bootstrap uncertainty to arbitrary new patients.
"""

import threading
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backend.services.model_registry import registry
from backend.settings import ARTIFACTS_DIR, PROCESSED_DIR


class _StageBootstrap:
    def __init__(self, stage: str):
        npz = np.load(ARTIFACTS_DIR / f"{stage}_bootstrap_predictions.npz")
        self.predictions = npz["predictions"]
        self.row_indices = npz["row_indices"]
        self.dataset = pd.read_csv(PROCESSED_DIR / f"{stage}.csv")
        if len(self.dataset) != len(self.row_indices):
            raise RuntimeError(
                "Bootstrap artifact and processed dataset have different numbers of rows; "
                "re-run scripts/08_uncertainty_analysis.py after retraining."
            )


class UncertaintyService:
    def __init__(self):
        self._cache: Dict[str, Optional[_StageBootstrap]] = {}
        self._lock = threading.Lock()

    def _bootstrap(self, stage: str) -> Optional[_StageBootstrap]:
        with self._lock:
            if stage not in self._cache:
                npz = ARTIFACTS_DIR / f"{stage}_bootstrap_predictions.npz"
                data = PROCESSED_DIR / f"{stage}.csv"
                self._cache[stage] = _StageBootstrap(stage) if npz.exists() and data.exists() else None
            return self._cache[stage]

    def _find_matching_row(self, dataset: pd.DataFrame, features: Dict[str, Any], model_features):
        mask = pd.Series(True, index=dataset.index)
        for feature in model_features:
            supplied = features.get(feature)
            column = pd.to_numeric(dataset[feature], errors="coerce")
            if supplied is None:
                mask &= column.isna()
            else:
                mask &= np.isclose(column, float(supplied), equal_nan=False)
        matches = dataset.index[mask].tolist()
        return matches[0] if matches else None

    def estimate(self, features: Dict[str, Any], stage: str = "stage1") -> Dict[str, Any]:
        model = registry.get(stage)
        if model is None:
            return {"available": False, "stage": stage, "reason": f"No {stage} model is deployed."}

        production_risk = model.predict(features)["risk_score"]
        base = {
            "stage": stage,
            "production_calibrated_risk": production_risk,
            "model": model.model_name,
        }

        bootstrap = self._bootstrap(stage)
        if bootstrap is None:
            return {
                **base,
                "available": False,
                "reason": (
                    f"No bootstrap artifact for {stage}. Generate it with "
                    f"scripts/08_uncertainty_analysis.py --stage {stage}."
                ),
            }

        row_index = self._find_matching_row(bootstrap.dataset, features, model.features)
        if row_index is None:
            return {
                **base,
                "available": False,
                "reason": (
                    "The submitted assessment does not match a participant row in the "
                    "retrospective bootstrap cohort. A patient-specific bootstrap interval "
                    "is therefore not reported."
                ),
                "method": "No extrapolation is performed for unmatched assessments.",
                "interpretation": "Research uncertainty only; not a clinical confidence interval.",
            }

        position = np.where(bootstrap.row_indices == row_index)[0]
        values = bootstrap.predictions[:, int(position[0])]
        replicates = int(bootstrap.predictions.shape[0])

        return {
            **base,
            "available": True,
            "row_index": int(row_index),
            "bootstrap_raw_probability_mean": float(np.mean(values)),
            "bootstrap_raw_probability_median": float(np.median(values)),
            "bootstrap_raw_probability_lower_95": float(np.percentile(values, 2.5)),
            "bootstrap_raw_probability_upper_95": float(np.percentile(values, 97.5)),
            "bootstrap_raw_probability_std": float(np.std(values, ddof=1)),
            "bootstrap_replicates": replicates,
            "method": (
                f"Empirical interval from {replicates} bootstrap {model.model_name} refits. "
                "Bootstrap quantities are raw model probabilities and are not the production "
                "calibrated risk score."
            ),
            "interpretation": (
                "The bootstrap interval describes empirical model-estimate uncertainty in the "
                "retrospective research cohort. It is not a clinical confidence interval and "
                "should not be interpreted as diagnostic certainty or treatment guidance."
            ),
        }


uncertainty_service = UncertaintyService()
