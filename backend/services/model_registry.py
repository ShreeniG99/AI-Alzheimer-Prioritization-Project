"""Loads the trained stage models (stage1..stage4) that exist on disk.

Missing stages are simply unavailable; the API keeps working with the stages
that have been trained, so a fresh deployment without artifacts still starts.
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from backend.settings import ARTIFACTS_DIR
from ml.model_utils import priority_from_risk, speed_tier

logger = logging.getLogger("alzheimer.models")

STAGES = ["stage1", "stage2", "stage3", "stage4"]

STAGE_INFO = {
    "stage1": {
        "name": "Cognitive screening",
        "description": "Age, sex, education, MMSE, ADAS-Cog, FAQ and baseline comorbidities (pre-MRI).",
        "modality_features": [
            "age", "education", "mmse", "adas", "faq", "sex",
            "bmi", "sbp", "gds", "hx_cardiovascular", "hx_endocrine",
            "hx_psychiatric", "hx_neurologic", "hx_smoking",
        ],
        # Core screening inputs; comorbidities are optional extras.
        "trigger_features": ["age", "education", "mmse", "adas", "faq", "sex"],
    },
    "stage2": {
        "name": "Blood-based biomarkers",
        "description": "Plasma p-tau217, p-tau181, Aβ42/40, NfL, GFAP and APOE ε4 refine the screening score.",
        "modality_features": ["ptau217", "ab42_ab40", "nfl", "gfap", "ptau181", "nfl_legacy", "apoe4"],
        # Genotype alone does not mean the patient has had the blood biomarker test.
        "trigger_features": ["ptau217", "ab42_ab40", "nfl", "gfap", "ptau181", "nfl_legacy"],
    },
    "stage3": {
        "name": "MRI evaluation",
        "description": "MRI-derived eTIV, nWBV, ASF and hippocampal volume re-rank patients after imaging.",
        "modality_features": ["eTIV", "nWBV", "ASF", "hippocampus_icv"],
    },
    "stage4": {
        "name": "PET imaging",
        "description": "Amyloid PET (Centiloid) and FDG-PET metabolism for the narrowed-down candidates.",
        "modality_features": ["amyloid_centiloid", "fdg"],
    },
}


class StageModel:
    def __init__(self, stage: str, bundle: Dict[str, Any], metadata: Dict[str, Any]):
        self.stage = stage
        self.bundle = bundle
        self.metadata = metadata
        self.pipeline = bundle["pipeline"]
        self.calibrator = bundle["calibrator"]
        self.features: List[str] = bundle["features"]
        self.thresholds: Dict[str, float] = bundle["priority_thresholds"]
        self.model_name: str = bundle["selected_model"]
        self.model_version: str = bundle.get("model_version", f"{stage}-{self.model_name}")
        self.cohort: Dict[str, Any] = bundle.get("cohort") or metadata.get("cohort") or {}
        self._explainer = None
        self._explainer_lock = threading.Lock()

    def frame(self, features: Dict[str, Any]) -> pd.DataFrame:
        row = {f: features.get(f) for f in self.features}
        return pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce").astype(float)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.pipeline.predict_proba(self.frame(features))[:, 1]
        risk = float(self.calibrator.transform(raw)[0])
        return {
            "stage": self.stage,
            "risk_score": risk,
            "priority": priority_from_risk(risk, self.thresholds),
            "model": self.model_name,
            "model_version": self.model_version,
        }

    def explain(self, features: Dict[str, Any], top_n: Optional[int] = None) -> Dict[str, Any]:
        with self._explainer_lock:
            if self._explainer is None:
                from ml.explain import ShapExplainer

                self._explainer = ShapExplainer(self.bundle)
        clean = {f: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                 for f, v in ((f, features.get(f)) for f in self.features)}
        result = self._explainer.explain(clean, top_n=top_n)
        result.update({"stage": self.stage, "model": self.model_name})
        return result

    def info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            **{k: v for k, v in STAGE_INFO[self.stage].items() if not k.endswith("_features")},
            "selected_model": self.model_name,
            "model_version": self.model_version,
            "features": self.features,
            "priority_thresholds": self.thresholds,
            "cohort": self.cohort,
            "test_metrics": self.metadata.get("untouched_test_metrics"),
        }


class TempoModel(StageModel):
    """TEMPO: predicted annual change in CDR-SB for one pathway stage."""

    def __init__(self, stage: str, bundle: Dict[str, Any], metadata: Dict[str, Any]):
        self.stage = stage
        self.bundle = bundle
        self.metadata = metadata
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]
        self.model_name = bundle["selected_model"]
        self.model_version = bundle["model_version"]
        self.cohort = bundle.get("cohort", {})
        self.halfwidth = float(bundle["interval_halfwidth"])
        self.coverage = float(bundle["interval_coverage"])
        self.speed_thresholds = bundle["speed_thresholds"]
        self._explainer = None
        self._explainer_lock = threading.Lock()

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        slope = float(self.pipeline.predict(self.frame(features))[0])
        return {
            "stage": self.stage,
            "predicted_cdrsb_per_year": slope,
            "interval_low": slope - self.halfwidth,
            "interval_high": slope + self.halfwidth,
            "interval_coverage": self.coverage,
            "speed_tier": speed_tier(slope, self.speed_thresholds),
            "model": self.model_name,
            "model_version": self.model_version,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "selected_model": self.model_name,
            "model_version": self.model_version,
            "cohort": self.cohort,
            "speed_thresholds": self.speed_thresholds,
            "interval_halfwidth": self.halfwidth,
            "test_metrics": self.metadata.get("untouched_test_metrics"),
            "baselines": self.metadata.get("baselines"),
        }


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, StageModel] = {}
        self._tempo: Dict[str, TempoModel] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            for stage in STAGES:
                for prefix, cls, target in (("", StageModel, self._models), ("tempo_", TempoModel, self._tempo)):
                    path = ARTIFACTS_DIR / f"{prefix}{stage}_model.joblib"
                    if not path.exists():
                        continue
                    try:
                        bundle = joblib.load(path)
                        meta_path = ARTIFACTS_DIR / f"{prefix}{stage}_metadata.json"
                        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                        target[stage] = cls(stage, bundle, metadata)
                        logger.info("Loaded %s%s model from %s", prefix, stage, path)
                    except Exception:  # a corrupt artifact must not take the API down
                        logger.exception("Could not load %s", path)
            self._loaded = True

    def reload(self) -> None:
        with self._lock:
            self._models = {}
            self._tempo = {}
            self._loaded = False
        self._load()

    def available(self) -> List[str]:
        self._load()
        return [s for s in STAGES if s in self._models]

    def get(self, stage: str) -> Optional[StageModel]:
        self._load()
        return self._models.get(stage)

    def tempo_available(self) -> List[str]:
        self._load()
        return [s for s in STAGES if s in self._tempo]

    def get_tempo(self, stage: str) -> Optional[TempoModel]:
        self._load()
        return self._tempo.get(stage)


registry = ModelRegistry()
