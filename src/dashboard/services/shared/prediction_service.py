"""Access helpers for dashboard-ready volatility bundles."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.model_loader import LoadedModelBundle, ModelLoader
from src.dashboard.services.shared.model_registry import ModelRegistry


class PredictionPipelineError(RuntimeError):
    """Raised when the dashboard requests data outside the precalculated bundle."""


class PredictionService:
    """Expose precomputed dashboard artifacts for the selected model."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        feature_schema: FeatureSchema,
    ) -> None:
        self.model_registry = model_registry
        self.model_loader = model_loader
        self.feature_schema = feature_schema

    def load_bundle(self, model_id: str) -> LoadedModelBundle:
        discovered = self.model_registry.get_model(model_id)
        if discovered is None:
            raise FileNotFoundError(f"Model '{model_id}' was not found.")
        return self.model_loader.load(discovered)

    def load_dashboard_model(self, model_id: str):
        bundle = self.load_bundle(model_id)
        return bundle.dashboard_model

    def predict_frame(self, model_id: str, frame: pd.DataFrame) -> pd.Series:
        bundle = self.load_bundle(model_id)
        if self._can_use_precomputed_predictions(bundle.dashboard_model, frame):
            return bundle.dashboard_model.predictions_for_indices(frame.index)
        raise PredictionPipelineError(
            "Runtime prediction is disabled in the dashboard. "
            "Only samples already exported inside dashboard_model can be scored here."
        )

    def call_manual_prediction_api(
        self,
        model_id: str,
        sample_payload: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = self.load_bundle(model_id)
        return {
            "status": "stubbed",
            "prediction": float(bundle.dashboard_model.manual_api_stub.prediction),
            "summary": bundle.dashboard_model.manual_api_stub.summary,
            "reference_sample_index": bundle.dashboard_model.manual_api_stub.reference_sample_index,
            "payload_echo": dict(sample_payload),
        }

    @staticmethod
    def _can_use_precomputed_predictions(dashboard_model, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        return pd.Index(frame.index).isin(dashboard_model.dataset_frame.index).all()

