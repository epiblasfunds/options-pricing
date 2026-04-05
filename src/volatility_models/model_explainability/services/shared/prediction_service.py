"""Prediction helpers for selected volatility models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_SETTINGS
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.services.shared.model_loader import (
    LoadedModelBundle,
    ModelLoader,
)
from src.volatility_models.model_explainability.services.shared.model_registry import (
    ModelRegistry,
)
from src.volatility_models.model_explainability.utils.validation import ensure_columns


class PredictionPipelineError(RuntimeError):
    """Raised when the model cannot be executed consistently with training."""


class PredictionService:
    """Run predictions through the discovered model and optional preprocessor."""

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

    def resolve_model_input_features(self, bundle: LoadedModelBundle) -> list[str]:
        metadata_features = bundle.metadata.get("model_input_features")
        if metadata_features:
            return list(metadata_features)
        return list(DEFAULT_SETTINGS.model_input_features)

    def predict_frame(self, model_id: str, frame: pd.DataFrame) -> pd.Series:
        bundle = self.load_bundle(model_id)
        feature_names = self.resolve_model_input_features(bundle)
        ensure_columns(frame, feature_names)
        model_frame = frame[feature_names].copy()
        transformed = self._transform_inputs(bundle, model_frame)
        predictions = bundle.model.predict(transformed, verbose=0)
        values = np.asarray(predictions).reshape(-1)
        return pd.Series(values, index=frame.index, name="PredictedVolatility")

    def _transform_inputs(self, bundle: LoadedModelBundle, model_frame: pd.DataFrame) -> Any:
        if bundle.preprocessor is not None:
            return bundle.preprocessor.transform(model_frame)

        metadata_mappings = bundle.metadata.get("categorical_mappings", {})
        transformed = model_frame.copy()
        for column, mapping in metadata_mappings.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].map(mapping)

        non_numeric_columns = [
            column
            for column in transformed.columns
            if not pd.api.types.is_numeric_dtype(transformed[column])
        ]
        if non_numeric_columns:
            raise PredictionPipelineError(
                "The selected model requires preprocessing artifacts or categorical "
                f"mappings for columns: {non_numeric_columns}."
            )
        return transformed.to_numpy(dtype=float)
