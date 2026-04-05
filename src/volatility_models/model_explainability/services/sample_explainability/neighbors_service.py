"""Nearest-neighbor local comparison service."""

from __future__ import annotations

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.volatility_models.model_explainability.config import DEFAULT_SETTINGS
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)
from src.volatility_models.model_explainability.utils.preprocessing import (
    build_similarity_preprocessor,
)
from src.volatility_models.model_explainability.utils.sampling import sample_frame


class NeighborsService:
    """Find similar observations under a mixed-type distance."""

    def __init__(
        self,
        prediction_service: PredictionService,
        feature_schema: FeatureSchema,
    ) -> None:
        self.prediction_service = prediction_service
        self.feature_schema = feature_schema

    def find_neighbors(
        self,
        model_id: str,
        dataset: pd.DataFrame,
        sample: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        feature_names = self.prediction_service.resolve_model_input_features(
            self.prediction_service.load_bundle(model_id)
        )
        sampled_dataset = sample_frame(
            dataset,
            max_rows=DEFAULT_SETTINGS.neighbors_sample_size,
            random_state=DEFAULT_SETTINGS.random_state,
        )
        sampled_dataset = self._prepare_feature_frame(sampled_dataset, feature_names)
        sample = self._prepare_feature_frame(sample, feature_names)
        preprocessor = build_similarity_preprocessor(self.feature_schema, feature_names)
        transformed_dataset = preprocessor.fit_transform(sampled_dataset[feature_names])
        transformed_sample = preprocessor.transform(sample[feature_names])

        estimator = NearestNeighbors(n_neighbors=min(k, len(sampled_dataset)))
        estimator.fit(transformed_dataset)
        distances, indices = estimator.kneighbors(transformed_sample)

        neighbors = sampled_dataset.iloc[indices[0]].copy()
        neighbors["distance"] = distances[0]
        neighbors["PredictedVolatility"] = self.prediction_service.predict_frame(
            model_id, neighbors
        ).values
        return neighbors

    def _prepare_feature_frame(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        prepared = frame.copy()
        for feature in self.feature_schema.numerical_features(raw_only=True):
            if feature.name in feature_names and feature.name in prepared.columns:
                prepared[feature.name] = pd.to_numeric(prepared[feature.name], errors="coerce")
        for feature in self.feature_schema.categorical_features(raw_only=True):
            if feature.name in feature_names and feature.name in prepared.columns:
                prepared[feature.name] = prepared[feature.name].astype("object")
        return prepared
