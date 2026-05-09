"""Nearest-neighbor local comparison service backed by precalculated artifacts."""

import numpy as np
import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.prediction_service import PredictionService


class NeighborsService:
    """Expose precomputed nearest neighbors for dashboard samples."""

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
        sample: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        bundle = self.prediction_service.load_bundle(model_id)
        if sample.empty:
            return pd.DataFrame()
        row_index = sample.index[0]
        neighbors = bundle.dashboard_model.neighbors_for_index(row_index)
        if neighbors.empty:
            raise KeyError(
                f"No precomputed neighbors were exported for sample index {row_index!r}."
            )
        return neighbors.head(k).copy()

    def rank_neighbors(
        self,
        model_id: str,
        sample: pd.DataFrame,
    ) -> pd.DataFrame:
        bundle = self.prediction_service.load_bundle(model_id)
        if sample.empty:
            return pd.DataFrame()
        reference_frame = bundle.dashboard_model.training_reference_frame.copy()
        feature_names = self._neighbor_feature_names(
            bundle.dashboard_model,
            sample,
        )
        if not feature_names:
            return pd.DataFrame()

        dataset_matrix = reference_frame.loc[:, feature_names].apply(
            pd.to_numeric,
            errors="coerce",
        )
        sample_vector = (
            sample.loc[:, feature_names]
            .iloc[0]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )
        center = dataset_matrix.mean()
        scale = dataset_matrix.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        normalized_dataset = (dataset_matrix.fillna(center) - center) / scale
        normalized_sample = (sample_vector.fillna(center) - center) / scale
        distances = np.sqrt(
            ((normalized_dataset - normalized_sample.to_numpy()) ** 2).mean(axis=1)
        )
        ranked = reference_frame.copy()
        ranked["distance"] = distances.to_numpy()
        return ranked.sort_values("distance", kind="stable")

    @staticmethod
    def _neighbor_feature_names(dashboard_model, sample_frame: pd.DataFrame) -> list[str]:
        candidates = (
            dashboard_model.transformed_feature_names
            or dashboard_model.metadata.get("model_input_features", [])
        )
        return [
            name
            for name in candidates
            if name in dashboard_model.training_reference_frame.columns
            and name in sample_frame.columns
        ]
