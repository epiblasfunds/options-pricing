"""Nearest-neighbor local comparison service backed by precalculated artifacts."""

from __future__ import annotations

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
        dataset: pd.DataFrame,
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

