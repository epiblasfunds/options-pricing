"""Access to precomputed equivalent explainable models."""

from __future__ import annotations

from src.python_models.explainable_model import SurrogateTreeModel
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)


class EquivalentModelsService:
    """Expose the persisted surrogate tree for explainable bundles."""

    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def load_surrogates(self, model_id: str) -> dict[int, SurrogateTreeModel]:
        bundle = self.prediction_service.load_bundle(model_id)
        if bundle.explainable_model is None:
            raise ValueError(
                "The selected model is not an explainable bundle with persisted surrogate trees."
            )
        return dict(sorted(bundle.explainable_model.tree_models.items()))
