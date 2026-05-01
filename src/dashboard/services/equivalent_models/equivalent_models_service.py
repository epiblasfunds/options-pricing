"""Access to precomputed equivalent explainable models."""

import typing as t

from src.dashboard.services.shared.prediction_service import PredictionService
from src.python_models.dashboard.artifacts import SurrogateTreeModel
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


class EquivalentModelsService:
    """Expose the persisted surrogate trees from dashboard bundles."""

    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def load_surrogates(self, model_id: str) -> dict[int, SurrogateTreeModel]:
        bundle = self.prediction_service.load_bundle(model_id)
        return dict(sorted(bundle.dashboard_model.tree_models.items()))

    def load_equivalent_models(
        self,
        model_id: str,
    ) -> tuple[SymbolicRegressorModel | None, dict[int, SurrogateTreeModel]]:
        bundle = self.prediction_service.load_bundle(model_id)
        return (
            t.cast(
                SymbolicRegressorModel | None, bundle.dashboard_model.symbolic_model
            ),
            dict(sorted(bundle.dashboard_model.tree_models.items())),
        )
