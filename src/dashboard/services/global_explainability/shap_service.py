"""SHAP-based explainability services backed by precalculated dashboard artifacts."""

from dataclasses import dataclass

import pandas as pd
import shap

from src.dashboard.services.shared.prediction_service import PredictionService
from src.python_models.dashboard.artifacts import StoredShapExplanation


@dataclass
class ShapExplanationResult:
    """Container for global and local explanation outputs."""

    method: str
    explanation: shap.Explanation
    explain_frame: pd.DataFrame
    predictions: pd.Series
    mean_abs_shap: pd.Series
    feature_names: list[str]


class ShapService:
    """Serve SHAP explanations persisted inside dashboard_model artifacts."""

    def __init__(
        self,
        prediction_service: PredictionService,
    ) -> None:
        self.prediction_service = prediction_service

    def explain(self, model_id: str) -> ShapExplanationResult:
        bundle = self.prediction_service.load_bundle(model_id)
        return self._from_stored(bundle.dashboard_model.global_shap)

    def explain_sample(
        self,
        model_id: str,
        sample_to_explain: pd.DataFrame,
    ) -> ShapExplanationResult:
        """Return the precomputed local SHAP explanation for the selected sample."""

        bundle = self.prediction_service.load_bundle(model_id)
        row_index = sample_to_explain.index[0]
        if row_index in bundle.dashboard_model.local_shap.index:
            return self._from_stored(
                bundle.dashboard_model.local_shap_for_index(row_index)
            )
        return self._from_stored(bundle.dashboard_model.local_shap)

    def _from_stored(self, stored: StoredShapExplanation) -> ShapExplanationResult:
        explain_frame = pd.DataFrame(
            stored.data,
            index=stored.index,
            columns=stored.feature_names,
        )
        return ShapExplanationResult(
            method=stored.method,
            explanation=stored.to_explanation(),
            explain_frame=explain_frame,
            predictions=pd.Series(
                stored.waterfall_predictions(),
                index=stored.index,
                name="PredictedVolatility",
            ),
            mean_abs_shap=pd.Series(stored.mean_abs_shap).sort_values(ascending=False),
            feature_names=list(stored.feature_names),
        )
