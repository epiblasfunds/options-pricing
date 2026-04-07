"""SHAP-based explainability services backed by precalculated dashboard artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import shap

from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.prediction_service import PredictionService
from src.python_models.dashboard.dashboard_artifacts import StoredShapExplanation


@dataclass
class ShapExplanationResult:
    """Container for global and local explanation outputs."""

    method: str
    explanation: shap.Explanation
    explain_frame: pd.DataFrame
    predictions: pd.Series
    mean_abs_shap: pd.Series
    feature_names: list[str]

    @property
    def shap_values(self) -> pd.DataFrame:
        values = self.explanation.values
        if getattr(values, "ndim", 1) == 1:
            values = values.reshape(1, -1)
        return pd.DataFrame(values, index=self.explain_frame.index, columns=self.feature_names)

    @property
    def base_value(self) -> float:
        base_values = self.explanation.base_values
        if hasattr(base_values, "__len__") and not isinstance(base_values, (str, bytes)):
            return float(pd.Series(base_values).astype(float).mean())
        return float(base_values)


class ShapService:
    """Serve SHAP explanations persisted inside dashboard_model artifacts."""

    def __init__(
        self,
        prediction_service: PredictionService,
        feature_schema: FeatureSchema,
    ) -> None:
        self.prediction_service = prediction_service
        self.feature_schema = feature_schema

    def explain(self, model_id: str, frame: pd.DataFrame) -> ShapExplanationResult:
        bundle = self.prediction_service.load_bundle(model_id)
        return self._from_stored(bundle.dashboard_model.global_shap)

    def explain_sample(
        self,
        model_id: str,
        sample_to_explain: pd.DataFrame,
        reference_frame: pd.DataFrame,
    ) -> ShapExplanationResult:
        """Return the precomputed local SHAP explanation for the selected sample."""

        bundle = self.prediction_service.load_bundle(model_id)
        row_index = sample_to_explain.index[0]
        if row_index in bundle.dashboard_model.local_shap.index:
            return self._from_stored(bundle.dashboard_model.local_shap_for_index(row_index))
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
                stored.predictions,
                index=stored.index,
                name="PredictedVolatility",
            ),
            mean_abs_shap=pd.Series(stored.mean_abs_shap).sort_values(ascending=False),
            feature_names=list(stored.feature_names),
        )

