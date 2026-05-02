"""SHAP-based explainability services backed by precalculated dashboard artifacts."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.dashboard.services.shared.prediction_service import PredictionService
from src.model2dashboard.features import MAIN_EXPLAINABILITY_FEATURE_NAMES
from src.python_models.dashboard.artifacts import StoredShapExplanation

FULL_FEATURE_SCOPE = "full"
MAIN_FEATURE_SCOPE = "main"
AUXILIARY_FEATURE_LABEL = "Auxiliar Features"


@dataclass
class ShapExplanationResult:
    """Container for global and local explanation outputs."""

    method: str
    explanation: shap.Explanation
    explain_frame: pd.DataFrame
    predictions: pd.Series
    mean_abs_shap: pd.Series
    feature_names: list[str]
    feature_scope: str = FULL_FEATURE_SCOPE


class ShapService:
    """Serve SHAP explanations persisted inside dashboard_model artifacts."""

    def __init__(
        self,
        prediction_service: PredictionService,
    ) -> None:
        self.prediction_service = prediction_service

    def explain(
        self,
        model_id: str,
        feature_scope: str = FULL_FEATURE_SCOPE,
    ) -> ShapExplanationResult:
        bundle = self.prediction_service.load_bundle(model_id)
        return self._apply_feature_scope(
            self._from_stored(bundle.dashboard_model.global_shap),
            feature_scope,
        )

    def explain_sample(
        self,
        model_id: str,
        sample_to_explain: pd.DataFrame,
        feature_scope: str = FULL_FEATURE_SCOPE,
    ) -> ShapExplanationResult:
        """Return the precomputed local SHAP explanation for the selected sample."""

        bundle = self.prediction_service.load_bundle(model_id)
        row_index = sample_to_explain.index[0]
        if row_index in bundle.dashboard_model.local_shap.index:
            stored = bundle.dashboard_model.local_shap_for_index(row_index)
        else:
            stored = bundle.dashboard_model.local_shap
        return self._apply_feature_scope(
            self._from_stored(stored),
            feature_scope,
        )

    def from_payload(
        self,
        payload: dict[str, Any],
        feature_scope: str = FULL_FEATURE_SCOPE,
    ) -> ShapExplanationResult:
        stored = StoredShapExplanation(
            method=str(payload["method"]),
            feature_names=[str(name) for name in payload["feature_names"]],
            index=list(payload["index"]),
            values=np.asarray(payload["values"]),
            base_values=np.asarray(payload["base_values"]),
            data=np.asarray(payload["data"]),
            display_data=(
                None
                if payload.get("display_data") is None
                else np.asarray(payload["display_data"], dtype=object)
            ),
            predictions=np.asarray(payload["predictions"]),
            mean_abs_shap={
                str(name): float(value)
                for name, value in dict(payload["mean_abs_shap"]).items()
            },
        )
        return self._apply_feature_scope(self._from_stored(stored), feature_scope)

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
            feature_scope=FULL_FEATURE_SCOPE,
        )

    def _apply_feature_scope(
        self,
        result: ShapExplanationResult,
        feature_scope: str,
    ) -> ShapExplanationResult:
        normalized_scope = (
            MAIN_FEATURE_SCOPE
            if feature_scope == MAIN_FEATURE_SCOPE
            else FULL_FEATURE_SCOPE
        )
        if normalized_scope == FULL_FEATURE_SCOPE:
            return ShapExplanationResult(
                method=result.method,
                explanation=result.explanation,
                explain_frame=result.explain_frame.copy(),
                predictions=result.predictions.copy(),
                mean_abs_shap=result.mean_abs_shap.copy(),
                feature_names=list(result.feature_names),
                feature_scope=FULL_FEATURE_SCOPE,
            )

        main_feature_names = [
            feature_name
            for feature_name in MAIN_EXPLAINABILITY_FEATURE_NAMES
            if feature_name in result.feature_names
        ]
        auxiliary_feature_names = [
            feature_name
            for feature_name in result.feature_names
            if feature_name not in main_feature_names
        ]
        if not auxiliary_feature_names:
            return ShapExplanationResult(
                method=result.method,
                explanation=result.explanation,
                explain_frame=result.explain_frame.copy(),
                predictions=result.predictions.copy(),
                mean_abs_shap=result.mean_abs_shap.copy(),
                feature_names=list(result.feature_names),
                feature_scope=MAIN_FEATURE_SCOPE,
            )

        values = np.asarray(result.explanation.values, dtype="float64")
        if values.ndim == 1:
            values = values.reshape(1, -1)
        feature_positions = {
            feature_name: position
            for position, feature_name in enumerate(result.feature_names)
        }
        main_positions = [feature_positions[name] for name in main_feature_names]
        auxiliary_positions = [
            feature_positions[name] for name in auxiliary_feature_names
        ]

        main_values = values[:, main_positions]
        auxiliary_values = values[:, auxiliary_positions].sum(axis=1, keepdims=True)
        scoped_values = np.concatenate([main_values, auxiliary_values], axis=1)
        scoped_feature_names = [*main_feature_names, AUXILIARY_FEATURE_LABEL]

        scoped_explain_frame = result.explain_frame.loc[:, main_feature_names].copy()
        scoped_explain_frame[AUXILIARY_FEATURE_LABEL] = 0.0

        explanation = shap.Explanation(
            values=scoped_values,
            base_values=np.asarray(result.explanation.base_values),
            data=self._select_data_matrix(
                matrix=result.explanation.data,
                positions=main_positions,
                row_count=len(scoped_explain_frame),
                auxiliary_fill=0.0,
            ),
            display_data=self._select_display_matrix(
                matrix=result.explanation.display_data,
                positions=main_positions,
                row_count=len(scoped_explain_frame),
                auxiliary_fill="Aggregated hidden inputs",
            ),
            instance_names=result.explanation.instance_names,
            feature_names=scoped_feature_names,
            output_names=result.explanation.output_names,
        )
        mean_abs_shap = pd.Series(
            np.abs(scoped_values).mean(axis=0),
            index=scoped_feature_names,
        ).sort_values(ascending=False)
        return ShapExplanationResult(
            method=result.method,
            explanation=explanation,
            explain_frame=scoped_explain_frame,
            predictions=result.predictions.copy(),
            mean_abs_shap=mean_abs_shap,
            feature_names=scoped_feature_names,
            feature_scope=MAIN_FEATURE_SCOPE,
        )

    @staticmethod
    def _select_data_matrix(
        *,
        matrix,
        positions: list[int],
        row_count: int,
        auxiliary_fill: float,
    ):
        if matrix is None:
            return None
        selected = np.asarray(matrix)[:, positions]
        auxiliary = np.full((row_count, 1), auxiliary_fill, dtype="float64")
        return np.concatenate([selected, auxiliary], axis=1)

    @staticmethod
    def _select_display_matrix(
        *,
        matrix,
        positions: list[int],
        row_count: int,
        auxiliary_fill: str,
    ):
        if matrix is None:
            return None
        selected = np.asarray(matrix, dtype=object)[:, positions]
        auxiliary = np.full((row_count, 1), auxiliary_fill, dtype=object)
        return np.concatenate([selected, auxiliary], axis=1)


__all__ = [
    "AUXILIARY_FEATURE_LABEL",
    "FULL_FEATURE_SCOPE",
    "MAIN_FEATURE_SCOPE",
    "ShapExplanationResult",
    "ShapService",
]
