"""SHAP-based explainability services."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.volatility_models.model_explainability.config import DEFAULT_SETTINGS
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)
from src.volatility_models.model_explainability.utils.sampling import sample_frame


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
    """Compute SHAP explanations on the model-ready feature space."""

    def __init__(
        self,
        prediction_service: PredictionService,
        feature_schema: FeatureSchema,
    ) -> None:
        self.prediction_service = prediction_service
        self.feature_schema = feature_schema

    def explain(self, model_id: str, frame: pd.DataFrame) -> ShapExplanationResult:
        bundle = self.prediction_service.load_bundle(model_id)
        raw_feature_names = self.prediction_service.resolve_model_input_features(bundle)
        feature_names = self._resolve_transformed_feature_names(bundle, raw_feature_names)
        explain_frame = sample_frame(
            frame,
            max_rows=DEFAULT_SETTINGS.shap_explain_size,
            random_state=DEFAULT_SETTINGS.random_state,
        )
        background = sample_frame(
            frame,
            max_rows=DEFAULT_SETTINGS.shap_background_size,
            random_state=DEFAULT_SETTINGS.random_state + 1,
        )
        prepared_explain = self._prepare_feature_frame(explain_frame, raw_feature_names)
        prepared_background = self._prepare_feature_frame(background, raw_feature_names)
        transformed_explain = self._transform_frame(
            bundle=bundle,
            frame=prepared_explain,
            raw_feature_names=raw_feature_names,
            transformed_feature_names=feature_names,
        )
        transformed_background = self._transform_frame(
            bundle=bundle,
            frame=prepared_background,
            raw_feature_names=raw_feature_names,
            transformed_feature_names=feature_names,
        )
        explainer = self._build_explainer(
            model=bundle.model,
            feature_names=feature_names,
            background_frame=transformed_background,
        )
        explanation = explainer(
            transformed_explain,
            max_evals=self._max_evals(len(feature_names)),
            silent=True,
        )
        predictions = self.prediction_service.predict_frame(model_id, explain_frame)
        mean_abs_shap = (
            pd.Series(
                explanation.abs.mean(0).values,
                index=feature_names,
            )
            .sort_values(ascending=False)
        )
        return ShapExplanationResult(
            method="shap.Explainer(permutation)",
            explanation=self._with_display_data(
                explanation=explanation,
                raw_frame=prepared_explain,
                transformed_feature_names=feature_names,
            ),
            explain_frame=transformed_explain,
            predictions=predictions,
            mean_abs_shap=mean_abs_shap,
            feature_names=feature_names,
        )

    def explain_sample(
        self,
        model_id: str,
        sample_to_explain: pd.DataFrame,
        reference_frame: pd.DataFrame,
    ) -> ShapExplanationResult:
        """Explain one selected sample against a sampled background."""

        bundle = self.prediction_service.load_bundle(model_id)
        raw_feature_names = self.prediction_service.resolve_model_input_features(bundle)
        feature_names = self._resolve_transformed_feature_names(bundle, raw_feature_names)
        explain_frame = sample_to_explain.copy()
        background = sample_frame(
            reference_frame,
            max_rows=DEFAULT_SETTINGS.shap_background_size,
            random_state=DEFAULT_SETTINGS.random_state + 1,
        )
        prepared_explain = self._prepare_feature_frame(explain_frame, raw_feature_names)
        prepared_background = self._prepare_feature_frame(background, raw_feature_names)
        transformed_explain = self._transform_frame(
            bundle=bundle,
            frame=prepared_explain,
            raw_feature_names=raw_feature_names,
            transformed_feature_names=feature_names,
        )
        transformed_background = self._transform_frame(
            bundle=bundle,
            frame=prepared_background,
            raw_feature_names=raw_feature_names,
            transformed_feature_names=feature_names,
        )
        explainer = self._build_explainer(
            model=bundle.model,
            feature_names=feature_names,
            background_frame=transformed_background,
        )
        explanation = explainer(
            transformed_explain,
            max_evals=self._max_evals(len(feature_names)),
            silent=True,
        )
        predictions = self.prediction_service.predict_frame(model_id, explain_frame)
        return ShapExplanationResult(
            method="shap.Explainer(permutation)",
            explanation=self._with_display_data(
                explanation=explanation,
                raw_frame=prepared_explain,
                transformed_feature_names=feature_names,
            ),
            explain_frame=transformed_explain,
            predictions=predictions,
            mean_abs_shap=(
                pd.Series(
                    explanation.abs.mean(0).values,
                    index=feature_names,
                ).sort_values(ascending=False)
            ),
            feature_names=feature_names,
        )

    def _build_explainer(
        self,
        model,
        background_frame: pd.DataFrame,
        feature_names: list[str],
    ) -> shap.Explainer:
        return shap.Explainer(
            lambda values: self._predict_transformed_values(model, feature_names, values),
            masker=background_frame,
            algorithm="permutation",
            feature_names=feature_names,
            seed=DEFAULT_SETTINGS.random_state,
        )

    def _max_evals(self, n_features: int) -> int:
        return max(
            2 * n_features + 1,
            DEFAULT_SETTINGS.shap_permutations * n_features,
        )

    def _predict_transformed_values(
        self,
        model,
        feature_names: list[str],
        values,
    ):
        frame = values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame(values, columns=feature_names)
        matrix = frame.to_numpy(dtype=np.float32, copy=False)
        predictions = model.predict(matrix, verbose=0)
        return np.asarray(predictions).reshape(-1)

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
        return prepared[feature_names]

    def _transform_frame(
        self,
        bundle,
        frame: pd.DataFrame,
        raw_feature_names: list[str],
        transformed_feature_names: list[str],
    ) -> pd.DataFrame:
        if bundle.preprocessor is not None:
            transformed = bundle.preprocessor.transform(frame[raw_feature_names])
            matrix = np.asarray(transformed, dtype=np.float32)
            return pd.DataFrame(matrix, index=frame.index, columns=transformed_feature_names)

        return self._prepare_feature_frame(frame, transformed_feature_names)

    @staticmethod
    def _resolve_transformed_feature_names(
        bundle,
        raw_feature_names: list[str],
    ) -> list[str]:
        metadata_names = bundle.metadata.get("transformed_feature_names")
        if metadata_names:
            return [str(name) for name in metadata_names]
        if bundle.preprocessor is not None and hasattr(bundle.preprocessor, "get_feature_names_out"):
            try:
                return [str(name) for name in bundle.preprocessor.get_feature_names_out()]
            except Exception:
                pass
        return list(raw_feature_names)

    def _with_display_data(
        self,
        explanation: shap.Explanation,
        raw_frame: pd.DataFrame,
        transformed_feature_names: list[str],
    ) -> shap.Explanation:
        display_frame = pd.DataFrame(index=raw_frame.index)
        for feature_name in transformed_feature_names:
            display_frame[feature_name] = self._display_series(raw_frame, feature_name)
        explanation.display_data = display_frame.to_numpy()
        return explanation

    def _display_series(
        self,
        raw_frame: pd.DataFrame,
        feature_name: str,
    ) -> pd.Series:
        if feature_name in raw_frame.columns:
            return raw_frame[feature_name]

        if "__" not in feature_name:
            return pd.Series([None] * len(raw_frame), index=raw_frame.index)

        _, transformed_name = feature_name.split("__", 1)
        if transformed_name in raw_frame.columns:
            return raw_frame[transformed_name]

        for feature in self.feature_schema.categorical_features(raw_only=True):
            prefix = f"{feature.name}_"
            if transformed_name.startswith(prefix) and feature.name in raw_frame.columns:
                expected_value = transformed_name[len(prefix) :]
                return (
                    raw_frame[feature.name].astype(str).str.upper()
                    == str(expected_value).upper()
                ).astype(int)

        return pd.Series([None] * len(raw_frame), index=raw_frame.index)
