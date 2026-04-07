import typing as t

import numpy as np
import pandas as pd
import shap

from src.config.config import config
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.volatility_models import build_feature_frame_from_trades


def transform_feature_frame(
    frame: pd.DataFrame,
    preprocessor: t.Any,
    feature_names: list[str],
) -> pd.DataFrame:
    ordered = frame.loc[:, feature_names].copy()
    if preprocessor is None:
        return ordered
    transformed = preprocessor.transform(ordered)
    matrix = np.asarray(transformed, dtype=np.float32)
    return pd.DataFrame(matrix, index=ordered.index, columns=feature_names)


def predict_raw_frame(trained_model: t.Any, raw_frame: pd.DataFrame) -> np.ndarray:
    feature_frame = build_feature_frame_from_trades(raw_frame)
    transformed = transform_feature_frame(
        feature_frame,
        trained_model.preprocessor,
        list(trained_model.metadata.feature_names),
    )
    predictions = trained_model.model.predict(
        transformed.to_numpy(dtype=np.float32, copy=False),
        verbose=0,
    )
    return np.asarray(predictions).reshape(-1)


def build_shap_explainer(
    model: t.Any,
    background_frame: pd.DataFrame,
    feature_names: list[str],
) -> shap.Explainer:
    return shap.Explainer(
        lambda values: predict_transformed_values(model, feature_names, values),
        masker=background_frame,
        algorithm="permutation",
        feature_names=feature_names,
        seed=config.dashboard_models_config.random_state,
    )


def predict_transformed_values(
    model: t.Any,
    feature_names: list[str],
    values: t.Any,
) -> np.ndarray:
    frame = (
        values.copy()
        if isinstance(values, pd.DataFrame)
        else pd.DataFrame(values, columns=feature_names)
    )
    predictions = model.predict(frame.to_numpy(dtype=np.float32, copy=False), verbose=0)
    return np.asarray(predictions).reshape(-1)


def max_evals(n_features: int) -> int:
    return max(
        2 * n_features + 1,
        config.dashboard_models_config.shap_permutations * n_features,
    )


def serialize_shap_result(
    *,
    method: str,
    explanation: shap.Explanation,
    transformed_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
    predictions: pd.Series,
) -> StoredShapExplanation:
    explanation.display_data = raw_frame.loc[:, feature_names].to_numpy()
    mean_abs_shap = pd.Series(
        explanation.abs.mean(0).values,
        index=feature_names,
    ).sort_values(ascending=False)
    return StoredShapExplanation(
        method=method,
        feature_names=list(feature_names),
        index=list(transformed_frame.index),
        values=np.asarray(explanation.values),
        base_values=np.asarray(explanation.base_values),
        data=np.asarray(transformed_frame.to_numpy()),
        display_data=np.asarray(explanation.display_data),
        predictions=predictions.to_numpy(),
        mean_abs_shap={
            str(name): float(value) for name, value in mean_abs_shap.items()
        },
    )
