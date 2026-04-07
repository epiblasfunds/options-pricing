import numpy as np
import pandas as pd

from src.config.config import config
from src.dashboard.utils.sampling import quantile_grid, sample_frame
from src.enums.data_enums import VolatilityDBEnum
from src.functionalities.dashboard_models.behaviour_artifacts import (
    build_ale_frame,
    build_ice_frame,
    build_neighbors_frame,
    build_surfaces_frame,
    financial_checks_from_surface,
)
from src.functionalities.dashboard_models.diagnosis_artifacts import (
    build_diagnosis_artifact,
)
from src.functionalities.dashboard_models.runtime import (
    build_shap_explainer,
    max_evals,
    predict_raw_frame,
    serialize_shap_result,
    transform_feature_frame,
)
from src.functionalities.dashboard_models.tree_models import build_surrogate_tree_models
from src.python_models.dashboard.artifacts import ManualApiStubResponse
from src.volatility_models import (
    ANALYSIS_FEATURE_NAMES,
    TARGET_COLUMN,
    build_feature_frame_from_trades,
    build_model_dataset,
    select_trade_columns,
)


def build_dashboard_model(
    cls,
    *,
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
):
    raw_frame = select_trade_columns(X)
    raw_frame[str(TARGET_COLUMN)] = pd.Series(y, index=X.index)
    dataset_frame = build_model_dataset(raw_frame)

    trained_model = model.main_model
    model_input_features = list(trained_model.metadata.feature_names)
    predictions = predict_raw_frame(trained_model, raw_frame)
    dataset_frame["PredictedVolatility"] = predictions
    dataset_frame["Residual"] = (
        dataset_frame[str(TARGET_COLUMN)] - dataset_frame["PredictedVolatility"]
    )
    dataset_frame["AbsoluteError"] = dataset_frame["Residual"].abs()

    build_config = config.dashboard_models_config.build_config
    tree_models = (
        dict(model.tree_models)
        if model.tree_models
        else build_surrogate_tree_models(
            trained_model=trained_model,
            dataset_frame=dataset_frame,
            raw_frame=raw_frame,
            model_input_features=model_input_features,
            surrogate_depths=build_config.surrogate_depths,
            sample_frame=sample_frame,
            transform_feature_frame=transform_feature_frame,
        )
    )
    sample_indices = sample_frame(
        dataset_frame,
        max_rows=build_config.sample_option_size,
        random_state=config.dashboard_models_config.random_state,
    ).index.tolist()
    behaviour_anchor_indices = sample_frame(
        dataset_frame,
        max_rows=min(build_config.behaviour_anchor_size, len(dataset_frame)),
        random_state=config.dashboard_models_config.random_state + 11,
    ).index.tolist()
    global_shap, local_shap = _build_shap_artifacts(
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        trained_model=trained_model,
        model_input_features=model_input_features,
        sample_indices=sample_indices,
    )
    neighbors_frame = build_neighbors_frame(
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        trained_model=trained_model,
        model_input_features=model_input_features,
        sample_indices=sample_indices,
        sample_frame=sample_frame,
        transform_feature_frame=transform_feature_frame,
    )
    surfaces_frame = build_surfaces_frame(
        trained_model=trained_model,
        raw_frame=raw_frame,
        anchor_indices=behaviour_anchor_indices,
        predict_raw_frame=predict_raw_frame,
    )
    ice_frame = build_ice_frame(
        trained_model=trained_model,
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        feature_names=list(ANALYSIS_FEATURE_NAMES),
        sample_frame=sample_frame,
        quantile_grid=quantile_grid,
        predict_raw_frame=predict_raw_frame,
    )
    ale_frame = build_ale_frame(
        trained_model=trained_model,
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        feature_names=list(ANALYSIS_FEATURE_NAMES),
        predict_raw_frame=predict_raw_frame,
    )
    warnings = _diagnosis_warnings(
        surfaces_frame=surfaces_frame,
        behaviour_anchor_indices=behaviour_anchor_indices,
    )
    diagnosis = build_diagnosis_artifact(
        dataset_frame,
        financial_warnings=warnings,
        sample_frame=sample_frame,
    )
    manual_api_stub = ManualApiStubResponse(
        prediction=float(dataset_frame["PredictedVolatility"].mean()),
        summary="Reference prediction based on the persisted explainability bundle.",
        reference_sample_index=sample_indices[0] if sample_indices else None,
    )
    return cls(
        model_id=model.metadata.model_id,
        model_name=model.metadata.name,
        metadata=dict(model.metadata.metadata),
        dataset_frame=dataset_frame,
        raw_feature_names=[
            str(VolatilityDBEnum.EXEC_DATETIME),
            str(VolatilityDBEnum.OPTION_TYPE),
            str(VolatilityDBEnum.QUANTITY),
            str(VolatilityDBEnum.STRIKE_PRICE),
            str(VolatilityDBEnum.TRADE_TYPE),
            str(VolatilityDBEnum.UNDERLYING_LAG_MINUTES),
            str(VolatilityDBEnum.UNDERLYING_PRICE),
            str(VolatilityDBEnum.TIME_TO_EXPIRATION),
            str(VolatilityDBEnum.RATE),
        ],
        transformed_feature_names=model_input_features,
        tree_models=tree_models,
        sample_indices=sample_indices,
        behaviour_anchor_indices=behaviour_anchor_indices,
        global_shap=global_shap,
        local_shap=local_shap,
        neighbors_frame=neighbors_frame,
        surfaces_frame=surfaces_frame,
        ice_frame=ice_frame,
        ale_frame=ale_frame,
        diagnosis=diagnosis,
        manual_api_stub=manual_api_stub,
    )


def _build_shap_artifacts(
    *,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    trained_model,
    model_input_features: list[str],
    sample_indices: list,
):
    background_raw = raw_frame.loc[
        sample_frame(
            dataset_frame,
            max_rows=config.dashboard_models_config.shap_background_size,
            random_state=config.dashboard_models_config.random_state + 1,
        ).index
    ].copy()
    global_raw = raw_frame.loc[
        sample_frame(
            dataset_frame,
            max_rows=config.dashboard_models_config.shap_explain_size,
            random_state=config.dashboard_models_config.random_state,
        ).index
    ].copy()
    background_features = build_feature_frame_from_trades(background_raw)
    global_features = build_feature_frame_from_trades(global_raw)
    transformed_background = transform_feature_frame(
        background_features,
        trained_model.preprocessor,
        model_input_features,
    )
    transformed_global = transform_feature_frame(
        global_features,
        trained_model.preprocessor,
        model_input_features,
    )
    explainer = build_shap_explainer(
        model=trained_model.model,
        background_frame=transformed_background,
        feature_names=model_input_features,
    )
    global_explanation = explainer(
        transformed_global,
        max_evals=max_evals(len(model_input_features)),
        silent=True,
    )
    global_shap = serialize_shap_result(
        method="shap.Explainer(permutation)",
        explanation=global_explanation,
        transformed_frame=transformed_global,
        raw_frame=global_features,
        feature_names=model_input_features,
        predictions=dataset_frame.loc[global_raw.index, "PredictedVolatility"],
    )
    local_raw = raw_frame.loc[sample_indices].copy()
    local_features = build_feature_frame_from_trades(local_raw)
    transformed_local = transform_feature_frame(
        local_features,
        trained_model.preprocessor,
        model_input_features,
    )
    local_explanation = explainer(
        transformed_local,
        max_evals=max_evals(len(model_input_features)),
        silent=True,
    )
    local_shap = serialize_shap_result(
        method="shap.Explainer(permutation)",
        explanation=local_explanation,
        transformed_frame=transformed_local,
        raw_frame=local_features,
        feature_names=model_input_features,
        predictions=dataset_frame.loc[local_raw.index, "PredictedVolatility"],
    )
    return global_shap, local_shap


def _diagnosis_warnings(
    *,
    surfaces_frame: pd.DataFrame,
    behaviour_anchor_indices: list,
) -> list[str]:
    if behaviour_anchor_indices and not surfaces_frame.empty:
        return financial_checks_from_surface(
            surfaces_frame.loc[
                surfaces_frame["anchor_index"] == behaviour_anchor_indices[0]
            ]
        )
    return ["No large discontinuities were detected by the heuristic checks."]
