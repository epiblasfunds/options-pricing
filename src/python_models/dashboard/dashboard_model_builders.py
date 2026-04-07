from __future__ import annotations

import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.dashboard.utils.sampling import quantile_grid, sample_frame
from src.enums.data_enums import VolatilityDBEnum
from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.dashboard.dashboard_artifacts import (
    DiagnosisArtifact,
    ManualApiStubResponse,
    StoredShapExplanation,
)
from src.python_models.explainable_model import (
    ExplainableModel,
    SingleModelMetadata,
    SurrogateTreeModel,
)
from src.volatility_models import (
    ANALYSIS_FEATURE_NAMES,
    TARGET_COLUMN,
    add_dashboard_derived_features,
    apply_feature_override,
    build_feature_frame_from_trades,
    build_model_dataset,
    select_trade_columns,
)

METRICS_REGISTRY = build_metrics_registry()


def build_dashboard_model(
    cls,
    *,
    model: ExplainableModel,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None,
):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input features X must be a pandas DataFrame.")
    if y is None:
        raise ValueError("Target variable y cannot be None.")

    raw_frame = select_trade_columns(X)
    raw_frame[str(TARGET_COLUMN)] = pd.Series(y, index=X.index)
    dataset_frame = build_model_dataset(raw_frame)

    trained_model = model.main_model
    model_input_features = list(trained_model.metadata.feature_names)
    predictions = predict_raw_frame(trained_model, raw_frame)
    dataset_frame["PredictedVolatility"] = predictions
    dataset_frame["Residual"] = (
        pd.to_numeric(dataset_frame[str(TARGET_COLUMN)], errors="coerce")
        - dataset_frame["PredictedVolatility"]
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

    neighbors_frame = build_neighbors_frame(
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        trained_model=trained_model,
        model_input_features=model_input_features,
        sample_indices=sample_indices,
        neighbors_k=build_config.neighbors_k,
    )
    surfaces_frame = build_surfaces_frame(
        trained_model=trained_model,
        raw_frame=raw_frame,
        anchor_indices=behaviour_anchor_indices,
    )
    ice_frame = build_ice_frame(
        trained_model=trained_model,
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        feature_names=list(ANALYSIS_FEATURE_NAMES),
    )
    ale_frame = build_ale_frame(
        trained_model=trained_model,
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        feature_names=list(ANALYSIS_FEATURE_NAMES),
    )
    diagnosis = build_diagnosis_artifact(
        dataset_frame,
        financial_warnings=(
            financial_checks_from_surface(
                surfaces_frame.loc[
                    surfaces_frame["anchor_index"] == behaviour_anchor_indices[0]
                ]
            )
            if behaviour_anchor_indices and not surfaces_frame.empty
            else ["No large discontinuities were detected by the heuristic checks."]
        ),
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


def load_dashboard_tree_models(
    bundle_path: Path, payload: dict[str, t.Any]
) -> dict[int, SurrogateTreeModel]:
    tree_models: dict[int, SurrogateTreeModel] = {}
    trees_root = bundle_path / "tree_models"
    if trees_root.exists():
        for tree_path in sorted(trees_root.iterdir(), key=lambda path: path.name):
            if not tree_path.is_dir():
                continue
            depth = int(tree_path.name.removeprefix("depth_"))
            tree_models[depth] = SurrogateTreeModel.load(
                metadata=SingleModelMetadata(
                    model_id=f"{payload['model_id']}_dashboard_tree_{depth}",
                    name=f"{payload['model_name']}_dashboard_tree_{depth}",
                    path=tree_path,
                    format=ModelFormatEnum.JOBLIB,
                    metadata={},
                )
            )
    return tree_models


def transform_feature_frame(
    frame: pd.DataFrame,
    preprocessor,
    feature_names: list[str],
) -> pd.DataFrame:
    ordered = frame.loc[:, feature_names].copy()
    if preprocessor is None:
        return ordered
    transformed = preprocessor.transform(ordered)
    matrix = np.asarray(transformed, dtype=np.float32)
    return pd.DataFrame(matrix, index=ordered.index, columns=feature_names)


def predict_raw_frame(trained_model, raw_frame: pd.DataFrame) -> np.ndarray:
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
    model,
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


def predict_transformed_values(model, feature_names: list[str], values):
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
):
    explanation.display_data = raw_frame.loc[:, feature_names].to_numpy()
    mean_abs_shap = pd.Series(
        explanation.abs.mean(0).values, index=feature_names
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


def build_neighbors_frame(
    *,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    trained_model,
    model_input_features: list[str],
    sample_indices: list[t.Any],
    neighbors_k: int,
) -> pd.DataFrame:
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.neighbors_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    dataset_features = build_feature_frame_from_trades(raw_frame.loc[sampled_dataset.index])
    sample_features = build_feature_frame_from_trades(raw_frame.loc[sample_indices])
    transformed_dataset = transform_feature_frame(
        dataset_features,
        trained_model.preprocessor,
        model_input_features,
    )
    transformed_samples = transform_feature_frame(
        sample_features,
        trained_model.preprocessor,
        model_input_features,
    )
    estimator = NearestNeighbors(n_neighbors=min(neighbors_k, len(sampled_dataset)))
    estimator.fit(transformed_dataset.to_numpy())
    distances, indices = estimator.kneighbors(transformed_samples.to_numpy())
    rows: list[dict[str, t.Any]] = []
    dataset_indices = sampled_dataset.index.to_numpy()
    for sample_position, sample_index in enumerate(sample_indices):
        for rank, neighbor_position in enumerate(indices[sample_position]):
            rows.append(
                {
                    "sample_index": sample_index,
                    "neighbor_index": dataset_indices[neighbor_position],
                    "rank": int(rank),
                    "distance": float(distances[sample_position, rank]),
                }
            )
    return pd.DataFrame(rows)


def build_surfaces_frame(
    *,
    trained_model,
    raw_frame: pd.DataFrame,
    anchor_indices: list[t.Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    surface_grid_size = config.dashboard_models_config.surface_grid_size
    for anchor_index in anchor_indices:
        anchor = raw_frame.loc[[anchor_index]].copy()
        base_underlying = float(anchor[str(VolatilityDBEnum.UNDERLYING_PRICE)].iloc[0])
        anchor_tte = float(anchor[str(VolatilityDBEnum.TIME_TO_EXPIRATION)].iloc[0])
        moneyness_values = np.linspace(0.8, 1.2, surface_grid_size)
        maturity_values = np.linspace(1.0, max(anchor_tte * 1.5, 30.0), surface_grid_size)
        grid_rows: list[pd.DataFrame] = []
        for maturity in maturity_values:
            for moneyness in moneyness_values:
                row = anchor.copy()
                row[str(VolatilityDBEnum.TIME_TO_EXPIRATION)] = maturity
                row[str(VolatilityDBEnum.UNDERLYING_PRICE)] = base_underlying
                row[str(VolatilityDBEnum.STRIKE_PRICE)] = base_underlying / moneyness
                grid_rows.append(add_dashboard_derived_features(row))
        surface_raw = pd.concat(grid_rows, ignore_index=True)
        surface = build_model_dataset(surface_raw)
        surface[str(TARGET_COLUMN)] = np.nan
        surface["anchor_index"] = anchor_index
        surface["PredictedVolatility"] = predict_raw_frame(trained_model, surface_raw)
        rows.append(surface)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_ice_frame(
    *,
    trained_model,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    curve_points = config.dashboard_models_config.curve_points
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.ice_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name not in sampled_dataset.columns:
            continue
        values = quantile_grid(sampled_dataset[feature_name], curve_points)
        if not values:
            continue
        for sample_id, sample_index in enumerate(sampled_dataset.index):
            base_raw = raw_frame.loc[[sample_index]].copy()
            for value in values:
                adjusted = apply_feature_override(base_raw, feature_name, value)
                prediction = float(predict_raw_frame(trained_model, adjusted)[0])
                rows.append(
                    {
                        "feature_name": feature_name,
                        "sample_id": int(sample_id),
                        "feature_value": float(value),
                        "prediction": prediction,
                    }
                )
    return pd.DataFrame(rows)


def build_ale_frame(
    *,
    trained_model,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name not in dataset_frame.columns:
            continue
        series = pd.to_numeric(dataset_frame[feature_name], errors="coerce")
        edges = (
            series.dropna()
            .quantile(np.linspace(0.05, 0.95, 13))
            .drop_duplicates()
            .tolist()
        )
        if len(edges) < 2:
            continue
        increments: list[float] = []
        centers: list[float] = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            bucket_index = dataset_frame.loc[(series >= lower) & (series <= upper)].index
            if len(bucket_index) == 0:
                continue
            bucket_raw = raw_frame.loc[bucket_index]
            lower_frame = apply_feature_override(bucket_raw, feature_name, lower)
            upper_frame = apply_feature_override(bucket_raw, feature_name, upper)
            delta = (
                pd.Series(predict_raw_frame(trained_model, upper_frame), index=bucket_index)
                - pd.Series(predict_raw_frame(trained_model, lower_frame), index=bucket_index)
            ).mean()
            increments.append(float(delta))
            centers.append(float((lower + upper) / 2.0))
        if not increments:
            continue
        ale = np.cumsum(increments)
        ale = ale - ale.mean()
        for center, value in zip(centers, ale):
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_value": float(center),
                    "ale": float(value),
                }
            )
    return pd.DataFrame(rows)


def build_diagnosis_artifact(
    dataset_frame: pd.DataFrame,
    *,
    financial_warnings: list[str],
):
    sampled = sample_frame(
        dataset_frame.dropna(subset=[str(TARGET_COLUMN)]),
        max_rows=config.dashboard_models_config.diagnosis_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    metrics = METRICS_REGISTRY.compute_metrics(
        sampled[str(TARGET_COLUMN)].astype(float).reset_index(drop=True),
        sampled["PredictedVolatility"].astype(float).reset_index(drop=True),
        config.dashboard_models_config.error_metrics,
    )
    plot_frame = sample_frame(
        sampled,
        max_rows=min(2500, len(sampled)),
        random_state=config.dashboard_models_config.random_state + 7,
    )
    error_heatmap = (
        sampled.assign(
            moneyness_bin=pd.cut(sampled["Moneyness"], bins=12),
            maturity_bin=pd.cut(sampled[str(VolatilityDBEnum.TIME_TO_EXPIRATION)], bins=12),
        )
        .groupby(["moneyness_bin", "maturity_bin"], observed=False)["AbsoluteError"]
        .mean()
        .reset_index()
    )
    return DiagnosisArtifact(
        metrics={str(name): float(value) for name, value in metrics.items()},
        plot_frame=plot_frame,
        error_heatmap=error_heatmap,
        financial_warnings=list(financial_warnings),
    )


def financial_checks_from_surface(surface_frame: pd.DataFrame) -> list[str]:
    if {"TimeToExpiration", "Moneyness", "PredictedVolatility"}.issubset(
        surface_frame.columns
    ):
        pivot = surface_frame.pivot_table(
            index="TimeToExpiration",
            columns="Moneyness",
            values="PredictedVolatility",
        ).sort_index()
        smile_diff = pivot.diff(axis=1).abs().max().max()
        term_diff = pivot.diff(axis=0).abs().max().max()
        warnings: list[str] = []
        if pd.notna(smile_diff) and smile_diff > 0.20:
            warnings.append(
                "Heuristic warning: adjacent smile points show large volatility jumps."
            )
        if pd.notna(term_diff) and term_diff > 0.20:
            warnings.append(
                "Heuristic warning: adjacent maturity points show large term-structure jumps."
            )
        if warnings:
            return warnings
    return ["No large discontinuities were detected by the heuristic checks."]


def build_surrogate_tree_models(
    *,
    trained_model,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    model_input_features: list[str],
    surrogate_depths: tuple[int, ...],
) -> dict[int, SurrogateTreeModel]:
    sampled = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.surrogate_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    sampled_features = build_feature_frame_from_trades(raw_frame.loc[sampled.index])
    transformed = transform_feature_frame(
        sampled_features,
        trained_model.preprocessor,
        model_input_features,
    )
    predictions = pd.Series(
        predict_raw_frame(trained_model, raw_frame.loc[sampled.index]),
        index=sampled.index,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        transformed,
        predictions,
        test_size=0.2,
        random_state=config.dashboard_models_config.random_state,
    )
    tree_models: dict[int, SurrogateTreeModel] = {}
    for depth in surrogate_depths:
        surrogate = DecisionTreeRegressor(
            max_depth=int(depth),
            min_samples_leaf=config.dashboard_models_config.surrogate_min_samples_leaf,
            random_state=config.dashboard_models_config.random_state,
        )
        surrogate.fit(X_train, y_train)
        y_pred = pd.Series(surrogate.predict(X_test), index=y_test.index)
        metrics = METRICS_REGISTRY.compute_metrics(
            y_test.reset_index(drop=True),
            y_pred.reset_index(drop=True),
            config.dashboard_models_config.error_metrics,
        )
        importances = pd.Series(
            surrogate.feature_importances_,
            index=model_input_features,
        ).sort_values(ascending=False)
        top_features = importances[importances > 0].head(3).index.tolist()
        tree_models[int(depth)] = SurrogateTreeModel(
            model=surrogate,
            feature_importances=importances,
            tree_depth=surrogate.get_depth(),
            n_leaves=surrogate.get_n_leaves(),
            text_rules=export_text(surrogate, feature_names=model_input_features),
            interpretation=(
                "The surrogate approximates the trained volatility model "
                f"with RMSE {metrics['rmse']:.4f} at max depth {int(depth)}. "
                f"The dominant decision logic is driven by {', '.join(top_features) or 'no features'}."
            ),
            fidelity_frame=pd.DataFrame(
                {
                    "model_prediction": y_test.reset_index(drop=True),
                    "surrogate_prediction": y_pred.reset_index(drop=True),
                }
            ),
            feature_names=list(model_input_features),
            metrics={str(name): float(value) for name, value in metrics.items()},
        )
    return tree_models


def load_preprocessor(
    base_path: Path, metadata: dict[str, t.Any]
):
    explicit_preprocessor = metadata.get("preprocessor_path")
    candidates: list[Path] = []
    if explicit_preprocessor:
        explicit_path = Path(explicit_preprocessor)
        if explicit_path.is_absolute():
            candidates.append(explicit_path)
        else:
            parent = base_path if base_path.is_dir() else base_path.parent
            candidates.append(parent / explicit_path)
            candidates.append(parent.parent / explicit_path)
    if base_path.is_dir():
        candidates.extend(
            [
                base_path / "preprocessor.joblib",
                base_path.parent / "preprocessor.joblib",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return joblib.load(candidate)
    return None
