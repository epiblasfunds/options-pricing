from __future__ import annotations

import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor, export_text

from src.python_models.explainable_model import SurrogateTreeModel
from src.volatility_models.model_explainability.config import (
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_METRICS_REGISTRY,
    DEFAULT_SETTINGS,
)
from src.volatility_models.model_explainability.utils.feature_utils import (
    add_derived_features,
    apply_feature_override,
)
from src.volatility_models.model_explainability.utils.preprocessing import (
    build_similarity_preprocessor,
    build_tree_preprocessor,
)
from src.volatility_models.model_explainability.utils.sampling import (
    quantile_grid,
    sample_frame,
)


def build_dashboard_model_from_runtime(
    cls,
    *,
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None,
    preprocessor: ColumnTransformer | None,
    build_config,
):
    from src.python_models.dashboard_models import ManualApiStubResponse

    explainable_model = (
        model if model.__class__.__name__ == "ExplainableModel" else None
    )
    epi_model = model.main_model if explainable_model is not None else model
    metadata = dict(getattr(epi_model.metadata, "metadata", {}) or {})
    model_id = getattr(epi_model.metadata, "model_id", "dashboard_model")
    model_name = getattr(epi_model.metadata, "name", model_id)
    raw_feature_names = list(
        metadata.get(
            "model_input_features", list(DEFAULT_SETTINGS.model_input_features)
        )
    )
    target_column = metadata.get("target_column", DEFAULT_SETTINGS.target_column)
    transformed_feature_names = list(
        metadata.get("transformed_feature_names", raw_feature_names)
    )

    working = X.copy()
    if y is not None:
        target_series = (
            y
            if isinstance(y, pd.Series)
            else pd.Series(y, index=working.index, name=target_column)
        )
        working[target_column] = target_series
    working = add_derived_features(working, DEFAULT_FEATURE_SCHEMA)
    prepared = prepare_feature_frame(working, raw_feature_names)
    resolved_preprocessor = preprocessor or load_preprocessor(
        epi_model.metadata.path, metadata
    )
    transformed = transform_frame(
        prepared,
        raw_feature_names,
        transformed_feature_names,
        resolved_preprocessor,
    )
    predictions = predict_transformed(epi_model.model, transformed)
    dataset_frame = working.copy()
    dataset_frame["PredictedVolatility"] = predictions
    if target_column in dataset_frame.columns:
        dataset_frame["Residual"] = (
            dataset_frame[target_column].astype(float)
            - dataset_frame["PredictedVolatility"]
        )
        dataset_frame["AbsoluteError"] = dataset_frame["Residual"].abs()

    tree_models = (
        dict(explainable_model.tree_models)
        if explainable_model is not None
        else build_surrogate_tree_models(
            model=epi_model.model,
            preprocessor=resolved_preprocessor,
            reference_frame=dataset_frame,
            model_input_features=raw_feature_names,
            surrogate_depths=build_config.surrogate_depths,
        )
    )
    sample_indices = sample_frame(
        dataset_frame,
        max_rows=build_config.sample_option_size,
        random_state=DEFAULT_SETTINGS.random_state,
    ).index.tolist()
    behaviour_anchor_indices = sample_frame(
        dataset_frame,
        max_rows=min(build_config.behaviour_anchor_size, len(dataset_frame)),
        random_state=DEFAULT_SETTINGS.random_state + 11,
    ).index.tolist()

    background = sample_frame(
        dataset_frame,
        max_rows=DEFAULT_SETTINGS.shap_background_size,
        random_state=DEFAULT_SETTINGS.random_state + 1,
    )
    global_rows = sample_frame(
        dataset_frame,
        max_rows=DEFAULT_SETTINGS.shap_explain_size,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    explainer = build_shap_explainer(
        model=epi_model.model,
        background_frame=transform_frame(
            prepare_feature_frame(background, raw_feature_names),
            raw_feature_names,
            transformed_feature_names,
            resolved_preprocessor,
        ),
        feature_names=transformed_feature_names,
    )
    global_shap = serialize_shap_result(
        method="shap.Explainer(permutation)",
        explanation=explainer(
            transform_frame(
                prepare_feature_frame(global_rows, raw_feature_names),
                raw_feature_names,
                transformed_feature_names,
                resolved_preprocessor,
            ),
            max_evals=max_evals(len(transformed_feature_names)),
            silent=True,
        ),
        transformed_frame=transform_frame(
            prepare_feature_frame(global_rows, raw_feature_names),
            raw_feature_names,
            transformed_feature_names,
            resolved_preprocessor,
        ),
        raw_frame=prepare_feature_frame(global_rows, raw_feature_names),
        feature_names=transformed_feature_names,
        predictions=dataset_frame.loc[global_rows.index, "PredictedVolatility"],
    )
    local_rows = dataset_frame.loc[sample_indices]
    local_shap = serialize_shap_result(
        method="shap.Explainer(permutation)",
        explanation=explainer(
            transform_frame(
                prepare_feature_frame(local_rows, raw_feature_names),
                raw_feature_names,
                transformed_feature_names,
                resolved_preprocessor,
            ),
            max_evals=max_evals(len(transformed_feature_names)),
            silent=True,
        ),
        transformed_frame=transform_frame(
            prepare_feature_frame(local_rows, raw_feature_names),
            raw_feature_names,
            transformed_feature_names,
            resolved_preprocessor,
        ),
        raw_frame=prepare_feature_frame(local_rows, raw_feature_names),
        feature_names=transformed_feature_names,
        predictions=dataset_frame.loc[local_rows.index, "PredictedVolatility"],
    )

    neighbors_frame = build_neighbors_frame(
        dataset_frame=dataset_frame,
        raw_feature_names=raw_feature_names,
        sample_indices=sample_indices,
        neighbors_k=build_config.neighbors_k,
    )
    surfaces_frame = build_surfaces_frame(
        model=epi_model.model,
        preprocessor=resolved_preprocessor,
        dataset_frame=dataset_frame,
        raw_feature_names=raw_feature_names,
        anchor_indices=behaviour_anchor_indices,
        target_column=target_column,
    )
    numerical_features = [
        feature.name
        for feature in DEFAULT_FEATURE_SCHEMA.numerical_features(raw_only=False)
        if feature.name in dataset_frame.columns
        or feature.name in {"Moneyness", "LogMoneyness"}
    ]
    ice_frame = build_ice_frame(
        model=epi_model.model,
        preprocessor=resolved_preprocessor,
        dataset_frame=dataset_frame,
        raw_feature_names=raw_feature_names,
        feature_names=numerical_features,
    )
    ale_frame = build_ale_frame(
        model=epi_model.model,
        preprocessor=resolved_preprocessor,
        dataset_frame=dataset_frame,
        raw_feature_names=raw_feature_names,
        feature_names=numerical_features,
    )
    diagnosis = build_diagnosis_artifact(
        dataset_frame,
        target_column=target_column,
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
        summary="Placeholder response from the manual-input prediction API.",
        reference_sample_index=sample_indices[0] if sample_indices else None,
    )
    return cls(
        model_id=model_id,
        model_name=model_name,
        metadata=metadata,
        dataset_frame=dataset_frame,
        raw_feature_names=raw_feature_names,
        transformed_feature_names=transformed_feature_names,
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
                metadata=single_metadata_from_path(
                    tree_path,
                    model_id=f"{payload['model_id']}_dashboard_tree_{depth}",
                    name=f"{payload['model_name']}_dashboard_tree_{depth}",
                )
            )
    return tree_models


def single_metadata_from_path(path: Path, model_id: str, name: str):
    from src.enums.volatility_model_enums import ModelFormatEnum
    from src.python_models.explainable_model import SingleModelMetadata

    return SingleModelMetadata(
        model_id=model_id,
        name=name,
        path=path,
        format=ModelFormatEnum.JOBLIB,
        metadata={},
    )


def prepare_feature_frame(
    frame: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for feature in DEFAULT_FEATURE_SCHEMA.numerical_features(raw_only=True):
        if feature.name in feature_names and feature.name in prepared.columns:
            prepared[feature.name] = pd.to_numeric(
                prepared[feature.name], errors="coerce"
            )
    for feature in DEFAULT_FEATURE_SCHEMA.categorical_features(raw_only=True):
        if feature.name in feature_names and feature.name in prepared.columns:
            prepared[feature.name] = prepared[feature.name].astype("object")
    return prepared[feature_names]


def transform_frame(
    frame: pd.DataFrame,
    raw_feature_names: list[str],
    transformed_feature_names: list[str],
    preprocessor: ColumnTransformer | None,
) -> pd.DataFrame:
    if preprocessor is None:
        return frame[transformed_feature_names].copy()
    transformed = preprocessor.transform(frame[raw_feature_names])
    matrix = np.asarray(transformed, dtype=np.float32)
    return pd.DataFrame(matrix, index=frame.index, columns=transformed_feature_names)


def predict_transformed(model, transformed_frame: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(
        transformed_frame.to_numpy(dtype=np.float32, copy=False), verbose=0
    )
    return np.asarray(predictions).reshape(-1)


def predict_frame(
    model,
    preprocessor: ColumnTransformer | None,
    frame: pd.DataFrame,
    raw_feature_names: list[str],
    transformed_feature_names: list[str],
) -> pd.Series:
    prepared = prepare_feature_frame(frame, raw_feature_names)
    transformed = transform_frame(
        prepared,
        raw_feature_names,
        transformed_feature_names,
        preprocessor,
    )
    return pd.Series(
        predict_transformed(model, transformed),
        index=frame.index,
        name="PredictedVolatility",
    )


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
        seed=DEFAULT_SETTINGS.random_state,
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
        DEFAULT_SETTINGS.shap_permutations * n_features,
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
    from src.python_models.dashboard_models import StoredShapExplanation

    display_frame = pd.DataFrame(index=raw_frame.index)
    for feature_name in feature_names:
        display_frame[feature_name] = display_series(raw_frame, feature_name)
    explanation.display_data = display_frame.to_numpy()
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


def display_series(raw_frame: pd.DataFrame, feature_name: str) -> pd.Series:
    if feature_name in raw_frame.columns:
        return raw_frame[feature_name]
    if "__" not in feature_name:
        return pd.Series([None] * len(raw_frame), index=raw_frame.index)
    _, transformed_name = feature_name.split("__", 1)
    if transformed_name in raw_frame.columns:
        return raw_frame[transformed_name]
    for feature in DEFAULT_FEATURE_SCHEMA.categorical_features(raw_only=True):
        prefix = f"{feature.name}_"
        if transformed_name.startswith(prefix) and feature.name in raw_frame.columns:
            expected_value = transformed_name[len(prefix):]
            return (
                raw_frame[feature.name].astype(str).str.upper()
                == str(expected_value).upper()
            ).astype(int)
    return pd.Series([None] * len(raw_frame), index=raw_frame.index)


def build_neighbors_frame(
    *,
    dataset_frame: pd.DataFrame,
    raw_feature_names: list[str],
    sample_indices: list[t.Any],
    neighbors_k: int,
) -> pd.DataFrame:
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=DEFAULT_SETTINGS.neighbors_sample_size,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    prepared_dataset = prepare_feature_frame(sampled_dataset, raw_feature_names)
    prepared_samples = prepare_feature_frame(
        dataset_frame.loc[sample_indices], raw_feature_names
    )
    preprocessor = build_similarity_preprocessor(
        DEFAULT_FEATURE_SCHEMA, raw_feature_names
    )
    transformed_dataset = preprocessor.fit_transform(
        prepared_dataset[raw_feature_names]
    )
    transformed_samples = preprocessor.transform(prepared_samples[raw_feature_names])
    estimator = NearestNeighbors(n_neighbors=min(neighbors_k, len(sampled_dataset)))
    estimator.fit(transformed_dataset)
    distances, indices = estimator.kneighbors(transformed_samples)
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
    model,
    preprocessor: ColumnTransformer | None,
    dataset_frame: pd.DataFrame,
    raw_feature_names: list[str],
    anchor_indices: list[t.Any],
    target_column: str,
) -> pd.DataFrame:
    transformed_feature_names = resolve_transformed_feature_names(
        raw_feature_names, preprocessor
    )
    rows: list[pd.DataFrame] = []
    for anchor_index in anchor_indices:
        anchor = dataset_frame.loc[anchor_index, raw_feature_names].copy()
        anchor_frame = anchor.to_frame().T
        base_underlying = float(anchor_frame["UnderlyingPrice"].iloc[0])
        moneyness_values = np.linspace(0.8, 1.2, DEFAULT_SETTINGS.surface_grid_size)
        maturity_values = np.linspace(
            1.0,
            max(float(anchor_frame["TimeToExpiration"].iloc[0]) * 1.5, 30.0),
            DEFAULT_SETTINGS.surface_grid_size,
        )
        grid_rows: list[pd.DataFrame] = []
        for maturity in maturity_values:
            for moneyness in moneyness_values:
                row = anchor_frame.copy()
                row["TimeToExpiration"] = maturity
                row["UnderlyingPrice"] = base_underlying
                row["StrikePrice"] = base_underlying / moneyness
                row["Moneyness"] = moneyness
                row["LogMoneyness"] = np.log(moneyness)
                row["AbsLogMoneyness"] = abs(np.log(moneyness))
                if target_column in row.columns:
                    row[target_column] = np.nan
                row["anchor_index"] = anchor_index
                grid_rows.append(row)
        surface = pd.concat(grid_rows, ignore_index=True)
        surface["PredictedVolatility"] = predict_frame(
            model,
            preprocessor,
            surface,
            raw_feature_names,
            transformed_feature_names,
        ).to_numpy()
        rows.append(surface)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_ice_frame(
    *,
    model,
    preprocessor: ColumnTransformer | None,
    dataset_frame: pd.DataFrame,
    raw_feature_names: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    transformed_feature_names = resolve_transformed_feature_names(
        raw_feature_names, preprocessor
    )
    sampled = sample_frame(
        dataset_frame,
        max_rows=DEFAULT_SETTINGS.ice_sample_size,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name in sampled.columns:
            values = quantile_grid(sampled[feature_name], DEFAULT_SETTINGS.curve_points)
        elif feature_name in {"Moneyness", "LogMoneyness"}:
            values = quantile_grid(sampled[feature_name], DEFAULT_SETTINGS.curve_points)
        else:
            continue
        for sample_id, (_, sample_row) in enumerate(sampled.iterrows()):
            base_frame = sample_row.to_frame().T
            for value in values:
                adjusted = apply_feature_override(base_frame, feature_name, value)
                prediction = float(
                    predict_frame(
                        model,
                        preprocessor,
                        adjusted,
                        raw_feature_names,
                        transformed_feature_names,
                    ).iloc[0]
                )
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
    model,
    preprocessor: ColumnTransformer | None,
    dataset_frame: pd.DataFrame,
    raw_feature_names: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    transformed_feature_names = resolve_transformed_feature_names(
        raw_feature_names, preprocessor
    )
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name not in dataset_frame.columns:
            continue
        series = dataset_frame[feature_name]
        edges = (
            pd.Series(series)
            .dropna()
            .quantile(np.linspace(0.05, 0.95, 13))
            .drop_duplicates()
            .tolist()
        )
        if len(edges) < 2:
            continue
        increments: list[float] = []
        centers: list[float] = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            bucket = dataset_frame.loc[(series >= lower) & (series <= upper)]
            if bucket.empty:
                continue
            lower_frame = apply_feature_override(bucket, feature_name, lower)
            upper_frame = apply_feature_override(bucket, feature_name, upper)
            delta = (
                predict_frame(
                    model,
                    preprocessor,
                    upper_frame,
                    raw_feature_names,
                    transformed_feature_names,
                )
                - predict_frame(
                    model,
                    preprocessor,
                    lower_frame,
                    raw_feature_names,
                    transformed_feature_names,
                )
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
    target_column: str,
    financial_warnings: list[str],
):
    from src.python_models.dashboard_models import DiagnosisArtifact

    sampled = sample_frame(
        dataset_frame.dropna(subset=[target_column]),
        max_rows=DEFAULT_SETTINGS.diagnosis_sample_size,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    metrics = DEFAULT_METRICS_REGISTRY.compute_metrics(
        sampled[target_column].astype(float).reset_index(drop=True),
        sampled["PredictedVolatility"].astype(float).reset_index(drop=True),
        DEFAULT_SETTINGS.error_metrics,
    )
    plot_frame = sample_frame(
        sampled,
        max_rows=min(2500, DEFAULT_SETTINGS.diagnosis_sample_size),
        random_state=DEFAULT_SETTINGS.random_state + 7,
    )
    error_heatmap = (
        sampled.assign(
            moneyness_bin=pd.cut(sampled["Moneyness"], bins=12),
            maturity_bin=pd.cut(sampled["TimeToExpiration"], bins=12),
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
    model,
    preprocessor: ColumnTransformer | None,
    reference_frame: pd.DataFrame,
    model_input_features: list[str],
    surrogate_depths: tuple[int, ...],
) -> dict[int, SurrogateTreeModel]:
    sampled = sample_frame(
        reference_frame,
        max_rows=DEFAULT_SETTINGS.surrogate_sample_size,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    prepared = prepare_feature_frame(sampled, model_input_features)
    predictions = predict_frame(
        model,
        preprocessor,
        sampled,
        model_input_features,
        resolve_transformed_feature_names(model_input_features, preprocessor),
    )
    tree_preprocessor = build_tree_preprocessor(
        DEFAULT_FEATURE_SCHEMA, model_input_features
    )
    transformed = tree_preprocessor.fit_transform(prepared)
    X_train, X_test, y_train, y_test = train_test_split(
        transformed,
        predictions,
        test_size=0.2,
        random_state=DEFAULT_SETTINGS.random_state,
    )
    tree_models: dict[int, SurrogateTreeModel] = {}
    for depth in surrogate_depths:
        surrogate = DecisionTreeRegressor(
            max_depth=int(depth),
            min_samples_leaf=DEFAULT_SETTINGS.surrogate_min_samples_leaf,
            random_state=DEFAULT_SETTINGS.random_state,
        )
        surrogate.fit(X_train, y_train)
        y_pred = pd.Series(surrogate.predict(X_test), index=y_test.index)
        metrics = DEFAULT_METRICS_REGISTRY.compute_metrics(
            y_test.reset_index(drop=True),
            y_pred.reset_index(drop=True),
            DEFAULT_SETTINGS.error_metrics,
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
                "The surrogate approximates the main volatility model "
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


def resolve_transformed_feature_names(
    raw_feature_names: list[str],
    preprocessor: ColumnTransformer | None,
) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass
    return list(raw_feature_names)


def load_preprocessor(
    base_path: Path, metadata: dict[str, t.Any]
) -> ColumnTransformer | None:
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
            loaded = joblib.load(candidate)
            if isinstance(loaded, ColumnTransformer):
                return loaded
            return t.cast(ColumnTransformer, loaded)
    return None
