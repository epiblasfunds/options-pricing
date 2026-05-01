import typing as t

import numpy as np
import pandas as pd
import shap
import sympy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.dashboard.utils.sampling import quantile_grid
from src.dashboard.utils.sampling import sample_frame
from src.enums.data_enums import VolatilityDBEnum
from src.model2dashboard.features import ANALYSIS_FEATURE_NAMES
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.model2dashboard.features import TARGET_COLUMN
from src.model2dashboard.features import add_dashboard_derived_features
from src.model2dashboard.features import apply_feature_override
from src.model2dashboard.features import build_dashboard_dataset
from src.model2dashboard.features import build_explainability_encoder
from src.model2dashboard.features import build_explainability_frame
from src.model2dashboard.features import build_feature_frame_from_trades
from src.model2dashboard.features import column_name
from src.model2dashboard.model_io import TrainingModelRuntime
from src.model2dashboard.model_io import predict_feature_frame
from src.model2dashboard.model_io import predict_raw_frame
from src.model2dashboard.model_io import transform_feature_frame
from src.model2dashboard.surface_checks import financial_checks_from_surface
from src.python_models.dashboard.artifacts import DiagnosisArtifact
from src.python_models.dashboard.artifacts import ManualApiStubResponse
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.python_models.dashboard.artifacts import SurrogateTreeModel
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


METRICS_REGISTRY = build_metrics_registry()


def build_dashboard_artifacts(
    *,
    runtime: TrainingModelRuntime,
    raw_test_frame: pd.DataFrame,
) -> dict[str, t.Any]:
    feature_frame = build_feature_frame_from_trades(raw_test_frame)
    predictions = pd.Series(
        predict_feature_frame(runtime, feature_frame),
        index=raw_test_frame.index,
        name="PredictedVolatility",
    )
    dataset_frame = build_dashboard_dataset(raw_test_frame, predictions)
    sample_indices = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.build_config.sample_option_size,
        random_state=config.dashboard_models_config.random_state,
    ).index.tolist()
    behaviour_anchor_indices = sample_frame(
        dataset_frame,
        max_rows=min(
            config.dashboard_models_config.build_config.behaviour_anchor_size,
            len(dataset_frame),
        ),
        random_state=config.dashboard_models_config.random_state + 11,
    ).index.tolist()
    global_shap, local_shap = build_shap_artifacts(
        runtime=runtime,
        dataset_frame=dataset_frame,
        raw_frame=raw_test_frame,
        predictions=predictions,
        sample_indices=sample_indices,
    )
    surfaces_frame = build_surfaces_frame(
        runtime=runtime,
        raw_frame=raw_test_frame,
        anchor_indices=behaviour_anchor_indices,
    )
    warnings = financial_checks_from_surface(
        surfaces_frame.loc[
            surfaces_frame["anchor_index"] == behaviour_anchor_indices[0]
        ]
        if behaviour_anchor_indices and not surfaces_frame.empty
        else pd.DataFrame()
    )
    return {
        "dataset_frame": dataset_frame,
        "feature_frame": feature_frame,
        "predictions": predictions,
        "sample_indices": sample_indices,
        "behaviour_anchor_indices": behaviour_anchor_indices,
        "tree_models": build_surrogate_tree_models(
            runtime=runtime,
            feature_frame=feature_frame,
            predictions=predictions,
            dataset_frame=dataset_frame,
        ),
        "symbolic_model": build_symbolic_regressor_model(
            runtime=runtime,
            feature_frame=feature_frame,
            predictions=predictions,
            dataset_frame=dataset_frame,
        ),
        "global_shap": global_shap,
        "local_shap": local_shap,
        "neighbors_frame": build_neighbors_frame(
            runtime=runtime,
            dataset_frame=dataset_frame,
            feature_frame=feature_frame,
            sample_indices=sample_indices,
        ),
        "surfaces_frame": surfaces_frame,
        "ice_frame": build_ice_frame(
            runtime=runtime,
            dataset_frame=dataset_frame,
            raw_frame=raw_test_frame,
            feature_names=ANALYSIS_FEATURE_NAMES,
        ),
        "ale_frame": build_ale_frame(
            runtime=runtime,
            dataset_frame=dataset_frame,
            raw_frame=raw_test_frame,
            feature_names=ANALYSIS_FEATURE_NAMES,
        ),
        "diagnosis": build_diagnosis_artifact(
            dataset_frame=dataset_frame,
            financial_warnings=warnings,
        ),
        "manual_api_stub": ManualApiStubResponse(
            prediction=float(predictions.mean()),
            summary="Reference prediction based on the final-test dashboard bundle.",
            reference_sample_index=sample_indices[0] if sample_indices else None,
        ),
    }


def build_shap_artifacts(
    *,
    runtime: TrainingModelRuntime,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    predictions: pd.Series,
    sample_indices: list[t.Any],
) -> tuple[StoredShapExplanation, StoredShapExplanation]:
    explainability_frame = build_explainability_frame(
        raw_frame,
        feature_names=EXPLAINABILITY_FEATURE_NAMES,
    )
    background_indices = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.shap_background_size,
        random_state=config.dashboard_models_config.random_state + 1,
    ).index
    global_indices = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.shap_explain_size,
        random_state=config.dashboard_models_config.random_state,
    ).index
    method = "shap.Explainer(permutation)"
    background_explain_frame = explainability_frame.loc[background_indices]
    global_shap = _build_stored_shap_for_rows(
        method=method,
        runtime=runtime,
        raw_frame=raw_frame,
        explainability_frame=explainability_frame,
        background_explain_frame=background_explain_frame,
        row_indices=list(global_indices),
        predictions=predictions,
    )
    local_shap = _build_stored_shap_for_rows(
        method=method,
        runtime=runtime,
        raw_frame=raw_frame,
        explainability_frame=explainability_frame,
        background_explain_frame=background_explain_frame,
        row_indices=list(sample_indices),
        predictions=predictions,
    )
    return global_shap, local_shap


def _build_stored_shap_for_rows(
    *,
    method: str,
    runtime: TrainingModelRuntime,
    raw_frame: pd.DataFrame,
    explainability_frame: pd.DataFrame,
    background_explain_frame: pd.DataFrame,
    row_indices: list[t.Any],
    predictions: pd.Series,
) -> StoredShapExplanation:
    stored_rows: list[StoredShapExplanation] = []
    max_evals = _max_evals(len(EXPLAINABILITY_FEATURE_NAMES))
    for row_index in row_indices:
        row_raw = raw_frame.loc[[row_index]].copy()
        row_explain_frame = explainability_frame.loc[[row_index]].copy()
        row_encoder = build_explainability_encoder(
            pd.concat([background_explain_frame, row_raw], axis=0),
            feature_names=EXPLAINABILITY_FEATURE_NAMES,
            defaults_override=row_raw.iloc[0].to_dict(),
        )
        encoded_background = row_encoder.encode_frame(background_explain_frame)
        encoded_row = row_encoder.encode_frame(row_explain_frame)
        explainer = shap.Explainer(
            lambda values, encoder=row_encoder: _predict_explainability_values(
                runtime, encoder, values
            ),
            masker=encoded_background,
            algorithm="permutation",
            feature_names=list(row_encoder.feature_names),
            seed=config.dashboard_models_config.random_state,
        )
        explanation = explainer(
            encoded_row,
            max_evals=max_evals,
            silent=True,
        )
        stored_rows.append(
            serialize_shap_result(
                method=method,
                explanation=explanation,
                transformed_frame=encoded_row,
                display_frame=row_explain_frame,
                feature_names=list(row_encoder.feature_names),
                predictions=predictions.loc[[row_index]],
            )
        )
    return _merge_stored_shap_rows(
        method="shap.Explainer(permutation)",
        feature_names=list(EXPLAINABILITY_FEATURE_NAMES),
        stored_rows=stored_rows,
    )


def _merge_stored_shap_rows(
    *,
    method: str,
    feature_names: list[str],
    stored_rows: list[StoredShapExplanation],
) -> StoredShapExplanation:
    feature_count = len(feature_names)
    if not stored_rows:
        return StoredShapExplanation(
            method=method,
            feature_names=list(feature_names),
            index=[],
            values=np.empty((0, feature_count), dtype="float64"),
            base_values=np.empty((0,), dtype="float64"),
            data=np.empty((0, feature_count), dtype="float64"),
            display_data=np.empty((0, feature_count), dtype="float64"),
            predictions=np.empty((0,), dtype="float64"),
            mean_abs_shap={str(name): 0.0 for name in feature_names},
        )
    values = np.concatenate([np.asarray(row.values) for row in stored_rows], axis=0)
    base_values = np.concatenate(
        [np.asarray(row.base_values).reshape(-1) for row in stored_rows],
        axis=0,
    )
    data = np.concatenate([np.asarray(row.data) for row in stored_rows], axis=0)
    display_data = (
        None
        if any(row.display_data is None for row in stored_rows)
        else np.concatenate(
            [np.asarray(row.display_data) for row in stored_rows],
            axis=0,
        )
    )
    predictions = np.concatenate(
        [np.asarray(row.predictions).reshape(-1) for row in stored_rows],
        axis=0,
    )
    mean_abs_shap = pd.Series(
        np.abs(values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)
    return StoredShapExplanation(
        method=method,
        feature_names=list(feature_names),
        index=[index for row in stored_rows for index in row.index],
        values=values,
        base_values=base_values,
        data=data,
        display_data=display_data,
        predictions=predictions,
        mean_abs_shap={
            str(name): float(value) for name, value in mean_abs_shap.items()
        },
    )


def serialize_shap_result(
    *,
    method: str,
    explanation: shap.Explanation,
    transformed_frame: pd.DataFrame,
    display_frame: pd.DataFrame,
    feature_names: list[str],
    predictions: pd.Series,
) -> StoredShapExplanation:
    explanation.display_data = display_frame.loc[:, feature_names].to_numpy()
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


def build_surrogate_tree_models(
    *,
    runtime: TrainingModelRuntime,
    feature_frame: pd.DataFrame,
    predictions: pd.Series,
    dataset_frame: pd.DataFrame | None = None,
) -> dict[int, SurrogateTreeModel]:
    reference_frame = dataset_frame if dataset_frame is not None else feature_frame
    explainability_frame = build_explainability_frame(
        reference_frame,
        feature_names=EXPLAINABILITY_FEATURE_NAMES,
    )
    encoder = build_explainability_encoder(
        explainability_frame,
        feature_names=EXPLAINABILITY_FEATURE_NAMES,
    )
    transformed = encoder.encode_frame(explainability_frame)
    sampled = sample_frame(
        transformed,
        max_rows=config.dashboard_models_config.surrogate_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    sampled_predictions = predictions.loc[sampled.index]
    X_train, X_test, y_train, y_test = train_test_split(
        sampled,
        sampled_predictions,
        test_size=0.2,
        random_state=config.dashboard_models_config.random_state,
    )
    tree_models: dict[int, SurrogateTreeModel] = {}
    for depth in config.dashboard_models_config.build_config.surrogate_depths:
        surrogate = DecisionTreeRegressor(
            max_depth=int(depth),
            min_samples_leaf=config.dashboard_models_config.surrogate_min_samples_leaf,
            random_state=config.dashboard_models_config.random_state,
        )
        surrogate.fit(X_train, y_train)
        surrogate_predictions = pd.Series(surrogate.predict(X_test), index=y_test.index)
        metrics = METRICS_REGISTRY.compute_metrics(
            y_test.reset_index(drop=True),
            surrogate_predictions.reset_index(drop=True),
            config.dashboard_models_config.error_metrics,
        )
        importances = pd.Series(
            surrogate.feature_importances_,
            index=EXPLAINABILITY_FEATURE_NAMES,
        ).sort_values(ascending=False)
        top_features = importances[importances > 0].head(3).index.tolist()
        tree_models[int(depth)] = SurrogateTreeModel(
            model=surrogate,
            feature_importances=importances,
            tree_depth=surrogate.get_depth(),
            n_leaves=surrogate.get_n_leaves(),
            text_rules=export_text(
                surrogate,
                feature_names=EXPLAINABILITY_FEATURE_NAMES,
            ),
            interpretation=(
                f"The surrogate approximates {runtime.family_name} on final-test "
                f"predictions with RMSE {metrics['rmse']:.4f} at max depth {int(depth)}. "
                f"The dominant decision logic is driven by "
                f"{', '.join(top_features) or 'no features'}."
            ),
            fidelity_frame=pd.DataFrame(
                {
                    "model_prediction": y_test.reset_index(drop=True),
                    "surrogate_prediction": surrogate_predictions.reset_index(drop=True),
                }
            ),
            feature_names=list(EXPLAINABILITY_FEATURE_NAMES),
            metrics={str(name): float(value) for name, value in metrics.items()},
        )
    return tree_models


def build_symbolic_regressor_model(
    *,
    runtime: TrainingModelRuntime,
    feature_frame: pd.DataFrame,
    predictions: pd.Series,
    dataset_frame: pd.DataFrame | None = None,
) -> SymbolicRegressorModel:
    from pysr import PySRRegressor

    reference_frame = dataset_frame if dataset_frame is not None else feature_frame
    explainability_frame = build_explainability_frame(
        reference_frame,
        feature_names=EXPLAINABILITY_FEATURE_NAMES,
    )
    encoder = build_explainability_encoder(
        explainability_frame,
        feature_names=EXPLAINABILITY_FEATURE_NAMES,
    )
    transformed = encoder.encode_frame(explainability_frame)
    sampled = sample_frame(
        transformed,
        max_rows=config.dashboard_models_config.symbolic_sample_size,
        random_state=config.dashboard_models_config.random_state + 5,
    )
    sampled_predictions = predictions.loc[sampled.index]
    selected_features = list(EXPLAINABILITY_FEATURE_NAMES)
    X_train, X_test, y_train, y_test = train_test_split(
        sampled.loc[:, selected_features],
        sampled_predictions,
        test_size=0.2,
        random_state=config.dashboard_models_config.random_state,
    )
    regressor = PySRRegressor(
        model_selection="best",
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["square", "cube"],
        constraints={"^": (-1, 1)},
        niterations=config.dashboard_models_config.symbolic_niterations,
        populations=config.dashboard_models_config.symbolic_populations,
        population_size=config.dashboard_models_config.symbolic_population_size,
        topn=config.dashboard_models_config.symbolic_topn,
        ncycles_per_iteration=(
            config.dashboard_models_config.symbolic_ncycles_per_iteration
        ),
        maxsize=config.dashboard_models_config.symbolic_maxsize,
        maxdepth=config.dashboard_models_config.symbolic_maxdepth,
        timeout_in_seconds=config.dashboard_models_config.symbolic_timeout_seconds,
        random_state=config.dashboard_models_config.random_state,
        deterministic=False,
        batching=True,
        batch_size=min(256, len(X_train)),
        precision=32,
        progress=False,
        verbosity=0,
        update=False,
        parallelism="multithreading",
    )
    regressor.fit(
        X_train.to_numpy(dtype="float32"),
        y_train.to_numpy(dtype="float32"),
        variable_names=selected_features,
    )
    symbolic_predictions = pd.Series(
        regressor.predict(X_test.to_numpy(dtype="float32")),
        index=y_test.index,
        name="symbolic_prediction",
    )
    metrics = METRICS_REGISTRY.compute_metrics(
        y_test.reset_index(drop=True),
        symbolic_predictions.reset_index(drop=True),
        config.dashboard_models_config.error_metrics,
    )
    candidate_equations = _normalize_symbolic_equation_table(
        regressor,
        min_equations=config.dashboard_models_config.symbolic_min_candidate_equations,
    )
    best_equation = regressor.get_best()
    sympy_expression = regressor.sympy()
    expression = sympy.sympify(str(sympy_expression))
    used_feature_names = [
        feature_name
        for feature_name in selected_features
        if sympy.Symbol(feature_name) in expression.free_symbols
    ]
    return SymbolicRegressorModel(
        equation=str(best_equation["equation"]),
        sympy_expression=str(sympy_expression),
        latex_expression=str(regressor.latex(precision=4)),
        interpretation=(
            f"Symbolic PySR surrogate fitted on final-test predictions. "
            f"It uses {', '.join(used_feature_names) or 'an intercept-like constant'} "
            f"and approximates {runtime.family_name} with RMSE {metrics['rmse']:.4f} "
            f"at complexity {int(best_equation['complexity'])}."
        ),
        feature_names=list(selected_features),
        used_feature_names=list(used_feature_names),
        complexity=int(best_equation["complexity"]),
        model_selection=str(regressor.model_selection),
        metrics={name: float(value) for name, value in metrics.items()},
        candidate_equations=candidate_equations,
        fidelity_frame=pd.DataFrame(
            {
                "model_prediction": y_test,
                "symbolic_prediction": symbolic_predictions,
                "residual": y_test - symbolic_predictions,
            }
        ).reset_index(drop=True),
    )


def build_neighbors_frame(
    *,
    runtime: TrainingModelRuntime,
    dataset_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    sample_indices: list[t.Any],
) -> pd.DataFrame:
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.neighbors_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    transformed_dataset = transform_feature_frame(
        runtime,
        feature_frame.loc[sampled_dataset.index],
    )
    transformed_samples = transform_feature_frame(
        runtime,
        feature_frame.loc[sample_indices],
    )
    scaler = StandardScaler()
    dataset_values = scaler.fit_transform(transformed_dataset.to_numpy())
    sample_values = scaler.transform(transformed_samples.to_numpy())
    estimator = NearestNeighbors(
        n_neighbors=min(
            config.dashboard_models_config.build_config.neighbors_k,
            len(sampled_dataset),
        )
    )
    estimator.fit(dataset_values)
    distances, indices = estimator.kneighbors(sample_values)
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
    runtime: TrainingModelRuntime,
    raw_frame: pd.DataFrame,
    anchor_indices: list[t.Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    surface_grid_size = config.dashboard_models_config.surface_grid_size
    for anchor_index in anchor_indices:
        anchor = raw_frame.loc[anchor_index].copy()
        base_underlying = float(anchor[column_name(VolatilityDBEnum.UNDERLYING_PRICE)])
        anchor_tte = float(anchor[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)])
        moneyness_values = np.linspace(0.8, 1.2, surface_grid_size)
        maturity_values = np.linspace(
            1.0,
            max(anchor_tte * 1.5, 30.0),
            surface_grid_size,
        )
        maturity_grid = np.repeat(maturity_values, surface_grid_size)
        moneyness_grid = np.tile(moneyness_values, surface_grid_size)
        surface_raw = pd.DataFrame(
            {
                column: np.repeat(anchor[column], len(maturity_grid))
                for column in raw_frame.columns
            }
        )
        surface_raw[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)] = maturity_grid
        surface_raw[column_name(VolatilityDBEnum.UNDERLYING_PRICE)] = base_underlying
        surface_raw[column_name(VolatilityDBEnum.STRIKE_PRICE)] = (
            base_underlying / moneyness_grid
        )
        surface = add_dashboard_derived_features(surface_raw)
        surface[TARGET_COLUMN] = np.nan
        surface["anchor_index"] = anchor_index
        surface["PredictedVolatility"] = predict_raw_frame(runtime, surface_raw)
        rows.append(surface)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_ice_frame(
    *,
    runtime: TrainingModelRuntime,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.ice_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    rows: list[dict[str, t.Any]] = []
    sampled_raw = raw_frame.loc[sampled_dataset.index].copy()
    for feature_name in feature_names:
        if feature_name not in sampled_dataset.columns:
            continue
        values = quantile_grid(
            sampled_dataset[feature_name],
            config.dashboard_models_config.curve_points,
        )
        for value in values:
            adjusted = apply_feature_override(sampled_raw, feature_name, value)
            predictions = predict_raw_frame(runtime, adjusted)
            for sample_id, prediction in enumerate(predictions):
                rows.append(
                    {
                        "feature_name": feature_name,
                        "sample_id": int(sample_id),
                        "feature_value": float(value),
                        "prediction": float(prediction),
                    }
                )
    return pd.DataFrame(rows)


def build_ale_frame(
    *,
    runtime: TrainingModelRuntime,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name not in dataset_frame.columns:
            continue
        series = dataset_frame[feature_name]
        edges = (
            series.dropna()
            .astype(float)
            .quantile(np.linspace(0.05, 0.95, 13))
            .drop_duplicates()
            .tolist()
        )
        if len(edges) < 2:
            continue
        increments: list[float] = []
        centers: list[float] = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            bucket_index = dataset_frame.loc[
                (series >= lower) & (series <= upper)
            ].index
            if len(bucket_index) == 0:
                continue
            bucket_raw = raw_frame.loc[bucket_index]
            lower_frame = apply_feature_override(bucket_raw, feature_name, lower)
            upper_frame = apply_feature_override(bucket_raw, feature_name, upper)
            delta = (
                pd.Series(predict_raw_frame(runtime, upper_frame), index=bucket_index)
                - pd.Series(predict_raw_frame(runtime, lower_frame), index=bucket_index)
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
    *,
    dataset_frame: pd.DataFrame,
    financial_warnings: list[str],
) -> DiagnosisArtifact:
    sampled = sample_frame(
        dataset_frame.dropna(subset=[TARGET_COLUMN]),
        max_rows=config.dashboard_models_config.diagnosis_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    metrics = METRICS_REGISTRY.compute_metrics(
        sampled[TARGET_COLUMN].astype(float).reset_index(drop=True),
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
            maturity_bin=pd.cut(sampled[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)], bins=12),
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


def _predict_transformed_values(
    runtime: TrainingModelRuntime,
    values: t.Any,
) -> np.ndarray:
    frame = (
        values.copy()
        if isinstance(values, pd.DataFrame)
        else pd.DataFrame(values, columns=runtime.model_input_features)
    )
    matrix = frame.to_numpy(dtype="float32", copy=False)
    if runtime.is_keras:
        predictions = runtime.model.predict(matrix, verbose=0)
    else:
        predictions = runtime.model.predict(matrix)
    return np.asarray(predictions, dtype="float64").reshape(-1)


def _predict_explainability_values(
    runtime: TrainingModelRuntime,
    encoder,
    values: t.Any,
) -> np.ndarray:
    raw_frame = encoder.reconstruct_raw_frame(values)
    return predict_raw_frame(runtime, raw_frame)


def _max_evals(n_features: int) -> int:
    return 2 * n_features + 1


def _normalize_symbolic_equation_table(
    regressor,
    *,
    min_equations: int,
) -> pd.DataFrame:
    equations = regressor.equations_
    frame = equations[0].copy() if isinstance(equations, list) else equations.copy()
    keep_columns = [
        column_name
        for column_name in ("complexity", "loss", "score", "equation")
        if column_name in frame.columns
    ]
    normalized = frame.loc[:, keep_columns].copy().reset_index(drop=True)
    normalized["complexity"] = normalized["complexity"].astype("int64")
    normalized["loss"] = normalized["loss"].astype("float64")
    if "score" in normalized.columns:
        normalized["score"] = normalized["score"].astype("float64")
    normalized["equation"] = normalized["equation"].astype(str)
    normalized = normalized.drop_duplicates(subset=["equation"]).reset_index(drop=True)
    selected_equation = str(regressor.get_best()["equation"])
    selected_complexity = int(regressor.get_best()["complexity"])

    if normalized.empty:
        normalized["selected"] = pd.Series(dtype="bool")
        return normalized

    primary = (
        normalized.sort_values(["complexity", "loss"], ascending=[True, True])
        .drop_duplicates(subset=["complexity"], keep="first")
        .reset_index(drop=True)
    )
    if len(primary) < min_equations:
        remaining = normalized.loc[
            ~normalized["equation"].isin(primary["equation"])
        ].sort_values(["loss", "complexity"], ascending=[True, True])
        needed = min_equations - len(primary)
        primary = pd.concat(
            [primary, remaining.head(max(0, needed))],
            ignore_index=True,
        )

    selected_row = normalized.loc[
        normalized["equation"] == selected_equation
    ].head(1)
    if selected_row.empty:
        selected_row = normalized.loc[
            normalized["complexity"] == selected_complexity
        ].head(1)
    if not selected_row.empty and selected_row.iloc[0]["equation"] not in set(primary["equation"]):
        primary = pd.concat([selected_row, primary], ignore_index=True)

    primary = (
        primary.drop_duplicates(subset=["equation"])
        .sort_values(["loss", "complexity"], ascending=[True, True])
        .reset_index(drop=True)
    )
    primary["selected"] = primary["equation"] == selected_equation
    return primary
