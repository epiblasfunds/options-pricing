import typing as t

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config.config import config
from src.enums.data_enums import VolatilityDBEnum
from src.volatility_models import (
    TARGET_COLUMN,
    add_dashboard_derived_features,
    apply_feature_override,
    build_feature_frame_from_trades,
    build_model_dataset,
)


def build_neighbors_frame(
    *,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    trained_model: t.Any,
    model_input_features: list[str],
    sample_indices: list[t.Any],
    sample_frame: t.Callable[..., pd.DataFrame],
    transform_feature_frame: t.Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    sampled_dataset = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.neighbors_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    dataset_features = build_feature_frame_from_trades(
        raw_frame.loc[sampled_dataset.index]
    )
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
    estimator = NearestNeighbors(
        n_neighbors=min(
            config.dashboard_models_config.build_config.neighbors_k,
            len(sampled_dataset),
        )
    )
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
    trained_model: t.Any,
    raw_frame: pd.DataFrame,
    anchor_indices: list[t.Any],
    predict_raw_frame: t.Callable[[t.Any, pd.DataFrame], np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    surface_grid_size = config.dashboard_models_config.surface_grid_size
    for anchor_index in anchor_indices:
        anchor = raw_frame.loc[[anchor_index]].copy()
        base_underlying = float(anchor[str(VolatilityDBEnum.UNDERLYING_PRICE)].iloc[0])
        anchor_tte = float(anchor[str(VolatilityDBEnum.TIME_TO_EXPIRATION)].iloc[0])
        moneyness_values = np.linspace(0.8, 1.2, surface_grid_size)
        maturity_values = np.linspace(
            1.0, max(anchor_tte * 1.5, 30.0), surface_grid_size
        )
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
    trained_model: t.Any,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
    sample_frame: t.Callable[..., pd.DataFrame],
    quantile_grid: t.Callable[[pd.Series, int], list[float]],
    predict_raw_frame: t.Callable[[t.Any, pd.DataFrame], np.ndarray],
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
    trained_model: t.Any,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    feature_names: list[str],
    predict_raw_frame: t.Callable[[t.Any, pd.DataFrame], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, t.Any]] = []
    for feature_name in feature_names:
        if feature_name not in dataset_frame.columns:
            continue
        series = dataset_frame[feature_name]
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
            bucket_index = dataset_frame.loc[
                (series >= lower) & (series <= upper)
            ].index
            if len(bucket_index) == 0:
                continue
            bucket_raw = raw_frame.loc[bucket_index]
            lower_frame = apply_feature_override(bucket_raw, feature_name, lower)
            upper_frame = apply_feature_override(bucket_raw, feature_name, upper)
            delta = (
                pd.Series(
                    predict_raw_frame(trained_model, upper_frame), index=bucket_index
                )
                - pd.Series(
                    predict_raw_frame(trained_model, lower_frame), index=bucket_index
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
