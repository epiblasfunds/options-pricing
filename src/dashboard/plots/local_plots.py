"""Local sample explainability figures."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.model2dashboard.features import VISIBLE_RAW_INPUT_FEATURE_NAMES
from src.dashboard.plots.plot_style import (
    HOVERLABEL_STYLE,
    STANDARD_MARGIN,
    STANDARD_TEMPLATE,
)


def neighbors_projection_figure(
    sample_frame,
    reference_frame,
    *,
    feature_names: list[str] | None = None,
    center_label: str = "Analyzed Sample",
    dimensions: int = 2,
):
    frame = reference_frame.copy()
    sample = sample_frame.head(1).copy()
    title = _projection_title(dimensions)
    if frame.empty or sample.empty:
        return _empty_projection_figure(dimensions=dimensions, title=title)

    frame["row_id"] = _neighbor_row_id_series(frame)
    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("distance", kind="stable").reset_index(drop=True)
    sample["row_id"] = "sample"
    sample["distance"] = 0.0

    projected = _project_neighbourhood_feature_space(
        sample_frame=sample,
        neighbors_frame=frame,
        feature_names=feature_names,
        dimensions=dimensions,
    )
    return _projection_figure_from_frame(
        projected,
        dimensions=dimensions,
        center_label=center_label,
        title=title,
    )


def neighbors_projection_3d_figure(
    sample_frame,
    reference_frame,
    *,
    feature_names: list[str] | None = None,
    center_label: str = "Analyzed Sample",
):
    return neighbors_projection_figure(
        sample_frame,
        reference_frame,
        feature_names=feature_names,
        center_label=center_label,
        dimensions=3,
    )


def neighbors_distance_figure(neighbors_frame, *, center_label: str = "Analyzed Sample"):
    """Backward-compatible wrapper around the feature-space projection figure."""

    if neighbors_frame.empty:
        return neighbors_projection_figure(
            pd.DataFrame(),
            neighbors_frame,
            center_label=center_label,
        )
    sample_stub = pd.DataFrame([{column: 0.0 for column in neighbors_frame.columns}])
    return neighbors_projection_figure(
        sample_stub,
        neighbors_frame,
        center_label=center_label,
    )


def _empty_projection_figure(*, dimensions: int, title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=_projection_margin(),
        title=title,
        hoverlabel=HOVERLABEL_STYLE,
        legend=_point_type_legend(),
    )
    if dimensions == 2:
        fig.update_layout(
            xaxis_title="Projection Component 1",
            yaxis_title="Projection Component 2",
        )
    else:
        fig.update_layout(
            scene={
                "xaxis_title": "Projection Component 1",
                "yaxis_title": "Projection Component 2",
                "zaxis_title": "Projection Component 3",
            }
        )
    return fig


def _projection_figure_from_frame(
    projected: pd.DataFrame,
    *,
    dimensions: int,
    center_label: str,
    title: str,
) -> go.Figure:
    sample_projected = projected.loc[projected["point_type"] == "sample"].copy()
    neighbors_projected = projected.loc[projected["point_type"] == "neighbor"].copy()
    color_column = _point_color_column(neighbors_projected)
    color_range = _point_color_range(neighbors_projected)
    hover_columns = _neighbor_hover_columns(neighbors_projected, color_column)

    fig = go.Figure()
    if dimensions == 2:
        fig.add_trace(
            go.Scatter(
                x=sample_projected["x"],
                y=sample_projected["y"],
                mode="markers+text",
                name=center_label,
                text=["Sample"],
                textposition="top center",
                marker=_sample_marker(),
                hovertemplate=(
                    f"{center_label}: sample<br>"
                    "Relative distance: %{customdata:.4f}<extra></extra>"
                ),
                customdata=sample_projected["distance"],
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=neighbors_projected["x"],
                y=neighbors_projected["y"],
                mode="markers",
                name="Neighbours",
                marker=_neighbor_marker(
                    neighbors_projected,
                    color_range=color_range,
                ),
                customdata=neighbors_projected.loc[:, hover_columns].to_numpy(
                    dtype=object
                ),
                hovertemplate=_neighbor_hovertemplate(
                    neighbors_projected,
                    color_column=color_column,
                    hover_columns=hover_columns,
                ),
            )
        )
        fig.update_layout(
            xaxis_title="Projection Component 1",
            yaxis_title="Projection Component 2",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.add_trace(
            go.Scatter3d(
                x=sample_projected["x"],
                y=sample_projected["y"],
                z=sample_projected["z"],
                mode="markers+text",
                name=center_label,
                text=["Sample"],
                textposition="top center",
                marker=_sample_marker(size=7),
                hovertemplate=(
                    f"{center_label}: sample<br>"
                    "Relative distance: %{customdata:.4f}<extra></extra>"
                ),
                customdata=sample_projected["distance"],
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=neighbors_projected["x"],
                y=neighbors_projected["y"],
                z=neighbors_projected["z"],
                mode="markers",
                name="Neighbours",
                marker=_neighbor_marker(
                    neighbors_projected,
                    color_range=color_range,
                    size=5,
                ),
                customdata=neighbors_projected.loc[:, hover_columns].to_numpy(
                    dtype=object
                ),
                hovertemplate=_neighbor_hovertemplate(
                    neighbors_projected,
                    color_column=color_column,
                    hover_columns=hover_columns,
                ),
            )
        )
        fig.update_layout(
            scene={
                "xaxis_title": "Projection Component 1",
                "yaxis_title": "Projection Component 2",
                "zaxis_title": "Projection Component 3",
                "aspectmode": "data",
            }
        )

    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=_projection_margin(),
        title=title,
        hoverlabel=HOVERLABEL_STYLE,
        legend=_point_type_legend(),
        legend_title_text="Point Type",
    )
    return fig


def _sample_marker(*, size: int = 16) -> dict[str, object]:
    return {
        "size": size,
        "color": "#2e8b57",
        "symbol": "diamond",
        "line": {"width": 0},
    }


def _neighbor_marker(
    frame: pd.DataFrame,
    *,
    color_range: tuple[float, float],
    size: int = 8,
) -> dict[str, object]:
    return {
        "size": size,
        "opacity": 0.8,
        "color": _point_color_values(frame),
        "colorscale": _blue_volatility_colorscale(),
        "showscale": True,
        "colorbar": _point_colorbar(frame),
        "cmin": color_range[0],
        "cmax": color_range[1],
        "line": {"width": 0},
    }


def _point_colorbar(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "title": {"text": _point_color_title(frame), "side": "top"},
        "orientation": "h",
        "x": 0.48,
        "xanchor": "center",
        "y": 1.12,
        "yanchor": "bottom",
        "len": 0.62,
        "thickness": 14,
    }


def _point_type_legend() -> dict[str, object]:
    return {
        "x": 1.02,
        "xanchor": "left",
        "y": 1.0,
        "yanchor": "top",
        "bgcolor": "rgba(255,255,255,0.85)",
    }


def _projection_margin() -> dict[str, int]:
    return {
        **STANDARD_MARGIN,
        "t": 100,
        "r": 160,
    }


def _projection_title(dimensions: int) -> str:
    return f"Neighbourhood Distance Map ({dimensions}D)"


def _neighbor_row_id_series(frame: pd.DataFrame) -> pd.Series:
    if "row_id" in frame.columns:
        return frame["row_id"].astype(str)
    row_source = frame["index"] if "index" in frame.columns else frame.index
    return pd.Series(row_source, index=frame.index).astype(str)


def _project_neighbourhood_feature_space(
    *,
    sample_frame: pd.DataFrame,
    neighbors_frame: pd.DataFrame,
    feature_names: list[str] | None,
    dimensions: int,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            sample_frame.assign(point_type="sample"),
            neighbors_frame.assign(point_type="neighbor"),
        ],
        axis=0,
        ignore_index=True,
    )
    dimensions = max(2, int(dimensions))
    projection_features = (
        list(feature_names)
        if feature_names is not None
        else _default_projection_features(sample_frame, neighbors_frame)
    )
    if not projection_features:
        coords = np.zeros((len(combined), dimensions), dtype="float64")
        for index, distance in enumerate(neighbors_frame["distance"].tolist(), start=1):
            angle = 2.0 * math.pi * (index - 1) / max(1, len(neighbors_frame))
            coords[index, 0] = float(distance) * math.cos(angle)
            coords[index, 1] = float(distance) * math.sin(angle)
            if dimensions > 2:
                coords[index, 2] = float(distance)
        return _projected_frame_from_coords(combined, coords, dimensions=dimensions)

    matrix = combined.loc[:, projection_features].apply(pd.to_numeric, errors="coerce")
    fill_values = matrix.mean().fillna(0.0)
    standardized = matrix.fillna(fill_values)
    std = standardized.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = (standardized - fill_values) / std
    centered = standardized - standardized.mean(axis=0)

    if centered.shape[0] < 2 or centered.shape[1] == 0:
        coords = np.zeros((len(combined), dimensions), dtype="float64")
    elif centered.shape[1] == 1:
        coords = centered.iloc[:, [0]].to_numpy(dtype="float64")
    else:
        _, _, vh = np.linalg.svd(centered.to_numpy(dtype="float64"), full_matrices=False)
        component_count = min(dimensions, vh.shape[0])
        components = vh[:component_count].T
        coords = centered.to_numpy(dtype="float64") @ components

    if coords.shape[1] < dimensions:
        coords = np.column_stack(
            [
                coords,
                np.zeros((len(combined), dimensions - coords.shape[1]), dtype="float64"),
            ]
        )

    return _projected_frame_from_coords(combined, coords, dimensions=dimensions)


def _projected_frame_from_coords(
    combined: pd.DataFrame,
    coords: np.ndarray,
    *,
    dimensions: int,
) -> pd.DataFrame:
    sample_coord = coords[0].copy()
    coords = coords - sample_coord
    projected = combined.copy()
    projected["x"] = coords[:, 0]
    projected["y"] = coords[:, 1]
    if dimensions > 2:
        projected["z"] = coords[:, 2]
    return projected


def _default_projection_features(
    sample_frame: pd.DataFrame,
    neighbors_frame: pd.DataFrame,
) -> list[str]:
    excluded = {
        "PredictedVolatility",
        "ImpliedVolatility",
        "Residual",
        "AbsoluteError",
        "distance",
    }
    preferred_prefix = (
        "TTEYears",
        "sqrtTTEYears",
        "logMoneyness",
        "logMoneynessSq",
        "logMoneynessXSqrtTTE",
        "logForwardMoneyness",
        "rate",
        "isCall",
        "isPut",
        "Moneyness",
        "LogMoneyness",
        "AbsLogMoneyness",
        "StrikePrice",
        "UnderlyingPrice",
        "TimeToExpiration",
        "Rate",
    )
    common = [
        column
        for column in sample_frame.columns
        if column in neighbors_frame.columns and column not in excluded
    ]
    numeric_features = []
    for column in common:
        if column not in preferred_prefix:
            continue
        sample_numeric = pd.to_numeric(sample_frame[column], errors="coerce")
        neighbor_numeric = pd.to_numeric(neighbors_frame[column], errors="coerce")
        combined = pd.concat([sample_numeric, neighbor_numeric], axis=0)
        if combined.notna().sum() == 0:
            continue
        numeric_features.append(column)
    return numeric_features


def _point_color_column(frame: pd.DataFrame) -> str:
    if "ImpliedVolatility" in frame.columns:
        return "ImpliedVolatility"
    if "PredictedVolatility" in frame.columns:
        return "PredictedVolatility"
    return "distance"


def _point_color_values(frame: pd.DataFrame) -> pd.Series:
    column = _point_color_column(frame)
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().any():
        return values.fillna(values.median())
    return pd.Series(np.zeros(len(frame), dtype="float64"), index=frame.index)


def _point_color_range(frame: pd.DataFrame) -> tuple[float, float]:
    values = _point_color_values(frame)
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum, rel_tol=1e-9, abs_tol=1e-9):
        maximum = minimum + 1e-6
    return minimum, maximum


def _point_color_title(frame: pd.DataFrame) -> str:
    column = _point_color_column(frame)
    if column == "ImpliedVolatility":
        return "Implied Volatility"
    if column == "PredictedVolatility":
        return "Predicted Volatility"
    return "Distance"


def _neighbor_hover_columns(frame: pd.DataFrame, color_column: str) -> list[str]:
    columns = ["row_id", "distance", color_column]
    for column in VISIBLE_RAW_INPUT_FEATURE_NAMES:
        if column in frame.columns and column not in columns:
            columns.append(column)
    return columns


def _neighbor_hovertemplate(
    frame: pd.DataFrame,
    *,
    color_column: str,
    hover_columns: list[str],
) -> str:
    lines = [
        "ID: %{customdata[0]}",
        "Relative distance: %{customdata[1]:.4f}",
        f"{_point_color_title(frame)}: %{{customdata[2]:.4f}}",
    ]
    for index, column in enumerate(hover_columns[3:], start=3):
        lines.append(_hover_line_for_column(frame, column, index))
    return "<br>".join(lines) + "<extra></extra>"


def _hover_line_for_column(frame: pd.DataFrame, column: str, index: int) -> str:
    label = _hover_label(column)
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        return f"{label}: %{{customdata[{index}]:.4f}}"
    return f"{label}: %{{customdata[{index}]}}"


def _hover_label(column: str) -> str:
    return {
        "OptionType": "Option Type",
        "StrikePrice": "Strike",
        "UnderlyingPrice": "Underlying",
        "TimeToExpiration": "TTE",
        "Rate": "Rate",
        "OptionContractCode": "Contract",
    }.get(column, column)


def _blue_volatility_colorscale() -> list[list[object]]:
    return [
        [0.0, "#d9ecff"],
        [0.25, "#a9d0f5"],
        [0.5, "#7aa5d2"],
        [0.75, "#406f9f"],
        [1.0, "#17304f"],
    ]
