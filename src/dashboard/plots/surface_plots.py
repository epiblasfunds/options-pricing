"""Plotly figures for surface analysis."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.plots.plot_style import (
    HOVERLABEL_STYLE,
    STANDARD_MARGIN,
    STANDARD_TEMPLATE,
    VOLATILITY_COLORSCALE,
    safe_color_range,
    sample_scale_color,
    truncated_colorscale,
)

HEATMAP_DESCRIPTION = (
    "This heatmap represents the model-implied local volatility surface around the "
    "selected anchor observation. Each cell corresponds to one predicted volatility "
    "for a specific combination of moneyness and time to expiration. The color scale "
    "is shared with the 3D surface and the slice charts, so identical colors always "
    "indicate identical predicted-volatility levels across views."
)
SMILE_DESCRIPTION = (
    "This chart shows volatility-smile cross-sections extracted from the same local "
    "surface. Each curve keeps maturity fixed and varies moneyness, allowing you to "
    "compare curvature, skew, and level changes across expiries. Marker colors use "
    "the same volatility scale as the heatmap and 3D surface."
)
TERM_DESCRIPTION = (
    "This chart shows term-structure cross-sections extracted from the same local "
    "surface. Each curve keeps moneyness fixed and tracks how predicted volatility "
    "evolves with time to expiration. Marker colors use the same shared volatility "
    "scale as the rest of the surface views."
)
ICE_DESCRIPTION = (
    "This figure displays Individual Conditional Expectation curves. Each line tracks "
    "how the model prediction changes when the selected feature is perturbed for one "
    "sample while the remaining inputs are held fixed, which exposes heterogeneity in "
    "local sensitivities."
)
ALE_DESCRIPTION = (
    "This figure displays the Accumulated Local Effects profile for the selected "
    "feature. It summarizes the average local effect of moving the feature across its "
    "observed range while respecting the empirical data distribution, which makes it "
    "more robust than naive ceteris-paribus perturbations under feature dependence."
)
LOCAL_SURFACE_DESCRIPTION = (
    "This 3D surface is the same local volatility surface shown in the heatmap, now "
    "rendered as a continuous geometry. It helps assess slope, curvature, and local "
    "smoothness across moneyness and maturity while preserving the exact same shared "
    "volatility color range used in the other surface visualizations."
)
CURVE_LINE_COLOR = "#355070"
SMILE_CURVE_COLORSCALE = truncated_colorscale("PuBuGn", start=0.24, end=1.0)
TERM_CURVE_COLORSCALE = "Sunset"


def _surface_color_range(surface_frame, volatility_range=None):
    if volatility_range is not None:
        return volatility_range
    return safe_color_range(surface_frame["PredictedVolatility"].tolist())


def _legend_label(frame, column_name: str, precision: int) -> str:
    value = float(frame[column_name].iloc[0])
    return f"{value:.{precision}f}"


def _axis_padding(values, fallback: float = 0.05) -> float:
    if not values:
        return fallback
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if abs(span) < 1e-9:
        return max(abs(maximum) * 0.08, fallback)
    return span * 0.08


def _spread_positions(values):
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    gap = max((upper - lower) * 0.07, 0.004)
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    previous = None
    for index, value in indexed:
        adjusted_value = value if previous is None else max(value, previous + gap)
        adjusted[index] = adjusted_value
        previous = adjusted_value
    upper_limit = upper + gap
    overflow = max(adjusted) - upper_limit
    if overflow > 0:
        adjusted = [value - overflow for value in adjusted]
    lower_limit = lower - gap
    underflow = lower_limit - min(adjusted)
    if underflow > 0:
        adjusted = [value + underflow for value in adjusted]
    return adjusted


def _apply_direct_curve_labels(fig, curve_endpoints, xaxis_range):
    if not curve_endpoints:
        return
    x_min, x_max = xaxis_range
    x_pad = _axis_padding([x_min, x_max])
    label_y_positions = _spread_positions([point["y"] for point in curve_endpoints])
    annotations = []
    for point, label_y in zip(curve_endpoints, label_y_positions):
        annotations.append(
            dict(
                x=x_max + x_pad * 1.15,
                y=label_y,
                xref="x",
                yref="y",
                text=point["label"],
                showarrow=True,
                ax=point["x"],
                ay=point["y"],
                axref="x",
                ayref="y",
                arrowhead=0,
                arrowwidth=1,
                arrowcolor="#8da1b5",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#c8d6e5",
                borderwidth=1,
                borderpad=3,
                font=dict(size=11, color="#17304f"),
                align="left",
            )
        )
    fig.update_xaxes(range=[x_min, x_max + x_pad * 2.8])
    fig.update_layout(annotations=annotations)


def _add_curve_key(fig, text: str):
    existing = list(fig.layout.annotations) if fig.layout.annotations else []
    existing.append(
        dict(
            x=0.01,
            y=1.08,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            font=dict(size=11, color="#4a5a73"),
            align="left",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#d7e2ef",
            borderwidth=1,
            borderpad=3,
        )
    )
    fig.update_layout(annotations=existing)


def heatmap_figure(surface_frame, volatility_range=None):
    pivot = surface_frame.pivot_table(
        index="TimeToExpiration",
        columns="Moneyness",
        values="PredictedVolatility",
    ).sort_index().sort_index(axis=1)
    color_range = _surface_color_range(surface_frame, volatility_range)
    fig = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        color_continuous_scale=VOLATILITY_COLORSCALE,
        range_color=color_range,
        title="Predicted Volatility Surface Heatmap",
    )
    fig.update_traces(
        hovertemplate=(
            "Moneyness: %{x:.3f}<br>"
            "Time To Expiration: %{y:.2f}<br>"
            "Predicted volatility: %{z:.4f}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        xaxis_title="Moneyness",
        yaxis_title="Time To Expiration",
        margin=STANDARD_MARGIN,
        coloraxis_colorbar=dict(title="Predicted volatility"),
        hoverlabel=HOVERLABEL_STYLE,
    )
    fig.update_xaxes(side="bottom")
    return fig


def smile_figure(surface_frame, volatility_range=None):
    time_range = safe_color_range(surface_frame["TimeToExpiration"].tolist())
    fig = go.Figure()
    ordered_frame = surface_frame.sort_values(["TimeToExpiration", "Moneyness"])
    curve_endpoints = []
    for index, (_, slice_frame) in enumerate(
        ordered_frame.groupby("TimeToExpiration", sort=True)
    ):
        time_value = float(slice_frame["TimeToExpiration"].iloc[0])
        label = _legend_label(slice_frame, "TimeToExpiration", precision=1)
        curve_color = sample_scale_color(
            time_value,
            time_range,
            colorscale=SMILE_CURVE_COLORSCALE,
        )
        fig.add_trace(
            go.Scatter(
                x=slice_frame["Moneyness"],
                y=slice_frame["PredictedVolatility"],
                mode="lines+markers",
                name=label,
                line=dict(
                    color=curve_color,
                    width=2.4,
                ),
                marker=dict(
                    size=8,
                    symbol="circle",
                    color=[time_value] * len(slice_frame),
                    colorscale=SMILE_CURVE_COLORSCALE,
                    cmin=time_range[0],
                    cmax=time_range[1],
                    showscale=index == 0,
                    colorbar=(
                        dict(
                            title="Time To Expiration",
                            orientation="h",
                            x=0.5,
                            xanchor="center",
                            y=-0.32,
                            len=0.62,
                            thickness=14,
                        )
                        if index == 0
                        else None
                    ),
                    line=dict(color="#ffffff", width=0.6),
                ),
                customdata=slice_frame[["TimeToExpiration"]].to_numpy(),
                hovertemplate=(
                    "Moneyness: %{x:.3f}<br>"
                    "Predicted volatility: %{y:.4f}<br>"
                    "Time To Expiration: %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )
        curve_endpoints.append(
            {
                "x": float(slice_frame["Moneyness"].iloc[-1]),
                "y": float(slice_frame["PredictedVolatility"].iloc[-1]),
                "label": f"T: {label}",
            }
        )
    x_values = ordered_frame["Moneyness"].astype(float).tolist()
    _apply_direct_curve_labels(
        fig,
        curve_endpoints,
        (min(x_values), max(x_values)),
    )
    _add_curve_key(fig, "T = Time To Expiration")
    curve_margin = dict(STANDARD_MARGIN)
    curve_margin["b"] = 104
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        xaxis_title="Moneyness",
        yaxis_title="Predicted volatility",
        showlegend=False,
        margin=curve_margin,
        hoverlabel=HOVERLABEL_STYLE,
        title="Smile Slices",
    )
    return fig


def term_figure(surface_frame, volatility_range=None):
    moneyness_range = safe_color_range(surface_frame["Moneyness"].tolist())
    fig = go.Figure()
    ordered_frame = surface_frame.sort_values(["Moneyness", "TimeToExpiration"])
    curve_endpoints = []
    for index, (_, slice_frame) in enumerate(
        ordered_frame.groupby("Moneyness", sort=True)
    ):
        moneyness_value = float(slice_frame["Moneyness"].iloc[0])
        label = _legend_label(slice_frame, "Moneyness", precision=3)
        curve_color = sample_scale_color(
            moneyness_value,
            moneyness_range,
            colorscale=TERM_CURVE_COLORSCALE,
        )
        fig.add_trace(
            go.Scatter(
                x=slice_frame["TimeToExpiration"],
                y=slice_frame["PredictedVolatility"],
                mode="lines+markers",
                name=label,
                line=dict(
                    color=curve_color,
                    width=2.4,
                ),
                marker=dict(
                    size=8,
                    symbol="circle",
                    color=[moneyness_value] * len(slice_frame),
                    colorscale=TERM_CURVE_COLORSCALE,
                    cmin=moneyness_range[0],
                    cmax=moneyness_range[1],
                    showscale=index == 0,
                    colorbar=(
                        dict(
                            title="Moneyness",
                            orientation="h",
                            x=0.5,
                            xanchor="center",
                            y=-0.32,
                            len=0.62,
                            thickness=14,
                        )
                        if index == 0
                        else None
                    ),
                    line=dict(color="#ffffff", width=0.6),
                ),
                customdata=slice_frame[["Moneyness"]].to_numpy(),
                hovertemplate=(
                    "Time To Expiration: %{x:.2f}<br>"
                    "Predicted volatility: %{y:.4f}<br>"
                    "Moneyness: %{customdata[0]:.3f}<extra></extra>"
                ),
            )
        )
        curve_endpoints.append(
            {
                "x": float(slice_frame["TimeToExpiration"].iloc[-1]),
                "y": float(slice_frame["PredictedVolatility"].iloc[-1]),
                "label": f"M: {label}",
            }
        )
    x_values = ordered_frame["TimeToExpiration"].astype(float).tolist()
    _apply_direct_curve_labels(
        fig,
        curve_endpoints,
        (min(x_values), max(x_values)),
    )
    _add_curve_key(fig, "M = Moneyness")
    curve_margin = dict(STANDARD_MARGIN)
    curve_margin["b"] = 104
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        xaxis_title="Time To Expiration",
        yaxis_title="Predicted volatility",
        showlegend=False,
        margin=curve_margin,
        hoverlabel=HOVERLABEL_STYLE,
        title="Term Structure Slices",
    )
    return fig


def ice_figure(ice_frame, feature_label: str):
    fig = px.line(
        ice_frame,
        x="feature_value",
        y="prediction",
        color="sample_id",
        title=f"ICE Curves: {feature_label}",
    )
    fig.update_traces(
        hovertemplate=(
            f"{feature_label}: %{{x:.4f}}<br>"
            "Predicted volatility: %{y:.4f}<br>"
            "Sample: %{fullData.name}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        xaxis_title=feature_label,
        yaxis_title="Predicted volatility",
        legend_title_text="Sample",
        margin=STANDARD_MARGIN,
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def ale_figure(ale_frame, feature_label: str):
    fig = px.line(
        ale_frame,
        x="feature_value",
        y="ale",
        markers=True,
        title=f"ALE Plot: {feature_label}",
    )
    fig.update_traces(
        hovertemplate=(
            f"{feature_label}: %{{x:.4f}}<br>"
            "Accumulated local effect: %{y:.4f}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        xaxis_title=feature_label,
        yaxis_title="Accumulated local effect",
        margin=STANDARD_MARGIN,
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def local_surface_figure(surface_frame, volatility_range=None):
    pivot = surface_frame.pivot_table(
        index="TimeToExpiration",
        columns="Moneyness",
        values="PredictedVolatility",
    ).sort_index().sort_index(axis=1)
    color_range = _surface_color_range(surface_frame, volatility_range)
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns,
                y=pivot.index,
                z=pivot.values,
                colorscale=VOLATILITY_COLORSCALE,
                cmin=color_range[0],
                cmax=color_range[1],
                colorbar=dict(title="Predicted volatility"),
                hovertemplate=(
                    "Moneyness: %{x:.3f}<br>"
                    "Time To Expiration: %{y:.2f}<br>"
                    "Predicted volatility: %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Local Volatility Surface",
        template=STANDARD_TEMPLATE,
        margin=dict(l=0, r=0, t=48, b=0),
        hoverlabel=HOVERLABEL_STYLE,
        scene=dict(
            xaxis=dict(
                title="Moneyness",
                range=[float(pivot.columns.min()), float(pivot.columns.max())],
            ),
            yaxis=dict(
                title="Time To Expiration",
                range=[float(pivot.index.min()), float(pivot.index.max())],
            ),
            zaxis_title="Predicted volatility",
        ),
    )
    return fig

