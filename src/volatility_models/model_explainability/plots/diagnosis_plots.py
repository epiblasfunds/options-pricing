"""Plotly figures for model diagnostics."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from src.volatility_models.model_explainability.plots.plot_style import (
    ERROR_COLORSCALE,
    HOVERLABEL_STYLE,
    STANDARD_MARGIN,
    STANDARD_TEMPLATE,
    safe_color_range,
)

def real_vs_predicted_figure(frame):
    fig = px.scatter(
        frame,
        x="ImpliedVolatility",
        y="PredictedVolatility",
        color="OptionType" if "OptionType" in frame.columns else None,
        title="Actual vs Predicted Implied Volatility",
        opacity=0.6,
        render_mode="webgl",
    )
    minimum = min(frame["ImpliedVolatility"].min(), frame["PredictedVolatility"].min())
    maximum = max(frame["ImpliedVolatility"].max(), frame["PredictedVolatility"].max())
    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Diagonal",
            line=dict(color="#333333", dash="dash"),
        )
    )
    fig.update_traces(
        hovertemplate=(
            "Observed implied volatility: %{x:.4f}<br>"
            "Predicted volatility: %{y:.4f}<br>"
            "Series: %{fullData.name}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        legend_title_text="Option Type",
        margin=STANDARD_MARGIN,
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def residual_by_feature_figure(frame, feature_name: str, title: str):
    fig = px.scatter(
        frame,
        x=feature_name,
        y="Residual",
        color="OptionType" if "OptionType" in frame.columns else None,
        title=title,
        opacity=0.6,
        render_mode="webgl",
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#4f5d75")
    fig.update_traces(
        hovertemplate=(
            f"{feature_name}: %{{x}}<br>"
            "Residual: %{y:.4f}<br>"
            "Series: %{fullData.name}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        yaxis_title="Residual",
        legend_title_text="Option Type",
        margin=STANDARD_MARGIN,
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def error_heatmap_figure(error_heatmap):
    pivot = error_heatmap.pivot_table(
        index="maturity_bin",
        columns="moneyness_bin",
        values="AbsoluteError",
    ).sort_index().sort_index(axis=1)
    fig = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        color_continuous_scale=ERROR_COLORSCALE,
        range_color=safe_color_range(error_heatmap["AbsoluteError"].tolist()),
        title="Average Absolute Error Heatmap",
    )
    fig.update_traces(
        hovertemplate=(
            "Maturity bin: %{y}<br>"
            "Moneyness bin: %{x}<br>"
            "Average absolute error: %{z:.4f}<extra></extra>"
        )
    )
    return fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Moneyness Bin",
        yaxis_title="Maturity Bin",
        coloraxis_colorbar=dict(title="Absolute error"),
        hoverlabel=HOVERLABEL_STYLE,
    )
