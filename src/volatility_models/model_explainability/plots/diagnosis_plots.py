"""Plotly figures for model diagnostics."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


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
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Option Type",
        margin=dict(l=30, r=20, t=48, b=30),
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
    fig.update_layout(
        template="plotly_white",
        yaxis_title="Residual",
        legend_title_text="Option Type",
        margin=dict(l=30, r=20, t=48, b=30),
    )
    return fig


def error_heatmap_figure(error_heatmap):
    pivot = error_heatmap.pivot_table(
        index="maturity_bin",
        columns="moneyness_bin",
        values="AbsoluteError",
    )
    return px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="OrRd",
        title="Average Absolute Error Heatmap",
    ).update_layout(
        template="plotly_white",
        margin=dict(l=30, r=20, t=48, b=30),
        xaxis_title="Moneyness Bin",
        yaxis_title="Maturity Bin",
    )
