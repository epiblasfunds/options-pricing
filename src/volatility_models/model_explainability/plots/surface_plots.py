"""Plotly figures for surface analysis."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


def heatmap_figure(surface_frame):
    pivot = surface_frame.pivot_table(
        index="TimeToExpiration",
        columns="Moneyness",
        values="PredictedVolatility",
    ).sort_index()
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Predicted Volatility Surface Heatmap",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Moneyness",
        yaxis_title="Time To Expiration",
        margin=dict(l=30, r=20, t=48, b=30),
    )
    return fig


def smile_figure(surface_frame):
    fig = px.line(
        surface_frame,
        x="Moneyness",
        y="PredictedVolatility",
        color="TimeToExpiration",
        title="Smile Slices",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Moneyness",
        yaxis_title="Predicted volatility",
        legend_title_text="Time To Expiration",
        margin=dict(l=30, r=20, t=48, b=30),
    )
    return fig


def term_figure(surface_frame):
    fig = px.line(
        surface_frame,
        x="TimeToExpiration",
        y="PredictedVolatility",
        color="Moneyness",
        title="Term Structure Slices",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time To Expiration",
        yaxis_title="Predicted volatility",
        legend_title_text="Moneyness",
        margin=dict(l=30, r=20, t=48, b=30),
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
    fig.update_layout(
        template="plotly_white",
        xaxis_title=feature_label,
        yaxis_title="Predicted volatility",
        legend_title_text="Sample",
        margin=dict(l=30, r=20, t=48, b=30),
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
    fig.update_layout(
        template="plotly_white",
        xaxis_title=feature_label,
        yaxis_title="Accumulated local effect",
        margin=dict(l=30, r=20, t=48, b=30),
    )
    return fig


def local_surface_figure(surface_frame):
    pivot = surface_frame.pivot_table(
        index="TimeToExpiration",
        columns="Moneyness",
        values="PredictedVolatility",
    ).sort_index()
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns,
                y=pivot.index,
                z=pivot.values,
                colorscale="Viridis",
            )
        ]
    )
    fig.update_layout(
        title="Local Volatility Surface",
        template="plotly_white",
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            xaxis_title="Moneyness",
            yaxis_title="Time To Expiration",
            zaxis_title="Predicted volatility",
        ),
    )
    return fig
