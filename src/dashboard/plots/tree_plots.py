"""Tree plots and visual exports."""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import tree

from src.python_models.explainable_model import SurrogateTreeModel
from src.dashboard.plots.plot_style import (
    HOVERLABEL_STYLE,
    STANDARD_MARGIN,
    STANDARD_TEMPLATE,
)

def feature_importance_figure(result: SurrogateTreeModel):
    frame = result.feature_importances.rename("importance").reset_index()
    frame.columns = ["feature", "importance"]
    fig = px.bar(
        frame,
        x="importance",
        y="feature",
        orientation="h",
        title="Surrogate Feature Importances",
    )
    fig.update_traces(
        hovertemplate=(
            "Feature: %{y}<br>"
            "Relative importance: %{x:.4f}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Relative importance",
        yaxis_title="Feature",
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def fidelity_figure(result: SurrogateTreeModel):
    fig = px.scatter(
        result.fidelity_frame,
        x="model_prediction",
        y="surrogate_prediction",
        title="Surrogate Fidelity",
        opacity=0.6,
    )
    fig.update_traces(
        hovertemplate=(
            "Original model prediction: %{x:.4f}<br>"
            "Surrogate prediction: %{y:.4f}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Original model prediction",
        yaxis_title="Surrogate prediction",
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def tree_png_base64(result: SurrogateTreeModel) -> str:
    width = max(20, min(48, result.n_leaves * 2.1))
    height = max(8, min(24, result.tree_depth * 2.2 + result.n_leaves * 0.28))
    font_size = max(7, 12 - min(result.tree_depth, 6) // 2)
    figure = plt.figure(figsize=(width, height))
    tree.plot_tree(
        result.model,
        feature_names=result.feature_names or result.feature_importances.index.tolist(),
        filled=True,
        rounded=True,
        fontsize=font_size,
    )
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

