"""Local sample explainability figures."""

from __future__ import annotations

import plotly.express as px

from src.dashboard.plots.plot_style import (
    HOVERLABEL_STYLE,
    STANDARD_MARGIN,
    STANDARD_TEMPLATE,
)

def neighbors_distance_figure(neighbors_frame):
    frame = neighbors_frame.copy()
    frame["row_id"] = frame.index.astype(str)
    fig = px.bar(
        frame,
        x="row_id",
        y="distance",
        title="Nearest Neighbour Distances",
    )
    fig.update_traces(
        hovertemplate=(
            "Neighbour row: %{x}<br>"
            "Distance: %{y:.4f}<extra></extra>"
        )
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Neighbour row",
        yaxis_title="Distance",
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig

