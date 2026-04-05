"""Local sample explainability figures."""

from __future__ import annotations

import plotly.express as px


def neighbors_distance_figure(neighbors_frame):
    frame = neighbors_frame.copy()
    frame["row_id"] = frame.index.astype(str)
    return px.bar(
        frame,
        x="row_id",
        y="distance",
        title="Nearest Neighbour Distances",
    )
