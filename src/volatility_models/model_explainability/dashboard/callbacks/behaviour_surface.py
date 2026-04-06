"""Callbacks for behaviour and surface analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, html

from src.volatility_models.model_explainability.dashboard.ids import IDS
from src.volatility_models.model_explainability.plots.plot_style import (
    safe_color_range,
)
from src.volatility_models.model_explainability.plots.surface_plots import (
    ale_figure,
    heatmap_figure,
    ice_figure,
    local_surface_figure,
    smile_figure,
    term_figure,
)


def _empty_figure():
    return go.Figure()


def register_behaviour_callbacks(app, services) -> None:
    """Register behaviour callbacks."""

    @app.callback(
        Output(IDS.SURFACE_HEATMAP, "figure"),
        Output(IDS.SMILE_GRAPH, "figure"),
        Output(IDS.TERM_GRAPH, "figure"),
        Output(IDS.ICE_GRAPH, "figure"),
        Output(IDS.ALE_GRAPH, "figure"),
        Output(IDS.LOCAL_SURFACE_GRAPH, "figure"),
        Output(IDS.BEHAVIOUR_WARNINGS, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.BEHAVIOUR_ANCHOR_INDEX, "value"),
        Input(IDS.BEHAVIOUR_ICE_FEATURE, "value"),
    )
    def render_behaviour(model_id, anchor_index, feature_name):
        if not model_id:
            return (_empty_figure(),) * 6 + ("Select a model.",)

        dataset = services.data_provider.load_dataset(model_id=model_id)
        try:
            dashboard_model = services.prediction_service.load_dashboard_model(model_id)
            if anchor_index is None and dashboard_model.behaviour_anchor_indices:
                anchor_index = dashboard_model.behaviour_anchor_indices[0]
            if anchor_index is not None and anchor_index in dataset.index:
                anchor = dataset.loc[
                    anchor_index, list(services.settings.model_input_features)
                ]
            else:
                anchor = services.surface_service.default_anchor(dataset)
            surface = services.surface_service.build_surface(model_id, anchor)
            unique_maturities = sorted(surface["TimeToExpiration"].unique().tolist())
            keep_maturities = [
                unique_maturities[index]
                for index in np.linspace(
                    0,
                    len(unique_maturities) - 1,
                    min(5, len(unique_maturities)),
                    dtype=int,
                )
            ]
            smile_frame = surface[
                surface["TimeToExpiration"].isin(keep_maturities)
            ].copy()
            unique_moneyness = sorted(surface["Moneyness"].unique().tolist())
            keep_moneyness = [
                unique_moneyness[index]
                for index in np.linspace(
                    0,
                    len(unique_moneyness) - 1,
                    min(5, len(unique_moneyness)),
                    dtype=int,
                )
            ]
            term_frame = surface[surface["Moneyness"].isin(keep_moneyness)].copy()
            selected_feature = feature_name or "Moneyness"
            ice_frame = services.surface_service.compute_ice_curves(
                model_id, dataset, selected_feature
            )
            ale_frame = services.surface_service.compute_ale(
                model_id, dataset, selected_feature
            )
            warnings = html.Ul(
                [
                    html.Li(message)
                    for message in services.surface_service.financial_checks(surface)
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return (_empty_figure(),) * 6 + (
                html.Div(str(exc), style={"color": "#8a1c1c"}),
            )

        feature_label = (
            services.feature_schema.get(selected_feature).label
            if selected_feature in services.feature_schema.names()
            else selected_feature
        )
        volatility_range = safe_color_range(
            surface["PredictedVolatility"].tolist()
        )
        return (
            heatmap_figure(surface, volatility_range=volatility_range),
            smile_figure(smile_frame, volatility_range=volatility_range),
            term_figure(term_frame, volatility_range=volatility_range),
            ice_figure(ice_frame, feature_label),
            ale_figure(ale_frame, feature_label),
            local_surface_figure(surface, volatility_range=volatility_range),
            warnings,
        )
