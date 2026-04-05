"""Callbacks for model diagnosis."""

from __future__ import annotations

from dash import Input, Output, html
import plotly.graph_objects as go

from src.volatility_models.model_explainability.dashboard.ids import IDS
from src.volatility_models.model_explainability.plots.diagnosis_plots import (
    error_heatmap_figure,
    real_vs_predicted_figure,
    residual_by_feature_figure,
)


def _empty_figure():
    return go.Figure()


def register_diagnosis_callbacks(app, services) -> None:
    """Register diagnosis callbacks."""

    @app.callback(
        Output(IDS.DIAGNOSIS_METRICS, "children"),
        Output(IDS.DIAGNOSIS_SCATTER, "figure"),
        Output(IDS.DIAGNOSIS_MONEYNESS, "figure"),
        Output(IDS.DIAGNOSIS_MATURITY, "figure"),
        Output(IDS.DIAGNOSIS_HEATMAP, "figure"),
        Output(IDS.DIAGNOSIS_WARNINGS, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
    )
    def render_diagnosis(model_id):
        if not model_id:
            return "Select a model.", _empty_figure(), _empty_figure(), _empty_figure(), _empty_figure(), ""

        dataset = services.data_provider.load_dataset(model_id=model_id)
        try:
            diagnosis = services.cache.get_or_compute(
                "diagnosis",
                {"model_id": model_id},
                lambda: services.diagnosis_service.diagnose(model_id, dataset),
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return html.Div(str(exc), style={"color": "#8a1c1c"}), _empty_figure(), _empty_figure(), _empty_figure(), _empty_figure(), ""

        metric_cards = html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "8px"},
            children=[
                html.Div(
                    style={
                        "padding": "14px 16px",
                        "background": "#eef3fb",
                        "borderRadius": "12px",
                        "minWidth": "140px",
                        "border": "1px solid rgba(23,48,79,0.08)",
                    },
                    children=[
                        html.Div(services.metrics_registry.get(name).label, style={"fontWeight": "700", "marginBottom": "4px"}),
                        html.Div(services.metrics_registry.format_metric(name, value), style={"fontSize": "1.05rem"}),
                    ],
                )
                for name, value in diagnosis["metrics"].items()
            ],
        )
        plot_frame = diagnosis.get("plot_frame", diagnosis["diagnosis_frame"])
        warnings = (
            html.Ul([html.Li(message) for message in diagnosis["financial_warnings"]])
            if diagnosis["financial_warnings"]
            else html.P("No financial consistency warnings detected.", style={"margin": "0", "color": "#355070"})
        )
        return (
            metric_cards,
            real_vs_predicted_figure(plot_frame),
            residual_by_feature_figure(
                plot_frame,
                "Moneyness",
                "Residuals By Moneyness",
            ),
            residual_by_feature_figure(
                plot_frame,
                "TimeToExpiration",
                "Residuals By Maturity",
            ),
            error_heatmap_figure(diagnosis["error_heatmap"]),
            warnings,
        )
