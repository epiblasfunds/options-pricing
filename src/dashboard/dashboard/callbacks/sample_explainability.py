"""Callbacks for local sample explainability."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.local_plots import neighbors_distance_figure
from src.dashboard.plots.shap_plots import waterfall_image


def _empty_figure():
    return go.Figure()


def _manual_field(feature, default_value):
    component_id = {"type": "manual-feature", "feature": feature.name}
    if feature.widget == "dropdown" and feature.allowed_values:
        options = [
            {"label": str(value), "value": value}
            for value in feature.allowed_values or ()
        ]
        input_component = dcc.Dropdown(
            id=component_id, options=options, value=default_value
        )
    elif feature.dtype == "datetime":
        input_component = dcc.Input(
            id=component_id,
            type="text",
            value="" if default_value is None else str(default_value),
            placeholder="YYYY-MM-DD HH:MM:SS",
        )
    else:
        input_component = dcc.Input(
            id=component_id,
            type="number" if feature.is_numerical else "text",
            value=default_value,
            min=feature.min_value,
            max=feature.max_value,
            step=1 if feature.dtype == "int" else "any",
        )
    return html.Div(
        children=[
            html.Label(feature.label),
            input_component,
        ]
    )


def _neighbors_table(frame: pd.DataFrame):
    columns = [
        "OptionType",
        "TimeToExpiration",
        "UnderlyingPrice",
        "StrikePrice",
        "ImpliedVolatility",
        "PredictedVolatility",
        "distance",
    ]
    present_columns = [column for column in columns if column in frame.columns]
    cell_style = {
        "padding": "10px 12px",
        "textAlign": "center",
        "borderRight": "1px solid rgba(23,48,79,0.10)",
    }
    header = html.Tr(
        [
            html.Th(
                column,
                style={
                    **cell_style,
                    "fontWeight": "700",
                    "background": "#f5f8fc",
                },
            )
            for column in present_columns
        ]
    )
    body_rows = [
        html.Tr(
            [
                html.Td(
                    (
                        f"{row[column]:,.4f}"
                        if isinstance(row[column], float)
                        else row[column]
                    ),
                    style=cell_style,
                )
                for column in present_columns
            ]
        )
        for _, row in frame.head(10).iterrows()
    ]
    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        style={
            "width": "100%",
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "border": "1px solid rgba(23,48,79,0.10)",
            "borderRadius": "12px",
            "overflow": "hidden",
            "background": "#ffffff",
        },
    )


def register_sample_callbacks(app, services) -> None:
    """Register sample-explainability callbacks."""

    @app.callback(
        Output(IDS.SAMPLE_MANUAL_FORM, "children"),
        Output(IDS.SAMPLE_INDEX_CONTAINER, "style"),
        Input(IDS.SAMPLE_MODE, "value"),
        Input(IDS.MODEL_SELECTOR, "value"),
    )
    def render_manual_form(mode, _model_id):
        if mode != "manual":
            return html.Div(), {"marginTop": "14px"}
        dataset = services.data_provider.load_dataset(model_id=_model_id)
        defaults = services.feature_schema.defaults_from_frame(dataset, raw_only=True)
        return (
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "10px",
                    "margin": "16px 0",
                },
                children=[
                    _manual_field(feature, defaults.get(feature.name))
                    for feature in services.feature_schema.raw_input_features()
                ],
            ),
            {"display": "none"},
        )

    @app.callback(
        Output(IDS.SAMPLE_OUTPUT, "children"),
        Output(IDS.SAMPLE_WATERFALL, "src"),
        Output(IDS.SAMPLE_NEIGHBORS, "children"),
        Output(IDS.SAMPLE_COMPARISON, "figure"),
        Input(IDS.SAMPLE_RUN_BUTTON, "n_clicks"),
        State(IDS.MODEL_SELECTOR, "value"),
        State(IDS.SAMPLE_MODE, "value"),
        State(IDS.SAMPLE_INDEX, "value"),
        State({"type": "manual-feature", "feature": ALL}, "id"),
        State({"type": "manual-feature", "feature": ALL}, "value"),
    )
    def analyze_sample(_, model_id, mode, sample_index, manual_ids, manual_values):
        if not model_id:
            return "Select a model.", None, html.Div(), _empty_figure()
        dataset = services.data_provider.load_dataset(model_id=model_id)
        try:
            if mode == "dataset":
                if sample_index is None:
                    sample_frame = dataset.head(1).copy()
                else:
                    sample_frame = dataset.loc[[sample_index]].copy()
            else:
                sample_payload = {
                    component_id["feature"]: value
                    for component_id, value in zip(manual_ids, manual_values)
                }
                sample_payload = services.feature_schema.normalize_sample(
                    sample_payload
                )
                errors = services.feature_schema.validate_sample(sample_payload)
                if errors:
                    return (
                        html.Ul(
                            [
                                html.Li(f"{key}: {message}")
                                for key, message in errors.items()
                            ]
                        ),
                        None,
                        html.Div(),
                        _empty_figure(),
                    )
                api_result = services.prediction_service.call_manual_prediction_api(
                    model_id,
                    sample_payload,
                )
                reference_index = api_result.get("reference_sample_index")
                reference_sample = (
                    dataset.loc[[reference_index]].copy()
                    if reference_index is not None and reference_index in dataset.index
                    else dataset.head(1).copy()
                )
                explanation = services.shap_service.explain_sample(
                    model_id,
                    reference_sample,
                    dataset,
                )
                neighbors = services.neighbors_service.find_neighbors(
                    model_id,
                    dataset,
                    reference_sample,
                    k=10,
                )
                comparison = neighbors_distance_figure(neighbors)
                sample_summary = html.Div(
                    [
                        html.H3("Manual Input"),
                        html.P(
                            f"Predicted volatility (API stub): {float(api_result['prediction']):.4f}"
                        ),
                        html.P(str(api_result.get("summary", ""))),
                    ]
                )
                return (
                    sample_summary,
                    waterfall_image(
                        explanation, reference_sample.index[0], services.feature_schema
                    ),
                    _neighbors_table(neighbors),
                    comparison,
                )

            prediction = float(
                services.prediction_service.predict_frame(model_id, sample_frame).iloc[
                    0
                ]
            )
            explanation = services.shap_service.explain_sample(
                model_id, sample_frame, dataset
            )
            neighbors = services.neighbors_service.find_neighbors(
                model_id, dataset, sample_frame, k=10
            )
            comparison = neighbors_distance_figure(neighbors)
            actual = (
                f" | actual IV: {float(sample_frame['ImpliedVolatility'].iloc[0]):.4f}"
                if "ImpliedVolatility" in sample_frame.columns
                else ""
            )
            sample_summary = html.Div(
                [
                    html.H3("Selected Sample"),
                    html.P(f"Predicted volatility: {prediction:.4f}{actual}"),
                ]
            )
            return (
                sample_summary,
                waterfall_image(
                    explanation, sample_frame.index[0], services.feature_schema
                ),
                _neighbors_table(neighbors),
                comparison,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return (
                html.Div(str(exc), style={"color": "#8a1c1c"}),
                None,
                html.Div(),
                _empty_figure(),
            )
