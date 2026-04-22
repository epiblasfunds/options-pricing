"""Callbacks for local sample explainability."""

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from src.config.config import config
from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.local_plots import neighbors_distance_figure
from src.dashboard.plots.shap_plots import waterfall_image


MANUAL_INPUT_DEFAULTS = {
    "OptionType": "CALL",
    "StrikePrice": 9100.0,
    "UnderlyingPrice": 9000.0,
    "TimeToExpiration": 20.0,
    "Rate": -0.6,
}


def _empty_figure():
    return go.Figure()


def _prediction_indicator(
    *,
    title: str,
    prediction: float,
    actual: float | None = None,
) -> html.Div:
    supporting_items = []
    if actual is not None:
        supporting_items.append(
            html.Div(
                [
                    html.Span("Actual implied volatility"),
                    html.Strong(f"{actual:.4f}"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "gap": "18px",
                    "fontSize": "0.96rem",
                    "color": "#35506e",
                },
            )
        )
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "minmax(0, 1fr) auto",
            "gap": "18px",
            "alignItems": "center",
            "padding": "20px 22px",
            "borderRadius": "14px",
            "border": "1px solid rgba(33,75,122,0.18)",
            "background": "linear-gradient(135deg, #ffffff 0%, #edf5ff 100%)",
            "boxShadow": "0 12px 26px rgba(22,40,68,0.10)",
            "margin": "16px 0",
        },
        children=[
            html.Div(
                [
                    html.H3(
                        title,
                        style={
                            "margin": "0 0 8px 0",
                            "fontSize": "1.05rem",
                            "color": "#17304f",
                        },
                    ),
                    html.Div(
                        "Predicted volatility",
                        style={
                            "fontSize": "0.92rem",
                            "fontWeight": "700",
                            "textTransform": "uppercase",
                            "color": "#48617f",
                        },
                    ),
                    html.Div(supporting_items, style={"marginTop": "12px"}),
                ]
            ),
            html.Div(
                f"{prediction:.4f}",
                style={
                    "minWidth": "180px",
                    "textAlign": "right",
                    "fontSize": "2.55rem",
                    "fontWeight": "800",
                    "lineHeight": "1",
                    "color": "#17304f",
                },
            ),
        ],
    )


def _manual_field(feature, default_value):
    component_id = {"type": "manual-feature", "feature": feature.name}
    if feature.name == "OptionType":
        input_component = dcc.Dropdown(
            id=component_id,
            options=[
                {"label": "CALL", "value": "CALL"},
                {"label": "PUT", "value": "PUT"},
            ],
            value=_manual_option_type_value(default_value),
            clearable=False,
        )
    elif feature.widget == "dropdown" and feature.allowed_values:
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
    elif feature.is_numerical:
        input_component = _manual_number_control(feature, default_value)
    else:
        input_component = dcc.Input(
            id=component_id,
            type="text",
            value=default_value,
        )
    return html.Div(
        children=[
            html.Label(feature.label),
            input_component,
        ]
    )


def _manual_number_control(feature, default_value):
    feature_name = feature.name
    button_style = {
        "width": "34px",
        "height": "34px",
        "border": "1px solid rgba(23,48,79,0.18)",
        "background": "#edf5ff",
        "color": "#17304f",
        "fontWeight": "800",
        "cursor": "pointer",
    }
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "34px minmax(0, 1fr) 34px",
            "alignItems": "center",
            "width": "100%",
        },
        children=[
            html.Button(
                "-",
                id={
                    "type": "manual-step",
                    "feature": feature_name,
                    "direction": "down",
                },
                n_clicks=0,
                type="button",
                style={
                    **button_style,
                    "borderRadius": "8px 0 0 8px",
                    "borderRight": "0",
                },
            ),
            dcc.Input(
                id={"type": "manual-feature", "feature": feature_name},
                type="text",
                value=_manual_numeric_value(default_value),
                style={
                    "height": "34px",
                    "width": "100%",
                    "boxSizing": "border-box",
                    "borderRadius": "0",
                    "border": "1px solid rgba(23,48,79,0.18)",
                    "textAlign": "right",
                    "padding": "0 10px",
                },
            ),
            html.Button(
                "+",
                id={
                    "type": "manual-step",
                    "feature": feature_name,
                    "direction": "up",
                },
                n_clicks=0,
                type="button",
                style={
                    **button_style,
                    "borderRadius": "0 8px 8px 0",
                    "borderLeft": "0",
                },
            ),
        ],
    )


def _manual_option_type_value(default_value):
    value = default_value.value if hasattr(default_value, "value") else default_value
    text = str(value).strip().upper()
    if text in {"CALL", "C"}:
        return "CALL"
    if text in {"PUT", "P"}:
        return "PUT"
    return "CALL"


def _manual_numeric_step(feature):
    if feature.dtype == "int":
        return 1
    return {
        "StrikePrice": 1,
        "UnderlyingPrice": 1,
        "TimeToExpiration": 1,
        "UnderlyingLagMinutes": 1,
        "Quantity": 1,
        "Rate": 0.01,
    }.get(feature.name, 0.01 if feature.is_numerical else None)


def _manual_numeric_value(default_value):
    if default_value in (None, ""):
        return None
    try:
        numeric_value = float(default_value)
    except (TypeError, ValueError):
        return default_value
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def _stepped_manual_value(feature_schema, feature_name, current_value, direction):
    feature = feature_schema.get(feature_name)
    step = float(_manual_numeric_step(feature) or 1)
    try:
        value = float(current_value)
    except (TypeError, ValueError):
        value = float(MANUAL_INPUT_DEFAULTS.get(feature_name, 0.0))
    value = value + step if direction == "up" else value - step
    if feature.min_value is not None:
        value = max(float(feature.min_value), value)
    if feature.max_value is not None:
        value = min(float(feature.max_value), value)
    if feature.dtype == "int" or float(value).is_integer():
        return int(round(value))
    return round(value, 10)


def _neighbors_table(frame: pd.DataFrame):
    columns = [
        "index",
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


def _validate_manual_payload(services, sample_payload: dict) -> dict[str, str]:
    errors: dict[str, str] = {}
    mandatory_names = config.clientserver_config.dashboard_manual_input_features
    normalized = services.feature_schema.normalize_sample(sample_payload)
    for feature_name in mandatory_names:
        feature = services.feature_schema.get(feature_name)
        value = normalized.get(feature_name)
        if value in (None, ""):
            errors[feature_name] = "Value is required."
            continue
        if feature.allowed_values is not None and value not in feature.allowed_values:
            errors[feature_name] = "Value is outside the allowed set."
            continue
        if feature.is_numerical:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors[feature_name] = "Value must be numeric."
                continue
            if feature.min_value is not None and numeric_value < feature.min_value:
                errors[feature_name] = "Value is below the minimum."
            if feature.max_value is not None and numeric_value > feature.max_value:
                errors[feature_name] = "Value is above the maximum."
    return errors


def register_sample_callbacks(app, services) -> None:
    """Register sample-explainability callbacks."""

    @app.callback(
        Output({"type": "manual-feature", "feature": MATCH}, "value"),
        Input({"type": "manual-step", "feature": MATCH, "direction": ALL}, "n_clicks"),
        State({"type": "manual-feature", "feature": MATCH}, "id"),
        State({"type": "manual-feature", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def step_manual_numeric_value(_, component_id, current_value):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        feature_name = component_id["feature"]
        return _stepped_manual_value(
            services.feature_schema,
            feature_name,
            current_value,
            triggered.get("direction"),
        )

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
        defaults.update(MANUAL_INPUT_DEFAULTS)
        mandatory_names = set(
            config.clientserver_config.dashboard_manual_input_features
        )
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
                    if feature.name in mandatory_names
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
                errors = _validate_manual_payload(services, sample_payload)
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
                api_result = (
                    services.prediction_service.call_manual_sample_explainability_api(
                        model_id,
                        sample_payload,
                    )
                )
                neighbors = pd.DataFrame(api_result.get("neighbors", []))
                comparison = (
                    neighbors_distance_figure(neighbors)
                    if not neighbors.empty
                    else _empty_figure()
                )
                sample_summary = _prediction_indicator(
                    title="Manual Input",
                    prediction=float(api_result["prediction"]),
                )
                return (
                    sample_summary,
                    api_result.get("waterfall_image"),
                    _neighbors_table(neighbors),
                    comparison,
                )

            prediction = float(
                services.prediction_service.predict_frame(model_id, sample_frame).iloc[
                    0
                ]
            )
            explanation = services.shap_service.explain_sample(model_id, sample_frame)
            neighbors = services.neighbors_service.find_neighbors(
                model_id, sample_frame, k=10
            )
            comparison = neighbors_distance_figure(neighbors)
            actual = (
                float(sample_frame["ImpliedVolatility"].iloc[0])
                if "ImpliedVolatility" in sample_frame.columns
                else None
            )
            sample_summary = _prediction_indicator(
                title="Selected Sample",
                prediction=prediction,
                actual=actual,
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
