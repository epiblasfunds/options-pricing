"""Callbacks for local sample explainability."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from src.config.config import config
from src.dashboard.dashboard.ids import IDS
from src.dashboard.dashboard.styles import PILL_BUTTON_STYLE
from src.dashboard.plots.local_plots import neighbors_distance_figure
from src.dashboard.plots.shap_plots import waterfall_image
from src.dashboard.services.global_explainability import AUXILIARY_FEATURE_LABEL
from src.dashboard.utils.feature_utils import display_feature_label
from src.dashboard.utils.feature_utils import format_feature_value
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.model2dashboard.features import MAIN_EXPLAINABILITY_FEATURE_NAMES
from src.model2dashboard.features import VISIBLE_RAW_INPUT_FEATURE_NAMES


MANUAL_INPUT_DEFAULTS = {
    "OptionType": "CALL",
    "StrikePrice": 9100.0,
    "UnderlyingPrice": 9000.0,
    "TimeToExpiration": 20.0,
    "Rate": -0.6,
    "Quantity": 1,
    "UnderlyingLagMinutes": 0.0,
}

MANUAL_INPUT_LABEL_OVERRIDES = {
    "OptionType": "Type",
    "StrikePrice": "Strike",
    "UnderlyingPrice": "Underlying",
}

AUXILIARY_MANUAL_FEATURE_NAMES = [
    feature_name
    for feature_name in EXPLAINABILITY_FEATURE_NAMES
    if feature_name not in MAIN_EXPLAINABILITY_FEATURE_NAMES
]


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
    elif feature.name == "ExecDatetime":
        input_component = html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) auto", "gap": "8px"},
            children=[
                dcc.Input(
                    id=component_id,
                    type="text",
                    value=_manual_datetime_value(default_value),
                    placeholder="YYYY-MM-DDTHH:MM:SS+02:00",
                    style={
                        "height": "34px",
                        "width": "100%",
                        "boxSizing": "border-box",
                        "borderRadius": "8px",
                        "border": "1px solid rgba(23,48,79,0.18)",
                        "padding": "0 10px",
                    },
                ),
                html.Button(
                    "now",
                    id={"type": "manual-now", "feature": feature.name},
                    n_clicks=0,
                    type="button",
                    style={
                        **PILL_BUTTON_STYLE,
                        "height": "34px",
                        "padding": "0 12px",
                        "alignSelf": "center",
                    },
                ),
            ],
        )
    elif feature.allowed_values:
        options = [
            {"label": str(value), "value": value}
            for value in feature.allowed_values or ()
        ]
        input_component = dcc.Dropdown(
            id=component_id,
            options=options,
            value=default_value,
            clearable=False,
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
            html.Label(
                MANUAL_INPUT_LABEL_OVERRIDES.get(feature.name, feature.label),
                title=feature.description,
            ),
            input_component,
            html.Div(
                feature.description,
                style={
                    "marginTop": "6px",
                    "fontSize": "0.82rem",
                    "color": "#5b6d85",
                },
            )
            if feature.description
            else html.Div(),
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


def _manual_datetime_value(default_value):
    if default_value in (None, ""):
        return _current_local_hour_timestamp()
    timestamp = pd.to_datetime(default_value, errors="coerce")
    if pd.isna(timestamp):
        return _current_local_hour_timestamp()
    if getattr(timestamp, "tzinfo", None) is None:
        return timestamp.isoformat(timespec="seconds")
    return timestamp.isoformat(timespec="seconds")


def _current_local_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _current_local_hour_timestamp():
    return (
        datetime.now()
        .astimezone()
        .replace(minute=0, second=0, microsecond=0)
        .isoformat(timespec="milliseconds")
    )


def _current_local_midnight_timestamp():
    return (
        datetime.now()
        .astimezone()
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat(timespec="milliseconds")
    )


def _manual_hidden_defaults() -> dict[str, object]:
    return {
        "ExecDatetime": _current_local_midnight_timestamp(),
        "Quantity": MANUAL_INPUT_DEFAULTS["Quantity"],
        "UnderlyingLagMinutes": MANUAL_INPUT_DEFAULTS["UnderlyingLagMinutes"],
    }


def _manual_feature_names(feature_scope: str) -> list[str]:
    if feature_scope == "full":
        return list(EXPLAINABILITY_FEATURE_NAMES)
    return list(VISIBLE_RAW_INPUT_FEATURE_NAMES)


def _manual_section(title: str, feature_names: list[str], defaults: dict, services, *, subtle: bool = False):
    border = "1px dashed rgba(33,75,122,0.16)" if subtle else "1px solid rgba(33,75,122,0.12)"
    background = "#fbfcfe" if subtle else "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)"
    title_color = "#54708f" if subtle else "#17304f"
    description_color = "#6a7b92" if subtle else "#48617f"
    return html.Div(
        style={
            "display": "grid",
            "gap": "12px",
            "padding": "14px 16px",
            "borderRadius": "14px",
            "border": border,
            "background": background,
            "boxShadow": "0 8px 18px rgba(22,40,68,0.04)" if not subtle else "none",
        },
        children=[
            html.Div(
                [
                    html.Div(
                        title,
                        style={
                            "fontSize": "0.82rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": title_color,
                        },
                    ),
                    html.P(
                        (
                            "Primary visible drivers used in the dashboard."
                            if not subtle
                            else "Hidden model inputs available only in Full Features mode."
                        ),
                        style={
                            "margin": "4px 0 0 0",
                            "fontSize": "0.84rem",
                            "color": description_color,
                        },
                    ),
                ]
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "10px",
                },
                children=[
                    _manual_field(
                        services.feature_schema.get(feature_name),
                        defaults.get(feature_name),
                    )
                    for feature_name in feature_names
                ],
            ),
        ],
    )


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


def _feature_value_chip(
    label: str,
    value: str,
    *,
    subtle: bool = False,
) -> html.Div:
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "800",
                    "textTransform": "uppercase",
                    "color": "#6a7b92" if subtle else "#5b6d85",
                },
            ),
            html.Strong(
                value,
                style={
                    "fontSize": "0.9rem" if subtle else "0.96rem",
                    "color": "#415775" if subtle else "#17304f",
                    "fontWeight": "700" if subtle else "800",
                },
            ),
        ],
        style={
            "display": "grid",
            "gap": "4px",
            "padding": "10px 12px",
            "borderRadius": "12px",
            "border": (
                "1px dashed rgba(33,75,122,0.16)"
                if subtle
                else "1px solid rgba(33,75,122,0.12)"
            ),
            "background": "#fbfcfe" if subtle else "#ffffff",
            "minWidth": "150px",
            "opacity": "0.82" if subtle else "1",
        },
    )


def _sample_feature_preview_card(services, sample_payload: dict) -> html.Div:
    main_features = [
        feature_name
        for feature_name in MAIN_EXPLAINABILITY_FEATURE_NAMES
        if feature_name in sample_payload
    ]
    auxiliary_features = [
        feature_name
        for feature_name in EXPLAINABILITY_FEATURE_NAMES
        if feature_name not in main_features and feature_name in sample_payload
    ]
    return html.Div(
        style={
            "display": "grid",
            "gap": "14px",
            "padding": "16px 18px",
            "borderRadius": "14px",
            "border": "1px solid rgba(33,75,122,0.14)",
            "background": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
            "boxShadow": "0 8px 18px rgba(22,40,68,0.05)",
        },
        children=[
            html.Div(
                [
                    html.Div(
                        "Sample Feature Snapshot",
                        style={
                            "fontSize": "0.78rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#5b6d85",
                        },
                    ),
                    html.P(
                        (
                            "Main Features are the five visible drivers. Auxiliar "
                            "Features are the hidden inputs that still enter the model."
                        ),
                        style={"margin": "4px 0 0 0", "color": "#48617f"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        "Main Features",
                        style={
                            "fontSize": "0.82rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#17304f",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "10px",
                        },
                        children=[
                            _feature_value_chip(
                                display_feature_label(feature_name, services.feature_schema),
                                format_feature_value(
                                    feature_name,
                                    sample_payload.get(feature_name),
                                ),
                            )
                            for feature_name in main_features
                        ],
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        AUXILIARY_FEATURE_LABEL,
                        style={
                            "fontSize": "0.82rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#54708f",
                            "marginBottom": "8px",
                        },
                    ),
                    html.P(
                        (
                            "Hidden model inputs grouped with a lighter visual "
                            "weight so the main explanatory drivers remain primary."
                        ),
                        style={
                            "margin": "0 0 8px 0",
                            "fontSize": "0.84rem",
                            "color": "#6a7b92",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "10px",
                        },
                        children=[
                            _feature_value_chip(
                                display_feature_label(feature_name, services.feature_schema),
                                format_feature_value(
                                    feature_name,
                                    sample_payload.get(feature_name),
                                ),
                                subtle=True,
                            )
                            for feature_name in auxiliary_features
                        ],
                    ),
                ]
            ),
        ],
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
        Output(
            {"type": "manual-feature", "feature": MATCH}, "value", allow_duplicate=True
        ),
        Input({"type": "manual-now", "feature": MATCH}, "n_clicks"),
        prevent_initial_call=True,
    )
    def set_manual_datetime_now(_):
        return _current_local_timestamp()

    @app.callback(
        Output(IDS.SAMPLE_FEATURE_PREVIEW, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.SAMPLE_MODE, "value"),
        Input(IDS.SAMPLE_INDEX, "value"),
    )
    def render_sample_feature_preview(
        model_id,
        mode,
        sample_index,
    ):
        if not model_id:
            return html.Div()
        if mode == "manual":
            return html.Div()
        dataset = services.data_provider.load_dataset(model_id=model_id)
        sample_frame = (
            dataset.head(1).copy()
            if sample_index is None
            else dataset.loc[[sample_index]].copy()
        )
        sample_payload = {
            feature_name: sample_frame.iloc[0].get(feature_name)
            for feature_name in EXPLAINABILITY_FEATURE_NAMES
        }
        sample_payload = services.feature_schema.normalize_sample(sample_payload)
        return _sample_feature_preview_card(services, sample_payload)

    @app.callback(
        Output(IDS.SAMPLE_MANUAL_FORM, "children"),
        Output(IDS.SAMPLE_INDEX_CONTAINER, "style"),
        Input(IDS.SAMPLE_MODE, "value"),
        Input(IDS.SAMPLE_SHAP_FEATURE_SCOPE, "value"),
        Input(IDS.MODEL_SELECTOR, "value"),
    )
    def render_manual_form(mode, shap_feature_scope, _model_id):
        if mode != "manual":
            return html.Div(), {"marginTop": "14px"}
        dataset = services.data_provider.load_dataset(model_id=_model_id)
        defaults = services.feature_schema.defaults_from_frame(dataset, raw_only=True)
        defaults.update(MANUAL_INPUT_DEFAULTS)
        defaults.update(_manual_hidden_defaults())
        defaults["ExecDatetime"] = _current_local_hour_timestamp()
        main_feature_names = list(MAIN_EXPLAINABILITY_FEATURE_NAMES)
        auxiliary_feature_names = list(AUXILIARY_MANUAL_FEATURE_NAMES)
        form_sections = [
            _manual_section("Main Features", main_feature_names, defaults, services)
        ]
        if shap_feature_scope == "full":
            form_sections.append(
                _manual_section(
                    AUXILIARY_FEATURE_LABEL,
                    auxiliary_feature_names,
                    defaults,
                    services,
                    subtle=True,
                )
            )
        return (
            html.Div(
                style={
                    "display": "grid",
                    "gap": "14px",
                    "margin": "16px 0",
                },
                children=form_sections,
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
        State(IDS.SAMPLE_SHAP_FEATURE_SCOPE, "value"),
        State(IDS.SAMPLE_MODE, "value"),
        State(IDS.SAMPLE_INDEX, "value"),
        State({"type": "manual-feature", "feature": ALL}, "id"),
        State({"type": "manual-feature", "feature": ALL}, "value"),
    )
    def analyze_sample(
        _,
        model_id,
        shap_feature_scope,
        mode,
        sample_index,
        manual_ids,
        manual_values,
    ):
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
                sample_payload = _manual_hidden_defaults()
                sample_payload.update(
                    {
                        component_id["feature"]: value
                        for component_id, value in zip(manual_ids, manual_values)
                    }
                )
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
                explanation = services.shap_service.from_payload(
                    api_result["local_explanation"],
                    feature_scope=shap_feature_scope,
                )
                sample_summary = _prediction_indicator(
                    title="Manual Input",
                    prediction=float(api_result["prediction"]),
                )
                return (
                    sample_summary,
                    waterfall_image(
                        explanation,
                        explanation.explain_frame.index[0],
                        services.feature_schema,
                    ),
                    _neighbors_table(neighbors),
                    comparison,
                )

            explanation = services.shap_service.explain_sample(
                model_id,
                sample_frame,
                feature_scope=shap_feature_scope,
            )
            prediction = float(explanation.predictions.iloc[0])
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
