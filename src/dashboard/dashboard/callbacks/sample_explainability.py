"""Callbacks for local sample explainability."""

import numbers

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from src.config.config import config
from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.local_plots import neighbors_projection_3d_figure
from src.dashboard.plots.shap_plots import waterfall_image
from src.dashboard.utils.feature_utils import display_feature_label
from src.dashboard.utils.feature_utils import format_feature_value
from src.model2dashboard.features import add_dashboard_derived_features
from src.model2dashboard.features import build_feature_frame_from_trades
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.model2dashboard.features import VISIBLE_RAW_INPUT_FEATURE_NAMES


MANUAL_INPUT_DEFAULTS = {
    "OptionType": "CALL",
    "StrikePrice": 9100.0,
    "UnderlyingPrice": 9000.0,
    "TimeToExpiration": 20.0,
    "Rate": -0.6,
}

MANUAL_INPUT_LABEL_OVERRIDES = {
    "OptionType": "Type",
    "StrikePrice": "Strike",
    "UnderlyingPrice": "Underlying",
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


def _manual_hidden_defaults() -> dict[str, object]:
    return {}


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
        "ID",
        "OptionType",
        "TimeToExpiration",
        "UnderlyingPrice",
        "StrikePrice",
        "ImpliedVolatility",
        "PredictedVolatility",
    ]
    present_columns = [column for column in columns if column in frame.columns]
    if "Distance" not in present_columns:
        present_columns.append("Distance")
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
                _neighbors_table_cell(row, column, cell_style)
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
) -> html.Div:
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "fontSize": "0.75rem",
                    "fontWeight": "800",
                    "textTransform": "uppercase",
                    "color": "#5b6d85",
                },
            ),
            html.Strong(
                value,
                style={
                    "fontSize": "0.96rem",
                    "color": "#17304f",
                    "fontWeight": "800",
                },
            ),
        ],
        style={
            "display": "grid",
            "gap": "4px",
            "padding": "10px 12px",
            "borderRadius": "12px",
            "border": "1px solid rgba(33,75,122,0.12)",
            "background": "#ffffff",
            "minWidth": "150px",
        },
    )


def _neighbors_table_cell(row: pd.Series, column: str, cell_style: dict) -> html.Td:
    if column == "ID":
        return html.Td(_format_neighbor_id(row.get("row_id")), style=cell_style)
    if column == "Distance":
        distance_value = pd.to_numeric(row.get("distance"), errors="coerce")
        distance = 0.0 if pd.isna(distance_value) else float(distance_value)
        max_distance_value = pd.to_numeric(row.get("_distance_max"), errors="coerce")
        max_distance = 0.0 if pd.isna(max_distance_value) else float(max_distance_value)
        width_percent = (
            100.0 if max_distance <= 0.0 else 100.0 * distance / max_distance
        )
        fill_percent = 0.0 if distance <= 0.0 else max(8.0, width_percent)
        return html.Td(
            [
                html.Div(
                    f"{distance:.4f}",
                    style={
                        "fontWeight": "700",
                        "color": "#17304f",
                        "marginBottom": "6px",
                    },
                ),
                html.Div(
                    style={
                        "height": "10px",
                        "borderRadius": "999px",
                        "background": "rgba(122,165,210,0.18)",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Div(
                            style={
                                "height": "100%",
                                "width": f"{fill_percent:.1f}%",
                                "borderRadius": "999px",
                                "background": (
                                    "linear-gradient(90deg, #7aa5d2 0%, #17304f 100%)"
                                ),
                            }
                        )
                    ],
                ),
            ],
            style={**cell_style, "minWidth": "180px"},
        )

    value = row.get(column)
    formatted = (
        f"{float(value):,.4f}"
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
        else value
    )
    return html.Td(formatted, style=cell_style)


def _prepare_neighbors_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    neighbors = frame.copy()
    if neighbors.empty:
        return neighbors
    if "index" not in neighbors.columns:
        neighbors = neighbors.reset_index()
    neighbors["row_id"] = neighbors["index"].apply(_format_neighbor_id)
    neighbors["distance"] = pd.to_numeric(neighbors["distance"], errors="coerce")
    neighbors["_distance_max"] = float(neighbors["distance"].max(skipna=True) or 0.0)
    return neighbors


def _format_neighbor_id(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(float(numeric)))
    return str(value)


def _sample_projection_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sample = frame.head(1).copy()
    if sample.empty:
        return sample
    if "index" not in sample.columns:
        sample = sample.reset_index()
    return sample


def _manual_sample_projection_frame(sample_payload: dict[str, object]) -> pd.DataFrame:
    raw_frame = pd.DataFrame([sample_payload])
    feature_frame = build_feature_frame_from_trades(raw_frame)
    for column in feature_frame.columns:
        raw_frame[column] = feature_frame[column].to_numpy()
    return add_dashboard_derived_features(raw_frame)


def _projection_feature_names(
    dashboard_model,
    sample_frame: pd.DataFrame,
    neighbors: pd.DataFrame,
) -> list[str]:
    candidate_names = [
        name
        for name in (
            dashboard_model.transformed_feature_names
            or dashboard_model.metadata.get("model_input_features", [])
        )
        if name in sample_frame.columns and name in neighbors.columns
    ]
    feature_names: list[str] = []
    for feature_name in candidate_names:
        sample_numeric = pd.to_numeric(sample_frame[feature_name], errors="coerce")
        neighbor_numeric = pd.to_numeric(neighbors[feature_name], errors="coerce")
        combined = pd.concat([sample_numeric, neighbor_numeric], axis=0)
        if combined.notna().sum() == 0:
            continue
        if np.isclose(float(combined.max()), float(combined.min()), equal_nan=True):
            continue
        feature_names.append(feature_name)
    return feature_names


def _sample_feature_preview_card(services, sample_payload: dict) -> html.Div:
    feature_names = [
        feature_name
        for feature_name in EXPLAINABILITY_FEATURE_NAMES
        if feature_name in sample_payload
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
                        "Original input variables used by the model for this sample.",
                        style={"margin": "4px 0 0 0", "color": "#48617f"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        "Sample Features",
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
                            for feature_name in feature_names
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
        Output(IDS.SAMPLE_FEATURE_PREVIEW, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.SAMPLE_MODE, "value"),
        Input(IDS.SAMPLE_INDEX, "value"),
        Input(IDS.MODEL_REFRESH_TOKEN, "data"),
    )
    def render_sample_feature_preview(
        model_id,
        mode,
        sample_index,
        _refresh_token,
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
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.MODEL_REFRESH_TOKEN, "data"),
    )
    def render_manual_form(mode, _model_id, _refresh_token):
        if mode != "manual":
            return html.Div(), {"marginTop": "14px"}
        dataset = services.data_provider.load_dataset(model_id=_model_id)
        defaults = services.feature_schema.defaults_from_frame(dataset, raw_only=True)
        defaults.update(MANUAL_INPUT_DEFAULTS)
        defaults.update(_manual_hidden_defaults())
        return (
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "10px",
                    "margin": "16px 0",
                },
                children=[
                    _manual_field(
                        services.feature_schema.get(feature_name),
                        defaults.get(feature_name),
                    )
                    for feature_name in VISIBLE_RAW_INPUT_FEATURE_NAMES
                ],
            ),
            {"display": "none"},
        )

    @app.callback(
        Output(IDS.SAMPLE_OUTPUT, "children"),
        Output(IDS.SAMPLE_WATERFALL, "src"),
        Output(IDS.SAMPLE_NEIGHBORS, "children"),
        Output(IDS.SAMPLE_COMPARISON_3D, "figure"),
        Input(IDS.SAMPLE_RUN_BUTTON, "n_clicks"),
        State(IDS.MODEL_SELECTOR, "value"),
        State(IDS.SAMPLE_MODE, "value"),
        State(IDS.SAMPLE_INDEX, "value"),
        State({"type": "manual-feature", "feature": ALL}, "id"),
        State({"type": "manual-feature", "feature": ALL}, "value"),
    )
    def analyze_sample(
        _,
        model_id,
        mode,
        sample_index,
        manual_ids,
        manual_values,
    ):
        if not model_id:
            return "Select a model.", None, html.Div(), _empty_figure()
        dataset = services.data_provider.load_dataset(model_id=model_id)
        try:
            dashboard_model = services.prediction_service.load_dashboard_model(model_id)
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
                projection_sample = _manual_sample_projection_frame(sample_payload)
                ranked_neighbors = services.neighbors_service.rank_neighbors(
                    model_id,
                    projection_sample,
                )
                map_neighbors = ranked_neighbors.head(
                    config.dashboard_models_config.build_config.neighbors_k
                )
                neighbors = _prepare_neighbors_for_display(ranked_neighbors.head(10))
                projection_features = _projection_feature_names(
                    dashboard_model,
                    projection_sample,
                    map_neighbors,
                )
                comparison_3d = (
                    neighbors_projection_3d_figure(
                        projection_sample,
                        map_neighbors,
                        feature_names=projection_features,
                        center_label="Manual Input",
                    )
                    if not map_neighbors.empty
                    else _empty_figure()
                )
                explanation = services.shap_service.from_payload(
                    api_result["local_explanation"],
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
                    comparison_3d,
                )

            explanation = services.shap_service.explain_sample(
                model_id,
                sample_frame,
            )
            prediction = float(explanation.predictions.iloc[0])
            ranked_neighbors = services.neighbors_service.rank_neighbors(
                model_id,
                sample_frame,
            )
            map_neighbors = ranked_neighbors.head(
                config.dashboard_models_config.build_config.neighbors_k
            )
            neighbors = _prepare_neighbors_for_display(ranked_neighbors.head(10))
            projection_sample = _sample_projection_frame(sample_frame)
            projection_features = _projection_feature_names(
                dashboard_model,
                projection_sample,
                map_neighbors,
            )
            comparison_3d = neighbors_projection_3d_figure(
                projection_sample,
                map_neighbors,
                feature_names=projection_features,
                center_label="Selected Sample",
            )
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
                comparison_3d,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return (
                html.Div(str(exc), style={"color": "#8a1c1c"}),
                None,
                html.Div(),
                _empty_figure(),
            )
