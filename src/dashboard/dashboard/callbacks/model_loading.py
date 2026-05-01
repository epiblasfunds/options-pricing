"""Callbacks related to model discovery and shared dashboard context."""

from dash import Input, Output, State, html

from src.config.config import config
from src.dashboard.dashboard.ids import IDS
from src.dashboard.utils.feature_utils import build_manual_input_sample_label
from src.dashboard.utils.feature_utils import build_sample_label, display_feature_label
from src.dashboard.utils.sampling import sample_frame
from src.model2dashboard.features import ANALYSIS_FEATURE_NAMES


def _default_ice_feature(ice_options: list[dict]) -> str | None:
    option_values = [option["value"] for option in ice_options]
    if "StrikePrice" in option_values:
        return "StrikePrice"
    return option_values[0] if option_values else None


def _model_info_panel(
    *,
    model_name: str,
    format_label: str,
    explained_features: list[str],
    context_features: list[str],
    metric_names: list[str],
    feature_schema,
) -> html.Div:
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "minmax(220px, 0.9fr) minmax(320px, 1.8fr)",
            "gap": "16px",
            "alignItems": "stretch",
            "padding": "14px 16px",
            "border": "1px solid rgba(33,75,122,0.14)",
            "borderRadius": "14px",
            "background": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
            "boxShadow": "0 8px 18px rgba(22,40,68,0.05)",
        },
        children=[
            html.Div(
                children=[
                    html.Div(
                        "Selected model",
                        style={
                            "fontSize": "0.78rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#5b6d85",
                            "marginBottom": "5px",
                        },
                    ),
                    html.H3(
                        model_name,
                        style={
                            "margin": "0 0 10px 0",
                            "fontSize": "1.18rem",
                            "lineHeight": "1.25",
                            "color": "#17304f",
                        },
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                        children=[
                            _metadata_chip("Format", format_label),
                            *[_metadata_chip("Metric", metric) for metric in metric_names],
                        ],
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Div(
                        "Explained variables",
                        style={
                            "fontSize": "0.78rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#5b6d85",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "overflowX": "auto",
                            "padding": "2px 2px 8px 2px",
                            "maxWidth": "100%",
                            "scrollbarWidth": "thin",
                        },
                        children=[
                            html.Span(
                                (
                                    feature_schema.get(str(feature_name)).label
                                    if str(feature_name) in feature_schema.names()
                                    else str(feature_name)
                                ),
                                title=(
                                    feature_schema.get(str(feature_name)).description
                                    if str(feature_name) in feature_schema.names()
                                    else str(feature_name)
                                ),
                                style={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "whiteSpace": "nowrap",
                                    "padding": "6px 10px",
                                    "borderRadius": "999px",
                                    "border": "1px solid rgba(33,75,122,0.14)",
                                    "background": "#ffffff",
                                    "color": "#243b5a",
                                    "fontSize": "0.86rem",
                                    "fontWeight": "600",
                                },
                            )
                            for feature_name in explained_features
                        ],
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Div(
                        "Context variable",
                        style={
                            "fontSize": "0.78rem",
                            "fontWeight": "800",
                            "textTransform": "uppercase",
                            "color": "#5b6d85",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Span(
                                (
                                    feature_schema.get(str(feature_name)).label
                                    if str(feature_name) in feature_schema.names()
                                    else str(feature_name)
                                ),
                                title=(
                                    feature_schema.get(str(feature_name)).description
                                    if str(feature_name) in feature_schema.names()
                                    else str(feature_name)
                                ),
                                style={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "whiteSpace": "nowrap",
                                    "padding": "6px 10px",
                                    "borderRadius": "999px",
                                    "border": "1px solid rgba(33,75,122,0.14)",
                                    "background": "#ffffff",
                                    "color": "#243b5a",
                                    "fontSize": "0.86rem",
                                    "fontWeight": "600",
                                },
                            )
                            for feature_name in context_features
                        ]
                        or [
                            html.Span(
                                "None",
                                style={
                                    "color": "#5b6d85",
                                    "fontSize": "0.86rem",
                                },
                            )
                        ],
                    ),
                ]
            ),
        ],
    )


def _metadata_chip(label: str, value: str) -> html.Span:
    return html.Span(
        [
            html.Span(
                f"{label}:",
                style={"color": "#5b6d85", "fontWeight": "700"},
            ),
            html.Span(str(value), style={"color": "#17304f", "fontWeight": "800"}),
        ],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "4px",
            "padding": "6px 10px",
            "borderRadius": "999px",
            "background": "#edf5ff",
            "border": "1px solid rgba(47,93,138,0.18)",
            "fontSize": "0.84rem",
        },
    )


def register_model_loading_callbacks(app, services) -> None:
    """Register model-loading callbacks."""

    @app.callback(
        Output(IDS.MODEL_SELECTOR, "options"),
        Output(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.MODEL_REFRESH_BUTTON, "n_clicks"),
        State(IDS.MODEL_SELECTOR, "value"),
    )
    def refresh_models(_, current_value):
        models = services.model_registry.discover_models()
        options = [
            {
                "label": f"{model.name} ({model.format.value})",
                "value": model.model_id,
            }
            for model in models
        ]
        valid_values = {option["value"] for option in options}
        value = (
            current_value
            if current_value in valid_values
            else (options[0]["value"] if options else None)
        )
        return options, value

    @app.callback(
        Output(IDS.MODEL_INFO, "children"),
        Output(IDS.GLOBAL_DEPENDENCE_FEATURE, "options"),
        Output(IDS.GLOBAL_DEPENDENCE_FEATURE, "value"),
        Output(IDS.BEHAVIOUR_ICE_FEATURE, "options"),
        Output(IDS.BEHAVIOUR_ICE_FEATURE, "value"),
        Output(IDS.BEHAVIOUR_ANCHOR_INDEX, "options"),
        Output(IDS.SAMPLE_INDEX, "options"),
        Input(IDS.MODEL_SELECTOR, "value"),
    )
    def update_shared_context(model_id):
        dataset = services.data_provider.load_dataset(model_id=model_id)
        sampled = sample_frame(
            dataset,
            max_rows=250,
            random_state=config.dashboard_models_config.random_state,
        )
        sample_options = []
        anchor_options = []
        ice_options = [
            {"label": feature.label, "value": feature.name}
            for feature in services.feature_schema.numerical_features(raw_only=False)
            if feature.name in ANALYSIS_FEATURE_NAMES
            and feature.name in dataset.columns
        ]

        shap_options = []

        if not model_id:
            sample_options = [
                {"label": build_manual_input_sample_label(row), "value": int(index)}
                for index, row in sampled.iterrows()
            ]
            return (
                html.Div(
                    "No model selected. Publish explainable-model bundles to the configured storage backend and select one.",
                    style={"color": "#8a1c1c"},
                ),
                shap_options,
                None,
                ice_options,
                _default_ice_feature(ice_options),
                sample_options,
                sample_options,
            )

        model = services.model_registry.get_model(model_id)
        metadata = model.metadata if model else {}
        bundle = services.prediction_service.load_bundle(model_id)
        sample_options = [
            {
                "label": build_manual_input_sample_label(dataset.loc[index]),
                "value": int(index),
            }
            for index in bundle.dashboard_model.sample_indices
            if index in dataset.index
        ]
        anchor_options = [
            {"label": build_sample_label(dataset.loc[index]), "value": int(index)}
            for index in bundle.dashboard_model.behaviour_anchor_indices
            if index in dataset.index
        ]
        explained_feature_names = metadata.get("explainability_feature_names") or metadata.get(
            "raw_feature_names",
            [],
        )
        context_feature_names = metadata.get("context_feature_names", [])
        shap_feature_names = [
            str(feature_name)
            for feature_name in explained_feature_names
            if (
                str(feature_name) in services.feature_schema.names()
                and services.feature_schema.get(str(feature_name)).is_numerical
            )
        ]
        shap_options = [
            {
                "label": display_feature_label(
                    str(feature_name), services.feature_schema
                ),
                "value": str(feature_name),
            }
            for feature_name in shap_feature_names
        ]
        format_label = model.format.value if model else "unknown"
        metric_names = list(
            metadata.get("error_metrics", config.dashboard_models_config.error_metrics)
        )
        info_panel = _model_info_panel(
            model_name=model.name if model else str(model_id),
            format_label=format_label,
            explained_features=[str(feature) for feature in explained_feature_names],
            context_features=[str(feature) for feature in context_feature_names],
            metric_names=[str(metric) for metric in metric_names],
            feature_schema=services.feature_schema,
        )
        return (
            info_panel,
            shap_options,
            shap_options[0]["value"] if shap_options else None,
            ice_options,
            _default_ice_feature(ice_options),
            anchor_options,
            sample_options,
        )
