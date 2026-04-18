"""Callbacks related to model discovery and shared dashboard context."""

from dash import Input, Output, State, html

from src.config.config import config
from src.dashboard.dashboard.ids import IDS
from src.dashboard.utils.feature_utils import build_sample_label, display_feature_label
from src.dashboard.utils.sampling import sample_frame
from src.model2dashboard.features import ANALYSIS_FEATURE_NAMES


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
                {"label": build_sample_label(row), "value": int(index)}
                for index, row in sampled.iterrows()
            ]
            return (
                html.Div(
                    "No model selected. Add explainable-model bundles to src/dashboard/saved_models/ and select one.",
                    style={"color": "#8a1c1c"},
                ),
                shap_options,
                None,
                ice_options,
                ice_options[0]["value"] if ice_options else None,
                sample_options,
                sample_options,
            )

        model = services.model_registry.get_model(model_id)
        metadata = model.metadata if model else {}
        bundle = services.prediction_service.load_bundle(model_id)
        sample_options = [
            {"label": build_sample_label(dataset.loc[index]), "value": int(index)}
            for index in bundle.dashboard_model.sample_indices
            if index in dataset.index
        ]
        anchor_options = [
            {"label": build_sample_label(dataset.loc[index]), "value": int(index)}
            for index in bundle.dashboard_model.behaviour_anchor_indices
            if index in dataset.index
        ]
        model_features = metadata.get("model_input_features", [])
        shap_feature_names = metadata.get("transformed_feature_names", model_features)
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
        info_children = [
            html.H3(model.name if model else model_id, style={"marginBottom": "6px"}),
            html.P(f"Format: {format_label}"),
            html.P(f"Inputs: {', '.join(model_features)}"),
            html.P(
                "Metadata metrics: "
                + ", ".join(
                    metadata.get(
                        "error_metrics", config.dashboard_models_config.error_metrics
                    )
                )
            ),
        ]
        return (
            html.Div(info_children),
            shap_options,
            shap_options[0]["value"] if shap_options else None,
            ice_options,
            ice_options[0]["value"] if ice_options else None,
            anchor_options,
            sample_options,
        )
