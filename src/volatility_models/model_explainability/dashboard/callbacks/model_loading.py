"""Callbacks related to model discovery and shared dashboard context."""

from __future__ import annotations

from dash import Input, Output, State, dcc, html

from src.volatility_models.model_explainability.dashboard.ids import IDS
from src.volatility_models.model_explainability.utils.feature_utils import (
    build_sample_label,
    display_feature_label,
)
from src.volatility_models.model_explainability.utils.sampling import sample_frame


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
                "label": f"{model.name} ({model.format})",
                "value": model.model_id,
            }
            for model in models
        ]
        valid_values = {option["value"] for option in options}
        value = current_value if current_value in valid_values else (options[0]["value"] if options else None)
        return options, value

    @app.callback(
        Output(IDS.MODEL_INFO, "children"),
        Output(IDS.GLOBAL_DEPENDENCE_FEATURE, "options"),
        Output(IDS.GLOBAL_DEPENDENCE_FEATURE, "value"),
        Output(IDS.BEHAVIOUR_ICE_FEATURE, "options"),
        Output(IDS.BEHAVIOUR_ICE_FEATURE, "value"),
        Output(IDS.BEHAVIOUR_ANCHOR_INDEX, "options"),
        Output(IDS.SAMPLE_INDEX, "options"),
        Output(IDS.GLOBAL_EQUIVALENT_DEPTH_TABS, "children"),
        Output(IDS.GLOBAL_EQUIVALENT_DEPTH_TABS, "value"),
        Input(IDS.MODEL_SELECTOR, "value"),
    )
    def update_shared_context(model_id):
        dataset = services.data_provider.load_dataset(model_id=model_id)
        sampled = sample_frame(
            dataset,
            max_rows=250,
            random_state=services.settings.random_state,
        )
        sample_options = []
        anchor_options = []
        ice_options = [
            {"label": feature.label, "value": feature.name}
            for feature in services.feature_schema.numerical_features(raw_only=False)
            if feature.name in dataset.columns or feature.name in {"Moneyness", "LogMoneyness"}
        ]

        shap_options = []
        tree_tabs = []
        tree_value = None

        if not model_id:
            sample_options = [
                {"label": build_sample_label(row), "value": int(index)}
                for index, row in sampled.iterrows()
            ]
            return (
                html.Div(
                    "No model selected. Add explainable-model bundles to src/volatility_models/saved_models/ and select one.",
                    style={"color": "#8a1c1c"},
                ),
                shap_options,
                None,
                ice_options,
                ice_options[0]["value"] if ice_options else None,
                sample_options,
                sample_options,
                tree_tabs,
                tree_value,
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
        model_features = metadata.get("model_input_features", list(services.settings.model_input_features))
        shap_feature_names = metadata.get("transformed_feature_names", model_features)
        shap_options = [
            {
                "label": display_feature_label(str(feature_name), services.feature_schema),
                "value": str(feature_name),
            }
            for feature_name in shap_feature_names
        ]
        available_depths = metadata.get("available_surrogate_depths", [])
        tree_tabs = [
            dcc.Tab(label=f"Depth {int(depth)}", value=str(int(depth)))
            for depth in available_depths
        ]
        tree_value = tree_tabs[0].value if tree_tabs else None
        format_label = model.format.value if model else "unknown"
        info_children = [
            html.H3(model.name if model else model_id, style={"marginBottom": "6px"}),
            html.P(f"Format: {format_label}"),
            html.P(f"Inputs: {', '.join(model_features)}"),
            html.P(
                "Metadata metrics: "
                + ", ".join(metadata.get("error_metrics", services.settings.error_metrics))
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
            tree_tabs,
            tree_value,
        )
