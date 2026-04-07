"""Callbacks for global explainability plots."""

from __future__ import annotations

from dash import Input, Output

from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.shap_plots import (
    bar_image,
    beeswarm_image,
    dependence_image,
    heatmap_image,
)
from src.dashboard.utils.feature_utils import display_feature_label


def _empty_image():
    return None


def _pick_dependence_feature(result, requested_feature):
    if requested_feature in result.feature_names:
        series = result.explain_frame[requested_feature]
        if int(series.nunique(dropna=False)) > 1:
            return requested_feature

    for feature_name in result.mean_abs_shap.index.tolist():
        if feature_name not in result.explain_frame.columns:
            continue
        if int(result.explain_frame[feature_name].nunique(dropna=False)) > 1:
            return feature_name
    return result.feature_names[0]


def register_global_callbacks(app, services) -> None:
    """Register global explainability callbacks."""

    @app.callback(
        Output(IDS.GLOBAL_SUMMARY_GRAPH, "src"),
        Output(IDS.GLOBAL_BAR_GRAPH, "src"),
        Output(IDS.GLOBAL_DEPENDENCE_GRAPH, "src"),
        Output(IDS.GLOBAL_INTERACTION_GRAPH, "src"),
        Output(IDS.GLOBAL_NOTE, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.GLOBAL_DEPENDENCE_FEATURE, "value"),
    )
    def render_global_explainability(model_id, selected_feature):
        if not model_id:
            return _empty_image(), _empty_image(), _empty_image(), _empty_image(), "Select a model."
        dataset = services.data_provider.load_dataset(model_id=model_id)
        try:
            result = services.cache.get_or_compute(
                "shap_global",
                {"model_id": model_id},
                lambda: services.shap_service.explain(model_id, dataset),
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return _empty_image(), _empty_image(), _empty_image(), _empty_image(), str(exc)

        feature_name = _pick_dependence_feature(result, selected_feature)
        top_driver_labels = [
            display_feature_label(name, services.feature_schema)
            for name in result.mean_abs_shap.head(5).index.tolist()
        ]
        note = (
            f"Method: {result.method}. Dependence feature: "
            + f"{display_feature_label(feature_name, services.feature_schema)}. Top drivers: "
            + ", ".join(top_driver_labels)
        )
        return (
            beeswarm_image(result, services.feature_schema),
            bar_image(result, services.feature_schema),
            dependence_image(result, feature_name, services.feature_schema),
            heatmap_image(result, services.feature_schema),
            note,
        )

