"""Callbacks for equivalent explainable models."""

from dash import Input, Output, html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.dashboard.symbolic_model_components import (
    build_symbolic_panel,
)
from src.dashboard.dashboard.tree_model_components import (
    build_tree_panel_content,
)
from src.dashboard.dashboard.tree_model_components import (
    build_tree_shell,
)


def register_equivalent_callbacks(app, services) -> None:
    """Register equivalent-model callbacks."""

    @app.callback(
        Output(IDS.GLOBAL_EQUIVALENT_CONTENT, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.MODEL_REFRESH_TOKEN, "data"),
    )
    def render_equivalent_models(model_id, _refresh_token):
        if not model_id:
            return html.Div("Select an explainable-model bundle.")
        try:
            symbolic_model, tree_models = (
                services.equivalent_models_service.load_equivalent_models(model_id)
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            return html.Div(str(exc), style={"color": "#8a1c1c"})

        if symbolic_model is None and not tree_models:
            return html.Div(
                "No persisted equivalent models are available for this bundle."
            )

        children = []
        if symbolic_model is not None:
            children.append(build_symbolic_panel(symbolic_model, services))
        if tree_models:
            requested_depth = next(iter(tree_models))
            children.append(build_tree_shell(sorted(tree_models), requested_depth))
        return html.Div(style={"display": "grid", "gap": "18px"}, children=children)

    @app.callback(
        Output(IDS.GLOBAL_EQUIVALENT_TREE_PANEL, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.GLOBAL_EQUIVALENT_DEPTH_TABS, "value"),
        Input(IDS.MODEL_REFRESH_TOKEN, "data"),
    )
    def render_tree_panel(model_id, selected_depth, _refresh_token):
        if not model_id:
            return html.Div()
        try:
            tree_models = services.equivalent_models_service.load_surrogates(model_id)
        except Exception as exc:  # pragma: no cover - defensive UI path
            return html.Div(str(exc), style={"color": "#8a1c1c"})
        if not tree_models:
            return html.Div("No persisted surrogate trees are available.")
        requested_depth = (
            int(selected_depth)
            if selected_depth is not None and int(selected_depth) in tree_models
            else next(iter(tree_models))
        )
        return build_tree_panel_content(
            tree_models[requested_depth],
            requested_depth,
            services,
        )

    @app.callback(
        Output(IDS.GLOBAL_EQUIVALENT_TREE_IMAGE, "style"),
        Input(IDS.GLOBAL_EQUIVALENT_ZOOM, "value"),
        prevent_initial_call=True,
    )
    def update_tree_zoom(zoom_percent):
        zoom = int(zoom_percent or 120)
        return {
            "width": f"{zoom}%",
            "maxWidth": "none",
            "display": "block",
        }
