"""Callbacks for equivalent explainable decision trees."""

from __future__ import annotations

from dash import Input, Output, dcc, html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.tree_plots import (
    feature_importance_figure,
    fidelity_figure,
    tree_png_base64,
)


def register_equivalent_callbacks(app, services) -> None:
    """Register equivalent-model callbacks."""

    @app.callback(
        Output(IDS.GLOBAL_EQUIVALENT_CONTENT, "children"),
        Input(IDS.MODEL_SELECTOR, "value"),
        Input(IDS.GLOBAL_EQUIVALENT_DEPTH_TABS, "value"),
    )
    def render_equivalent_models(model_id, selected_depth):
        if not model_id:
            return html.Div("Select an explainable-model bundle.")
        try:
            results = services.equivalent_models_service.load_surrogates(model_id)
        except Exception as exc:  # pragma: no cover - defensive UI path
            return html.Div(str(exc), style={"color": "#8a1c1c"})

        if not results:
            return html.Div("No persisted surrogate trees are available for this model.")

        requested_depth = (
            int(selected_depth)
            if selected_depth is not None and int(selected_depth) in results
            else next(iter(results))
        )
        result = results[requested_depth]
        image_src = f"data:image/png;base64,{tree_png_base64(result)}"
        metric_cards = [
            html.Div(
                style={
                    "padding": "12px 14px",
                    "background": "#eef4fb",
                    "borderRadius": "12px",
                    "minWidth": "120px",
                },
                children=[
                    html.Div(
                        services.metrics_registry.get(name).label,
                        style={"fontWeight": "700", "marginBottom": "4px"},
                    ),
                    html.Div(services.metrics_registry.format_metric(name, value)),
                ],
            )
            for name, value in result.metrics.items()
        ]
        return html.Div(
            children=[
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(340px, 1fr))",
                        "gap": "18px",
                        "alignItems": "start",
                    },
                    children=[
                        html.Div(
                            style={
                                "padding": "16px",
                                "border": "1px solid rgba(23,48,79,0.10)",
                                "borderRadius": "14px",
                                "background": "#fbfdff",
                            },
                            children=[
                                html.H4(
                                    f"Tree Depth {requested_depth}",
                                    style={"margin": "0 0 8px 0"},
                                ),
                                html.P(result.interpretation, style={"marginTop": "0"}),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "10px",
                                        "flexWrap": "wrap",
                                        "marginBottom": "12px",
                                    },
                                    children=metric_cards,
                                ),
                                html.Div(
                                    style={
                                        "padding": "12px 14px",
                                        "border": "1px solid rgba(23,48,79,0.10)",
                                        "borderRadius": "12px",
                                        "background": "#fbfdff",
                                    },
                                    children=[
                                        html.Div(
                                            f"Effective depth: {result.tree_depth}",
                                            style={"marginBottom": "6px"},
                                        ),
                                        html.Div(f"Leaves: {result.n_leaves}"),
                                    ],
                                ),
                                html.H4("Feature Importance", style={"margin": "16px 0 8px 0"}),
                                html.P(
                                    "Relative importance of each surrogate split feature in the selected decision tree.",
                                    style={"marginTop": "0", "color": "#4a5a73"},
                                ),
                                dcc.Graph(figure=feature_importance_figure(result)),
                                html.H4("Surrogate Fidelity", style={"margin": "16px 0 8px 0"}),
                                html.P(
                                    "Agreement between the neural model prediction and the surrogate tree prediction on the sampled points.",
                                    style={"marginTop": "0", "color": "#4a5a73"},
                                ),
                                dcc.Graph(figure=fidelity_figure(result)),
                            ]
                        ),
                        html.Div(
                            style={
                                "padding": "16px",
                                "border": "1px solid rgba(23,48,79,0.10)",
                                "borderRadius": "14px",
                                "background": "#ffffff",
                            },
                            children=[
                                html.H4("Decision Tree", style={"margin": "0 0 8px 0"}),
                                html.P(
                                    "Scrollable surrogate tree rendering. Increase zoom to inspect deep branches without compressing the node layout.",
                                    style={"marginTop": "0", "color": "#4a5a73"},
                                ),
                                html.Label("Tree Zoom"),
                                dcc.Slider(
                                    id=IDS.GLOBAL_EQUIVALENT_ZOOM,
                                    min=80,
                                    max=220,
                                    step=10,
                                    value=120,
                                    marks={
                                        80: "80%",
                                        120: "120%",
                                        160: "160%",
                                        220: "220%",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "overflow": "auto",
                                        "maxHeight": "820px",
                                        "borderRadius": "12px",
                                        "border": "1px solid rgba(23,48,79,0.10)",
                                        "background": "white",
                                        "marginBottom": "12px",
                                    },
                                    children=[
                                        html.Img(
                                            id=IDS.GLOBAL_EQUIVALENT_TREE_IMAGE,
                                            src=image_src,
                                            style={
                                                "width": "120%",
                                                "maxWidth": "none",
                                                "display": "block",
                                            },
                                        )
                                    ],
                                ),
                                html.H4("Decision Rules", style={"margin": "0 0 8px 0"}),
                                html.P(
                                    "Text export of the same tree to read split conditions sequentially.",
                                    style={"marginTop": "0", "color": "#4a5a73"},
                                ),
                                html.Pre(
                                    result.text_rules,
                                    style={
                                        "whiteSpace": "pre-wrap",
                                        "background": "#0f1728",
                                        "color": "#e8eef8",
                                        "padding": "14px",
                                        "borderRadius": "12px",
                                        "overflowX": "auto",
                                    },
                                ),
                            ]
                        ),
                    ],
                )
            ]
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

