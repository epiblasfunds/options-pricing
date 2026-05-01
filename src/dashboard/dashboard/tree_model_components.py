from dash import dcc
from dash import html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.tree_plots import feature_importance_figure
from src.dashboard.plots.tree_plots import fidelity_figure
from src.dashboard.plots.tree_plots import tree_png_base64
from src.dashboard.utils.feature_utils import replace_feature_names_in_text


def build_tree_shell(available_depths, requested_depth):
    return html.Div(
        style=_panel_style(),
        children=[
            html.H3("Decision-Tree Surrogates", style={"margin": "0 0 8px 0"}),
            html.P(
                "Depth-controlled surrogate trees fitted on the same explainability bundle. "
                "Use them to inspect explicit split logic and compare fidelity across depths.",
                style={"margin": "0 0 16px 0", "color": "#4a5a73"},
            ),
            dcc.Tabs(
                id=IDS.GLOBAL_EQUIVALENT_DEPTH_TABS,
                value=str(int(requested_depth)),
                children=[
                    dcc.Tab(label=f"Depth {int(depth)}", value=str(int(depth)))
                    for depth in available_depths
                ],
            ),
            html.Div(
                id=IDS.GLOBAL_EQUIVALENT_TREE_PANEL,
                style={"marginTop": "16px"},
            ),
        ],
    )


def build_tree_panel_content(result, requested_depth, services):
    image_src = f"data:image/png;base64,{tree_png_base64(result, services.feature_schema)}"
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
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(340px, 1fr))",
            "gap": "18px",
            "alignItems": "start",
        },
        children=[
            html.Div(
                style=_subcard_style(),
                children=[
                    html.H4(f"Tree Depth {requested_depth}", style={"margin": "0 0 8px 0"}),
                    html.P(
                        replace_feature_names_in_text(
                            result.interpretation,
                            services.feature_schema,
                        ),
                        style={"marginTop": "0"},
                    ),
                    html.Div(style=_metric_row_style(), children=metric_cards),
                    html.Div(
                        style=_summary_card_style(),
                        children=[
                            html.Div(
                                f"Effective depth: {result.tree_depth}",
                                style={"marginBottom": "6px"},
                            ),
                            html.Div(f"Leaves: {result.n_leaves}"),
                        ],
                    ),
                    html.H4("Feature Importance", style={"margin": "16px 0 8px 0"}),
                    dcc.Graph(
                        figure=feature_importance_figure(
                            result,
                            schema=services.feature_schema,
                        )
                    ),
                    html.H4("Surrogate Fidelity", style={"margin": "16px 0 8px 0"}),
                    dcc.Graph(figure=fidelity_figure(result)),
                ],
            ),
            html.Div(
                style=_subcard_style(),
                children=[
                    html.H4("Decision Tree", style={"margin": "0 0 8px 0"}),
                    html.Label("Tree Zoom"),
                    dcc.Slider(
                        id=IDS.GLOBAL_EQUIVALENT_ZOOM,
                        min=80,
                        max=220,
                        step=10,
                        value=120,
                        marks={80: "80%", 120: "120%", 160: "160%", 220: "220%"},
                    ),
                    html.Div(
                        style=_tree_viewport_style(),
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
                    html.Pre(
                        replace_feature_names_in_text(
                            result.text_rules,
                            services.feature_schema,
                        ),
                        style=_rules_style(),
                    ),
                ],
            ),
        ],
    )


def _panel_style():
    return {
        "padding": "18px",
        "border": "1px solid rgba(23,48,79,0.10)",
        "borderRadius": "16px",
        "background": "#fbfdff",
    }


def _subcard_style():
    return {
        "padding": "16px",
        "border": "1px solid rgba(23,48,79,0.10)",
        "borderRadius": "14px",
        "background": "#ffffff",
    }


def _metric_row_style():
    return {
        "display": "flex",
        "gap": "10px",
        "flexWrap": "wrap",
        "marginBottom": "12px",
    }


def _summary_card_style():
    return {
        "padding": "12px 14px",
        "border": "1px solid rgba(23,48,79,0.10)",
        "borderRadius": "12px",
        "background": "#fbfdff",
    }


def _tree_viewport_style():
    return {
        "overflow": "auto",
        "maxHeight": "820px",
        "borderRadius": "12px",
        "border": "1px solid rgba(23,48,79,0.10)",
        "background": "white",
        "marginBottom": "12px",
    }


def _rules_style():
    return {
        "whiteSpace": "pre-wrap",
        "background": "#102542",
        "color": "#f8fbff",
        "padding": "16px",
        "borderRadius": "14px",
        "fontSize": "0.96rem",
        "overflowX": "auto",
    }
