from dash import dcc
from dash import html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.plots.symbolic_plots import symbolic_fidelity_figure
from src.dashboard.plots.symbolic_plots import symbolic_frontier_figure
from src.dashboard.plots.tree_plots import feature_importance_figure
from src.dashboard.plots.tree_plots import fidelity_figure
from src.dashboard.plots.tree_plots import tree_png_base64


def build_symbolic_panel(symbolic_model, services):
    metric_cards = _metric_cards(symbolic_model.metrics, services)
    model_cards = [
        _metric_card("Complexity", str(symbolic_model.complexity)),
        _metric_card("Selection", symbolic_model.model_selection.title()),
        _metric_card("Used Features", str(len(symbolic_model.used_feature_names))),
    ]
    equation_rows = _candidate_rows(symbolic_model)
    return html.Div(
        style={
            "padding": "20px",
            "borderRadius": "18px",
            "border": "1px solid rgba(23,48,79,0.12)",
            "background": "linear-gradient(135deg, #eef4fb 0%, #ffffff 48%, #f5f9ff 100%)",
            "boxShadow": "0 14px 32px rgba(18,36,62,0.08)",
        },
        children=[
            html.H3("Symbolic Regressor", style={"margin": "0 0 8px 0"}),
            html.P(
                "Closed-form surrogate fitted with PySR over the engineered model inputs. "
                "It complements the tree panel by exposing a compact analytic approximation.",
                style={"margin": "0 0 16px 0", "color": "#4a5a73"},
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(340px, 1.1fr) minmax(340px, 1fr)",
                    "gap": "18px",
                },
                children=[
                    html.Div(
                        style=_subcard_style(),
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "marginBottom": "12px",
                                },
                                children=model_cards + metric_cards,
                            ),
                            html.H4("Selected Equation", style={"margin": "0 0 8px 0"}),
                            html.Pre(symbolic_model.equation, style=_dark_pre_style()),
                            html.H4("Interpretation", style={"margin": "14px 0 8px 0"}),
                            html.P(symbolic_model.interpretation, style={"margin": "0 0 12px 0"}),
                            html.H4("LaTeX Export", style={"margin": "14px 0 8px 0"}),
                            html.Pre(
                                symbolic_model.latex_expression,
                                style=_light_pre_style(),
                            ),
                            html.H4("Active Inputs", style={"margin": "14px 0 8px 0"}),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                children=[
                                    html.Span(feature_name, style=_chip_style())
                                    for feature_name in symbolic_model.used_feature_names
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gap": "16px"},
                        children=[
                            html.Div(
                                style=_subcard_style(),
                                children=[
                                    dcc.Graph(
                                        figure=symbolic_frontier_figure(symbolic_model)
                                    )
                                ],
                            ),
                            html.Div(
                                style=_subcard_style(),
                                children=[
                                    dcc.Graph(
                                        figure=symbolic_fidelity_figure(symbolic_model)
                                    )
                                ],
                            ),
                            html.Div(
                                style=_subcard_style(),
                                children=[
                                    html.H4(
                                        "Candidate Equations",
                                        style={"margin": "0 0 10px 0"},
                                    ),
                                    html.Table(
                                        style={
                                            "width": "100%",
                                            "borderCollapse": "collapse",
                                            "fontSize": "0.92rem",
                                        },
                                        children=equation_rows,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_tree_panel(result, requested_depth, services):
    image_src = f"data:image/png;base64,{tree_png_base64(result)}"
    metric_cards = _metric_cards(result.metrics, services)
    return html.Div(
        style={
            "padding": "18px",
            "border": "1px solid rgba(23,48,79,0.10)",
            "borderRadius": "16px",
            "background": "#fbfdff",
        },
        children=[
            html.H3("Decision-Tree Surrogates", style={"margin": "0 0 8px 0"}),
            html.P(
                "Depth-controlled surrogate trees fitted on the same explainability bundle. "
                "Use them to inspect explicit split logic and compare fidelity across depths.",
                style={"margin": "0 0 16px 0", "color": "#4a5a73"},
            ),
            html.Div(
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
                            html.H4(
                                "Feature Importance",
                                style={"margin": "16px 0 8px 0"},
                            ),
                            dcc.Graph(figure=feature_importance_figure(result)),
                            html.H4(
                                "Surrogate Fidelity",
                                style={"margin": "16px 0 8px 0"},
                            ),
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
                            html.Pre(result.text_rules, style=_dark_pre_style()),
                        ],
                    ),
                ],
            ),
        ],
    )


def _candidate_rows(symbolic_model):
    header = html.Tr(
        children=[
            html.Th("Complexity", style=_table_header_style()),
            html.Th("Loss", style=_table_header_style()),
            html.Th("Score", style=_table_header_style()),
            html.Th("Equation", style=_table_header_style()),
        ]
    )
    body = []
    for _, row in symbolic_model.candidate_equations.head(6).iterrows():
        body.append(
            html.Tr(
                style={"background": "#eef4fb" if bool(row["selected"]) else "transparent"},
                children=[
                    html.Td(str(int(row["complexity"])), style=_table_cell_style()),
                    html.Td(f"{float(row['loss']):.6f}", style=_table_cell_style()),
                    html.Td(f"{float(row.get('score', 0.0)):.4f}", style=_table_cell_style()),
                    html.Td(str(row["equation"]), style=_table_cell_style()),
                ],
            )
        )
    return [header, *body]


def _metric_cards(metrics, services):
    return [
        _metric_card(
            services.metrics_registry.get(name).label,
            services.metrics_registry.format_metric(name, value),
        )
        for name, value in metrics.items()
    ]


def _metric_card(label, value):
    return html.Div(
        style={
            "padding": "12px 14px",
            "background": "#eef4fb",
            "borderRadius": "12px",
            "minWidth": "120px",
        },
        children=[
            html.Div(label, style={"fontWeight": "700", "marginBottom": "4px"}),
            html.Div(value),
        ],
    )


def _subcard_style():
    return {"padding": "16px", "border": "1px solid rgba(23,48,79,0.10)", "borderRadius": "14px", "background": "#ffffff"}

def _dark_pre_style():
    return {
        "whiteSpace": "pre-wrap",
        "background": "#102542",
        "color": "#f8fbff",
        "padding": "16px",
        "borderRadius": "14px",
        "fontSize": "0.96rem",
        "overflowX": "auto",
    }

def _light_pre_style():
    return {"whiteSpace": "pre-wrap", "background": "#f6f9fd", "padding": "14px", "borderRadius": "12px", "border": "1px solid rgba(23,48,79,0.10)", "overflowX": "auto"}

def _chip_style():
    return {"padding": "6px 10px", "borderRadius": "999px", "background": "#dce9f8", "color": "#17304f", "fontWeight": "600"}

def _table_header_style():
    return {"textAlign": "left", "padding": "10px 8px", "borderBottom": "1px solid rgba(23,48,79,0.12)"}

def _table_cell_style():
    return {"padding": "10px 8px", "verticalAlign": "top", "borderBottom": "1px solid rgba(23,48,79,0.08)"}
