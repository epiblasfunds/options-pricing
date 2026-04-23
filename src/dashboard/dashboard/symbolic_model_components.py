from dash import dcc
from dash import html

from src.dashboard.plots.symbolic_plots import symbolic_formula_aliases
from src.dashboard.plots.symbolic_plots import symbolic_formula_image_src
from src.dashboard.plots.symbolic_plots import symbolic_expression_tree_figure
from src.dashboard.plots.symbolic_plots import symbolic_fidelity_figure
from src.dashboard.plots.symbolic_plots import symbolic_frontier_figure
from src.dashboard.utils.feature_utils import display_feature_label
from src.dashboard.utils.feature_utils import replace_feature_names_in_text


def build_symbolic_panel(symbolic_model, services):
    metric_cards = _metric_cards(symbolic_model.metrics, services)
    model_cards = [
        _metric_card("Complexity", str(symbolic_model.complexity)),
        _metric_card("Selection", symbolic_model.model_selection.title()),
        _metric_card("Used Features", str(len(symbolic_model.used_feature_names))),
    ]
    equation_rows = _candidate_rows(symbolic_model, services=services)
    formula_aliases = symbolic_formula_aliases(
        symbolic_model,
        schema=services.feature_schema,
    )
    return html.Div(
        style=_panel_style(),
        children=[
            html.H3("Symbolic Regressor", style={"margin": "0 0 8px 0"}),
            html.P(
                "Closed-form surrogate fitted over the engineered model inputs. "
                "It complements the tree panel by exposing a compact analytic approximation.",
                style={"margin": "0 0 16px 0", "color": "#4a5a73"},
            ),
            html.Div(
                style=_subcard_style(),
                children=[
                    html.Div(
                        style=_metric_row_style(),
                        children=model_cards + metric_cards,
                    ),
                    html.H4("Mathematical Formula", style={"margin": "0 0 8px 0"}),
                    html.Div(
                        style=_formula_style(),
                        children=[
                            html.Div(
                                style=_formula_canvas_style(),
                                children=[
                                    html.Img(
                                        src=symbolic_formula_image_src(
                                            symbolic_model,
                                            schema=services.feature_schema,
                                        ),
                                        style={
                                            "display": "block",
                                            "minWidth": "980px",
                                            "width": "100%",
                                        },
                                    )
                                ],
                            ),
                            html.Div(
                                style=_alias_grid_style(),
                                children=[
                                    html.Div(
                                        style=_alias_card_style(),
                                        children=[
                                            html.Code(alias, style=_alias_code_style()),
                                            html.Span(
                                                feature_name,
                                                style={"color": "#4a5a73"},
                                            ),
                                        ],
                                    )
                                    for alias, feature_name in formula_aliases
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style=_grid_style(),
                children=[
                    html.Div(
                        style=_subcard_style(),
                        children=[
                            html.H4("Equation Source", style={"margin": "0 0 8px 0"}),
                            html.Pre(
                                replace_feature_names_in_text(
                                    symbolic_model.equation,
                                    services.feature_schema,
                                ),
                                style=_dark_pre_style(),
                            ),
                            html.H4("Interpretation", style={"margin": "14px 0 8px 0"}),
                            html.P(
                                replace_feature_names_in_text(
                                    symbolic_model.interpretation,
                                    services.feature_schema,
                                ),
                                style={"margin": "0 0 12px 0"},
                            ),
                            html.H4("Active Inputs", style={"margin": "14px 0 8px 0"}),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                children=[
                                    html.Span(
                                        display_feature_label(
                                            feature_name,
                                            services.feature_schema,
                                        ),
                                        style=_chip_style(),
                                    )
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
                                        figure=symbolic_expression_tree_figure(symbolic_model)
                                    )
                                ],
                            ),
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
                                    html.H4("Candidate Equations", style={"margin": "0 0 10px 0"}),
                                    html.Table(
                                        style=_table_style(),
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


def _candidate_rows(symbolic_model, services=None):
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
                    html.Td(
                        replace_feature_names_in_text(
                            str(row["equation"]),
                            services.feature_schema,
                        )
                        if services is not None
                        else str(row["equation"]),
                        style=_table_cell_style(),
                    ),
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


def _panel_style():
    return {
        "padding": "20px",
        "borderRadius": "18px",
        "border": "1px solid rgba(23,48,79,0.12)",
        "background": "linear-gradient(135deg, #eef4fb 0%, #ffffff 48%, #f5f9ff 100%)",
        "boxShadow": "0 14px 32px rgba(18,36,62,0.08)",
    }


def _grid_style():
    return {
        "display": "grid",
        "gridTemplateColumns": "minmax(340px, 1.1fr) minmax(340px, 1fr)",
        "gap": "18px",
    }


def _metric_row_style():
    return {
        "display": "flex",
        "gap": "10px",
        "flexWrap": "wrap",
        "marginBottom": "12px",
    }


def _formula_style():
    return {
        "padding": "18px",
        "borderRadius": "14px",
        "background": "#f7fbff",
        "border": "1px solid rgba(23,48,79,0.10)",
        "minHeight": "84px",
        "display": "grid",
        "gap": "14px",
    }


def _subcard_style():
    return {
        "padding": "16px",
        "border": "1px solid rgba(23,48,79,0.10)",
        "borderRadius": "14px",
        "background": "#ffffff",
    }


def _formula_canvas_style():
    return {
        "overflowX": "auto",
        "borderRadius": "12px",
        "border": "1px solid rgba(23,48,79,0.08)",
        "background": "#ffffff",
        "padding": "10px 12px",
    }


def _alias_grid_style():
    return {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
        "gap": "10px",
    }


def _alias_card_style():
    return {
        "display": "flex",
        "alignItems": "center",
        "gap": "8px",
        "padding": "8px 10px",
        "borderRadius": "10px",
        "background": "#ffffff",
        "border": "1px solid rgba(23,48,79,0.08)",
    }


def _alias_code_style():
    return {
        "padding": "3px 8px",
        "borderRadius": "999px",
        "background": "#dce9f8",
        "color": "#17304f",
        "fontWeight": "700",
    }


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


def _chip_style():
    return {
        "padding": "6px 10px",
        "borderRadius": "999px",
        "background": "#dce9f8",
        "color": "#17304f",
        "fontWeight": "600",
    }


def _table_style():
    return {"width": "100%", "borderCollapse": "collapse", "fontSize": "0.92rem"}


def _table_header_style():
    return {
        "textAlign": "left",
        "padding": "10px 8px",
        "borderBottom": "1px solid rgba(23,48,79,0.12)",
    }


def _table_cell_style():
    return {
        "padding": "10px 8px",
        "verticalAlign": "top",
        "borderBottom": "1px solid rgba(23,48,79,0.08)",
    }
