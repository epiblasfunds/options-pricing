"""Dashboard layout construction."""

from __future__ import annotations

from dash import dcc, html

from src.volatility_models.model_explainability.dashboard.ids import IDS
from src.volatility_models.model_explainability.dashboard.styles import (
    BUTTON_STYLE,
    CARD_STYLE,
    CONTROL_ROW_STYLE,
    HEADER_STYLE,
    HELP_TEXT_STYLE,
    IMAGE_STYLE,
    PAGE_STYLE,
    SECTION_TITLE_STYLE,
    SUBCARD_STYLE,
)


def _bounded_image(image_id: str):
    return html.Div(
        style={"maxWidth": "860px", "margin": "0 auto"},
        children=[
            html.Img(
                id=image_id,
                style={
                    **IMAGE_STYLE,
                    "maxWidth": "860px",
                    "maxHeight": "520px",
                    "objectFit": "contain",
                    "margin": "0 auto",
                },
            )
        ],
    )


def _bounded_graph(graph_id: str, height: str = "520px"):
    return html.Div(
        style={"maxWidth": "860px", "margin": "0 auto"},
        children=[dcc.Graph(id=graph_id, style={"height": height})],
    )


def _behaviour_tab():
    return dcc.Tab(
        label="Behaviour And Surface",
        children=[
            dcc.Loading(
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.P(
                                "Explore how the model reacts to changes in moneyness, maturity and local perturbations around representative samples.",
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=CONTROL_ROW_STYLE,
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Anchor Sample"),
                                            dcc.Dropdown(id=IDS.BEHAVIOUR_ANCHOR_INDEX),
                                            html.P(
                                                "Reference observation used to build local smiles, terms and nearby surface slices.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.Label("ICE/ALE Feature"),
                                            dcc.Dropdown(id=IDS.BEHAVIOUR_ICE_FEATURE),
                                            html.P(
                                                "Feature perturbed for individual conditional expectation and accumulated local effects.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
                                    "gap": "16px",
                                },
                                children=[
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Surface Heatmap", style={"marginTop": "0"}),
                                            html.P(
                                                "2D volatility surface predicted by the model over a grid of moneyness and time to expiration.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.SURFACE_HEATMAP, height="520px"),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Local Surface Slice", style={"marginTop": "0"}),
                                            html.P(
                                                "3D local view of the predicted surface around the selected anchor sample.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.LOCAL_SURFACE_GRAPH, height="520px"),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Smile Curve", style={"marginTop": "0"}),
                                            html.P(
                                                "Cross-section showing how predicted volatility changes across moneyness at fixed maturities.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.SMILE_GRAPH, height="460px"),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Term Structure", style={"marginTop": "0"}),
                                            html.P(
                                                "Cross-section showing the evolution of predicted volatility across maturities.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.TERM_GRAPH, height="460px"),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("ICE", style={"marginTop": "0"}),
                                            html.P(
                                                "Individual Conditional Expectation curves for the selected feature across sampled observations.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.ICE_GRAPH, height="460px"),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("ALE", style={"marginTop": "0"}),
                                            html.P(
                                                "Accumulated Local Effects summarizing the average local sensitivity to the selected feature.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_graph(IDS.ALE_GRAPH, height="460px"),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H4("Surface Checks", style={"marginTop": "0"}),
                                    html.P(
                                        "Automatic financial consistency checks over the generated surface.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                    html.Div(id=IDS.BEHAVIOUR_WARNINGS),
                                ],
                            ),
                        ],
                    )
                ]
            )
        ],
    )


def _global_tab():
    return dcc.Tab(
        label="Global Explainability",
        children=[
            dcc.Loading(
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.P(
                                "Understand which transformed inputs drive the volatility model globally and inspect the precomputed surrogate trees that summarize its logic.",
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=CONTROL_ROW_STYLE,
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Dependence Feature"),
                                            dcc.Dropdown(id=IDS.GLOBAL_DEPENDENCE_FEATURE),
                                            html.P(
                                                "Choose the transformed feature to inspect its SHAP dependence profile.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.H3("SHAP Explainability", style=SECTION_TITLE_STYLE),
                            html.P(
                                "Native SHAP plots over the selected model and dataset sample.",
                                style={"margin": "0 0 14px 0", "opacity": "0.78"},
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                                    "gap": "16px",
                                    "marginBottom": "16px",
                                },
                                children=[
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Summary", style={"marginTop": "0"}),
                                            html.P(
                                                "Beeswarm plot showing the distribution of SHAP contributions for the most relevant features across the sampled observations.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_image(IDS.GLOBAL_SUMMARY_GRAPH),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Feature Importance", style={"marginTop": "0"}),
                                            html.P(
                                                "Mean absolute SHAP impact per feature, useful to rank the global drivers of the model.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_image(IDS.GLOBAL_BAR_GRAPH),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                                    "gap": "16px",
                                },
                                children=[
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Dependence", style={"marginTop": "0"}),
                                            html.P(
                                                "SHAP dependence plot for the selected feature, relating its value to its local contribution.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_image(IDS.GLOBAL_DEPENDENCE_GRAPH),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Heatmap", style={"marginTop": "0"}),
                                            html.P(
                                                "Observation-by-observation SHAP heatmap to compare attribution patterns across the sampled dataset.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            _bounded_image(IDS.GLOBAL_INTERACTION_GRAPH),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id=IDS.GLOBAL_NOTE, style=HELP_TEXT_STYLE),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H3("Equivalent Explainable Models", style=SECTION_TITLE_STYLE),
                                    html.P(
                                        "Precomputed surrogate trees for the selected model. Choose the depth to inspect the decision logic.",
                                        style={"margin": "0 0 14px 0", "opacity": "0.78"},
                                    ),
                                    dcc.Tabs(id=IDS.GLOBAL_EQUIVALENT_DEPTH_TABS),
                                    html.Div(
                                        id=IDS.GLOBAL_EQUIVALENT_CONTENT,
                                        style={"marginTop": "14px"},
                                    ),
                                ],
                            ),
                        ],
                    )
                ]
            )
        ],
    )


def _sample_tab():
    return dcc.Tab(
        label="Sample Explainability",
        children=[
            dcc.Loading(
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.P(
                                "Inspect one concrete option sample, explain its prediction locally and compare it with similar historical observations.",
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=CONTROL_ROW_STYLE,
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Mode"),
                                            dcc.RadioItems(
                                                id=IDS.SAMPLE_MODE,
                                                options=[
                                                    {"label": "Dataset sample", "value": "dataset"},
                                                    {"label": "Manual input", "value": "manual"},
                                                ],
                                                value="dataset",
                                                inline=True,
                                            ),
                                            html.P(
                                                "Choose between an existing dataset row or a manually defined input sample.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.Label("Dataset Sample"),
                                            dcc.Dropdown(id=IDS.SAMPLE_INDEX),
                                            html.P(
                                                "Observation index used when the dataset mode is selected.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(id=IDS.SAMPLE_MANUAL_FORM),
                            html.Button("Analyze Sample", id=IDS.SAMPLE_RUN_BUTTON),
                            html.Div(id=IDS.SAMPLE_OUTPUT, style={"marginTop": "12px"}),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H4("Local SHAP Waterfall", style={"marginTop": "0"}),
                                    html.P(
                                        "Waterfall decomposition of the selected prediction into baseline value plus feature contributions.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                    _bounded_image(IDS.SAMPLE_WATERFALL),
                                ],
                            ),
                            html.H4("Nearest Neighbours", style=SECTION_TITLE_STYLE),
                            html.P(
                                "Closest historical observations to the selected sample in the explainability feature space.",
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(id=IDS.SAMPLE_NEIGHBORS),
                            html.H4("Neighbour Distance Comparison", style=SECTION_TITLE_STYLE),
                            html.P(
                                "Distance profile of the retrieved neighbours to understand how isolated the sample is.",
                                style=HELP_TEXT_STYLE,
                            ),
                            dcc.Graph(id=IDS.SAMPLE_COMPARISON),
                        ],
                    )
                ]
            )
        ],
    )


def _diagnosis_tab():
    return dcc.Tab(
        label="Diagnosis",
        children=[
            dcc.Loading(
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.P(
                                "Review aggregate predictive quality, residual structure and error concentration across key market dimensions.",
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H4("Performance Summary", style={"marginTop": "0"}),
                                    html.P(
                                        "Headline error metrics for the selected model over the diagnosis sample.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                    html.Div(id=IDS.DIAGNOSIS_METRICS),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                                    "gap": "16px",
                                },
                                children=[
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Predicted vs Actual", style={"marginTop": "0"}),
                                            html.P(
                                                "Scatter plot comparing model outputs with observed implied volatilities.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            dcc.Graph(id=IDS.DIAGNOSIS_SCATTER),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Residual Heatmap", style={"marginTop": "0"}),
                                            html.P(
                                                "Heatmap of average residuals over the moneyness and maturity grid.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            dcc.Graph(id=IDS.DIAGNOSIS_HEATMAP),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Error by Moneyness", style={"marginTop": "0"}),
                                            html.P(
                                                "Residual pattern across moneyness to detect smile-related under or overestimation.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            dcc.Graph(id=IDS.DIAGNOSIS_MONEYNESS),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            html.H4("Error by Maturity", style={"marginTop": "0"}),
                                            html.P(
                                                "Residual pattern across time to expiration to detect term-structure biases.",
                                                style=HELP_TEXT_STYLE,
                                            ),
                                            dcc.Graph(id=IDS.DIAGNOSIS_MATURITY),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H4("Financial Warnings", style={"marginTop": "0"}),
                                    html.P(
                                        "Consistency checks derived from the generated local volatility surface.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                    html.Div(id=IDS.DIAGNOSIS_WARNINGS),
                                ],
                            ),
                        ],
                    )
                ]
            )
        ],
    )


def build_layout():
    """Build the top-level layout."""

    return html.Div(
        style=PAGE_STYLE,
        children=[
            html.Div(
                style=HEADER_STYLE,
                children=[
                    html.H1("Volatility Model Explainability", style={"margin": "0 0 8px 0"}),
                    html.P(
                        "Configuration-driven explainability for IBEX option volatility models.",
                        style={"margin": "0", "opacity": "0.88"},
                    ),
                ],
            ),
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        style=CONTROL_ROW_STYLE,
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Model"),
                                    dcc.Dropdown(
                                        id=IDS.MODEL_SELECTOR,
                                        placeholder="Select a Keras model",
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Refresh"),
                                    html.Button(
                                        "Refresh Model Catalog",
                                        id=IDS.MODEL_REFRESH_BUTTON,
                                        style=BUTTON_STYLE,
                                    ),
                                    html.P(
                                        "Reload the saved explainable-model bundles discovered on disk.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(id=IDS.MODEL_INFO),
                ],
            ),
            dcc.Tabs(
                children=[
                    _behaviour_tab(),
                    _global_tab(),
                    _sample_tab(),
                    _diagnosis_tab(),
                ]
            ),
        ],
    )
