"""Dashboard layout construction."""

from dash import dcc, html

from src.dashboard.dashboard.ids import IDS
from src.dashboard.dashboard.styles import (
    BUTTON_STYLE,
    CARD_STYLE,
    CONTROL_ROW_STYLE,
    HEADER_STYLE,
    HELP_TEXT_STYLE,
    IMAGE_STYLE,
    INLINE_BUTTON_STYLE,
    INFO_ICON_STYLE,
    PAGE_STYLE,
    SECTION_CONTROL_CARD_STYLE,
    SECTION_PANEL_HEADER_STYLE,
    SECTION_PANEL_INTRO_STYLE,
    SECTION_PANEL_STYLE,
    SECTION_TITLE_STYLE,
    SUBCARD_STYLE,
    TITLE_WITH_INFO_STYLE,
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


def _bounded_graph(
    graph_id: str,
    height: str = "520px",
    max_width: str = "860px",
):
    return html.Div(
        style={"maxWidth": max_width, "width": "100%", "margin": "0 auto"},
        children=[dcc.Graph(id=graph_id, style={"height": height})],
    )


def _section_title_with_info(title: str, info_text: str, level: int = 4):
    header_class = getattr(html, f"H{level}")
    title_style = {"margin": "0"} if level == 4 else SECTION_TITLE_STYLE
    return html.Div(
        style=TITLE_WITH_INFO_STYLE,
        children=[
            header_class(title, style=title_style),
            html.Span(
                "i",
                title=info_text,
                style=INFO_ICON_STYLE,
                **{"aria-label": f"Information about {title}"},
            ),
        ],
    )


def _compact_control_box(title: str, control, help_text: str):
    return html.Div(
        style={
            "padding": "12px 14px",
            "borderRadius": "12px",
            "border": "1px solid rgba(33,75,122,0.12)",
            "background": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
            "boxShadow": "0 6px 16px rgba(22,40,68,0.04)",
        },
        children=[
            html.Label(
                title,
                style={
                    "display": "block",
                    "marginBottom": "6px",
                    "fontSize": "0.82rem",
                    "fontWeight": "800",
                    "textTransform": "uppercase",
                    "color": "#17304f",
                },
            ),
            control,
            html.P(help_text, style=HELP_TEXT_STYLE),
        ],
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
                                (
                                    "Explore how the model reacts to changes in "
                                    "moneyness, maturity and local perturbations "
                                    "around representative samples."
                                ),
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=SECTION_PANEL_STYLE,
                                children=[
                                    html.Div(
                                        style=SECTION_PANEL_HEADER_STYLE,
                                        children=[
                                            html.Div(
                                                children=[
                                                    _section_title_with_info(
                                                        "Surface Analysis",
                                                        (
                                                            "This block groups "
                                                            "every visualization "
                                                            "driven by the "
                                                            "selected anchor "
                                                            "observation. "
                                                            "Changing the sample "
                                                            "updates the "
                                                            "volatility heatmap, "
                                                            "the 3D surface, the "
                                                            "smile slices, the "
                                                            "term structure "
                                                            "slices and the "
                                                            "associated financial "
                                                            "checks."
                                                        ),
                                                        level=3,
                                                    ),
                                                    html.P(
                                                        (
                                                            "Analyse one "
                                                            "representative "
                                                            "sample and inspect "
                                                            "locally the full "
                                                            "volatility surface "
                                                            "built around it, "
                                                            "including its smile "
                                                            "and term-structure "
                                                            "cross-sections."
                                                        ),
                                                        style=SECTION_PANEL_INTRO_STYLE,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                style=SECTION_CONTROL_CARD_STYLE,
                                                children=[
                                                    html.Label("Sample"),
                                                    dcc.Dropdown(
                                                        id=IDS.BEHAVIOUR_ANCHOR_INDEX
                                                    ),
                                                    html.P(
                                                        (
                                                            "Reference "
                                                            "observation used "
                                                            "to build the "
                                                            "surface views in "
                                                            "this block."
                                                        ),
                                                        style=HELP_TEXT_STYLE,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": (
                                                "repeat(auto-fit, "
                                                "minmax(560px, 1fr))"
                                            ),
                                            "gap": "16px",
                                        },
                                        children=[
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Surface Heatmap",
                                                        (
                                                            "2D volatility "
                                                            "surface predicted "
                                                            "by the selected "
                                                            "model on a grid of "
                                                            "moneyness and time "
                                                            "to expiration. Use "
                                                            "it to identify "
                                                            "level, skew and "
                                                            "curvature patterns. "
                                                            "The axes and color "
                                                            "scale are aligned "
                                                            "with the 3D surface "
                                                            "and the smile and "
                                                            "term-structure "
                                                            "slices so visual "
                                                            "comparisons are "
                                                            "directly consistent."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.SURFACE_HEATMAP,
                                                        height="520px",
                                                        max_width="100%",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Surface Slice",
                                                        (
                                                            "Three-dimensional "
                                                            "rendering of the "
                                                            "same volatility "
                                                            "surface around the "
                                                            "selected sample. "
                                                            "This view is useful "
                                                            "for assessing "
                                                            "smoothness, slope "
                                                            "changes and "
                                                            "interaction effects "
                                                            "between moneyness "
                                                            "and maturity while "
                                                            "preserving the exact "
                                                            "same volatility "
                                                            "scale used by the "
                                                            "heatmap and related "
                                                            "slices."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.LOCAL_SURFACE_GRAPH,
                                                        height="520px",
                                                        max_width="100%",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Smile Curve",
                                                        (
                                                            "Cross-sections of "
                                                            "the surface at fixed "
                                                            "maturities. Each line "
                                                            "shows how predicted "
                                                            "volatility changes "
                                                            "across moneyness, "
                                                            "making it easier to "
                                                            "inspect smile shape, "
                                                            "skew asymmetry and "
                                                            "differences in "
                                                            "volatility level "
                                                            "between expiries."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.SMILE_GRAPH,
                                                        height="460px",
                                                        max_width="100%",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Term Structure",
                                                        (
                                                            "Cross-sections of "
                                                            "the surface at fixed "
                                                            "moneyness levels. "
                                                            "This chart shows how "
                                                            "predicted volatility "
                                                            "evolves with time to "
                                                            "expiration and helps "
                                                            "detect "
                                                            "maturity-dependent "
                                                            "regimes, slope "
                                                            "changes and "
                                                            "non-linear "
                                                            "term-structure "
                                                            "behaviour."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.TERM_GRAPH,
                                                        height="460px",
                                                        max_width="100%",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            _section_title_with_info(
                                                "Surface Checks",
                                                (
                                                    "Automatic heuristic checks "
                                                    "applied to the generated "
                                                    "surface to detect abrupt "
                                                    "smile jumps, maturity "
                                                    "discontinuities or other "
                                                    "signs of financially "
                                                    "implausible local "
                                                    "behaviour."
                                                ),
                                            ),
                                            html.Div(id=IDS.BEHAVIOUR_WARNINGS),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style=SECTION_PANEL_STYLE,
                                children=[
                                    html.Div(
                                        style=SECTION_PANEL_HEADER_STYLE,
                                        children=[
                                            html.Div(
                                                children=[
                                                    _section_title_with_info(
                                                        "Feature Response Analysis",
                                                        (
                                                            "This block "
                                                            "contains only the "
                                                            "feature-response "
                                                            "diagnostics driven "
                                                            "by the ICE/ALE "
                                                            "feature selector. "
                                                            "Changing the "
                                                            "feature here "
                                                            "updates the ICE and "
                                                            "ALE views without "
                                                            "affecting the "
                                                            "surface charts."
                                                        ),
                                                        level=3,
                                                    ),
                                                    html.P(
                                                        (
                                                            "Study how the model "
                                                            "reacts when one "
                                                            "explanatory variable "
                                                            "changes, both at the "
                                                            "individual-observation "
                                                            "level and on average."
                                                        ),
                                                        style=SECTION_PANEL_INTRO_STYLE,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                style=SECTION_CONTROL_CARD_STYLE,
                                                children=[
                                                    html.Label("ICE/ALE Feature"),
                                                    dcc.Dropdown(
                                                        id=IDS.BEHAVIOUR_ICE_FEATURE
                                                    ),
                                                    html.P(
                                                        (
                                                            "Feature perturbed "
                                                            "for the "
                                                            "response-analysis "
                                                            "charts in this "
                                                            "block."
                                                        ),
                                                        style=HELP_TEXT_STYLE,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                                            "gap": "16px",
                                        },
                                        children=[
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "ICE",
                                                        (
                                                            "Individual "
                                                            "Conditional "
                                                            "Expectation curves "
                                                            "for the selected "
                                                            "feature. Each line "
                                                            "isolates one "
                                                            "observation and "
                                                            "shows how the "
                                                            "prediction changes "
                                                            "when that feature is "
                                                            "perturbed while the "
                                                            "remaining inputs stay "
                                                            "fixed, which reveals "
                                                            "local heterogeneity "
                                                            "and potential "
                                                            "interaction effects "
                                                            "hidden by aggregate "
                                                            "averages."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.ICE_GRAPH, height="460px"
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "ALE",
                                                        (
                                                            "Accumulated Local "
                                                            "Effects for the "
                                                            "selected feature. "
                                                            "This chart summarizes "
                                                            "the average local "
                                                            "sensitivity of the "
                                                            "model while "
                                                            "respecting the "
                                                            "observed data "
                                                            "distribution, which "
                                                            "makes the "
                                                            "interpretation more "
                                                            "robust when "
                                                            "explanatory "
                                                            "variables are "
                                                            "correlated."
                                                        ),
                                                    ),
                                                    _bounded_graph(
                                                        IDS.ALE_GRAPH, height="460px"
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
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
                                (
                                    "Understand which inputs drive "
                                    "the volatility model globally and inspect "
                                    "the precomputed surrogate trees that "
                                    "summarize its logic."
                                ),
                                style=HELP_TEXT_STYLE,
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
                                            _section_title_with_info(
                                                "Summary",
                                                (
                                                    "SHAP beeswarm summary "
                                                    "across the sampled "
                                                    "observations. It shows both "
                                                    "the direction and "
                                                    "dispersion of local feature "
                                                    "contributions, which helps "
                                                    "identify the most "
                                                    "influential drivers of the "
                                                    "model and how their impact "
                                                    "varies across the dataset."
                                                ),
                                            ),
                                            _bounded_image(IDS.GLOBAL_SUMMARY_GRAPH),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            _section_title_with_info(
                                                "Feature Importance",
                                                (
                                                    "Ranking of features by mean "
                                                    "absolute SHAP value. This "
                                                    "offers a global view of "
                                                    "which transformed inputs "
                                                    "contribute most strongly to "
                                                    "the model predictions on "
                                                    "average."
                                                ),
                                            ),
                                            _bounded_image(IDS.GLOBAL_BAR_GRAPH),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(640px, 1fr))",
                                    "gap": "16px",
                                },
                                children=[
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            _section_title_with_info(
                                                "Dependence",
                                                (
                                                    "SHAP dependence plot for "
                                                    "the selected transformed "
                                                    "feature. It relates feature "
                                                    "value to local contribution "
                                                    "and is useful for spotting "
                                                    "non-linear response "
                                                    "patterns, threshold effects "
                                                    "and interactions with other "
                                                    "predictors."
                                                ),
                                            ),
                                            html.Div(
                                                style={"marginBottom": "12px"},
                                                children=[
                                                    html.Label("Dependence Feature"),
                                                    dcc.Dropdown(
                                                        id=IDS.GLOBAL_DEPENDENCE_FEATURE
                                                    ),
                                                    html.P(
                                                        (
                                                            "Choose the "
                                                            "transformed feature "
                                                            "to inspect its SHAP "
                                                            "dependence profile."
                                                        ),
                                                        style=HELP_TEXT_STYLE,
                                                    ),
                                                ],
                                            ),
                                            _bounded_image(IDS.GLOBAL_DEPENDENCE_GRAPH),
                                        ],
                                    ),
                                    html.Div(
                                        style=SUBCARD_STYLE,
                                        children=[
                                            _section_title_with_info(
                                                "Heatmap",
                                                (
                                                    "Observation-level SHAP "
                                                    "heatmap used to compare "
                                                    "attribution patterns across "
                                                    "the sampled dataset. It "
                                                    "highlights clusters of "
                                                    "observations that share "
                                                    "similar explanatory "
                                                    "structure and regions where "
                                                    "the model relies on "
                                                    "distinct combinations of "
                                                    "features."
                                                ),
                                            ),
                                            _bounded_image(
                                                IDS.GLOBAL_INTERACTION_GRAPH
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id=IDS.GLOBAL_NOTE, style=HELP_TEXT_STYLE),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    html.H3(
                                        "Equivalent Explainable Models",
                                        style=SECTION_TITLE_STYLE,
                                    ),
                                    html.P(
                                        (
                                            "Precomputed symbolic and tree "
                                            "surrogates for the selected model. "
                                            "The symbolic regressor provides a "
                                            "compact closed-form approximation, "
                                            "and the tree tabs let you inspect "
                                            "explicit split logic by depth."
                                        ),
                                        style={
                                            "margin": "0 0 14px 0",
                                            "opacity": "0.78",
                                        },
                                    ),
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
                                (
                                    "Inspect one concrete option sample, "
                                    "explain its prediction locally and compare "
                                    "it with similar historical observations."
                                ),
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style={
                                    "maxWidth": "320px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    _compact_control_box(
                                        "Mode",
                                        dcc.RadioItems(
                                            id=IDS.SAMPLE_MODE,
                                            options=[
                                                {
                                                    "label": "Dataset sample",
                                                    "value": "dataset",
                                                },
                                                {
                                                    "label": "Manual input",
                                                    "value": "manual",
                                                },
                                            ],
                                            value="dataset",
                                            inline=True,
                                        ),
                                        (
                                            "Choose between an existing dataset row "
                                            "or a manually defined input sample."
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(
                                id=IDS.SAMPLE_INDEX_CONTAINER,
                                style={"marginTop": "14px"},
                                children=[
                                    html.Label("Dataset Sample"),
                                    dcc.Dropdown(
                                        id=IDS.SAMPLE_INDEX,
                                        className="single-line-dropdown",
                                    ),
                                    html.P(
                                        "Observation index used when the dataset mode is selected.",
                                        style=HELP_TEXT_STYLE,
                                    ),
                                ],
                            ),
                            html.Div(
                                id=IDS.SAMPLE_FEATURE_PREVIEW,
                                style={"marginTop": "14px"},
                            ),
                            html.Div(id=IDS.SAMPLE_MANUAL_FORM),
                            html.Div(
                                style={"marginTop": "16px", "display": "flex"},
                                children=[
                                    html.Button(
                                        "Analyze Sample",
                                        id=IDS.SAMPLE_RUN_BUTTON,
                                        style=INLINE_BUTTON_STYLE,
                                    )
                                ],
                            ),
                            html.Div(id=IDS.SAMPLE_OUTPUT, style={"marginTop": "12px"}),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    _section_title_with_info(
                                        "Local SHAP Waterfall",
                                        (
                                            "Waterfall decomposition of one "
                                            "prediction into baseline value plus "
                                            "signed feature contributions. It "
                                            "explains, in order of magnitude, "
                                            "which inputs pushed the final "
                                            "volatility estimate upward or "
                                            "downward for the selected sample."
                                        ),
                                    ),
                                    _bounded_image(IDS.SAMPLE_WATERFALL),
                                ],
                            ),
                            _section_title_with_info(
                                "Nearest Neighbours",
                                (
                                    "Closest historical observations to the "
                                    "selected sample in the explainability "
                                    "feature space. These rows provide local "
                                    "context and help assess whether the "
                                    "explanation is supported by genuinely "
                                    "similar cases from the dataset."
                                ),
                            ),
                            html.Div(id=IDS.SAMPLE_NEIGHBORS),
                            _section_title_with_info(
                                "Neighbour Distance Comparison",
                                (
                                    "Distance profile of the retrieved "
                                    "neighbours relative to the selected "
                                    "sample. Lower values indicate stronger "
                                    "local support, while larger gaps may "
                                    "suggest the sample lies in a sparse or less "
                                    "represented region of the feature space."
                                ),
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
                                (
                                    "Review aggregate predictive quality, "
                                    "residual structure and error concentration "
                                    "across key market dimensions."
                                ),
                                style=HELP_TEXT_STYLE,
                            ),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    _section_title_with_info(
                                        "Performance Summary",
                                        (
                                            "Headline diagnostic metrics for "
                                            "the selected model over the "
                                            "sampled evaluation set. This panel "
                                            "provides a compact overview of "
                                            "average predictive accuracy before "
                                            "drilling down into residual "
                                            "structure and localized error "
                                            "concentration."
                                        ),
                                    ),
                                    html.Div(id=IDS.DIAGNOSIS_METRICS),
                                ],
                            ),
                            html.Div(
                                style={"display": "grid", "gap": "16px"},
                                children=[
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(640px, 1fr))",
                                            "gap": "16px",
                                        },
                                        children=[
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Predicted vs Actual",
                                                        (
                                                            "Scatter comparison "
                                                            "between observed "
                                                            "implied volatility "
                                                            "and model "
                                                            "prediction. The "
                                                            "closer the cloud is "
                                                            "to the diagonal, the "
                                                            "more accurate and "
                                                            "better calibrated "
                                                            "the model is across "
                                                            "the evaluated "
                                                            "observations."
                                                        ),
                                                    ),
                                                    dcc.Graph(id=IDS.DIAGNOSIS_SCATTER),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Residual Heatmap",
                                                        (
                                                            "Heatmap of average "
                                                            "absolute error "
                                                            "across the joint "
                                                            "moneyness and "
                                                            "maturity grid. It is "
                                                            "designed to reveal "
                                                            "where model accuracy "
                                                            "deteriorates "
                                                            "systematically on "
                                                            "the surface and "
                                                            "whether that "
                                                            "deterioration is "
                                                            "linked to specific "
                                                            "smile or "
                                                            "term-structure "
                                                            "regimes."
                                                        ),
                                                    ),
                                                    dcc.Graph(id=IDS.DIAGNOSIS_HEATMAP),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(640px, 1fr))",
                                            "gap": "16px",
                                        },
                                        children=[
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Error by Moneyness",
                                                        (
                                                            "Residual pattern "
                                                            "across moneyness. "
                                                            "This view helps "
                                                            "identify whether the "
                                                            "model "
                                                            "systematically "
                                                            "overestimates or "
                                                            "underestimates "
                                                            "volatility in "
                                                            "in-the-money, "
                                                            "at-the-money or "
                                                            "out-of-the-money "
                                                            "regions, which is "
                                                            "particularly "
                                                            "relevant for smile "
                                                            "diagnostics."
                                                        ),
                                                    ),
                                                    dcc.Graph(
                                                        id=IDS.DIAGNOSIS_MONEYNESS
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style=SUBCARD_STYLE,
                                                children=[
                                                    _section_title_with_info(
                                                        "Error by Maturity",
                                                        (
                                                            "Residual pattern "
                                                            "across time to "
                                                            "expiration. It helps "
                                                            "detect short-end "
                                                            "versus long-end "
                                                            "calibration issues "
                                                            "and broader "
                                                            "term-structure "
                                                            "biases in the model "
                                                            "predictions."
                                                        ),
                                                    ),
                                                    dcc.Graph(
                                                        id=IDS.DIAGNOSIS_MATURITY
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style=SUBCARD_STYLE,
                                children=[
                                    _section_title_with_info(
                                        "Financial Warnings",
                                        (
                                            "Consistency messages derived from "
                                            "the generated local volatility "
                                            "surface. They summarize whether the "
                                            "heuristic validation detected "
                                            "abrupt local discontinuities or "
                                            "other behaviour worth reviewing "
                                            "before trusting the surface "
                                            "economically."
                                        ),
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
                    html.H1(
                        "Volatility Model Explainability", style={"margin": "0 0 8px 0"}
                    ),
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
                                    dcc.Dropdown(
                                        id=IDS.MODEL_SELECTOR,
                                        placeholder="Select a Keras model",
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Button(
                                        "Refresh Model Catalog",
                                        id=IDS.MODEL_REFRESH_BUTTON,
                                        style=BUTTON_STYLE,
                                    )
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
