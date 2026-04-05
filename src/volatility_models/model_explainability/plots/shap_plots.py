"""Render SHAP native plots as embeddable images."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from src.volatility_models.model_explainability.services.global_explainability import (
    ShapExplanationResult,
)
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.utils.feature_utils import (
    display_feature_label,
)


def beeswarm_image(result: ShapExplanationResult, schema: FeatureSchema) -> str:
    explanation = _rename_explanation(result, schema)
    return _render_plot(
        lambda: shap.plots.beeswarm(explanation, max_display=12, show=False),
        figsize=(11, 6.5),
    )


def bar_image(result: ShapExplanationResult, schema: FeatureSchema) -> str:
    explanation = _rename_explanation(result, schema)
    return _render_plot(
        lambda: shap.plots.bar(explanation, max_display=12, show=False),
        figsize=(10, 6),
    )


def dependence_image(
    result: ShapExplanationResult,
    feature_name: str,
    schema: FeatureSchema,
) -> str:
    explanation = _rename_explanation(result, schema)
    feature_label = display_feature_label(feature_name, schema)
    return _render_plot(
        lambda: shap.plots.scatter(explanation[:, feature_label], show=False),
        figsize=(9.5, 6),
    )


def heatmap_image(result: ShapExplanationResult, schema: FeatureSchema) -> str:
    explanation = _rename_explanation(result, schema)
    return _render_plot(
        lambda: shap.plots.heatmap(explanation, max_display=12, show=False),
        figsize=(11, 6),
    )


def waterfall_image(
    result: ShapExplanationResult,
    row_index: int,
    schema: FeatureSchema,
) -> str:
    explanation = _rename_explanation(result, schema)
    row_position = result.explain_frame.index.get_loc(row_index)
    return _render_plot(
        lambda: shap.plots.waterfall(
            explanation[row_position],
            max_display=12,
            show=False,
        ),
        figsize=(10, 6),
    )


def _rename_explanation(
    result: ShapExplanationResult,
    schema: FeatureSchema,
) -> shap.Explanation:
    feature_names = [display_feature_label(name, schema) for name in result.feature_names]
    return shap.Explanation(
        values=result.explanation.values,
        base_values=result.explanation.base_values,
        data=result.explanation.data,
        display_data=result.explanation.display_data,
        instance_names=result.explanation.instance_names,
        feature_names=feature_names,
        output_names=result.explanation.output_names,
    )


def _figure_to_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _render_plot(plotter, figsize: tuple[float, float]) -> str:
    plt.close("all")
    with plt.rc_context({"figure.figsize": figsize}):
        plotter()
        figure = plt.gcf()
    return _figure_to_data_uri(figure)
