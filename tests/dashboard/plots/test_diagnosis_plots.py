import pandas as pd
from plotly.io import to_json

from src.dashboard.plots.diagnosis_plots import error_heatmap_figure
from src.dashboard.plots.diagnosis_plots import real_vs_predicted_figure
from src.dashboard.plots.diagnosis_plots import residual_by_feature_figure


def test_error_heatmap_figure_serializes_interval_bins():
    error_heatmap = pd.DataFrame(
        {
            "moneyness_bin": [
                pd.Interval(0.9, 1.0, closed="right"),
                pd.Interval(1.0, 1.1, closed="right"),
            ],
            "maturity_bin": [
                pd.Interval(10.0, 20.0, closed="right"),
                pd.Interval(20.0, 30.0, closed="right"),
            ],
            "AbsoluteError": [0.1, 0.2],
        }
    )

    figure = error_heatmap_figure(error_heatmap)
    payload = to_json(figure)

    assert "(0.9, 1.0]" in payload
    assert "(10.0, 20.0]" in payload


def test_diagnosis_figures_render_core_series():
    frame = pd.DataFrame(
        {
            "ImpliedVolatility": [0.1, 0.2],
            "PredictedVolatility": [0.11, 0.19],
            "Residual": [-0.01, 0.01],
            "Rate": [0.02, 0.03],
            "OptionType": ["C", "P"],
        }
    )

    scatter = real_vs_predicted_figure(frame)
    residuals = residual_by_feature_figure(frame, "Rate", "Residuals by Rate")

    assert len(scatter.data) >= 2
    assert residuals.layout.yaxis.title.text == "Residual"
