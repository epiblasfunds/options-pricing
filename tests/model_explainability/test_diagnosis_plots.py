import pandas as pd
from plotly.io import to_json

from src.dashboard.plots.diagnosis_plots import error_heatmap_figure


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
