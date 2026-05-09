import pandas as pd

from src.dashboard.plots.local_plots import neighbors_projection_figure
from src.dashboard.plots.local_plots import neighbors_projection_3d_figure


def test_neighbors_projection_figure_places_sample_at_origin_and_neighbors_as_points():
    figure = neighbors_projection_figure(
        pd.DataFrame(
            {
                "OptionType": ["C"],
                "StrikePrice": [100.0],
                "UnderlyingPrice": [101.0],
                "TimeToExpiration": [30.0],
                "Rate": [0.02],
                "distance": [0.0],
            }
        ),
        pd.DataFrame(
            {
                "index": [101, 205],
                "distance": [0.1, 0.3],
                "OptionType": ["C", "P"],
                "StrikePrice": [99.0, 103.0],
                "UnderlyingPrice": [100.0, 104.0],
                "TimeToExpiration": [28.0, 45.0],
                "Rate": [0.01, 0.03],
            }
        ),
        feature_names=["StrikePrice", "UnderlyingPrice"],
        center_label="Selected Sample",
    )

    assert figure.data[0].x[0] == 0.0
    assert figure.data[0].y[0] == 0.0
    assert figure.data[0].name == "Selected Sample"
    assert len(figure.data[1].x) == 2
    assert figure.data[1].name == "Neighbours"
    assert figure.data[0].marker.line.width == 0
    assert figure.data[1].marker.line.width == 0
    assert figure.data[1].marker.colorbar.orientation == "h"
    assert "Option Type" in figure.data[1].hovertemplate
    assert "Strike" in figure.data[1].hovertemplate
    assert figure.data[1].customdata[0][3] == "C"


def test_neighbors_projection_3d_figure_exposes_third_component():
    figure = neighbors_projection_3d_figure(
        pd.DataFrame(
            {
                "OptionType": ["C"],
                "StrikePrice": [100.0],
                "UnderlyingPrice": [101.0],
                "TimeToExpiration": [30.0],
                "Rate": [0.02],
                "distance": [0.0],
            }
        ),
        pd.DataFrame(
            {
                "index": [101, 205],
                "distance": [0.1, 0.3],
                "OptionType": ["C", "P"],
                "StrikePrice": [99.0, 103.0],
                "UnderlyingPrice": [100.0, 104.0],
                "TimeToExpiration": [28.0, 45.0],
                "Rate": [0.01, 0.03],
            }
        ),
        feature_names=["StrikePrice", "UnderlyingPrice", "TimeToExpiration"],
        center_label="Selected Sample",
    )

    assert figure.data[0].x[0] == 0.0
    assert figure.data[0].y[0] == 0.0
    assert figure.data[0].z[0] == 0.0
    assert len(figure.data[1].z) == 2
    assert figure.layout.legend.x > 1.0
    assert "Underlying" in figure.data[1].hovertemplate
