from types import SimpleNamespace

import pandas as pd

from src.dashboard.dashboard.callbacks.sample_explainability import (
    _projection_feature_names,
)


def test_projection_feature_names_use_processed_model_features():
    dashboard_model = SimpleNamespace(
        transformed_feature_names=[
            "TTEYears",
            "sqrtTTEYears",
            "isCall",
            "isPut",
        ],
        metadata={},
    )
    sample_frame = pd.DataFrame(
        {
            "StrikePrice": [100.0],
            "UnderlyingPrice": [101.0],
            "TTEYears": [0.08],
            "sqrtTTEYears": [0.28],
            "isCall": [1.0],
            "isPut": [0.0],
        }
    )
    neighbors = pd.DataFrame(
        {
            "StrikePrice": [99.0, 103.0],
            "UnderlyingPrice": [100.0, 104.0],
            "TTEYears": [0.07, 0.12],
            "sqrtTTEYears": [0.26, 0.35],
            "isCall": [1.0, 0.0],
            "isPut": [0.0, 1.0],
        }
    )

    feature_names = _projection_feature_names(
        dashboard_model,
        sample_frame,
        neighbors,
    )

    assert feature_names == ["TTEYears", "sqrtTTEYears", "isCall", "isPut"]
