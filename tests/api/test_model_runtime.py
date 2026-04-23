from types import SimpleNamespace

import pandas as pd

from src.api.services.model_runtime import ApiModelService
from src.dashboard.plots.local_plots import neighbors_distance_figure
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES


class _LinearModel:
    def predict(self, matrix):
        return 0.01 * matrix[:, 0] + 0.02 * matrix[:, 1]


def _raw_frame():
    return pd.DataFrame(
        [
            {
                "ExecDatetime": "2026-04-22T10:00:00.000Z",
                "OptionType": "C",
                "Quantity": 1,
                "StrikePrice": 10000.0,
                "TradeType": "M",
                "UnderlyingLagMinutes": 0.0,
                "UnderlyingPrice": 10100.0,
                "TimeToExpiration": 30.0,
                "Rate": 0.02,
            },
            {
                "ExecDatetime": "2026-04-22T11:00:00.000Z",
                "OptionType": "P",
                "Quantity": 5,
                "StrikePrice": 9900.0,
                "TradeType": "M",
                "UnderlyingLagMinutes": 2.0,
                "UnderlyingPrice": 10000.0,
                "TimeToExpiration": 45.0,
                "Rate": 0.03,
            },
        ]
    )


def test_runtime_shap_explanation_uses_manual_row_not_stored_sample():
    raw = _raw_frame()
    runtime = SimpleNamespace(
        model=_LinearModel(),
        model_input_features=["TTEYears", "sqrtTTEYears"],
        scaler=None,
        is_keras=False,
    )
    dashboard_model = SimpleNamespace(
        dataset_frame=raw,
        raw_feature_names=list(EXPLAINABILITY_FEATURE_NAMES),
        metadata={"explainability_feature_names": list(EXPLAINABILITY_FEATURE_NAMES)},
    )

    stored = ApiModelService._runtime_shap_explanation(
        training_runtime=runtime,
        dashboard_model=dashboard_model,
        raw_frame=raw.iloc[[0]],
        prediction=0.123,
    )

    assert stored.index == [0]
    assert stored.method == "shap.Explainer(permutation, runtime)"
    assert stored.feature_names == list(EXPLAINABILITY_FEATURE_NAMES)
    assert stored.values.shape == (1, len(EXPLAINABILITY_FEATURE_NAMES))
    assert stored.predictions.tolist() == [0.123]


def test_neighbors_distance_figure_uses_api_neighbor_index_column():
    figure = neighbors_distance_figure(
        pd.DataFrame({"index": [101, 205], "distance": [0.1, 0.3]})
    )

    assert list(figure.data[0].x) == ["101", "205"]
