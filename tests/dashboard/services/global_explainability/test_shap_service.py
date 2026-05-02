import numpy as np
import pytest

from src.dashboard.services.global_explainability import AUXILIARY_FEATURE_LABEL
from src.dashboard.services.global_explainability import FULL_FEATURE_SCOPE
from src.dashboard.services.global_explainability import MAIN_FEATURE_SCOPE
from src.dashboard.services.global_explainability import ShapService


def _payload():
    return {
        "method": "shap.Explainer(permutation)",
        "feature_names": [
            "ExecDatetime",
            "OptionType",
            "Quantity",
            "StrikePrice",
            "TradeType",
            "UnderlyingLagMinutes",
            "UnderlyingPrice",
            "TimeToExpiration",
            "Rate",
        ],
        "index": [7],
        "values": [
            [0.04, 0.02, 0.03, 0.10, -0.01, 0.05, -0.07, 0.06, -0.02],
        ],
        "base_values": [0.30],
        "data": [
            [1.0, 0.0, 3.0, 9100.0, 2.0, 1.5, 9050.0, 20.0, -0.6],
        ],
        "display_data": [
            [
                "2026-04-22T10:00:00Z",
                "C",
                3.0,
                9100.0,
                "M",
                1.5,
                9050.0,
                20.0,
                -0.6,
            ]
        ],
        "predictions": [0.50],
        "mean_abs_shap": {
            "ExecDatetime": 0.04,
            "OptionType": 0.02,
            "Quantity": 0.03,
            "StrikePrice": 0.10,
            "TradeType": 0.01,
            "UnderlyingLagMinutes": 0.05,
            "UnderlyingPrice": 0.07,
            "TimeToExpiration": 0.06,
            "Rate": 0.02,
        },
    }


def test_main_scope_aggregates_hidden_features_without_changing_base_value():
    service = ShapService(prediction_service=None)

    result = service.from_payload(_payload(), feature_scope=MAIN_FEATURE_SCOPE)

    assert result.feature_scope == MAIN_FEATURE_SCOPE
    assert result.feature_names == [
        "OptionType",
        "StrikePrice",
        "UnderlyingPrice",
        "TimeToExpiration",
        "Rate",
        AUXILIARY_FEATURE_LABEL,
    ]
    assert float(np.asarray(result.explanation.base_values).reshape(-1)[0]) == 0.30
    assert result.explanation.display_data[0, -1] == "Aggregated hidden inputs"
    assert result.explain_frame.columns.tolist() == result.feature_names
    assert float(result.explain_frame.iloc[0, -1]) == 0.0
    assert float(result.explanation.data[0, -1]) == 0.0
    assert result.predictions.loc[7] == 0.50
    assert result.explanation.values[0, -1] == pytest.approx(0.11)


def test_full_scope_preserves_full_feature_set():
    service = ShapService(prediction_service=None)

    result = service.from_payload(_payload(), feature_scope=FULL_FEATURE_SCOPE)

    assert result.feature_scope == FULL_FEATURE_SCOPE
    assert len(result.feature_names) == 9
    assert AUXILIARY_FEATURE_LABEL not in result.feature_names
