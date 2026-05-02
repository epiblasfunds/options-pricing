import numpy as np

from src.dashboard.services.global_explainability import ShapService


def _payload():
    return {
        "method": "shap.Explainer(permutation)",
        "feature_names": [
            "OptionType",
            "StrikePrice",
            "UnderlyingPrice",
            "TimeToExpiration",
            "Rate",
            "OptionContractCode",
        ],
        "index": [7],
        "values": [
            [0.02, 0.10, -0.07, 0.06, -0.02, 0.04],
        ],
        "base_values": [0.30],
        "data": [
            [0.0, 9100.0, 9050.0, 20.0, -0.6, 0.0],
        ],
        "display_data": [
            [
                "C",
                9100.0,
                9050.0,
                20.0,
                -0.6,
                "CIBX 9100X26",
            ]
        ],
        "mean_abs_shap": {
            "OptionType": 0.02,
            "StrikePrice": 0.10,
            "UnderlyingPrice": 0.07,
            "TimeToExpiration": 0.06,
            "Rate": 0.02,
            "OptionContractCode": 0.04,
        },
        "predictions": [0.50],
    }


def test_from_payload_preserves_full_feature_set():
    service = ShapService(prediction_service=None)

    result = service.from_payload(_payload())

    assert result.feature_names == _payload()["feature_names"]
    assert result.explain_frame.columns.tolist() == _payload()["feature_names"]
    assert float(np.asarray(result.explanation.base_values).reshape(-1)[0]) == 0.30
    assert result.predictions.loc[7] == 0.43
    assert result.mean_abs_shap.index.tolist()[0] == "StrikePrice"


def test_from_payload_preserves_display_data_and_values():
    service = ShapService(prediction_service=None)

    result = service.from_payload(_payload())

    assert result.explanation.display_data[0, -1] == "CIBX 9100X26"
    assert float(result.explanation.values[0, -1]) == 0.04
