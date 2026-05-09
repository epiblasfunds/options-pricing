import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.dashboard.services.global_explainability import ShapService
from src.python_models.dashboard.artifacts import StoredShapExplanation


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


def _stored(index: int, prediction: float) -> StoredShapExplanation:
    return StoredShapExplanation(
        method="stored",
        feature_names=["StrikePrice"],
        index=[index],
        values=np.asarray([[0.1]]),
        base_values=np.asarray([prediction - 0.1]),
        data=np.asarray([[9100.0]]),
        display_data=np.asarray([[9100.0]]),
        predictions=np.asarray([prediction]),
        mean_abs_shap={"StrikePrice": 0.1},
    )


def test_explain_and_explain_sample_use_persisted_bundle_content():
    global_stored = _stored(index=5, prediction=0.5)
    local_stored = _stored(index=7, prediction=0.7)
    fallback_stored = _stored(index=9, prediction=0.9)
    local_bundle = SimpleNamespace(
        dashboard_model=SimpleNamespace(
            global_shap=global_stored,
            local_shap=local_stored,
            local_shap_for_index=lambda row_index: local_stored,
        )
    )
    fallback_bundle = SimpleNamespace(
        dashboard_model=SimpleNamespace(
            global_shap=global_stored,
            local_shap=fallback_stored,
            local_shap_for_index=lambda row_index: local_stored,
        )
    )
    local_service = ShapService(
        prediction_service=SimpleNamespace(load_bundle=lambda model_id: local_bundle)
    )
    fallback_service = ShapService(
        prediction_service=SimpleNamespace(load_bundle=lambda model_id: fallback_bundle)
    )

    global_result = local_service.explain("rf")
    local_result = local_service.explain_sample("rf", pd.DataFrame(index=[7]))
    fallback_result = fallback_service.explain_sample(
        "rf",
        pd.DataFrame(index=[999]),
    )

    assert global_result.predictions.loc[5] == 0.5
    assert local_result.predictions.loc[7] == 0.7
    assert fallback_result.predictions.loc[9] == 0.9
