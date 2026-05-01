import io
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.error import URLError

import pandas as pd
import pytest

from src.dashboard.domain import build_feature_schema
from src.dashboard.services.shared.prediction_service import PredictionPipelineError
from src.dashboard.services.shared.prediction_service import PredictionService


def _service_with_bundle(dataset_frame):
    bundle = SimpleNamespace(
        dashboard_model=SimpleNamespace(
            dataset_frame=dataset_frame,
            predictions_for_indices=lambda indices: dataset_frame.loc[
                list(indices), "PredictedVolatility"
            ].copy(),
        )
    )
    registry = SimpleNamespace(get_model=lambda model_id: object())
    loader = SimpleNamespace(load=lambda discovered: bundle)
    return PredictionService(
        model_registry=registry,
        model_loader=loader,
        feature_schema=build_feature_schema(),
    )


def test_predict_frame_uses_precomputed_predictions_only_for_exported_rows():
    dataset = pd.DataFrame({"PredictedVolatility": [0.2, 0.3]}, index=[10, 11])
    service = _service_with_bundle(dataset)

    result = service.predict_frame("model", pd.DataFrame({"x": [1, 2]}, index=[10, 11]))

    assert result.to_dict() == {10: 0.2, 11: 0.3}
    with pytest.raises(PredictionPipelineError):
        service.predict_frame("model", pd.DataFrame(index=[99]))


def test_api_features_accept_manual_call_put_labels():
    service = PredictionService.__new__(PredictionService)

    call_payload = service._api_features_from_dashboard_sample(
        {
            "OptionContractCode": "",
            "OptionType": "CALL",
            "StrikePrice": 9100.0,
            "UnderlyingPrice": 9000.0,
            "TimeToExpiration": 20.0,
            "Rate": -0.6,
            "ImpliedVolatility": 0.21,
        }
    )
    put_payload = service._api_features_from_dashboard_sample({"OptionType": "P"})

    assert call_payload["optionType"] == "CALL"
    assert call_payload["strikePrice"] == 9100.0
    assert call_payload["underlyingPrice"] == 9000.0
    assert call_payload["timeToExpiration"] == 20.0
    assert call_payload["rate"] == -0.6
    assert "optionContractCode" not in call_payload
    assert call_payload["impliedVolatility"] == 0.21
    assert put_payload["optionType"] == "PUT"


def test_manual_api_calls_handle_success_and_common_failures(monkeypatch):
    service = PredictionService.__new__(PredictionService)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{\"prediction\": 0.25}'

    monkeypatch.setattr(
        "src.dashboard.services.shared.prediction_service.urlopen",
        lambda request, timeout: _Response(),
    )

    body = service._post_manual_api(
        endpoint="/run_model/predict/",
        model_id="random_forest",
        sample_payload={"OptionType": "C"},
    )

    assert body == {"prediction": 0.25}

    monkeypatch.setattr(
        "src.dashboard.services.shared.prediction_service.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            HTTPError(
                url="http://test",
                code=500,
                msg="boom",
                hdrs=None,
                fp=io.BytesIO(b"detail"),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Manual API call failed"):
        service._post_manual_api(
            endpoint="/run_model/predict/",
            model_id="random_forest",
            sample_payload={"OptionType": "C"},
        )

    monkeypatch.setattr(
        "src.dashboard.services.shared.prediction_service.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        service._post_manual_api(
            endpoint="/run_model/predict/",
            model_id="random_forest",
            sample_payload={"OptionType": "C"},
        )
