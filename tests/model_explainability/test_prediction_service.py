from src.dashboard.services.shared.prediction_service import PredictionService


def test_api_features_accept_manual_call_put_labels():
    service = PredictionService.__new__(PredictionService)

    call_payload = service._api_features_from_dashboard_sample(
        {
            "OptionType": "CALL",
            "StrikePrice": 9100.0,
            "UnderlyingPrice": 9000.0,
            "TimeToExpiration": 20.0,
            "Rate": -0.6,
        }
    )
    put_payload = service._api_features_from_dashboard_sample({"OptionType": "PUT"})

    assert call_payload["optionType"] == "CALL"
    assert call_payload["strikePrice"] == 9100.0
    assert call_payload["underlyingPrice"] == 9000.0
    assert call_payload["timeToExpiration"] == 20.0
    assert call_payload["rate"] == -0.6
    assert put_payload["optionType"] == "PUT"
