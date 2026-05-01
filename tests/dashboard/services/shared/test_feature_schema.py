import pandas as pd

from src.dashboard.domain import build_feature_schema
from src.enums.data_enums import OptionTypeEnum


def test_feature_schema_exposes_raw_and_derived_features():
    schema = build_feature_schema()

    raw_names = [feature.name for feature in schema.raw_input_features()]
    explain_names = [feature.name for feature in schema.explainability_features()]

    assert "OptionType" in raw_names
    assert "Moneyness" in explain_names
    assert "Moneyness" not in raw_names
    assert schema.labels(["OptionType"]) == {"OptionType": "Option Type"}


def test_feature_schema_defaults_normalization_and_validation():
    schema = build_feature_schema()
    frame = pd.DataFrame(
        {
            "ExecDatetime": ["2026-04-22T10:00:00Z", "2026-04-23T11:00:00Z"],
            "OptionType": ["C", "P"],
            "StrikePrice": [9000.0, 9100.0],
            "UnderlyingPrice": [9050.0, 9150.0],
            "TimeToExpiration": [10.0, 20.0],
            "Rate": [0.01, 0.02],
        }
    )

    defaults = schema.defaults_from_frame(frame)
    normalized = schema.normalize_sample(
        {
            "ExecDatetime": "2026-04-22T10:00:00Z",
            "OptionType": "call",
            "StrikePrice": "9000",
            "UnderlyingPrice": 9050,
            "TimeToExpiration": "10.4",
            "Rate": "0.01",
        }
    )

    assert defaults["OptionType"] in {"C", "P"}
    assert normalized["OptionType"] == OptionTypeEnum.CALL
    assert normalized["StrikePrice"] == 9000.0
    assert normalized["TimeToExpiration"] == 10.4
    assert not schema.validate_sample(normalized)


def test_feature_schema_validation_reports_required_invalid_and_bound_errors():
    schema = build_feature_schema()

    errors = schema.validate_sample(
        {
            "ExecDatetime": "not-a-date",
            "OptionType": "invalid",
            "StrikePrice": -1,
            "UnderlyingPrice": "",
            "TimeToExpiration": "bad",
            "Rate": 0.0,
        }
    )

    assert errors["ExecDatetime"] == "Value must be a valid datetime."
    assert errors["OptionType"] == "Value is outside the allowed set."
    assert errors["StrikePrice"] == "Value is below the minimum."
    assert errors["UnderlyingPrice"] == "Value is required."
    assert errors["TimeToExpiration"] == "Value must be numeric."
