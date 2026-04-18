import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_FEATURE_SCHEMA


def test_feature_schema_exposes_raw_and_derived_features():
    raw_names = [feature.name for feature in DEFAULT_FEATURE_SCHEMA.raw_input_features()]
    explain_names = [feature.name for feature in DEFAULT_FEATURE_SCHEMA.explainability_features()]

    assert "OptionType" in raw_names
    assert "Moneyness" in explain_names
    assert "Moneyness" not in raw_names


def test_feature_schema_defaults_and_validation():
    frame = pd.DataFrame(
        {
            "TimeToExpiration": [10.0, 20.0],
            "Rate": [1.0, 2.0],
            "UnderlyingPrice": [9000.0, 9100.0],
            "StrikePrice": [8900.0, 9200.0],
            "OptionType": ["C", "P"],
            "ExecHour": [9, 10],
            "ExecWeekday": [1, 2],
        }
    )

    defaults = DEFAULT_FEATURE_SCHEMA.defaults_from_frame(frame)
    errors = DEFAULT_FEATURE_SCHEMA.validate_sample(defaults)

    assert defaults["OptionType"] in {"C", "P"}
    assert errors == {}
