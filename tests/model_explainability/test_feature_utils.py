import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_FEATURE_SCHEMA
from src.volatility_models.model_explainability.utils.feature_utils import (
    add_derived_features,
    apply_feature_override,
)


def test_add_derived_features_creates_exec_and_moneyness_columns():
    frame = pd.DataFrame(
        {
            "ExecDatetime": ["2020-01-06 09:15:00.000000"],
            "UnderlyingPrice": [9000.0],
            "StrikePrice": [8900.0],
        }
    )

    enriched = add_derived_features(frame, DEFAULT_FEATURE_SCHEMA)

    assert enriched.loc[0, "ExecHour"] == 9
    assert enriched.loc[0, "ExecWeekday"] == 1
    assert "Moneyness" in enriched.columns
    assert "LogMoneyness" in enriched.columns


def test_apply_feature_override_updates_strike_for_moneyness():
    frame = pd.DataFrame({"UnderlyingPrice": [9000.0], "StrikePrice": [8900.0]})

    updated = apply_feature_override(frame, "Moneyness", 1.1)

    assert round(updated.loc[0, "StrikePrice"], 6) == round(9000.0 / 1.1, 6)
