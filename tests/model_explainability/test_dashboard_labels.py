import pandas as pd

from src.dashboard.utils.feature_utils import build_manual_input_sample_label
from src.dashboard.utils.feature_utils import build_sample_label


def test_build_sample_label_expands_abbreviations():
    row = pd.Series(
        {
            "OptionType": "C",
            "StrikePrice": 9100.0,
            "TimeToExpiration": 15.3,
            "Moneyness": 0.9,
        },
        name=42,
    )

    label = build_sample_label(row)

    assert label == (
        "Sample ID: 42 | Option type: CALL | "
        "Strike: 9,100 | Time to expiration: 15.3 days | Moneyness: 0.900"
    )


def test_build_manual_input_sample_label_uses_manual_input_fields():
    row = pd.Series(
        {
            "ExecDatetime": "2026-04-23T09:30:00",
            "OptionType": "P",
            "StrikePrice": 9100.0,
            "UnderlyingPrice": 9000.0,
            "TimeToExpiration": 20.0,
            "Rate": -0.6,
        },
        name=7,
    )

    label = build_manual_input_sample_label(row)

    assert label == (
        "ID: 7 | Exec datetime: 2026-04-23 09:30:00 | "
        "Type: PUT | Strike: 9,100 | "
        "Underlying: 9,000 | Time to expiration: 20.0 | "
        "Rate: -0.60"
    )
