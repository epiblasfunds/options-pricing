import pandas as pd

from src.dashboard.utils.feature_utils import build_sample_label


def test_build_sample_label_expands_abbreviations():
    row = pd.Series(
        {
            "OptionType": "C",
            "TimeToExpiration": 15.3,
            "Moneyness": 0.9,
        },
        name=42,
    )

    label = build_sample_label(row)

    assert label == (
        "Sample ID: 42 | Option type: CALL | "
        "Time to expiration: 15.3 days | Moneyness: 0.900"
    )
