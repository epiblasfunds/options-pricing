import pandas as pd

from src.dashboard.utils.feature_utils import build_manual_input_sample_label


def test_build_manual_input_sample_label_omits_exec_datetime():
    label = build_manual_input_sample_label(
        pd.Series(
            {
                "ExecDatetime": "2026-05-02T00:00:00.000+02:00",
                "OptionType": "C",
                "StrikePrice": 9100.0,
                "UnderlyingPrice": 9000.0,
                "TimeToExpiration": 20.0,
                "Rate": -0.6,
            },
            name=17,
        )
    )

    assert "Exec datetime" not in label
    assert label.startswith("ID: 17 | Type: CALL")
