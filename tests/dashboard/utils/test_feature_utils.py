import pandas as pd

from src.dashboard.domain import build_feature_schema
from src.dashboard.utils import feature_utils
from src.dashboard.utils.feature_utils import build_manual_input_sample_label
from src.dashboard.utils.feature_utils import build_sample_label
from src.dashboard.utils.feature_utils import display_feature_label
from src.dashboard.utils.feature_utils import format_feature_value
from src.dashboard.utils.feature_utils import replace_feature_names_in_text


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


def test_build_sample_label_formats_core_fields():
    label = build_sample_label(
        pd.Series(
            {
                "OptionType": "P",
                "StrikePrice": 9050.0,
                "TimeToExpiration": 12.4,
                "Moneyness": 1.015,
            },
            name=21,
        )
    )

    assert label.startswith("Sample ID: 21 | Option type: PUT")
    assert "Strike: 9,050" in label
    assert "Time to expiration: 12.4 days" in label
    assert "Moneyness: 1.015" in label


def test_feature_value_formatters_cover_supported_types_and_missing_values():
    assert format_feature_value("OptionType", "C") == "CALL"
    assert format_feature_value("StrikePrice", 9010.4) == "9,010"
    assert format_feature_value("UnderlyingPrice", 9005.4) == "9,005"
    assert format_feature_value("Rate", -0.625) == "-0.62"
    assert format_feature_value("TimeToExpiration", 10.25) == "10.2"
    assert format_feature_value("UnknownFeature", "") == "?"
    assert format_feature_value("UnknownFeature", None) == "?"
    assert format_feature_value("UnknownFeature", "abc") == "abc"


def test_display_feature_label_and_text_replacement_cover_transformed_features():
    schema = build_feature_schema()

    assert display_feature_label("OptionType", schema) == "Option Type"
    assert display_feature_label("prefix__StrikePrice", schema) == "Strike Price"
    assert display_feature_label("ohe__OptionType_C", schema) == "Option Type = C"
    assert display_feature_label("custom__SomethingElse", schema) == "SomethingElse"

    text = replace_feature_names_in_text(
        "OptionType and StrikePrice move together",
        schema,
    )

    assert "Option Type" in text
    assert "Strike Price" in text


def test_private_display_helpers_cover_datetime_and_invalid_numbers():
    assert feature_utils._display_option_type(" ") == "?"
    assert feature_utils._display_datetime("2026-05-09T10:11:12Z").startswith(
        "2026-05-09 10:11:12"
    )
    assert feature_utils._display_datetime("bad-value") == "?"
    assert feature_utils._display_number("bad") == "?"
