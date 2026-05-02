import numpy as np
import pandas as pd
import pytest

from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.model2dashboard.features import MODEL_INPUT_FEATURE_NAMES
from src.model2dashboard.features import TARGET_COLUMN
from src.model2dashboard.features import VISIBLE_RAW_INPUT_FEATURE_NAMES
from src.model2dashboard.features import add_dashboard_derived_features
from src.model2dashboard.features import apply_feature_override
from src.model2dashboard.features import build_dashboard_dataset
from src.model2dashboard.features import build_explainability_encoder
from src.model2dashboard.features import build_explainability_frame
from src.model2dashboard.features import build_feature_frame_from_trades


def _raw_trade_frame():
    return pd.DataFrame(
        [
            {
                "ExecDatetime": "2026-04-22T10:00:00Z",
                "OptionContractCode": "CIBX 9000X26",
                "OptionType": "C",
                "Quantity": 1,
                "StrikePrice": 9000.0,
                "UnderlyingPrice": 9050.0,
                "TimeToExpiration": 15.0,
                "Rate": -0.5,
                "ImpliedVolatility": 0.20,
            },
            {
                "ExecDatetime": "2026-04-23T15:00:00Z",
                "OptionContractCode": "PIBX 9100X26",
                "OptionType": "P",
                "Quantity": 7,
                "StrikePrice": 9100.0,
                "UnderlyingPrice": 9000.0,
                "TimeToExpiration": 20.0,
                "Rate": -0.6,
                "ImpliedVolatility": 0.21,
            },
        ],
        index=[10, 11],
    )


def test_explainability_encoder_roundtrip_preserves_supported_feature_types():
    raw_frame = _raw_trade_frame()
    encoder = build_explainability_encoder(raw_frame)

    explain_frame = build_explainability_frame(raw_frame)
    encoded = encoder.encode_frame(raw_frame)
    decoded = encoder.decode_values(encoded)
    reconstructed = encoder.reconstruct_raw_frame(encoded)

    assert list(explain_frame.columns) == list(EXPLAINABILITY_FEATURE_NAMES)
    assert encoded.shape == (2, len(EXPLAINABILITY_FEATURE_NAMES))
    assert decoded.loc[10, "OptionType"] == "C"
    assert decoded.loc[11, "OptionType"] == "P"
    assert reconstructed.loc[10, "Quantity"] == 1.0


def test_feature_and_dashboard_dataset_builders_add_expected_columns():
    raw_frame = _raw_trade_frame()

    feature_frame = build_feature_frame_from_trades(raw_frame)
    derived = add_dashboard_derived_features(raw_frame)
    dataset = build_dashboard_dataset(raw_frame, predictions=np.array([0.18, 0.22]))

    assert list(feature_frame.columns) == list(MODEL_INPUT_FEATURE_NAMES)
    assert feature_frame.loc[10, "isCall"] == 1.0
    assert feature_frame.loc[11, "isPut"] == 1.0
    assert "Moneyness" in derived.columns
    assert "ExecHour" in derived.columns
    assert "PredictedVolatility" in dataset.columns
    assert "Residual" in dataset.columns
    assert dataset.loc[0, "Residual"] == dataset.loc[0, TARGET_COLUMN] - 0.18


def test_exec_datetime_is_hidden_from_visible_manual_inputs():
    assert "ExecDatetime" not in VISIBLE_RAW_INPUT_FEATURE_NAMES


def test_apply_feature_override_updates_supported_fields():
    raw_frame = _raw_trade_frame().iloc[[0]]

    adjusted_moneyness = apply_feature_override(raw_frame, "Moneyness", 1.1)
    adjusted_rate = apply_feature_override(raw_frame, "Rate", 0.05)

    assert adjusted_moneyness.loc[10, "StrikePrice"] == pytest.approx(9050.0 / 1.1)
    assert adjusted_rate.loc[10, "Rate"] == 0.05
