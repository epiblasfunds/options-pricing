from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model2dashboard.model_io import TrainingModelRuntime
from src.model2dashboard.model_io import discover_model_families
from src.model2dashboard.model_io import predict_feature_frame
from src.model2dashboard.model_io import predict_raw_frame
from src.model2dashboard.model_io import transform_feature_frame
from src.model2dashboard.features import MODEL_INPUT_FEATURE_NAMES


class _Scaler:
    def transform(self, values):
        return values + 10.0


class _LinearModel:
    def predict(self, matrix, verbose=0):
        return matrix[:, :1]


def _feature_frame():
    frame = pd.DataFrame(0.0, index=[0, 1], columns=list(MODEL_INPUT_FEATURE_NAMES))
    frame.loc[:, "TTEYears"] = [0.1, 0.2]
    frame.loc[:, "sqrtTTEYears"] = [0.3, 0.4]
    frame.loc[:, "isCall"] = [1.0, 0.0]
    return frame


def test_discover_model_families_filters_supported_artifacts(tmp_path):
    for filename in [
        "alpha.joblib",
        "alpha_scaler.joblib",
        "beta.keras",
        "gamma.h5",
        "notes.txt",
    ]:
        (tmp_path / filename).write_text("x", encoding="utf-8")

    assert discover_model_families(tmp_path) == ["alpha", "beta", "gamma"]


def test_transform_and_predict_feature_frames_support_scalers_and_keras_flag():
    feature_frame = _feature_frame()
    sklearn_runtime = TrainingModelRuntime(
        family_name="alpha",
        model=_LinearModel(),
        model_path=Path("alpha.joblib"),
        scaler=_Scaler(),
        scaler_path=Path("alpha_scaler.joblib"),
        final_test_metadata={},
        train_val_metadata={},
        model_input_features=list(MODEL_INPUT_FEATURE_NAMES),
    )
    keras_runtime = TrainingModelRuntime(
        family_name="beta",
        model=_LinearModel(),
        model_path=Path("beta.keras"),
        scaler=None,
        scaler_path=None,
        final_test_metadata={},
        train_val_metadata={},
        model_input_features=list(MODEL_INPUT_FEATURE_NAMES),
    )

    transformed = transform_feature_frame(sklearn_runtime, feature_frame)
    sklearn_pred = predict_feature_frame(sklearn_runtime, feature_frame)
    keras_pred = predict_feature_frame(keras_runtime, feature_frame)

    assert transformed.loc[0, "TTEYears"] == np.float32(10.1)
    assert sklearn_pred.tolist() == pytest.approx([10.1, 10.2])
    assert keras_pred.tolist() == pytest.approx([0.1, 0.2])


def test_predict_raw_frame_builds_features_before_scoring():
    raw_frame = pd.DataFrame(
        {
            "ExecDatetime": ["2026-04-22T10:00:00Z"],
            "OptionContractCode": ["CIBX 9000X26"],
            "OptionType": ["C"],
            "Quantity": [1],
            "StrikePrice": [9000.0],
            "UnderlyingPrice": [9050.0],
            "TimeToExpiration": [15.0],
            "Rate": [-0.5],
            "ImpliedVolatility": [0.20],
        }
    )
    runtime = TrainingModelRuntime(
        family_name="alpha",
        model=_LinearModel(),
        model_path=Path("alpha.joblib"),
        scaler=None,
        scaler_path=None,
        final_test_metadata={},
        train_val_metadata={},
        model_input_features=list(MODEL_INPUT_FEATURE_NAMES),
    )

    prediction = predict_raw_frame(runtime, raw_frame)

    assert prediction.tolist() == pytest.approx([15.0 / 365.0])
