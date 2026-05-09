from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.model2dashboard import model_io
from src.model2dashboard.model_io import TrainingModelRuntime
from src.model2dashboard.model_io import discover_model_families
from src.model2dashboard.model_io import load_training_runtime
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


def test_load_training_runtime_reads_model_scaler_and_metadata(tmp_path, monkeypatch):
    trained_dir = tmp_path / "trained"
    metadata_dir = tmp_path / "metadata"
    trained_dir.mkdir()
    metadata_dir.mkdir()
    (trained_dir / "alpha.joblib").write_text("model", encoding="utf-8")
    (trained_dir / "alpha_scaler.joblib").write_text("scaler", encoding="utf-8")
    (metadata_dir / "alpha_final_test_retrained_metadata.json").write_text(
        json.dumps({"rmse": 0.12}),
        encoding="utf-8",
    )
    (metadata_dir / "alpha_train_val_retrained_metadata.json").write_text(
        json.dumps({"rows": 10}),
        encoding="utf-8",
    )

    loaded_scalers: list[Path] = []
    monkeypatch.setattr(
        model_io.joblib,
        "load",
        lambda path: loaded_scalers.append(Path(path)) or _Scaler(),
    )
    monkeypatch.setattr(model_io, "_load_model", lambda path: _LinearModel())
    monkeypatch.setattr(model_io, "_force_single_process_prediction", lambda model: None)

    runtime = load_training_runtime(
        family_name="alpha",
        trained_models_dir=trained_dir,
        retrained_metadata_dir=metadata_dir,
    )

    assert runtime.family_name == "alpha"
    assert runtime.final_test_metadata == {"rmse": 0.12}
    assert runtime.train_val_metadata == {"rows": 10}
    assert runtime.scaler_path == trained_dir / "alpha_scaler.joblib"
    assert loaded_scalers == [trained_dir / "alpha_scaler.joblib"]


def test_resolve_helpers_and_force_single_process_cover_edge_cases(tmp_path, monkeypatch):
    trained_dir = tmp_path / "trained"
    metadata_dir = tmp_path / "metadata"
    trained_dir.mkdir()
    metadata_dir.mkdir()
    (trained_dir / "beta.keras").write_text("x", encoding="utf-8")
    (metadata_dir / "beta_final_test_retrained_progressive_metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )

    resolved_model = model_io._resolve_model_path(trained_dir, "beta")
    resolved_meta = model_io._resolve_retrained_metadata_path(
        retrained_metadata_dir=metadata_dir,
        family_name="beta_retrained_progressive",
        phase="final_test",
    )

    assert resolved_model == trained_dir / "beta.keras"
    assert resolved_meta == metadata_dir / "beta_final_test_retrained_progressive_metadata.json"

    with pytest.raises(FileNotFoundError):
        model_io._resolve_model_path(trained_dir, "missing")

    class _SetParamsModel:
        def __init__(self) -> None:
            self.received = None

        def set_params(self, **kwargs):
            self.received = kwargs

    model = _SetParamsModel()
    model_io._force_single_process_prediction(model)
    assert model.received == {"n_jobs": 1}

    class _NJobsOnlyModel:
        def set_params(self, **kwargs):
            raise ValueError("unsupported")

    fallback_model = _NJobsOnlyModel()
    fallback_model.n_jobs = 4
    model_io._force_single_process_prediction(fallback_model)
    assert fallback_model.n_jobs == 1


def test_load_model_joblib_branch_and_read_json(tmp_path, monkeypatch):
    artifact = tmp_path / "gamma.joblib"
    artifact.write_text("x", encoding="utf-8")
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    monkeypatch.setattr(model_io.joblib, "load", lambda path: {"loaded": Path(path).name})

    loaded = model_io._load_model(artifact)

    assert loaded == {"loaded": "gamma.joblib"}
    assert model_io._read_json(payload_path) == {"a": 1}
