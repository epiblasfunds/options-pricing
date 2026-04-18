import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model2dashboard.features import BASE_NUMERIC_FEATURE_NAMES
from src.model2dashboard.features import MODEL_INPUT_FEATURE_NAMES
from src.model2dashboard.features import build_feature_frame_from_trades


@dataclass
class TrainingModelRuntime:
    family_name: str
    model: t.Any
    model_path: Path
    scaler: t.Any | None
    scaler_path: Path | None
    final_test_metadata: dict[str, t.Any]
    train_val_metadata: dict[str, t.Any]
    model_input_features: list[str]

    @property
    def is_keras(self) -> bool:
        return self.model_path.suffix.lower() in {".keras", ".h5"}


def discover_model_families(trained_models_dir: Path) -> list[str]:
    families: set[str] = set()
    for artifact in trained_models_dir.iterdir():
        if not artifact.is_file():
            continue
        if artifact.suffix.lower() not in {".joblib", ".keras", ".h5"}:
            continue
        stem = artifact.stem
        if stem.endswith("_scaler"):
            continue
        families.add(stem)
    return sorted(families)


def load_training_runtime(
    *,
    family_name: str,
    trained_models_dir: Path,
    retrained_metadata_dir: Path,
) -> TrainingModelRuntime:
    model_path = _resolve_model_path(trained_models_dir, family_name)
    final_test_metadata_path = (
        retrained_metadata_dir / f"{family_name}_final_test_retrained_metadata.json"
    )
    train_val_metadata_path = (
        retrained_metadata_dir / f"{family_name}_train_val_retrained_metadata.json"
    )
    if not final_test_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing final-test metadata for '{family_name}': {final_test_metadata_path}"
        )
    scaler_path = trained_models_dir / f"{family_name}_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    model = _load_model(model_path)
    _force_single_process_prediction(model)
    return TrainingModelRuntime(
        family_name=family_name,
        model=model,
        model_path=model_path,
        scaler=scaler,
        scaler_path=scaler_path if scaler_path.exists() else None,
        final_test_metadata=_read_json(final_test_metadata_path),
        train_val_metadata=(
            _read_json(train_val_metadata_path)
            if train_val_metadata_path.exists()
            else {}
        ),
        model_input_features=list(MODEL_INPUT_FEATURE_NAMES),
    )


def transform_feature_frame(
    runtime: TrainingModelRuntime,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    ordered = feature_frame.loc[:, runtime.model_input_features].copy()
    if runtime.scaler is None:
        return ordered.astype("float32")
    transformed = ordered.copy()
    numeric_columns = [
        feature for feature in BASE_NUMERIC_FEATURE_NAMES if feature in transformed.columns
    ]
    transformed.loc[:, numeric_columns] = runtime.scaler.transform(
        transformed.loc[:, numeric_columns].to_numpy(dtype="float64", copy=False)
    )
    return transformed.astype("float32")


def predict_feature_frame(
    runtime: TrainingModelRuntime,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    transformed = transform_feature_frame(runtime, feature_frame)
    matrix = transformed.to_numpy(dtype="float32", copy=False)
    if runtime.is_keras:
        predictions = runtime.model.predict(matrix, verbose=0)
    else:
        predictions = runtime.model.predict(matrix)
    return np.asarray(predictions, dtype="float64").reshape(-1)


def predict_raw_frame(
    runtime: TrainingModelRuntime,
    raw_frame: pd.DataFrame,
) -> np.ndarray:
    return predict_feature_frame(runtime, build_feature_frame_from_trades(raw_frame))


def _resolve_model_path(trained_models_dir: Path, family_name: str) -> Path:
    for suffix in (".joblib", ".keras", ".h5"):
        candidate = trained_models_dir / f"{family_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No model artifact found for '{family_name}' under {trained_models_dir}."
    )


def _load_model(path: Path):
    if path.suffix.lower() == ".joblib":
        return joblib.load(path)

    from keras.models import load_model
    from src.python_models.volatility_models.volatility_model_family import (
        TensorTrainLayer,
    )

    return load_model(
        path,
        compile=False,
        custom_objects={"TensorTrainLayer": TensorTrainLayer},
    )


def _force_single_process_prediction(model) -> None:
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
            return
        except (TypeError, ValueError):
            pass
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except AttributeError:
            pass


def _read_json(path: Path) -> dict[str, t.Any]:
    return json.loads(path.read_text(encoding="utf-8"))
