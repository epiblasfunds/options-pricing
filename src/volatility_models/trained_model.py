import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import joblib
from keras import models as keras_models
from sklearn.base import TransformerMixin

from src.enums.volatility_model_enums import ModelFormatEnum


@dataclass(frozen=True)
class TrainedModelMetadata:
    model_id: str
    name: str
    path: Path
    format: ModelFormatEnum
    feature_names: tuple[str, ...]
    target_column: str
    loss_name: str
    metadata: dict[str, t.Any] = field(default_factory=dict)

    @staticmethod
    def get_metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path.as_posix(),
            "format": self.format.value,
            "feature_names": list(self.feature_names),
            "target_column": self.target_column,
            "loss_name": self.loss_name,
            "metadata": self.metadata,
        }

    def save(self, path: Path | None = None) -> None:
        model_path = path or self.path
        model_path.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["path"] = model_path.as_posix()
        self.get_metadata_path(model_path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TrainedModelMetadata":
        payload = json.loads(cls.get_metadata_path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            name=payload["name"],
            path=path,
            format=ModelFormatEnum(payload["format"]),
            feature_names=tuple(payload["feature_names"]),
            target_column=payload["target_column"],
            loss_name=payload["loss_name"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class TrainingHistory:
    loss_name: str
    train_metrics: dict[str, list[float]]
    validation_metrics: dict[str, list[float]]

    @classmethod
    def from_keras_history(cls, history, loss_name: str = "loss") -> "TrainingHistory":
        history_dict = dict(history.history)
        train_metrics: dict[str, list[float]] = {}
        validation_metrics: dict[str, list[float]] = {}
        for metric_name, values in history_dict.items():
            normalized_values = [float(value) for value in values]
            if metric_name.startswith("val_"):
                validation_metrics[metric_name[4:]] = normalized_values
            else:
                train_metrics[metric_name] = normalized_values
        return cls(
            loss_name=loss_name,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )

    def save(self, path: Path) -> None:
        payload = {
            "loss_name": self.loss_name,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
        }
        path.mkdir(parents=True, exist_ok=True)
        (path / "history.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        payload = json.loads((path / "history.json").read_text(encoding="utf-8"))
        return cls(
            loss_name=payload["loss_name"],
            train_metrics={
                name: [float(value) for value in values]
                for name, values in payload.get("train_metrics", {}).items()
            },
            validation_metrics={
                name: [float(value) for value in values]
                for name, values in payload.get("validation_metrics", {}).items()
            },
        )


class TrainedModel:
    def __init__(
        self,
        *,
        model,
        metadata: TrainedModelMetadata,
        history: TrainingHistory,
        preprocessor: TransformerMixin | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.history = history
        self.preprocessor = preprocessor

    @staticmethod
    def get_model_path(path: Path, model_format: ModelFormatEnum) -> Path:
        if model_format == ModelFormatEnum.KERAS:
            return path / "model.keras"
        if model_format == ModelFormatEnum.H5:
            return path / "model.h5"
        raise ValueError(f"Unsupported trained-model format: {model_format.value}")

    @staticmethod
    def get_preprocessor_path(path: Path) -> Path:
        return path / "preprocessor.joblib"

    def save(self, path: Path | None = None) -> None:
        model_path = path or self.metadata.path
        model_path.mkdir(parents=True, exist_ok=True)
        self.metadata.save(model_path)
        self.history.save(model_path)
        self.model.save(self.get_model_path(model_path, self.metadata.format))
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, self.get_preprocessor_path(model_path))

    @classmethod
    def load(cls, metadata: TrainedModelMetadata) -> "TrainedModel":
        model = keras_models.load_model(
            cls.get_model_path(metadata.path, metadata.format)
        )
        preprocessor_path = cls.get_preprocessor_path(metadata.path)
        preprocessor = (
            joblib.load(preprocessor_path) if preprocessor_path.exists() else None
        )
        return cls(
            model=model,
            metadata=metadata,
            history=TrainingHistory.load(metadata.path),
            preprocessor=preprocessor,
        )
