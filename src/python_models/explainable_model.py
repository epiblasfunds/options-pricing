from __future__ import annotations

import json
import typing as t
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.enums.volatility_model_enums import ModelFormatEnum


def _import_keras_models():
    try:
        from tensorflow.keras import models as keras_models
    except ImportError:
        try:
            from keras import models as keras_models
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError(
                "TensorFlow/Keras is required to load persisted explainable models."
            ) from exc
    return keras_models


@dataclass
class AbstractModelMetadata(ABC):
    """Metadata for one model artifact on disk."""

    model_id: str
    name: str
    path: Path
    format: ModelFormatEnum
    metadata: dict

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path.as_posix(),
            "format": self.format.value,
            "metadata": self.metadata,
        }


@dataclass
class SingleModelMetadata(AbstractModelMetadata):
    """Metadata for a single model artifact."""


@dataclass
class ExplainableModelMetadata(AbstractModelMetadata):
    """Metadata for one explainable-model bundle."""

    epi_blas_model_format: ModelFormatEnum
    tree_format: ModelFormatEnum
    syr_format: t.Optional[ModelFormatEnum] = None

    @staticmethod
    def get_root_metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def get_epi_blas_model_path(self) -> Path:
        return self.path / "epi_blas_model"

    def get_epi_blas_model_id(self) -> str:
        return f"{self.model_id}_epi_blas_model"

    def get_epi_blas_model_name(self) -> str:
        return f"{self.name}_epi_blas_model"

    def get_tree_models_path(self) -> Path:
        return self.path / "tree_models"

    def get_tree_model_path(self, depth: int) -> Path:
        return self.get_tree_models_path() / f"depth_{int(depth)}"

    def get_legacy_tree_model_path(self) -> Path:
        return self.path / "tree_model"

    def get_tree_model_id(self, depth: int) -> str:
        return f"{self.model_id}_tree_model_depth_{int(depth)}"

    def get_tree_model_name(self, depth: int) -> str:
        return f"{self.name}_tree_model_depth_{int(depth)}"

    def get_epi_blas_model_metadata(self) -> SingleModelMetadata:
        return SingleModelMetadata(
            model_id=self.get_epi_blas_model_id(),
            name=self.get_epi_blas_model_name(),
            path=self.get_epi_blas_model_path(),
            format=self.epi_blas_model_format,
            metadata=self.metadata,
        )

    def get_tree_model_metadata(self, depth: int) -> SingleModelMetadata:
        return SingleModelMetadata(
            model_id=self.get_tree_model_id(depth),
            name=self.get_tree_model_name(depth),
            path=self.get_tree_model_path(depth),
            format=self.tree_format,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, t.Any]:
        payload = super().to_dict()
        payload.update(
            {
                "epi_blas_model_format": self.epi_blas_model_format.value,
                "tree_format": self.tree_format.value,
                "syr_format": self.syr_format.value if self.syr_format else None,
            }
        )
        return payload

    def save(self, path: Path | None = None) -> None:
        bundle_path = path or self.path
        bundle_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.get_root_metadata_path(bundle_path)
        metadata_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExplainableModelMetadata":
        metadata_path = cls.get_root_metadata_path(path)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            name=payload["name"],
            path=path,
            format=ModelFormatEnum(payload["format"]),
            epi_blas_model_format=ModelFormatEnum(payload["epi_blas_model_format"]),
            tree_format=ModelFormatEnum(payload["tree_format"]),
            syr_format=(
                ModelFormatEnum(payload["syr_format"])
                if payload.get("syr_format")
                else None
            ),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class SurrogateTreeModel:
    """Result bundle for the persisted decision-tree surrogate."""

    model: DecisionTreeRegressor
    feature_importances: pd.Series
    tree_depth: int
    n_leaves: int
    text_rules: str
    interpretation: str
    fidelity_frame: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _get_model_file_path(path: Path) -> Path:
        return path / "model_tree.joblib"

    @staticmethod
    def _get_feat_importance_file_path(path: Path) -> Path:
        return path / "feature_importances.csv"

    @staticmethod
    def _get_fidelity_frame_file_path(path: Path) -> Path:
        return path / "fidelity_frame.csv"

    @staticmethod
    def _get_attrs_file_path(path: Path) -> Path:
        return path / "attributes.json"

    @classmethod
    def load(cls, metadata: SingleModelMetadata) -> "SurrogateTreeModel":
        model = joblib.load(cls._get_model_file_path(path=metadata.path))
        feature_importances = pd.read_csv(
            cls._get_feat_importance_file_path(path=metadata.path), index_col=0
        ).iloc[:, 0]
        fidelity_frame = pd.read_csv(
            cls._get_fidelity_frame_file_path(path=metadata.path), index_col=0
        )
        attrs = json.loads(
            cls._get_attrs_file_path(path=metadata.path).read_text(encoding="utf-8")
        )

        return cls(
            model=model,
            feature_importances=feature_importances,
            tree_depth=int(attrs["tree_depth"]),
            n_leaves=int(attrs["n_leaves"]),
            text_rules=attrs["text_rules"],
            interpretation=attrs["interpretation"],
            fidelity_frame=fidelity_frame,
            feature_names=list(attrs.get("feature_names", feature_importances.index)),
            metrics={
                name: float(value) for name, value in attrs.get("metrics", {}).items()
            },
        )

    def save(self, path: Path) -> None:
        attrs = {
            "tree_depth": int(self.tree_depth),
            "n_leaves": int(self.n_leaves),
            "text_rules": self.text_rules,
            "interpretation": self.interpretation,
            "feature_names": list(self.feature_names or self.feature_importances.index),
            "metrics": {name: float(value) for name, value in self.metrics.items()},
        }

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self._get_model_file_path(path=path))
        self.feature_importances.to_csv(self._get_feat_importance_file_path(path=path))
        self.fidelity_frame.to_csv(self._get_fidelity_frame_file_path(path=path))
        self._get_attrs_file_path(path=path).write_text(
            json.dumps(attrs, indent=2),
            encoding="utf-8",
        )


@dataclass
class MetricStats:
    name: str
    values: t.List[float]


@dataclass
class TrainStats:
    loss: MetricStats
    metrics: t.List[MetricStats]

    @staticmethod
    def _get_path(path: Path) -> Path:
        return path / "train_stats.joblib"

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self._get_path(path=path))

    @classmethod
    def load(cls, path: Path) -> "TrainStats":
        obj = joblib.load(cls._get_path(path=path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls}, got {type(obj)}")
        return obj

    @classmethod
    def from_history(cls, history: dict[str, t.Iterable[t.Any]]) -> "TrainStats":
        return cls(
            loss=MetricStats(
                name="loss",
                values=[float(value) for value in history.get("loss", [])],
            ),
            metrics=[
                MetricStats(name=name, values=[float(value) for value in values])
                for name, values in history.items()
                if name != "loss"
            ],
        )


class EpiBlasModel:
    def __init__(
        self,
        model: t.Any,
        train_stats: TrainStats,
        metadata: SingleModelMetadata,
    ):
        self.model = model
        self.train_stats = train_stats
        self.metadata = metadata

    @staticmethod
    def _get_model_file_path(path: Path, model_format: ModelFormatEnum) -> Path:
        if model_format == ModelFormatEnum.KERAS:
            return path / "model.keras"
        if model_format == ModelFormatEnum.H5:
            return path / "model.h5"
        raise ValueError(f"Unsupported EpiBlas model format: {model_format}")

    @classmethod
    def load(cls, metadata: SingleModelMetadata) -> "EpiBlasModel":
        model_path = cls._get_model_file_path(metadata.path, metadata.format)
        keras_models = _import_keras_models()
        model = keras_models.load_model(model_path)
        train_stats = TrainStats.load(metadata.path)
        return cls(model=model, train_stats=train_stats, metadata=metadata)

    def save(self, path: Path | None = None) -> None:
        model_dir = path or self.metadata.path
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self._get_model_file_path(model_dir, self.metadata.format))
        self.train_stats.save(model_dir)


class ExplainableModel:
    def __init__(
        self,
        main_model: EpiBlasModel,
        tree_models: dict[int, SurrogateTreeModel],
        metadata: ExplainableModelMetadata,
    ):
        self.main_model = main_model
        self.tree_models = {int(depth): model for depth, model in tree_models.items()}
        self.metadata = metadata

    @property
    def tree_model(self) -> SurrogateTreeModel:
        if not self.tree_models:
            raise ValueError("The explainable model does not contain persisted surrogate trees.")
        return self.tree_models[max(self.tree_models)]

    @classmethod
    def load(cls, metadata: ExplainableModelMetadata) -> "ExplainableModel":
        main_model = EpiBlasModel.load(metadata=metadata.get_epi_blas_model_metadata())
        tree_models = cls._load_tree_models(metadata)
        return cls(
            main_model=main_model,
            tree_models=tree_models,
            metadata=metadata,
        )

    def save(self, path: Path | None = None) -> None:
        bundle_path = path or self.metadata.path
        bundle_path.mkdir(parents=True, exist_ok=True)
        self.metadata.save(bundle_path)
        self.main_model.save(bundle_path / "epi_blas_model")
        tree_models_path = bundle_path / "tree_models"
        tree_models_path.mkdir(parents=True, exist_ok=True)
        for depth, tree_model in sorted(self.tree_models.items()):
            tree_model.save(tree_models_path / f"depth_{int(depth)}")

    @staticmethod
    def _load_tree_models(
        metadata: ExplainableModelMetadata,
    ) -> dict[int, SurrogateTreeModel]:
        tree_models_path = metadata.get_tree_models_path()
        if tree_models_path.exists():
            loaded: dict[int, SurrogateTreeModel] = {}
            for tree_path in sorted(tree_models_path.iterdir(), key=lambda path: path.name):
                if not tree_path.is_dir():
                    continue
                try:
                    depth = int(tree_path.name.removeprefix("depth_"))
                except ValueError:
                    continue
                loaded[depth] = SurrogateTreeModel.load(
                    metadata=metadata.get_tree_model_metadata(depth)
                )
            if loaded:
                return loaded

        raise FileNotFoundError(
            f"No persisted surrogate trees were found under '{tree_models_path}'."
        )
