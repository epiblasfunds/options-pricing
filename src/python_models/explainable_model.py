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
from src.volatility_models.trained_model import TrainedModel, TrainedModelMetadata


@dataclass
class AbstractModelMetadata(ABC):
    """Metadata for one model artifact on disk."""

    model_id: str
    name: str
    path: Path
    format: ModelFormatEnum
    metadata: dict[str, t.Any]

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

    trained_model_format: ModelFormatEnum
    tree_format: ModelFormatEnum

    @staticmethod
    def get_root_metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def get_trained_model_path(self) -> Path:
        return self.path / "trained_model"

    def get_trained_model_metadata(self) -> TrainedModelMetadata:
        payload = dict(self.metadata)
        return TrainedModelMetadata(
            model_id=f"{self.model_id}_trained_model",
            name=f"{self.name}_trained_model",
            path=self.get_trained_model_path(),
            format=self.trained_model_format,
            feature_names=tuple(payload.get("model_input_features", [])),
            target_column=str(payload.get("target_column", "ImpliedVolatility")),
            loss_name=str(payload.get("loss_name", "loss")),
            metadata=payload,
        )

    def get_tree_models_path(self) -> Path:
        return self.path / "tree_models"

    def get_tree_model_path(self, depth: int) -> Path:
        return self.get_tree_models_path() / f"depth_{int(depth)}"

    def get_tree_model_metadata(self, depth: int) -> SingleModelMetadata:
        return SingleModelMetadata(
            model_id=f"{self.model_id}_tree_model_depth_{int(depth)}",
            name=f"{self.name}_tree_model_depth_{int(depth)}",
            path=self.get_tree_model_path(depth),
            format=self.tree_format,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, t.Any]:
        payload = super().to_dict()
        payload.update(
            {
                "trained_model_format": self.trained_model_format.value,
                "tree_format": self.tree_format.value,
            }
        )
        return payload

    def save(self, path: Path | None = None) -> None:
        bundle_path = path or self.path
        bundle_path.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["path"] = bundle_path.as_posix()
        self.get_root_metadata_path(bundle_path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ExplainableModelMetadata":
        payload = json.loads(cls.get_root_metadata_path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            name=payload["name"],
            path=path,
            format=ModelFormatEnum(payload["format"]),
            trained_model_format=ModelFormatEnum(payload["trained_model_format"]),
            tree_format=ModelFormatEnum(payload["tree_format"]),
            metadata=dict(payload.get("metadata", {})),
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


class ExplainableModel:
    def __init__(
        self,
        main_model: TrainedModel,
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
        main_model = TrainedModel.load(metadata=metadata.get_trained_model_metadata())
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
        self.main_model.save(bundle_path / "trained_model")
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
