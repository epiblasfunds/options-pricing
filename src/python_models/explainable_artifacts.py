import json
import typing as t
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.enums.volatility_model_enums import ModelFormatEnum
from src.volatility_models.trained_model import TrainedModelMetadata


@dataclass
class AbstractModelMetadata(ABC):
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
    pass


@dataclass
class ExplainableModelMetadata(AbstractModelMetadata):
    trained_model_format: ModelFormatEnum
    tree_format: ModelFormatEnum
    symbolic_format: ModelFormatEnum

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

    def get_symbolic_model_path(self) -> Path:
        return self.path / "symbolic_model"

    def get_symbolic_model_metadata(self) -> SingleModelMetadata:
        return SingleModelMetadata(
            model_id=f"{self.model_id}_symbolic_model",
            name=f"{self.name}_symbolic_model",
            path=self.get_symbolic_model_path(),
            format=self.symbolic_format,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, t.Any]:
        payload = super().to_dict()
        payload.update(
            {
                "trained_model_format": self.trained_model_format.value,
                "tree_format": self.tree_format.value,
                "symbolic_format": self.symbolic_format.value,
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
        payload = json.loads(
            cls.get_root_metadata_path(path).read_text(encoding="utf-8")
        )
        return cls(
            model_id=payload["model_id"],
            name=payload["name"],
            path=path,
            format=ModelFormatEnum(payload["format"]),
            trained_model_format=ModelFormatEnum(payload["trained_model_format"]),
            tree_format=ModelFormatEnum(payload["tree_format"]),
            symbolic_format=ModelFormatEnum(
                payload.get("symbolic_format", ModelFormatEnum.JOBLIB.value)
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class SurrogateTreeModel:
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
