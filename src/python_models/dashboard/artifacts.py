import typing as t
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.enums.volatility_model_enums import ModelFormatEnum


@dataclass
class DashboardBundleMetadata:
    model_id: str
    name: str
    path: Path
    format: ModelFormatEnum
    metadata: dict[str, t.Any]

    @staticmethod
    def get_root_metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path.as_posix(),
            "format": self.format.value,
            "metadata": self.metadata,
        }

    def save(self, path: Path | None = None) -> None:
        bundle_path = path or self.path
        bundle_path.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["path"] = bundle_path.as_posix()
        self.get_root_metadata_path(bundle_path).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "DashboardBundleMetadata":
        payload = json.loads(
            cls.get_root_metadata_path(path).read_text(encoding="utf-8-sig")
        )
        return cls(
            model_id=payload["model_id"],
            name=payload["name"],
            path=path,
            format=ModelFormatEnum(payload["format"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class StoredShapExplanation:
    method: str
    feature_names: list[str]
    index: list[t.Any]
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    display_data: np.ndarray | None
    predictions: np.ndarray
    mean_abs_shap: dict[str, float]

    def waterfall_predictions(self) -> np.ndarray:
        values = np.asarray(self.values, dtype="float64")
        if values.ndim == 1:
            values = values.reshape(1, -1)
        base_values = np.asarray(self.base_values, dtype="float64")
        if base_values.ndim == 0:
            base = np.full(values.shape[0], float(base_values), dtype="float64")
        else:
            base = base_values.reshape(-1).astype("float64", copy=False)
            if base.size == 1 and values.shape[0] != 1:
                base = np.repeat(base, values.shape[0])
        if base.shape[0] != values.shape[0]:
            raise ValueError(
                "Stored SHAP explanation has inconsistent base values and rows."
            )
        return base + values.sum(axis=1)

    def select(self, row_index: t.Any) -> t.Self:
        position = self.index.index(row_index)
        return type(self)(
            method=self.method,
            feature_names=list(self.feature_names),
            index=[row_index],
            values=np.asarray(self.values[position: position + 1]),
            base_values=np.asarray(self.base_values[position: position + 1]),
            data=np.asarray(self.data[position: position + 1]),
            display_data=(
                None
                if self.display_data is None
                else np.asarray(self.display_data[position: position + 1])
            ),
            predictions=np.asarray(self.predictions[position: position + 1]),
            mean_abs_shap=dict(self.mean_abs_shap),
        )

    def to_explanation(self):
        import shap

        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            display_data=self.display_data,
            feature_names=self.feature_names,
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
    def load(cls, path: Path) -> "SurrogateTreeModel":
        model = joblib.load(cls._get_model_file_path(path=path))
        feature_importances = pd.read_csv(
            cls._get_feat_importance_file_path(path=path), index_col=0
        ).iloc[:, 0]
        fidelity_frame = pd.read_csv(
            cls._get_fidelity_frame_file_path(path=path), index_col=0
        )
        attrs = json.loads(
            cls._get_attrs_file_path(path=path).read_text(encoding="utf-8")
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
class DiagnosisArtifact:
    metrics: dict[str, float]
    plot_frame: pd.DataFrame
    error_heatmap: pd.DataFrame
    financial_warnings: list[str]


@dataclass
class ManualApiStubResponse:
    prediction: float
    summary: str
    reference_sample_index: t.Any | None = None


@dataclass
class StoredNeighborsProjectionPca:
    feature_names: list[str]
    fill_values: dict[str, float]
    scale_values: dict[str, float]
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform_frame(
        self,
        frame: pd.DataFrame,
        *,
        dimensions: int = 3,
    ) -> np.ndarray:
        dimensions = max(1, int(dimensions))
        if frame.empty:
            return np.zeros((0, dimensions), dtype="float64")
        if not self.feature_names:
            return np.zeros((len(frame), dimensions), dtype="float64")

        matrix = frame.reindex(columns=self.feature_names).apply(
            pd.to_numeric,
            errors="coerce",
        )
        components = np.asarray(self.components, dtype="float64")
        fill = pd.Series(self.fill_values, index=self.feature_names, dtype="float64")
        scale = pd.Series(self.scale_values, index=self.feature_names, dtype="float64")
        scale = scale.replace(0.0, 1.0).fillna(1.0)
        standardized = (matrix.fillna(fill) - fill) / scale

        component_count = min(dimensions, int(components.shape[0]))
        if component_count == 0:
            return np.zeros((len(frame), dimensions), dtype="float64")
        coords = standardized.to_numpy(dtype="float64") @ components[:component_count].T
        if coords.shape[1] < dimensions:
            coords = np.column_stack(
                [
                    coords,
                    np.zeros(
                        (len(frame), dimensions - coords.shape[1]),
                        dtype="float64",
                    ),
                ]
            )
        return coords
