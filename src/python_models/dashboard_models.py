from __future__ import annotations

import json
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.python_models.dashboard_artifacts import DiagnosisArtifact
from src.python_models.dashboard_artifacts import ManualApiStubResponse
from src.python_models.dashboard_artifacts import StoredShapExplanation
from src.python_models.dashboard_model_builders import (
    build_dashboard_model,
    load_dashboard_tree_models,
)
from src.python_models.explainable_model import (
    ExplainableModel,
    SurrogateTreeModel,
)


class DashboardModel:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        metadata: dict[str, t.Any],
        dataset_frame: pd.DataFrame,
        raw_feature_names: list[str],
        transformed_feature_names: list[str],
        tree_models: dict[int, SurrogateTreeModel],
        sample_indices: list[t.Any],
        behaviour_anchor_indices: list[t.Any],
        global_shap: StoredShapExplanation,
        local_shap: StoredShapExplanation,
        neighbors_frame: pd.DataFrame,
        surfaces_frame: pd.DataFrame,
        ice_frame: pd.DataFrame,
        ale_frame: pd.DataFrame,
        diagnosis: DiagnosisArtifact,
        manual_api_stub: ManualApiStubResponse,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.metadata = metadata
        self.dataset_frame = dataset_frame
        self.raw_feature_names = raw_feature_names
        self.transformed_feature_names = transformed_feature_names
        self.tree_models = {int(depth): model for depth, model in tree_models.items()}
        self.sample_indices = list(sample_indices)
        self.behaviour_anchor_indices = list(behaviour_anchor_indices)
        self.global_shap = global_shap
        self.local_shap = local_shap
        self.neighbors_frame = neighbors_frame
        self.surfaces_frame = surfaces_frame
        self.ice_frame = ice_frame
        self.ale_frame = ale_frame
        self.diagnosis = diagnosis
        self.manual_api_stub = manual_api_stub

    @classmethod
    def from_model(
        cls,
        model: ExplainableModel,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        preprocessor: ColumnTransformer | None = None,
    ) -> "DashboardModel":
        del preprocessor
        return build_dashboard_model(
            cls,
            model=model,
            X=X,
            y=y,
        )

    @staticmethod
    def get_root_path(bundle_path: Path) -> Path:
        return bundle_path / "dashboard_model"

    @staticmethod
    def get_metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def save(self, bundle_path: Path) -> None:
        root = self.get_root_path(bundle_path)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "raw_feature_names": self.raw_feature_names,
            "transformed_feature_names": self.transformed_feature_names,
            "sample_indices": list(self.sample_indices),
            "behaviour_anchor_indices": list(self.behaviour_anchor_indices),
            "metadata": self.metadata,
        }
        self.get_metadata_path(root).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        joblib.dump(self.dataset_frame, root / "dataset_frame.joblib")
        joblib.dump(self.global_shap, root / "global_shap.joblib")
        joblib.dump(self.local_shap, root / "local_shap.joblib")
        joblib.dump(self.neighbors_frame, root / "neighbors_frame.joblib")
        joblib.dump(self.surfaces_frame, root / "surfaces_frame.joblib")
        joblib.dump(self.ice_frame, root / "ice_frame.joblib")
        joblib.dump(self.ale_frame, root / "ale_frame.joblib")
        joblib.dump(self.diagnosis, root / "diagnosis.joblib")
        joblib.dump(self.manual_api_stub, root / "manual_api_stub.joblib")
        trees_root = root / "tree_models"
        trees_root.mkdir(parents=True, exist_ok=True)
        for depth, tree_model in sorted(self.tree_models.items()):
            tree_model.save(trees_root / f"depth_{int(depth)}")

    @classmethod
    def load(cls, bundle_path: Path) -> "DashboardModel":
        root = cls.get_root_path(bundle_path)
        payload = json.loads(cls.get_metadata_path(root).read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            model_name=payload["model_name"],
            metadata=payload.get("metadata", {}),
            dataset_frame=joblib.load(root / "dataset_frame.joblib"),
            raw_feature_names=list(payload["raw_feature_names"]),
            transformed_feature_names=list(payload["transformed_feature_names"]),
            tree_models=load_dashboard_tree_models(root, payload),
            sample_indices=list(payload.get("sample_indices", [])),
            behaviour_anchor_indices=list(payload.get("behaviour_anchor_indices", [])),
            global_shap=joblib.load(root / "global_shap.joblib"),
            local_shap=joblib.load(root / "local_shap.joblib"),
            neighbors_frame=joblib.load(root / "neighbors_frame.joblib"),
            surfaces_frame=joblib.load(root / "surfaces_frame.joblib"),
            ice_frame=joblib.load(root / "ice_frame.joblib"),
            ale_frame=joblib.load(root / "ale_frame.joblib"),
            diagnosis=joblib.load(root / "diagnosis.joblib"),
            manual_api_stub=joblib.load(root / "manual_api_stub.joblib"),
        )

    def predictions_for_indices(self, indices: t.Iterable[t.Any]) -> pd.Series:
        return self.dataset_frame.loc[list(indices), "PredictedVolatility"].copy()

    def local_shap_for_index(self, row_index: t.Any) -> StoredShapExplanation:
        return self.local_shap.select(row_index)

    def neighbors_for_index(self, row_index: t.Any) -> pd.DataFrame:
        rows = self.neighbors_frame.loc[
            self.neighbors_frame["sample_index"] == row_index
        ].copy()
        if rows.empty:
            return pd.DataFrame()
        neighbors = self.dataset_frame.loc[rows["neighbor_index"]].copy()
        neighbors["distance"] = rows["distance"].to_numpy()
        return neighbors

    def surface_for_anchor(self, anchor_index: t.Any) -> pd.DataFrame:
        return self.surfaces_frame.loc[
            self.surfaces_frame["anchor_index"] == anchor_index
        ].copy()

    def ice_for_feature(self, feature_name: str) -> pd.DataFrame:
        return self.ice_frame.loc[self.ice_frame["feature_name"] == feature_name].copy()

    def ale_for_feature(self, feature_name: str) -> pd.DataFrame:
        return self.ale_frame.loc[self.ale_frame["feature_name"] == feature_name].copy()
