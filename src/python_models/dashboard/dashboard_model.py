import json
import typing as t
from pathlib import Path

import joblib
import pandas as pd

from src.python_models.dashboard.artifacts import (
    DiagnosisArtifact,
    ManualApiStubResponse,
    StoredShapExplanation,
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
        tree_models: dict[int, SurrogateTreeModel] | None = None,
        symbolic_model: t.Any | None = None,
        sample_indices: list[t.Any] | None = None,
        behaviour_anchor_indices: list[t.Any] | None = None,
        global_shap: StoredShapExplanation | None = None,
        local_shap: StoredShapExplanation | None = None,
        neighbors_frame: pd.DataFrame | None = None,
        surfaces_frame: pd.DataFrame | None = None,
        ice_frame: pd.DataFrame | None = None,
        ale_frame: pd.DataFrame | None = None,
        diagnosis: DiagnosisArtifact | None = None,
        manual_api_stub: ManualApiStubResponse | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.metadata = metadata
        self.dataset_frame = dataset_frame
        self.raw_feature_names = raw_feature_names
        self.transformed_feature_names = transformed_feature_names
        self.tree_models = {
            int(depth): model for depth, model in (tree_models or {}).items()
        }
        self.symbolic_model = symbolic_model
        self.sample_indices = list(sample_indices or [])
        self.behaviour_anchor_indices = list(behaviour_anchor_indices or [])
        self.global_shap = global_shap
        self.local_shap = local_shap
        self.neighbors_frame = (
            neighbors_frame.copy() if neighbors_frame is not None else pd.DataFrame()
        )
        self.surfaces_frame = (
            surfaces_frame.copy() if surfaces_frame is not None else pd.DataFrame()
        )
        self.ice_frame = ice_frame.copy() if ice_frame is not None else pd.DataFrame()
        self.ale_frame = ale_frame.copy() if ale_frame is not None else pd.DataFrame()
        self.diagnosis = diagnosis or DiagnosisArtifact(
            metrics={},
            plot_frame=pd.DataFrame(),
            error_heatmap=pd.DataFrame(),
            financial_warnings=[],
        )
        self.manual_api_stub = manual_api_stub or ManualApiStubResponse(
            prediction=0.0,
            summary="Manual runtime prediction is not available for this bundle.",
            reference_sample_index=self.sample_indices[0] if self.sample_indices else None,
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
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        joblib.dump(self.dataset_frame, root / "dataset_frame.joblib")
        self._dump_optional(root / "global_shap.joblib", self.global_shap)
        self._dump_optional(root / "local_shap.joblib", self.local_shap)
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
        if self.symbolic_model is not None:
            self.symbolic_model.save(root / "symbolic_model")

    @classmethod
    def load(cls, bundle_path: Path) -> "DashboardModel":
        root = cls.get_root_path(bundle_path)
        payload = json.loads(cls.get_metadata_path(root).read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            model_name=payload["model_name"],
            metadata=payload.get("metadata", {}),
            dataset_frame=joblib.load(root / "dataset_frame.joblib"),
            raw_feature_names=list(payload.get("raw_feature_names", [])),
            transformed_feature_names=list(payload.get("transformed_feature_names", [])),
            tree_models=cls._load_tree_models(root),
            symbolic_model=cls._load_symbolic_model(root),
            sample_indices=list(payload.get("sample_indices", [])),
            behaviour_anchor_indices=list(payload.get("behaviour_anchor_indices", [])),
            global_shap=cls._load_optional(root / "global_shap.joblib"),
            local_shap=cls._load_optional(root / "local_shap.joblib"),
            neighbors_frame=cls._load_optional_frame(root / "neighbors_frame.joblib"),
            surfaces_frame=cls._load_optional_frame(root / "surfaces_frame.joblib"),
            ice_frame=cls._load_optional_frame(root / "ice_frame.joblib"),
            ale_frame=cls._load_optional_frame(root / "ale_frame.joblib"),
            diagnosis=cls._load_optional(root / "diagnosis.joblib"),
            manual_api_stub=cls._load_optional(root / "manual_api_stub.joblib"),
        )

    def predictions_for_indices(self, indices: t.Iterable[t.Any]) -> pd.Series:
        return self.dataset_frame.loc[list(indices), "PredictedVolatility"].copy()

    def local_shap_for_index(self, row_index: t.Any) -> StoredShapExplanation:
        if self.local_shap is None:
            raise KeyError("No local SHAP artifact was exported for this model.")
        return self.local_shap.select(row_index)

    def neighbors_for_index(self, row_index: t.Any) -> pd.DataFrame:
        rows = self.neighbors_frame.loc[
            self.neighbors_frame["sample_index"] == row_index
        ].copy() if not self.neighbors_frame.empty else pd.DataFrame()
        if rows.empty:
            return pd.DataFrame()
        neighbors = self.dataset_frame.loc[rows["neighbor_index"]].copy()
        neighbors["distance"] = rows["distance"].to_numpy()
        return neighbors

    def surface_for_anchor(self, anchor_index: t.Any) -> pd.DataFrame:
        if self.surfaces_frame.empty:
            return pd.DataFrame()
        return self.surfaces_frame.loc[
            self.surfaces_frame["anchor_index"] == anchor_index
        ].copy()

    def ice_for_feature(self, feature_name: str) -> pd.DataFrame:
        if self.ice_frame.empty:
            return pd.DataFrame()
        return self.ice_frame.loc[self.ice_frame["feature_name"] == feature_name].copy()

    def ale_for_feature(self, feature_name: str) -> pd.DataFrame:
        if self.ale_frame.empty:
            return pd.DataFrame()
        return self.ale_frame.loc[self.ale_frame["feature_name"] == feature_name].copy()

    @staticmethod
    def _dump_optional(path: Path, value: t.Any | None) -> None:
        if value is not None:
            joblib.dump(value, path)

    @staticmethod
    def _load_optional(path: Path) -> t.Any | None:
        return joblib.load(path) if path.exists() else None

    @staticmethod
    def _load_optional_frame(path: Path) -> pd.DataFrame:
        return joblib.load(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _load_tree_models(root: Path) -> dict[int, SurrogateTreeModel]:
        tree_models: dict[int, SurrogateTreeModel] = {}
        trees_root = root / "tree_models"
        if not trees_root.exists():
            return tree_models
        for tree_path in sorted(trees_root.iterdir(), key=lambda path: path.name):
            if not tree_path.is_dir():
                continue
            try:
                depth = int(tree_path.name.removeprefix("depth_"))
            except ValueError:
                continue
            tree_models[depth] = SurrogateTreeModel.load(tree_path)
        return tree_models

    @staticmethod
    def _load_symbolic_model(root: Path) -> t.Any | None:
        symbolic_root = root / "symbolic_model"
        if not symbolic_root.exists():
            return None
        try:
            from src.python_models.symbolic_regressor_model import SymbolicRegressorModel
        except ImportError:
            return None
        return SymbolicRegressorModel.load(symbolic_root)
