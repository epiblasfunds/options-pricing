"""Model discovery utilities for dashboard-ready explainability bundles."""

import json
from pathlib import Path

from src.config.config import DASHBOARD_SAVED_MODELS_DIR_PATH
from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.explainable_artifacts import AbstractModelMetadata
from src.python_models.explainable_artifacts import ExplainableModelMetadata


class ModelRegistry:
    """Discover explainable-model bundles exported with dashboard artifacts."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or DASHBOARD_SAVED_MODELS_DIR_PATH
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def discover_models(self) -> list[AbstractModelMetadata]:
        models: list[AbstractModelMetadata] = []
        for artifact in sorted(
            self.model_dir.iterdir(), key=lambda path: path.name.lower()
        ):
            discovered = self._build_discovered_model(artifact)
            if discovered is not None:
                models.append(discovered)
        return models

    def get_model(self, model_id: str) -> AbstractModelMetadata | None:
        for model in self.discover_models():
            if model.model_id == model_id:
                return model
        return None

    def _build_discovered_model(self, artifact: Path) -> AbstractModelMetadata | None:
        if artifact.is_dir() and self._looks_like_dashboard_bundle_directory(artifact):
            return ExplainableModelMetadata.load(artifact)
        return None

    @staticmethod
    def _looks_like_dashboard_bundle_directory(path: Path) -> bool:
        metadata_path = ExplainableModelMetadata.get_root_metadata_path(path)
        if not metadata_path.exists():
            return False
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("format") != ModelFormatEnum.EXPLAINABLE_MODEL.value:
            return False
        return (path / "dashboard_model" / "metadata.json").exists()
