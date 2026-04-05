"""Model discovery utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.explainable_model import (
    AbstractModelMetadata,
    ExplainableModelMetadata,
    SingleModelMetadata,
)


class ModelRegistry:
    """Discover explainable-model bundles and standalone Keras artifacts."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def discover_models(self) -> list[AbstractModelMetadata]:
        models: list[AbstractModelMetadata] = []
        for artifact in sorted(self.model_dir.iterdir(), key=lambda path: path.name.lower()):
            discovered = self._build_discovered_model(artifact)
            if discovered is not None:
                models.append(discovered)
        return models

    def get_model(self, model_id: str) -> AbstractModelMetadata | None:
        for model in self.discover_models():
            if model.model_id == model_id:
                return model
        return None

    def _build_discovered_model(
        self, artifact: Path
    ) -> AbstractModelMetadata | None:
        if artifact.is_dir() and self._looks_like_explainable_model_directory(artifact):
            return ExplainableModelMetadata.load(artifact)

        if artifact.is_file() and artifact.suffix.lower() in {".keras", ".h5"}:
            format_name = ModelFormatEnum(artifact.suffix.lower().lstrip("."))
        else:
            return None

        metadata = self._load_metadata(artifact)
        model_id = artifact.relative_to(self.model_dir).as_posix()
        return SingleModelMetadata(
            model_id=model_id,
            name=artifact.stem,
            path=artifact,
            format=format_name,
            metadata=metadata,
        )

    @staticmethod
    def _looks_like_explainable_model_directory(path: Path) -> bool:
        metadata_path = ExplainableModelMetadata.get_root_metadata_path(path)
        if not metadata_path.exists():
            return False
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return payload.get("format") == ModelFormatEnum.EXPLAINABLE_MODEL.value

    def _load_metadata(self, artifact: Path) -> dict:
        candidates = [
            artifact.with_suffix(".metadata.json"),
            artifact.parent / f"{artifact.stem}.metadata.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        return {}
