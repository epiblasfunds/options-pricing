"""Lazy loading for explainable-model bundles and standalone Keras models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import joblib

from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.explainable_model import (
    AbstractModelMetadata,
    ExplainableModel,
    ExplainableModelMetadata,
    SingleModelMetadata,
)


@dataclass(frozen=True)
class LoadedModelBundle:
    """Loaded model plus optional preprocessing metadata."""

    discovered_model: AbstractModelMetadata
    model: object
    metadata: dict
    preprocessor: object | None
    explainable_model: ExplainableModel | None = None


class ModelLoader:
    """Load persisted model artifacts only when they are requested."""

    def __init__(self, cache_size: int = 16) -> None:
        self.cache_size = cache_size
        self._load_cached = lru_cache(maxsize=cache_size)(self._load_uncached)

    def load(self, discovered_model: AbstractModelMetadata) -> LoadedModelBundle:
        return self._load_cached(
            discovered_model.path.as_posix(),
            discovered_model.model_id,
            discovered_model.name,
            discovered_model.format.value,
            json.dumps(discovered_model.metadata, sort_keys=True, default=str),
        )

    def _load_uncached(
        self,
        model_path_str: str,
        model_id: str,
        name: str,
        model_format: str,
        metadata_json: str,
    ) -> LoadedModelBundle:
        model_path = Path(model_path_str)
        metadata = json.loads(metadata_json)
        format_enum = ModelFormatEnum(model_format)

        if format_enum == ModelFormatEnum.EXPLAINABLE_MODEL:
            discovered = ExplainableModelMetadata.load(model_path)
            explainable_model = ExplainableModel.load(discovered)
            preprocessor = self._load_preprocessor(discovered.path, discovered.metadata)
            return LoadedModelBundle(
                discovered_model=discovered,
                model=explainable_model.main_model.model,
                metadata=discovered.metadata,
                preprocessor=preprocessor,
                explainable_model=explainable_model,
            )

        discovered = SingleModelMetadata(
            model_id=model_id,
            name=name,
            path=model_path,
            format=format_enum,
            metadata=metadata,
        )
        keras_models = self._import_keras_models()
        model = keras_models.load_model(model_path)
        preprocessor = self._load_preprocessor(model_path, metadata)
        return LoadedModelBundle(
            discovered_model=discovered,
            model=model,
            metadata=metadata,
            preprocessor=preprocessor,
            explainable_model=None,
        )

    @staticmethod
    def _import_keras_models():
        try:
            from tensorflow.keras import models as keras_models
        except ImportError as exc:
            raise ImportError(
                "TensorFlow/Keras is required to load saved volatility models."
            ) from exc
        return keras_models

    @staticmethod
    def _load_preprocessor(base_path: Path, metadata: dict) -> object | None:
        explicit_preprocessor = metadata.get("preprocessor_path")
        candidates: list[Path] = []
        if explicit_preprocessor:
            explicit_path = Path(explicit_preprocessor)
            if explicit_path.is_absolute():
                candidates.append(explicit_path)
            else:
                parent = base_path if base_path.is_dir() else base_path.parent
                candidates.append(parent / explicit_path)

        if base_path.is_file():
            candidates.extend(
                [
                    base_path.with_suffix(".preprocessor.joblib"),
                    base_path.parent / f"{base_path.stem}.preprocessor.joblib",
                ]
            )
        else:
            candidates.extend(
                [
                    base_path / "preprocessor.joblib",
                    base_path / "epi_blas_model" / "preprocessor.joblib",
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return joblib.load(candidate)
        return None
