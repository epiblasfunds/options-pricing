"""Lazy loading for dashboard-ready explainable-model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.dashboard_models import DashboardModel
from src.python_models.explainable_model import AbstractModelMetadata
from src.python_models.explainable_model import ExplainableModelMetadata


@dataclass(frozen=True)
class LoadedModelBundle:
    """Loaded dashboard-ready bundle metadata and precalculated artifacts."""

    discovered_model: AbstractModelMetadata
    metadata: dict
    dashboard_model: DashboardModel


class ModelLoader:
    """Load persisted dashboard artifacts only when they are requested."""

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

        if format_enum != ModelFormatEnum.EXPLAINABLE_MODEL:
            raise ValueError(
                f"Unsupported model format for the dashboard runtime: {format_enum.value}."
            )
        discovered = ExplainableModelMetadata.load(model_path)
        dashboard_root = DashboardModel.get_root_path(discovered.path)
        if not dashboard_root.exists():
            raise FileNotFoundError(
                f"Dashboard artifacts were not found for bundle '{discovered.model_id}'. "
                "Re-export the model with DashboardModel artifacts."
            )
        return LoadedModelBundle(
            discovered_model=discovered,
            metadata=discovered.metadata,
            dashboard_model=DashboardModel.load(discovered.path),
        )
