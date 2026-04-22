import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GcpModelStorageConfig:
    credentials_path: Path | None
    trained_models_bucket: str
    trained_models_prefix: str
    retrained_metadata_bucket: str
    retrained_metadata_prefix: str
    explainability_artifacts_bucket: str
    dashboard_models_prefix: str


@dataclass(frozen=True)
class LocalModelStorageConfig:
    trained_models_dir: Path
    retrained_metadata_dir: Path
    dashboard_models_dir: Path


class ClientserverConfig:
    def __init__(self, clientserver_config_file_path: Path):
        with open(clientserver_config_file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        cloud_payload = self._load_cloud_config(
            clientserver_config_file_path.parent / "cloud_config.json"
        )

        project_root = clientserver_config_file_path.resolve().parent.parent
        api_config = payload["api"]
        storage_config = payload["model_storage"]
        dashboard_config = payload["dashboard"]

        env_api_base_url = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
        self.api_base_url = (
            env_api_base_url
            if env_api_base_url
            else str(api_config["base_url"]).rstrip("/")
        )
        env_timeout_seconds = os.environ.get("API_TIMEOUT_SECONDS", "").strip()
        self.api_timeout_seconds = (
            float(env_timeout_seconds)
            if env_timeout_seconds
            else float(api_config["timeout_seconds"])
        )
        self.api_cache_entries = int(api_config["cache_entries"])

        env_backend = os.environ.get("MODEL_STORAGE_BACKEND", "").strip().lower()
        self.model_storage_backend = env_backend if env_backend else str(storage_config["backend"]).lower()
        self.local_cache_dir = self._resolve_path(
            project_root,
            storage_config["local_cache_dir"],
        )
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        local_config = storage_config["local"]
        self.local_storage = LocalModelStorageConfig(
            trained_models_dir=self._resolve_path(
                project_root, local_config["trained_models_dir"]
            ),
            retrained_metadata_dir=self._resolve_path(
                project_root, local_config["retrained_metadata_dir"]
            ),
            dashboard_models_dir=self._resolve_path(
                project_root, local_config["dashboard_models_dir"]
            ),
        )

        gcp_config = storage_config["gcp"]
        volatility_bucket = self._first_non_empty(
            gcp_config.get("trained_models_bucket"),
            self._nested_get(cloud_payload, "storage", "volatility_models_bucket"),
        )
        retrained_bucket = self._first_non_empty(
            gcp_config.get("retrained_metadata_bucket"),
            self._nested_get(cloud_payload, "storage", "volatility_models_bucket"),
        )
        dashboard_bucket = self._first_non_empty(
            gcp_config.get("explainability_artifacts_bucket"),
            self._nested_get(cloud_payload, "storage", "explainability_artifacts_bucket"),
        )
        credentials_path = gcp_config.get("credentials_path")
        self.gcp_storage = GcpModelStorageConfig(
            credentials_path=(
                None
                if not credentials_path
                else self._resolve_path(project_root, credentials_path)
            ),
            trained_models_bucket=str(volatility_bucket or ""),
            trained_models_prefix=self._normalize_prefix(
                gcp_config["trained_models_prefix"]
            ),
            retrained_metadata_bucket=str(retrained_bucket or ""),
            retrained_metadata_prefix=self._normalize_prefix(
                gcp_config["retrained_metadata_prefix"]
            ),
            explainability_artifacts_bucket=str(dashboard_bucket or ""),
            dashboard_models_prefix=self._normalize_prefix(
                gcp_config["dashboard_models_prefix"]
            ),
        )

        self.dashboard_model_storage_dir = self._resolve_path(
            project_root,
            dashboard_config["model_storage_dir"],
        )
        self.dashboard_manual_input_features = tuple(
            str(feature) for feature in dashboard_config["manual_input_features"]
        )
        self.dashboard_sample_explainability_neighbors_k = int(
            dashboard_config["sample_explainability_neighbors_k"]
        )

    @staticmethod
    def _resolve_path(project_root: Path, value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else project_root / path

    @staticmethod
    def _normalize_prefix(value: str) -> str:
        return str(value).strip("/")

    @staticmethod
    def _load_cloud_config(cloud_config_path: Path) -> dict[str, Any]:
        if not cloud_config_path.exists():
            return {}
        with open(cloud_config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None
