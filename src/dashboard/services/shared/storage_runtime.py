"""Prepare dashboard model bundles for local or GCP-backed storage."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.clientserver_config import ClientserverConfig

logger = logging.getLogger(__name__)


class DashboardModelStorageRuntime:
    """Resolve a local dashboard model directory, downloading from GCS if needed."""

    def __init__(self, clientserver_config: ClientserverConfig) -> None:
        self.clientserver_config = clientserver_config

    def prepare_model_dir(self) -> Path:
        backend = self.clientserver_config.model_storage_backend
        if backend == "local":
            model_dir = self.clientserver_config.dashboard_model_storage_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir
        if backend == "gcp":
            return self._prepare_gcp()
        raise ValueError(f"Unsupported model storage backend: {backend!r}.")

    def _prepare_gcp(self) -> Path:
        gcp = self.clientserver_config.gcp_storage
        if not gcp.dashboard_models_bucket:
            raise ValueError(
                "model_storage.gcp.dashboard_models_bucket is empty. "
                "Set the dashboard bucket in resources/clientserver.json."
            )
        if not gcp.dashboard_models_prefix:
            raise ValueError(
                "model_storage.gcp.dashboard_models_prefix is empty. "
                "Set the prefix where dashboard bundles were uploaded."
            )

        target_dir = self.clientserver_config.local_cache_dir / "dashboard_saved_models"
        target_dir.mkdir(parents=True, exist_ok=True)

        downloaded = self._download_prefix(
            bucket_name=gcp.dashboard_models_bucket,
            prefix=gcp.dashboard_models_prefix,
            destination_dir=target_dir,
            strip_prefix=gcp.dashboard_models_prefix,
        )
        logger.info(
            "Downloaded %s dashboard artifact file(s) from gs://%s/%s into %s",
            downloaded,
            gcp.dashboard_models_bucket,
            gcp.dashboard_models_prefix,
            target_dir,
        )
        return self._resolve_bundle_root_dir(target_dir)

    def _resolve_bundle_root_dir(self, target_dir: Path) -> Path:
        if self._contains_bundle_dirs(target_dir):
            return target_dir

        nested_candidates = [
            path for path in target_dir.iterdir() if path.is_dir()
        ]
        for nested in nested_candidates:
            if self._contains_bundle_dirs(nested):
                logger.warning(
                    "Dashboard bundles were found under nested folder %s. "
                    "Using this folder as runtime model root.",
                    nested,
                )
                return nested

        raise FileNotFoundError(
            "No dashboard bundles were found after syncing from GCP. "
            "Expected files like '<model_id>/metadata.json' under the configured prefix."
        )

    @staticmethod
    def _contains_bundle_dirs(root: Path) -> bool:
        if not root.exists():
            return False
        return any(
            (path / "metadata.json").exists()
            for path in root.iterdir()
            if path.is_dir()
        )

    def _storage_client(self):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-storage is required when model_storage.backend is 'gcp'."
            ) from exc

        credentials_path = self.clientserver_config.gcp_storage.credentials_path
        if credentials_path is None:
            return storage.Client()

        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError(
                "google-auth is required to load GCP service account credentials."
            ) from exc

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return storage.Client(credentials=credentials, project=credentials.project_id)

    def _download_prefix(
        self,
        *,
        bucket_name: str,
        prefix: str,
        destination_dir: Path,
        strip_prefix: str,
    ) -> int:
        client = self._storage_client()
        blobs = list(client.list_blobs(bucket_name, prefix=f"{prefix.strip('/')}/"))
        if not blobs:
            raise FileNotFoundError(
                f"No GCP objects found under gs://{bucket_name}/{prefix}."
            )
        normalized_strip_prefix = strip_prefix.strip("/")
        downloaded = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative_name = blob.name.removeprefix(normalized_strip_prefix).lstrip("/")
            target_path = destination_dir / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target_path))
            downloaded += 1
        return downloaded