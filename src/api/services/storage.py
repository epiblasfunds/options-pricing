from dataclasses import dataclass
from pathlib import Path

from src.config.clientserver_config import ClientserverConfig


@dataclass(frozen=True)
class PreparedModelPaths:
    trained_models_dir: Path
    retrained_metadata_dir: Path
    dashboard_model_dir: Path


class ModelStorage:
    """Prepare model artifacts from local disk or Google Cloud Storage."""

    def __init__(self, clientserver_config: ClientserverConfig) -> None:
        self.clientserver_config = clientserver_config

    def prepare_model(self, model_name: str) -> PreparedModelPaths:
        backend = self.clientserver_config.model_storage_backend
        if backend == "local":
            return self._prepare_local(model_name)
        if backend == "gcp":
            return self._prepare_gcp(model_name)
        raise ValueError(f"Unsupported model storage backend: {backend!r}.")

    def _prepare_local(self, model_name: str) -> PreparedModelPaths:
        local = self.clientserver_config.local_storage
        dashboard_model_dir = local.dashboard_models_dir / model_name
        if not dashboard_model_dir.exists():
            raise FileNotFoundError(
                f"Dashboard model '{model_name}' was not found at {dashboard_model_dir}."
            )
        return PreparedModelPaths(
            trained_models_dir=local.trained_models_dir,
            retrained_metadata_dir=local.retrained_metadata_dir,
            dashboard_model_dir=dashboard_model_dir,
        )

    def _prepare_gcp(self, model_name: str) -> PreparedModelPaths:
        gcp = self.clientserver_config.gcp_storage
        target_root = self.clientserver_config.local_cache_dir / model_name
        trained_models_dir = target_root / "trained_models"
        retrained_metadata_dir = target_root / "retrained_metadata"
        dashboard_models_dir = target_root / "dashboard_saved_models"
        dashboard_model_dir = dashboard_models_dir / model_name

        trained_models_dir.mkdir(parents=True, exist_ok=True)
        retrained_metadata_dir.mkdir(parents=True, exist_ok=True)
        dashboard_model_dir.mkdir(parents=True, exist_ok=True)

        self._download_model_artifacts(
            bucket_name=gcp.trained_models_bucket,
            prefix=gcp.trained_models_prefix,
            model_name=model_name,
            destination_dir=trained_models_dir,
        )
        self._download_named_files(
            bucket_name=gcp.retrained_metadata_bucket,
            prefix=gcp.retrained_metadata_prefix,
            filenames=[f"{model_name}_final_test_retrained_metadata.json"],
            destination_dir=retrained_metadata_dir,
        )
        self._download_prefix(
            bucket_name=gcp.explainability_artifacts_bucket,
            prefix=f"{gcp.dashboard_models_prefix}/{model_name}",
            destination_dir=dashboard_model_dir,
            strip_prefix=f"{gcp.dashboard_models_prefix}/{model_name}",
        )

        return PreparedModelPaths(
            trained_models_dir=trained_models_dir,
            retrained_metadata_dir=retrained_metadata_dir,
            dashboard_model_dir=dashboard_model_dir,
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

    def _download_model_artifacts(
        self,
        *,
        bucket_name: str,
        prefix: str,
        model_name: str,
        destination_dir: Path,
    ) -> None:
        artifact_names = [
            f"{model_name}.joblib",
            f"{model_name}.keras",
            f"{model_name}.h5",
            f"{model_name}_scaler.joblib",
        ]
        self._download_named_files(
            bucket_name=bucket_name,
            prefix=prefix,
            filenames=artifact_names,
            destination_dir=destination_dir,
            missing_ok=True,
        )

    def _download_named_files(
        self,
        *,
        bucket_name: str,
        prefix: str,
        filenames: list[str],
        destination_dir: Path,
        missing_ok: bool = False,
    ) -> None:
        client = self._storage_client()
        bucket = client.bucket(bucket_name)
        missing: list[str] = []
        for filename in filenames:
            blob_name = f"{prefix}/{filename}".strip("/")
            blob = bucket.blob(blob_name)
            if not blob.exists():
                missing.append(blob_name)
                continue
            blob.download_to_filename(str(destination_dir / filename))
        if missing and not missing_ok:
            raise FileNotFoundError(
                "Missing required GCP object(s): " + ", ".join(missing)
            )

    def _download_prefix(
        self,
        *,
        bucket_name: str,
        prefix: str,
        destination_dir: Path,
        strip_prefix: str,
    ) -> None:
        client = self._storage_client()
        blobs = list(client.list_blobs(bucket_name, prefix=f"{prefix.strip('/')}/"))
        if not blobs:
            raise FileNotFoundError(
                f"No GCP objects found under gs://{bucket_name}/{prefix}."
            )
        normalized_strip_prefix = strip_prefix.strip("/")
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative_name = blob.name.removeprefix(normalized_strip_prefix).lstrip("/")
            target_path = destination_dir / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target_path))
