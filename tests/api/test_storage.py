from pathlib import Path
from types import SimpleNamespace

import pytest

from src.api.services.storage import ModelStorage


def _storage_config(tmp_path, backend: str):
    return SimpleNamespace(
        model_storage_backend=backend,
        local_cache_dir=tmp_path / "cache",
        local_storage=SimpleNamespace(
            trained_models_dir=tmp_path / "trained",
            retrained_metadata_dir=tmp_path / "metadata",
            dashboard_models_dir=tmp_path / "dashboard",
        ),
        gcp_storage=SimpleNamespace(
            credentials_path=None,
            trained_models_bucket="trained-bucket",
            trained_models_prefix="trained",
            retrained_metadata_bucket="metadata-bucket",
            retrained_metadata_prefix="metadata",
            explainability_artifacts_bucket="dashboard-bucket",
            dashboard_models_prefix="dashboards",
        ),
    )


def test_prepare_model_local_success_and_missing_path(tmp_path):
    config = _storage_config(tmp_path, backend="local")
    dashboard_model_dir = config.local_storage.dashboard_models_dir / "rf"
    dashboard_model_dir.mkdir(parents=True)
    storage = ModelStorage(config)

    prepared = storage.prepare_model("rf")

    assert prepared.dashboard_model_dir == dashboard_model_dir
    assert prepared.trained_models_dir == config.local_storage.trained_models_dir
    assert prepared.retrained_metadata_dir == config.local_storage.retrained_metadata_dir

    with pytest.raises(FileNotFoundError):
        storage.prepare_model("missing")


def test_prepare_model_rejects_unknown_backend(tmp_path):
    storage = ModelStorage(_storage_config(tmp_path, backend="unknown"))

    with pytest.raises(ValueError, match="Unsupported model storage backend"):
        storage.prepare_model("rf")


def test_prepare_gcp_creates_cache_and_delegates_downloads(tmp_path, monkeypatch):
    config = _storage_config(tmp_path, backend="gcp")
    storage = ModelStorage(config)
    calls: list[tuple[str, str, str, Path]] = []

    monkeypatch.setattr(
        storage,
        "_download_model_artifacts",
        lambda **kwargs: calls.append(
            (
                "artifacts",
                kwargs["bucket_name"],
                kwargs["model_name"],
                kwargs["destination_dir"],
            )
        ),
    )
    monkeypatch.setattr(
        storage,
        "_download_named_files",
        lambda **kwargs: calls.append(
            (
                "metadata",
                kwargs["bucket_name"],
                kwargs["filenames"][0],
                kwargs["destination_dir"],
            )
        ),
    )
    monkeypatch.setattr(
        storage,
        "_download_prefix",
        lambda **kwargs: calls.append(
            (
                "dashboard",
                kwargs["bucket_name"],
                kwargs["prefix"],
                kwargs["destination_dir"],
            )
        ),
    )

    prepared = storage.prepare_model("rf")

    assert prepared.trained_models_dir.exists()
    assert prepared.retrained_metadata_dir.exists()
    assert prepared.dashboard_model_dir.exists()
    assert calls == [
        ("artifacts", "trained-bucket", "rf", prepared.trained_models_dir),
        (
            "metadata",
            "metadata-bucket",
            "rf_final_test_retrained_metadata.json",
            prepared.retrained_metadata_dir,
        ),
        (
            "dashboard",
            "dashboard-bucket",
            "dashboards/rf",
            prepared.dashboard_model_dir,
        ),
    ]


class _FakeBlob:
    def __init__(self, name: str, exists: bool = True) -> None:
        self.name = name
        self._exists = exists
        self.downloads: list[str] = []

    def exists(self) -> bool:
        return self._exists

    def download_to_filename(self, target: str) -> None:
        self.downloads.append(target)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("x", encoding="utf-8")


class _FakeBucket:
    def __init__(self, blobs: dict[str, _FakeBlob]) -> None:
        self._blobs = blobs

    def blob(self, name: str) -> _FakeBlob:
        return self._blobs[name]


class _FakeClient:
    def __init__(self, bucket_map: dict[str, _FakeBucket], blobs=None) -> None:
        self._bucket_map = bucket_map
        self._list_blobs = blobs or []

    def bucket(self, name: str) -> _FakeBucket:
        return self._bucket_map[name]

    def list_blobs(self, bucket_name: str, prefix: str):
        assert bucket_name == "dashboard-bucket"
        assert prefix == "dashboards/rf/"
        return list(self._list_blobs)


def test_download_named_files_handles_missing_required_and_optional_objects(
    tmp_path,
    monkeypatch,
):
    existing = _FakeBlob("trained/rf.joblib", exists=True)
    missing = _FakeBlob("trained/rf.keras", exists=False)
    storage = ModelStorage(_storage_config(tmp_path, backend="gcp"))
    client = _FakeClient(
        {
            "trained-bucket": _FakeBucket(
                {
                    "trained/rf.joblib": existing,
                    "trained/rf.keras": missing,
                }
            )
        }
    )
    monkeypatch.setattr(storage, "_storage_client", lambda: client)

    with pytest.raises(FileNotFoundError, match="trained/rf.keras"):
        storage._download_named_files(
            bucket_name="trained-bucket",
            prefix="trained",
            filenames=["rf.joblib", "rf.keras"],
            destination_dir=tmp_path,
        )

    storage._download_named_files(
        bucket_name="trained-bucket",
        prefix="trained",
        filenames=["rf.joblib", "rf.keras"],
        destination_dir=tmp_path,
        missing_ok=True,
    )

    assert existing.downloads


def test_download_prefix_downloads_tree_and_rejects_empty_prefix(tmp_path, monkeypatch):
    storage = ModelStorage(_storage_config(tmp_path, backend="gcp"))
    blob_a = _FakeBlob("dashboards/rf/metadata.json")
    blob_b = _FakeBlob("dashboards/rf/subdir/model.joblib")
    client = _FakeClient({}, blobs=[blob_a, blob_b])
    monkeypatch.setattr(storage, "_storage_client", lambda: client)

    destination = tmp_path / "bundle"
    storage._download_prefix(
        bucket_name="dashboard-bucket",
        prefix="dashboards/rf",
        destination_dir=destination,
        strip_prefix="dashboards/rf",
    )

    assert (destination / "metadata.json").exists()
    assert (destination / "subdir" / "model.joblib").exists()

    empty_client = _FakeClient({}, blobs=[])
    monkeypatch.setattr(storage, "_storage_client", lambda: empty_client)
    with pytest.raises(FileNotFoundError, match="No GCP objects found"):
        storage._download_prefix(
            bucket_name="dashboard-bucket",
            prefix="dashboards/rf",
            destination_dir=destination,
            strip_prefix="dashboards/rf",
        )
