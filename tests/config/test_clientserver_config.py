import json
from pathlib import Path

from src.config.clientserver_config.clientserver_config import ClientserverConfig


def test_clientserver_config_reads_local_and_cloud_settings(tmp_path, monkeypatch):
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()
    config_path = resources_dir / "clientserver.json"
    config_path.write_text(
        json.dumps(
            {
                "api": {
                    "base_url": "http://localhost:8000/",
                    "timeout_seconds": 30,
                    "cache_entries": 2,
                },
                "model_storage": {
                    "backend": "gcp",
                    "local_cache_dir": "artifacts/cache",
                    "local": {
                        "trained_models_dir": "trained",
                        "retrained_metadata_dir": "metadata",
                        "dashboard_models_dir": "dashboard",
                    },
                    "gcp": {
                        "credentials_path": "creds.json",
                        "trained_models_bucket": "",
                        "trained_models_prefix": "/trained_models/",
                        "retrained_metadata_bucket": None,
                        "retrained_metadata_prefix": "retrained/",
                        "explainability_artifacts_bucket": "",
                        "dashboard_models_prefix": "/dashboards/",
                    },
                },
                "dashboard": {
                    "model_storage_dir": "dashboard",
                    "manual_input_features": ["OptionType", "Rate"],
                    "sample_explainability_neighbors_k": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    (resources_dir / "cloud_config.json").write_text(
        json.dumps(
            {
                "storage": {
                    "volatility_models_bucket": "vol-bucket",
                    "explainability_artifacts_bucket": "xai-bucket",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com/")
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("MODEL_STORAGE_BACKEND", "LOCAL")

    cfg = ClientserverConfig(config_path)

    project_root = tmp_path
    assert cfg.api_base_url == "https://api.example.com"
    assert cfg.api_timeout_seconds == 45.0
    assert cfg.api_cache_entries == 2
    assert cfg.model_storage_backend == "local"
    assert cfg.local_cache_dir == project_root / "artifacts" / "cache"
    assert cfg.local_storage.trained_models_dir == project_root / "trained"
    assert cfg.local_storage.retrained_metadata_dir == project_root / "metadata"
    assert cfg.local_storage.dashboard_models_dir == project_root / "dashboard"
    assert cfg.gcp_storage.credentials_path == project_root / "creds.json"
    assert cfg.gcp_storage.trained_models_bucket == "vol-bucket"
    assert cfg.gcp_storage.retrained_metadata_bucket == "vol-bucket"
    assert cfg.gcp_storage.explainability_artifacts_bucket == "xai-bucket"
    assert cfg.gcp_storage.trained_models_prefix == "trained_models"
    assert cfg.gcp_storage.retrained_metadata_prefix == "retrained"
    assert cfg.gcp_storage.dashboard_models_prefix == "dashboards"
    assert cfg.dashboard_manual_input_features == ("OptionType", "Rate")
    assert cfg.dashboard_sample_explainability_neighbors_k == 7


def test_clientserver_config_static_helpers_cover_edge_cases():
    assert ClientserverConfig._normalize_prefix("/a/b/") == "a/b"
    assert ClientserverConfig._first_non_empty(None, "", "value") == "value"
    assert ClientserverConfig._first_non_empty(None, " ", 3) == 3
    assert ClientserverConfig._nested_get({"a": {"b": {"c": 9}}}, "a", "b", "c") == 9
    assert ClientserverConfig._nested_get({"a": {}}, "a", "x") is None
    assert ClientserverConfig._resolve_path(Path("C:/root"), "nested/file.txt") == Path(
        "C:/root/nested/file.txt"
    )
