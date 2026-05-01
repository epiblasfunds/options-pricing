import json
from types import SimpleNamespace

import pytest

from src.dashboard.services.shared.model_loader import ModelLoader
from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.dashboard.artifacts import DashboardBundleMetadata


def _write_bundle_metadata(path, *, model_format=ModelFormatEnum.EXPLAINABLE_MODEL):
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": path.name,
                "name": path.name,
                "path": path.as_posix(),
                "format": model_format.value,
                "metadata": {"ok": True},
            }
        ),
        encoding="utf-8",
    )


def test_model_loader_uses_cache_and_loads_dashboard_model(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle"
    dashboard_root = bundle_path / "dashboard_model"
    dashboard_root.mkdir(parents=True)
    _write_bundle_metadata(bundle_path)

    discovered = DashboardBundleMetadata.load(bundle_path)
    loader = ModelLoader(cache_size=2)
    fake_dashboard_model = SimpleNamespace(dataset_frame="frame")
    calls = {"count": 0}

    monkeypatch.setattr(
        "src.dashboard.services.shared.model_loader.DashboardModel.load",
        lambda path: calls.__setitem__("count", calls["count"] + 1) or fake_dashboard_model,
    )

    first = loader.load(discovered)
    second = loader.load(discovered)

    assert first is second
    assert first.dashboard_model is fake_dashboard_model
    assert calls["count"] == 1


def test_model_loader_rejects_unsupported_formats(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle_metadata(bundle_path, model_format=ModelFormatEnum.JOBLIB)
    discovered = DashboardBundleMetadata.load(bundle_path)
    loader = ModelLoader()

    with pytest.raises(ValueError, match="Unsupported model format"):
        loader.load(discovered)


def test_model_loader_requires_dashboard_artifacts_directory(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle_metadata(bundle_path)
    discovered = DashboardBundleMetadata.load(bundle_path)
    loader = ModelLoader()

    with pytest.raises(FileNotFoundError, match="Dashboard artifacts were not found"):
        loader.load(discovered)
