import json

from src.dashboard.services.shared.model_registry import ModelRegistry


def _write_bundle(
        path,
        *,
        fmt="explainable_model",
        builder="src.model2dashboard.run_pipeline",
        with_dashboard_metadata=True
):
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": path.name,
                "name": path.name,
                "path": path.as_posix(),
                "format": fmt,
                "metadata": {"builder": builder},
            }
        ),
        encoding="utf-8",
    )
    if with_dashboard_metadata:
        dashboard_dir = path / "dashboard_model"
        dashboard_dir.mkdir(exist_ok=True)
        (dashboard_dir / "metadata.json").write_text("{}", encoding="utf-8")


def test_model_registry_discovers_only_valid_dashboard_bundles(tmp_path):
    _write_bundle(tmp_path / "valid_a")
    _write_bundle(tmp_path / "valid_b")
    _write_bundle(tmp_path / "wrong_format", fmt="joblib")
    _write_bundle(tmp_path / "wrong_builder", builder="other")
    _write_bundle(tmp_path / "missing_dashboard", with_dashboard_metadata=False)

    registry = ModelRegistry(tmp_path)
    discovered = registry.discover_models()

    assert [model.model_id for model in discovered] == ["valid_a", "valid_b"]
    assert registry.get_model("valid_b").model_id == "valid_b"
    assert registry.get_model("missing") is None
