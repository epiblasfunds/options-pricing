import json

import joblib

from src.volatility_models.model_explainability.services.model_loader import KerasModelLoader
from src.volatility_models.model_explainability.services.model_registry import ModelRegistry


class _FakeKerasModels:
    def __init__(self):
        self.loaded_paths = []

    def load_model(self, path):
        self.loaded_paths.append(str(path))
        return {"loaded": str(path)}


def test_model_registry_discovers_supported_artifacts(tmp_path):
    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    (model_dir / "vol_model.keras").write_text("dummy", encoding="utf-8")
    (model_dir / "vol_model.metadata.json").write_text(
        json.dumps({"model_input_features": ["a", "b"]}),
        encoding="utf-8",
    )
    saved_model_dir = model_dir / "saved_tree"
    saved_model_dir.mkdir()
    (saved_model_dir / "saved_model.pb").write_text("dummy", encoding="utf-8")

    discovered = ModelRegistry(model_dir).discover_models()

    assert len(discovered) == 2
    assert {model.format for model in discovered} == {"keras", "saved_model"}


def test_model_loader_uses_sidecar_preprocessor(tmp_path, monkeypatch):
    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    model_path = model_dir / "vol_model.keras"
    model_path.write_text("dummy", encoding="utf-8")
    preprocessor_path = model_dir / "vol_model.preprocessor.joblib"
    joblib.dump({"ok": True}, preprocessor_path)

    registry = ModelRegistry(model_dir)
    discovered = registry.discover_models()[0]
    loader = KerasModelLoader()
    fake_models = _FakeKerasModels()
    monkeypatch.setattr(loader, "_import_keras_models", lambda: fake_models)

    bundle = loader.load(discovered)

    assert bundle.model == {"loaded": str(model_path)}
    assert bundle.preprocessor == {"ok": True}
