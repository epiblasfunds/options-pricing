from pathlib import Path

import joblib

from src.python_models.dashboard.artifacts import (
    DiagnosisArtifact,
    ManualApiStubResponse,
    StoredShapExplanation,
)

ARTIFACT_FILE_NAMES = (
    "global_shap.joblib",
    "local_shap.joblib",
    "diagnosis.joblib",
    "manual_api_stub.joblib",
)
SUPPORTED_TYPES = (
    StoredShapExplanation,
    DiagnosisArtifact,
    ManualApiStubResponse,
)


def migrate_persisted_dashboard_artifacts(bundle_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for path in _iter_dashboard_artifact_paths(bundle_root):
        artifact = joblib.load(path)
        if not isinstance(artifact, SUPPORTED_TYPES):
            continue
        joblib.dump(artifact, path)
        rewritten_paths.append(path)
    return rewritten_paths


def _iter_dashboard_artifact_paths(bundle_root: Path):
    for artifact_name in ARTIFACT_FILE_NAMES:
        for path in bundle_root.rglob(artifact_name):
            yield path
