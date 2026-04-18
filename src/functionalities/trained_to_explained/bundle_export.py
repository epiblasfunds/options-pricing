import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.config import DASHBOARD_SAVED_MODELS_DIR_PATH
from src.config.config import VOLATILITY_TRAINED_MODELS_DIR_PATH
from src.config.config import config
from src.enums.volatility_model_enums import ModelFormatEnum
from src.functionalities.trained_to_explained.reference_data import (
    discover_trained_model_metadata,
)
from src.functionalities.trained_to_explained.reference_data import (
    load_reference_trade_frame,
)
from src.functionalities.trained_to_explained.symbolic_surrogate import (
    build_symbolic_regressor_model,
)
from src.python_models.dashboard.dashboard_model import DashboardModel
from src.python_models.explainable_model import ExplainableModel
from src.python_models.explainable_model import ExplainableModelMetadata
from src.volatility_models import TARGET_COLUMN
from src.volatility_models import TrainedModel
from src.volatility_models import select_trade_columns


@dataclass(frozen=True)
class ExportedBundlePaths:
    trained_model_path: Path
    explainable_bundle_path: Path
    dashboard_metadata_path: Path


def export_explainable_bundle(
    *,
    trained_model: TrainedModel,
    reference_frame: pd.DataFrame,
    bundle_name: str,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
) -> ExportedBundlePaths:
    bundle_path = bundle_dir / bundle_name
    raw_reference = select_trade_columns(reference_frame)
    y_reference = raw_reference[str(TARGET_COLUMN)]
    model_features = list(trained_model.metadata.feature_names)
    explainable_metadata = _build_explainable_metadata(
        trained_model=trained_model,
        bundle_path=bundle_path,
        model_features=model_features,
    )
    symbolic_model = build_symbolic_regressor_model(
        trained_model=trained_model,
        dataset_frame=raw_reference,
        raw_frame=raw_reference,
        model_input_features=model_features,
    )
    seed_explainable_model = ExplainableModel(
        main_model=trained_model,
        tree_models={},
        symbolic_model=symbolic_model,
        metadata=explainable_metadata,
    )
    dashboard_model = DashboardModel.from_model(
        seed_explainable_model,
        raw_reference.drop(columns=[str(TARGET_COLUMN)]),
        y_reference,
    )
    final_metadata = _build_final_metadata(
        trained_model=trained_model,
        bundle_path=bundle_path,
        base_metadata=explainable_metadata,
        dashboard_model=dashboard_model,
        symbolic_model=symbolic_model,
    )
    explainable_model = ExplainableModel(
        main_model=trained_model,
        tree_models=dashboard_model.tree_models,
        symbolic_model=symbolic_model,
        metadata=final_metadata,
    )
    explainable_model.save(bundle_path)
    dashboard_model.metadata = dict(final_metadata.metadata)
    dashboard_model.save(bundle_path)
    return ExportedBundlePaths(
        trained_model_path=trained_model.metadata.path,
        explainable_bundle_path=bundle_path,
        dashboard_metadata_path=DashboardModel.get_metadata_path(
            DashboardModel.get_root_path(bundle_path)
        ),
    )


def export_all_trained_models(
    *,
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
    reference_frame: pd.DataFrame | None = None,
) -> list[ExportedBundlePaths]:
    export_reference = (
        reference_frame if reference_frame is not None else load_reference_trade_frame()
    )
    exported: list[ExportedBundlePaths] = []
    for metadata in discover_trained_model_metadata(trained_models_dir):
        trained_model = TrainedModel.load(metadata)
        exported.append(
            export_explainable_bundle(
                trained_model=trained_model,
                reference_frame=export_reference,
                bundle_name=metadata.model_id,
                bundle_dir=bundle_dir,
            )
        )
    return exported


def rebuild_dashboard_saved_models(
    *,
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
    reference_frame: pd.DataFrame | None = None,
) -> list[ExportedBundlePaths]:
    backup_dir = _backup_saved_models(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    try:
        exported = export_all_trained_models(
            trained_models_dir=trained_models_dir,
            bundle_dir=bundle_dir,
            reference_frame=reference_frame,
        )
    except Exception:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        if backup_dir is not None and backup_dir.exists():
            backup_dir.rename(bundle_dir)
        raise
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)
    return exported


def _build_explainable_metadata(
    *,
    trained_model: TrainedModel,
    bundle_path: Path,
    model_features: list[str],
) -> ExplainableModelMetadata:
    return ExplainableModelMetadata(
        model_id=trained_model.metadata.model_id,
        name=trained_model.metadata.name,
        path=bundle_path,
        format=ModelFormatEnum.EXPLAINABLE_MODEL,
        trained_model_format=trained_model.metadata.format,
        tree_format=ModelFormatEnum.JOBLIB,
        symbolic_format=ModelFormatEnum.JOBLIB,
        metadata={
            "model_input_features": model_features,
            "transformed_feature_names": model_features,
            "target_column": str(TARGET_COLUMN),
            "error_metrics": list(config.dashboard_models_config.error_metrics),
            "loss_name": trained_model.metadata.loss_name,
            "trained_model_path": trained_model.metadata.path.as_posix(),
        },
    )


def _build_final_metadata(
    *,
    trained_model: TrainedModel,
    bundle_path: Path,
    base_metadata: ExplainableModelMetadata,
    dashboard_model: DashboardModel,
    symbolic_model,
) -> ExplainableModelMetadata:
    return ExplainableModelMetadata(
        model_id=trained_model.metadata.model_id,
        name=trained_model.metadata.name,
        path=bundle_path,
        format=ModelFormatEnum.EXPLAINABLE_MODEL,
        trained_model_format=trained_model.metadata.format,
        tree_format=ModelFormatEnum.JOBLIB,
        symbolic_format=ModelFormatEnum.JOBLIB,
        metadata={
            **base_metadata.metadata,
            "available_surrogate_depths": sorted(
                int(depth) for depth in dashboard_model.tree_models
            ),
            "surrogate_metrics_by_depth": {
                str(depth): tree.metrics
                for depth, tree in dashboard_model.tree_models.items()
            },
            "symbolic_metrics": dict(symbolic_model.metrics),
            "symbolic_feature_names": list(symbolic_model.used_feature_names),
            "symbolic_complexity": int(symbolic_model.complexity),
        },
    )


def _backup_saved_models(bundle_dir: Path) -> Path | None:
    if not bundle_dir.exists():
        return None
    if not any(bundle_dir.iterdir()):
        return None
    backup_dir = bundle_dir.parent / (
        f"{bundle_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    bundle_dir.rename(backup_dir)
    return backup_dir
