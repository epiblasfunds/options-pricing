import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config.config import (
    DASHBOARD_SAVED_MODELS_DIR_PATH,
    VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    VOLATILITY_TRAINED_MODELS_DIR_PATH,
    config,
)
from src.enums.volatility_model_enums import ModelFormatEnum
from src.enums.volatility_model_enums.model_name import display_model_name
from src.model2dashboard.artifact_builders import build_dashboard_artifacts
from src.model2dashboard.features import (
    ANALYSIS_FEATURE_NAMES,
    EXPLAINABILITY_FEATURE_NAMES,
    MODEL_INPUT_FEATURE_NAMES,
    TARGET_COLUMN,
    VISIBLE_RAW_INPUT_FEATURE_NAMES,
    load_train_trade_frame,
    load_test_trade_frame,
)
from src.model2dashboard.model_io import (
    _resolve_retrained_metadata_path,
    discover_model_families,
    load_training_runtime,
)
from src.python_models.dashboard.artifacts import DashboardBundleMetadata
from src.python_models.dashboard.dashboard_model import DashboardModel


@dataclass(frozen=True)
class ExportedDashboardBundle:
    model_id: str
    bundle_path: Path
    root_metadata_path: Path
    dashboard_metadata_path: Path


def run_pipeline(
    *,
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
    retrained_metadata_dir: Path = VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
    overwrite: bool = True,
) -> list[ExportedDashboardBundle]:
    return build_all_explainable_models(
        trained_models_dir=trained_models_dir,
        retrained_metadata_dir=retrained_metadata_dir,
        bundle_dir=bundle_dir,
        overwrite=overwrite,
    )


def build_all_explainable_models(
    *,
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
    retrained_metadata_dir: Path = VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
    overwrite: bool = True,
) -> list[ExportedDashboardBundle]:
    raw_train_frame = load_train_trade_frame(verbose=False)
    raw_test_frame = load_test_trade_frame(verbose=False)
    exported: list[ExportedDashboardBundle] = []
    for family_name in discover_model_families(trained_models_dir):
        exported.append(
            build_explainable_model(
                family_name=family_name,
                raw_train_frame=raw_train_frame,
                raw_test_frame=raw_test_frame,
                trained_models_dir=trained_models_dir,
                retrained_metadata_dir=retrained_metadata_dir,
                bundle_dir=bundle_dir,
                overwrite=overwrite,
            )
        )
    return exported


def build_explainable_model(
    *,
    family_name: str,
    raw_train_frame=None,
    raw_test_frame=None,
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
    retrained_metadata_dir: Path = VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    bundle_dir: Path = DASHBOARD_SAVED_MODELS_DIR_PATH,
    overwrite: bool = True,
) -> ExportedDashboardBundle:
    runtime = load_training_runtime(
        family_name=family_name,
        trained_models_dir=trained_models_dir,
        retrained_metadata_dir=retrained_metadata_dir,
    )
    test_frame = (
        raw_test_frame.copy()
        if raw_test_frame is not None
        else load_test_trade_frame(verbose=False)
    ).reset_index(drop=True)
    train_frame = (
        raw_train_frame.copy()
        if raw_train_frame is not None
        else load_train_trade_frame(verbose=False)
    ).reset_index(drop=True)
    artifacts = build_dashboard_artifacts(
        runtime=runtime,
        raw_train_frame=train_frame,
        raw_test_frame=test_frame,
    )

    bundle_path = bundle_dir / family_name
    if overwrite and bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    metadata_payload = _metadata_payload(runtime, artifacts, retrained_metadata_dir)
    bundle_metadata = DashboardBundleMetadata(
        model_id=family_name,
        name=_display_model_name(family_name),
        path=bundle_path,
        format=ModelFormatEnum.EXPLAINABLE_MODEL,
        metadata=metadata_payload,
    )
    dashboard_model = DashboardModel(
        model_id=family_name,
        model_name=_display_model_name(family_name),
        metadata=metadata_payload,
        dataset_frame=artifacts["dataset_frame"],
        training_reference_frame=artifacts["training_reference_frame"],
        neighbors_projection_pca=artifacts["neighbors_projection_pca"],
        raw_feature_names=list(VISIBLE_RAW_INPUT_FEATURE_NAMES),
        transformed_feature_names=list(runtime.model_input_features),
        tree_models=artifacts["tree_models"],
        symbolic_model=artifacts["symbolic_model"],
        sample_indices=artifacts["sample_indices"],
        behaviour_anchor_indices=artifacts["behaviour_anchor_indices"],
        global_shap=artifacts["global_shap"],
        local_shap=artifacts["local_shap"],
        neighbors_frame=artifacts["neighbors_frame"],
        surfaces_frame=artifacts["surfaces_frame"],
        ice_frame=artifacts["ice_frame"],
        ale_frame=artifacts["ale_frame"],
        diagnosis=artifacts["diagnosis"],
        manual_api_stub=artifacts["manual_api_stub"],
    )
    bundle_metadata.save(bundle_path)
    dashboard_model.save(bundle_path)
    return ExportedDashboardBundle(
        model_id=family_name,
        bundle_path=bundle_path,
        root_metadata_path=DashboardBundleMetadata.get_root_metadata_path(bundle_path),
        dashboard_metadata_path=DashboardModel.get_metadata_path(
            DashboardModel.get_root_path(bundle_path)
        ),
    )


def _metadata_payload(runtime, artifacts: dict, retrained_metadata_dir: Path) -> dict:
    final_test_metadata_path = _resolve_retrained_metadata_path(
        retrained_metadata_dir=retrained_metadata_dir,
        family_name=runtime.family_name,
        phase="final_test",
    )
    train_val_metadata_path = _resolve_retrained_metadata_path(
        retrained_metadata_dir=retrained_metadata_dir,
        family_name=runtime.family_name,
        phase="train_val",
    )
    symbolic_model = artifacts.get("symbolic_model")
    tree_models = artifacts.get("tree_models", {})
    return {
        "model_input_features": list(MODEL_INPUT_FEATURE_NAMES),
        "transformed_feature_names": list(MODEL_INPUT_FEATURE_NAMES),
        "raw_feature_names": list(VISIBLE_RAW_INPUT_FEATURE_NAMES),
        "explainability_feature_names": list(EXPLAINABILITY_FEATURE_NAMES),
        "analysis_feature_names": list(ANALYSIS_FEATURE_NAMES),
        "target_column": TARGET_COLUMN,
        "error_metrics": list(config.dashboard_models_config.error_metrics),
        "trained_model_path": runtime.model_path.as_posix(),
        "scaler_path": (
            runtime.scaler_path.as_posix() if runtime.scaler_path is not None else None
        ),
        "final_test_metadata_path": final_test_metadata_path.as_posix(),
        "train_val_metadata_path": (
            train_val_metadata_path.as_posix()
            if train_val_metadata_path.exists()
            else None
        ),
        "model_params": runtime.final_test_metadata.get("model_params", {}),
        "result_metrics": runtime.final_test_metadata.get("result_metrics", {}),
        "train_val_result_metrics": runtime.train_val_metadata.get(
            "result_metrics", {}
        ),
        "training_information": {
            "best_iteration": runtime.final_test_metadata.get(
                "training_information", {}
            ).get("best_iteration"),
            "best_score": runtime.final_test_metadata.get(
                "training_information", {}
            ).get("best_score"),
            "epoch_history": runtime.final_test_metadata.get(
                "training_information", {}
            ).get("epoch_history", {}),
        },
        "available_surrogate_depths": sorted(int(depth) for depth in tree_models),
        "surrogate_metrics_by_depth": {
            str(depth): tree.metrics for depth, tree in tree_models.items()
        },
        "symbolic_metrics": (
            dict(symbolic_model.metrics) if symbolic_model is not None else {}
        ),
        "symbolic_feature_names": (
            list(symbolic_model.used_feature_names)
            if symbolic_model is not None
            else []
        ),
        "symbolic_complexity": (
            int(symbolic_model.complexity) if symbolic_model is not None else None
        ),
        "data_split": "test",
        "data_source": "TrainingDataHandler.load_splitted_data()[-1]",
        "neighbor_reference_split": "train",
        "neighbor_reference_source": "TrainingDataHandler.load_splitted_data()[0]",
        "builder": "src.model2dashboard.run_pipeline",
    }


def _display_model_name(model_id: str) -> str:
    return display_model_name(model_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dashboard explainable artifacts from trained volatility models.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Model family name to process. If omitted, all discovered families are processed.",
    )
    args = parser.parse_args()

    if args.family:
        exported = [build_explainable_model(family_name=args.family)]
    else:
        exported = run_pipeline()

    for bundle in exported:
        print(f"{bundle.model_id}: {bundle.bundle_path}")


if __name__ == "__main__":
    main()
