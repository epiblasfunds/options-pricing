import typing as t
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.enums.volatility_model_enums import ModelFormatEnum
from src.functionalities.dashboard_models.runtime import predict_raw_frame
from src.python_models.explainable_artifacts import SingleModelMetadata
from src.python_models.explainable_artifacts import SurrogateTreeModel
from src.volatility_models import build_feature_frame_from_trades

METRICS_REGISTRY = build_metrics_registry()


def load_dashboard_tree_models(
    bundle_path: Path,
    payload: dict[str, t.Any],
) -> dict[int, SurrogateTreeModel]:
    tree_models: dict[int, SurrogateTreeModel] = {}
    trees_root = bundle_path / "tree_models"
    if not trees_root.exists():
        return tree_models
    for tree_path in sorted(trees_root.iterdir(), key=lambda path: path.name):
        if not tree_path.is_dir():
            continue
        depth = int(tree_path.name.removeprefix("depth_"))
        tree_models[depth] = SurrogateTreeModel.load(
            metadata=SingleModelMetadata(
                model_id=f"{payload['model_id']}_dashboard_tree_{depth}",
                name=f"{payload['model_name']}_dashboard_tree_{depth}",
                path=tree_path,
                format=ModelFormatEnum.JOBLIB,
                metadata={},
            )
        )
    return tree_models


def build_surrogate_tree_models(
    *,
    trained_model: t.Any,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    model_input_features: list[str],
    surrogate_depths: tuple[int, ...],
    sample_frame: t.Callable[..., pd.DataFrame],
    transform_feature_frame: t.Callable[..., pd.DataFrame],
) -> dict[int, SurrogateTreeModel]:
    sampled = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.surrogate_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    sampled_features = build_feature_frame_from_trades(raw_frame.loc[sampled.index])
    transformed = transform_feature_frame(
        sampled_features,
        trained_model.preprocessor,
        model_input_features,
    )
    predictions = pd.Series(
        predict_raw_frame(trained_model, raw_frame.loc[sampled.index]),
        index=sampled.index,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        transformed,
        predictions,
        test_size=0.2,
        random_state=config.dashboard_models_config.random_state,
    )
    tree_models: dict[int, SurrogateTreeModel] = {}
    for depth in surrogate_depths:
        surrogate = DecisionTreeRegressor(
            max_depth=int(depth),
            min_samples_leaf=config.dashboard_models_config.surrogate_min_samples_leaf,
            random_state=config.dashboard_models_config.random_state,
        )
        surrogate.fit(X_train, y_train)
        y_pred = pd.Series(surrogate.predict(X_test), index=y_test.index)
        metrics = METRICS_REGISTRY.compute_metrics(
            y_test.reset_index(drop=True),
            y_pred.reset_index(drop=True),
            config.dashboard_models_config.error_metrics,
        )
        importances = pd.Series(
            surrogate.feature_importances_,
            index=model_input_features,
        ).sort_values(ascending=False)
        top_features = importances[importances > 0].head(3).index.tolist()
        tree_models[int(depth)] = SurrogateTreeModel(
            model=surrogate,
            feature_importances=importances,
            tree_depth=surrogate.get_depth(),
            n_leaves=surrogate.get_n_leaves(),
            text_rules=export_text(surrogate, feature_names=model_input_features),
            interpretation=(
                "The surrogate approximates the trained volatility model "
                f"with RMSE {metrics['rmse']:.4f} at max depth {int(depth)}. "
                f"The dominant decision logic is driven by "
                f"{', '.join(top_features) or 'no features'}."
            ),
            fidelity_frame=pd.DataFrame(
                {
                    "model_prediction": y_test.reset_index(drop=True),
                    "surrogate_prediction": y_pred.reset_index(drop=True),
                }
            ),
            feature_names=list(model_input_features),
            metrics={str(name): float(value) for name, value in metrics.items()},
        )
    return tree_models
