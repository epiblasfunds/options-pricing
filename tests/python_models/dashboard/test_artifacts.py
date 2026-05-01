import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.enums.volatility_model_enums import ModelFormatEnum
from src.python_models.dashboard.artifacts import DashboardBundleMetadata
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.python_models.dashboard.artifacts import SurrogateTreeModel


def test_dashboard_bundle_metadata_roundtrip(tmp_path):
    metadata = DashboardBundleMetadata(
        model_id="rf",
        name="Random Forest",
        path=tmp_path / "rf",
        format=ModelFormatEnum.EXPLAINABLE_MODEL,
        metadata={"builder": "src.model2dashboard.run_pipeline"},
    )

    metadata.save()
    loaded = DashboardBundleMetadata.load(tmp_path / "rf")

    assert loaded.to_dict()["model_id"] == "rf"
    assert loaded.format is ModelFormatEnum.EXPLAINABLE_MODEL


def test_stored_shap_explanation_supports_waterfall_predictions_and_selection():
    stored = StoredShapExplanation(
        method="shap",
        feature_names=["a", "b"],
        index=[7, 8],
        values=[[0.2, -0.1], [0.1, 0.4]],
        base_values=[0.3, 0.1],
        data=[[1.0, 2.0], [3.0, 4.0]],
        display_data=None,
        predictions=[0.4, 0.6],
        mean_abs_shap={"a": 0.15},
    )

    selected = stored.select(8)

    assert stored.waterfall_predictions().tolist() == [0.4, 0.6]
    assert selected.index == [8]
    with pytest.raises(ValueError):
        StoredShapExplanation(
            method="shap",
            feature_names=["a"],
            index=[1, 2],
            values=[[0.1], [0.2]],
            base_values=[0.1, 0.2, 0.3],
            data=[[1.0], [2.0]],
            display_data=None,
            predictions=[0.2, 0.3],
            mean_abs_shap={},
        ).waterfall_predictions()


def test_surrogate_tree_model_save_and_load_roundtrip(tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 1.0]})
    y = pd.Series([0.1, 0.2, 0.3])
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    tree = SurrogateTreeModel(
        model=model,
        feature_importances=pd.Series({"a": 1.0, "b": 0.0}),
        tree_depth=model.get_depth(),
        n_leaves=model.get_n_leaves(),
        text_rules="rules",
        interpretation="interpretation",
        fidelity_frame=pd.DataFrame(
            {"model_prediction": [0.1], "surrogate_prediction": [0.1]}
        ),
        feature_names=["a", "b"],
        metrics={"rmse": 0.0},
    )

    tree.save(tmp_path / "tree")
    loaded = SurrogateTreeModel.load(tmp_path / "tree")

    assert loaded.tree_depth == tree.tree_depth
    assert loaded.n_leaves == tree.n_leaves
    assert loaded.feature_names == ["a", "b"]
    assert loaded.metrics == {"rmse": 0.0}
