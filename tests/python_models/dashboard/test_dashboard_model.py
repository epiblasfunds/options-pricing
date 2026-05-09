import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.python_models.dashboard.artifacts import DiagnosisArtifact
from src.python_models.dashboard.artifacts import ManualApiStubResponse
from src.python_models.dashboard.artifacts import StoredNeighborsProjectionPca
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.python_models.dashboard.artifacts import SurrogateTreeModel
from src.python_models.dashboard.dashboard_model import DashboardModel
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


def _tree_model():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 1.0]})
    y = pd.Series([0.1, 0.2, 0.3])
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    return SurrogateTreeModel(
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


def _dashboard_model():
    shap_values = StoredShapExplanation(
        method="shap",
        feature_names=["StrikePrice"],
        index=[5],
        values=[[0.2]],
        base_values=[0.1],
        data=[[9000.0]],
        display_data=None,
        predictions=[0.3],
        mean_abs_shap={"StrikePrice": 0.2},
    )
    return DashboardModel(
        model_id="rf",
        model_name="Random Forest",
        metadata={"a": 1},
        dataset_frame=pd.DataFrame(
            {"PredictedVolatility": [0.3], "OptionType": ["C"]},
            index=[5],
        ),
        raw_feature_names=["OptionType"],
        transformed_feature_names=["isCall"],
        training_reference_frame=pd.DataFrame(
            {
                "PredictedVolatility": [0.11],
                "ImpliedVolatility": [0.12],
                "OptionType": ["P"],
            },
            index=[17],
        ),
        neighbors_projection_pca=StoredNeighborsProjectionPca(
            feature_names=["isCall"],
            fill_values={"isCall": 0.5},
            scale_values={"isCall": 0.5},
            components=[[1.0]],
            explained_variance_ratio=[1.0],
        ),
        tree_models={2: _tree_model()},
        symbolic_model=SymbolicRegressorModel(
            equation="Rate",
            sympy_expression="Rate",
            latex_expression="Rate",
            interpretation="simple",
            feature_names=["Rate"],
            used_feature_names=["Rate"],
            complexity=1,
            model_selection="best",
        ),
        sample_indices=[5],
        behaviour_anchor_indices=[5],
        global_shap=shap_values,
        local_shap=shap_values,
        neighbors_frame=pd.DataFrame(
            {"sample_index": [5], "neighbor_index": [17], "distance": [0.0]}
        ),
        surfaces_frame=pd.DataFrame(
            {"anchor_index": [5], "Moneyness": [1.0], "PredictedVolatility": [0.3]}
        ),
        ice_frame=pd.DataFrame({"feature_name": ["Rate"], "value": [0.3]}),
        ale_frame=pd.DataFrame({"feature_name": ["Rate"], "value": [0.1]}),
        diagnosis=DiagnosisArtifact(
            metrics={"rmse": 0.1},
            plot_frame=pd.DataFrame({"x": [1]}),
            error_heatmap=pd.DataFrame({"y": [2]}),
            financial_warnings=["none"],
        ),
        manual_api_stub=ManualApiStubResponse(
            prediction=0.3,
            summary="ready",
            reference_sample_index=5,
        ),
    )


def test_dashboard_model_save_load_and_accessors(tmp_path):
    dashboard_model = _dashboard_model()

    dashboard_model.save(tmp_path / "bundle")
    loaded = DashboardModel.load(tmp_path / "bundle")

    assert loaded.model_id == "rf"
    assert loaded.predictions_for_indices([5]).to_dict() == {5: 0.3}
    assert loaded.local_shap_for_index(5).index == [5]
    assert loaded.training_reference_frame.index.tolist() == [17]
    assert loaded.neighbors_projection_pca is not None
    assert loaded.neighbors_projection_pca.feature_names == ["isCall"]
    assert loaded.neighbors_for_index(5)["distance"].tolist() == [0.0]
    assert loaded.neighbors_for_index(5).index.tolist() == [17]
    assert not loaded.surface_for_anchor(5).empty
    assert not loaded.ice_for_feature("Rate").empty
    assert not loaded.ale_for_feature("Rate").empty
    assert 2 in loaded.tree_models
    assert loaded.symbolic_model is not None


def test_dashboard_model_uses_safe_defaults_when_optional_artifacts_are_missing():
    model = DashboardModel(
        model_id="rf",
        model_name="Random Forest",
        metadata={},
        dataset_frame=pd.DataFrame({"PredictedVolatility": [0.1]}, index=[1]),
        raw_feature_names=[],
        transformed_feature_names=[],
    )

    assert model.diagnosis.metrics == {}
    assert model.manual_api_stub.reference_sample_index is None
    assert model.neighbors_for_index(1).empty
    with pytest.raises(KeyError):
        model.local_shap_for_index(1)
