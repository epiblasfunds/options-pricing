import pandas as pd

from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


def test_symbolic_regressor_model_roundtrip_and_predict(tmp_path):
    model = SymbolicRegressorModel(
        equation="Rate + StrikePrice",
        sympy_expression="Rate + StrikePrice",
        latex_expression="Rate + StrikePrice",
        interpretation="simple",
        feature_names=["Rate", "StrikePrice"],
        used_feature_names=["Rate", "StrikePrice"],
        complexity=2,
        model_selection="best",
        metrics={"rmse": 0.1},
        candidate_equations=pd.DataFrame({"equation": ["Rate + StrikePrice"]}),
        fidelity_frame=pd.DataFrame(
            {"model_prediction": [1.0], "symbolic_prediction": [1.0]}
        ),
    )

    model.save(tmp_path / "symbolic")
    loaded = SymbolicRegressorModel.load(tmp_path / "symbolic")
    prediction = loaded.predict(pd.DataFrame({"Rate": [0.1], "StrikePrice": [9.0]}))

    assert loaded.metrics == {"rmse": 0.1}
    assert prediction.tolist() == [9.1]
